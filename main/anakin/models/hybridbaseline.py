import os
import torch
import numpy as np
import torch.nn as nn
from typing import Dict
from anakin.opt import cfg
from collections import OrderedDict
from anakin.utils.builder import HEAD, MODEL, build_backbone, build_head, build_model
from anakin.utils.logger import logger
from anakin.utils.misc import enable_lower_param, param_size
from ..utils.transform import batch_uvd2xyz, compute_rotation_matrix_from_ortho6d
from anakin.datasets.hoquery import Queries
from anakin.postprocess.iknet import FittingUnit
from anakin.models.physics_discriminator import PhysicsMeshMLP


@MODEL.register_module
class HybridBaseline(nn.Module):
    @enable_lower_param
    def __init__(self, **cfg):
        super(HybridBaseline, self).__init__()

        if cfg["BACKBONE"]["PRETRAINED"] and cfg["PRETRAINED"]:
            logger.warning(
                f"{type(self).__name__}'s backbone {cfg['BACKBONE']['TYPE']} weights will be rewritten by {cfg['PRETRAINED']}"
            )

        self.center_idx = cfg["DATA_PRESET"].get("CENTER_IDX", 9)
        self.inp_res = cfg["DATA_PRESET"]["IMAGE_SIZE"]
        self.backbone = build_backbone(cfg["BACKBONE"], default_args=cfg["DATA_PRESET"])
        self.hybrid_head = build_head(cfg["HYBRID_HEAD"], default_args=cfg["DATA_PRESET"])  # IntegralDeconvHead
        self.box_head = build_model(cfg["BOX_HEAD"], default_args=cfg["DATA_PRESET"])  # box_head, mlp
        self.discriminator = PhysicsMeshMLP()
        self.fitting_unit = FittingUnit(aquire_gt=True) # generate GT fitting, note fitting unit is not a subclass of nn.Module so it's not trainable
        self.face = np.loadtxt("assets/postprocess/hand_close.npy").astype(np.int32) # for renderer
        self.init_weights(pretrained=cfg["PRETRAINED"])
        logger.info(f"{type(self).__name__} uses center_idx {self.center_idx}")
        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")

    def forward(self, inputs: Dict):
        batch_size, n_channel, height, width = inputs["image"].shape
        features = self.backbone(image=inputs["image"])
        pose_results = self.hybrid_head(feature=features["res_layer4"])
        box_rot_6d = self.box_head(features["res_layer4_mean"])

        pose_3d_abs = batch_uvd2xyz( # conversion has used root joint
            uvd=pose_results["kp3d"],
            root_joint=inputs[Queries.ROOT_JOINT],
            intr=inputs[Queries.CAM_INTR],
            inp_res=self.inp_res,
        )  # TENSOR (B, 22, 3)
        joints_3d_abs = pose_3d_abs[:, 0:21, :]  # TENSOR[B, 21, 3]
        optimized_pose, optimized_v, _ = self.fitting_unit(inputs, joints_3d_abs.detach()) # TENSOR[B, 778, 3]
        boxroot_3d_abs = pose_3d_abs[:, 21:22, :]  # TENSOR[B, 1, 3]
        corners_can_3d = inputs[Queries.CORNERS_CAN].to(boxroot_3d_abs.device)  # TENSOR[B, 8, 3]
        box_rot_rotmat = compute_rotation_matrix_from_ortho6d(box_rot_6d)

        corners_3d_abs = torch.matmul(box_rot_rotmat, corners_can_3d.permute(0, 2, 1)).permute(0, 2, 1) + boxroot_3d_abs
        obj_verts_3d_abs = torch.matmul(box_rot_rotmat, inputs[Queries.OBJ_VERTS_CAN].permute(0, 2, 1)).permute(0, 2, 1) + boxroot_3d_abs
        root_joint = joints_3d_abs[:, self.center_idx, :]  # (B, 3)
        joints_confd = pose_results["kp3d_confd"][:, :21]  # (B, 21)

        cam_intr = inputs[Queries.CAM_INTR].to(corners_3d_abs.device)  # [B, 3, 3]
        corners_2d = torch.matmul(cam_intr, corners_3d_abs.permute(0, 2, 1)).permute(0, 2, 1)  # [B, 8, 3], homogeneous
        corners_2d = corners_2d[:, :, 0:2] / corners_2d[:, :, 2:3]  # [B, 8, 2], 2d
        corners_2d[:, :, 0] /= width
        corners_2d[:, :, 1] /= height
        corners_2d_uvd = torch.cat(
            (corners_2d, torch.zeros_like(corners_2d[:, :, 0:1])), dim=2
        )  # [B, 8, 3], where[:, :, 2] all zeros
        final_2d_uvd = torch.cat(
            (pose_results["kp3d"][:, 0:21, :], corners_2d_uvd, pose_results["kp3d"][:, 21:22, :]), dim=1
        )

        return {
            # ↓ absolute value feed to criterion
            "joints_3d_abs": joints_3d_abs,
            "corners_3d_abs": corners_3d_abs,
            # ↓ root relative valus feed to evaluator
            "joints_3d": joints_3d_abs - root_joint.unsqueeze(1),
            "corners_3d": corners_3d_abs - root_joint.unsqueeze(1),
            "2d_uvd": final_2d_uvd,
            "boxroot_3d_abs": boxroot_3d_abs,
            "box_rot_rotmat": box_rot_rotmat,
            "hand_verts_3d_abs": optimized_v,
            "obj_center": boxroot_3d_abs,
            "obj_pred_rot": box_rot_rotmat,
            "discriminator": self.discriminator,
            "root_joint": root_joint,
            "obj_verts_3d_abs": obj_verts_3d_abs,
            "hand_face": self.face
        }
    
    def init_weights(self, pretrained=""):
        if pretrained == "":
            logger.warning(f"=> Init {type(self).__name__} weights in backbone and head")
            """
            Add init for other modules if has
            ...
            """
        elif os.path.isfile(pretrained):
            # pretrained_state_dict = torch.load(pretrained)
            logger.info(f"=> Loading {type(self).__name__} pretrained model from: {pretrained}")
            # self.load_state_dict(pretrained_state_dict, strict=False)
            # when loading pretrained, ignore discrinimator
            mlp_weights = cfg["ARCH"].get("MLP_PRETRAIN", "")
            
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict_old = checkpoint
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith("_model_list.0.") and (not cfg["ARCH"]["MLP_ONLY"] or 'discriminator' in key):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[14:]] = state_dict_old[key]  # delete "module." (in nn.parallel and ddp)
                    elif cfg["ARCH"]["MLP_ONLY"]:
                        continue
                    else:
                        state_dict[key] = state_dict_old[key]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict_old = checkpoint["state_dict"]
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith("module."):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]  # delete "module." (in nn.parallel and ddp)
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                logger.error(f"=> No state_dict found in checkpoint file {pretrained}")
                raise RuntimeError()
            if cfg["ARCH"]["MLP_ONLY"]:
                self.load_state_dict(state_dict, strict=False)
            else:
                self.load_state_dict(state_dict, strict=False) # the pretrained model will miss some keys in discriminator
            logger.info(f"Successfully loaded {type(self).__name__} pretrained model from: {pretrained}")

            if mlp_weights:
                logger.info(f"=> Loading additional MLP weights from {mlp_weights}")
                mlp_checkpoint = torch.load(mlp_weights)
                state_dict_old = mlp_checkpoint
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith("_model_list.0.") and 'discriminator' in key:
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[14:]] = state_dict_old[key]  # delete "module." (in nn.parallel and ddp)

                self.load_state_dict(state_dict, strict=False)

        else:
            logger.error(f"=> No {type(self).__name__} checkpoints file found in {pretrained}")
            raise FileNotFoundError()
