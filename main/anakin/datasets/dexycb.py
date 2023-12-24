import hashlib
import json
import os
import pickle
import torch
import trimesh
import numpy as np
import pytorch3d.io
from PIL import Image
from copy import deepcopy
from typing import List, Tuple
import manopth.manolayer as manolayer
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
from dex_ycb_toolkit.factory import get_dataset
from manotorch.manolayer import MANOOutput, ManoLayer # the manotorch implements A-MANO
from scipy.spatial.distance import cdist
from anakin.datasets.hodata import HOdata
from anakin.utils import transform
from anakin.utils.builder import DATASET
from anakin.utils.etqdm import etqdm
from anakin.utils.logger import logger
from anakin.utils.misc import enable_lower_param, CONST
from anakin.utils.transform import batch_ref_bone_len
from pytorch3d.transforms import random_quaternions, quaternion_to_axis_angle


class mesh:
    # a fake wrapper for trimesh Trimesh
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces


@DATASET.register_module
class DexYCB(HOdata):

    @enable_lower_param
    def __init__(self, **cfg):
        super().__init__(**cfg)

        self.split_mode = cfg["SPLIT_MODE"]
        self.use_left_hand = cfg["USE_LEFT_HAND"]
        self.filter_invisible_hand = cfg["FILTER_INVISIBLE_HAND"]
        self.dataset = None
        self.filter_gt = False

        self.dexycb_mano_right = ManoLayer(
            flat_hand_mean=False,
            side="right",
            mano_assets_root="assets/mano_v1_2",
            use_pca=True,
            ncomps=45,
        )

        self.standard_mano_layer_pca = manolayer.ManoLayer(flat_hand_mean=False,
                         ncomps=45,
                         side="right",
                         mano_root='assets/mano_v1_2/models',
                         use_pca=True)

        self.standard_mano_layer_axis = manolayer.ManoLayer(
                              joint_rot_mode="axisang",
                              use_pca=False,
                              mano_root='assets/mano_v1_2/models',
                              center_idx=None,
                              flat_hand_mean=True,
                              side="right",
                              )

        self.dexycb_mano_left = (ManoLayer(
            flat_hand_mean=False,
            side="left",
            mano_assets_root="assets/mano_v1_2",
            use_pca=True,
            ncomps=45,
        ) if self.use_left_hand else None)

        self.load_dataset()

    def _preload(self):
        self.name = "DexYCB"
        self.root = os.path.join(self.data_root, self.name)
        os.environ["DEX_YCB_DIR"] = self.root

        self.cache_identifier_dict = {
            "filter_thresh": float(self.filter_thresh),
            "data_split": self.data_split,
            "split_mode": self.split_mode,
            "fliter_no_contact": self.filter_no_contact,
            "use_left_hand": self.use_left_hand,
            "filter_invisible_hand": self.filter_invisible_hand,
        }
        self.cache_identifier_raw = json.dumps(self.cache_identifier_dict, sort_keys=True)
        self.cache_identifier = hashlib.md5(self.cache_identifier_raw.encode("ascii")).hexdigest()
        if self.data_split == 'test':
            self.cache_path = os.path.join("common", "cache", self.name, "{}.pkl".format('9fb79af8c22cdc2b8cc7cda1142caca3'))
        else:
            self.cache_path = os.path.join("common", "cache", self.name, "{}.pkl".format('631ddeca7bece94076718d32d0a9049a'))

    def load_dataset(self):
        self._preload()
        cache_folder = os.path.dirname(self.cache_path)
        os.makedirs(cache_folder, exist_ok=True)

        dexycb_name = f"{self.split_mode}_{self.data_split}"
        logger.info(f"DexYCB use split: {dexycb_name}")
        self.dataset: DexYCBDataset = get_dataset(dexycb_name)
        self.raw_size = (640, 480)
        self.load_obj_mesh()

        if self.use_left_hand and not self.filter_no_contact and not self.filter_invisible_hand:
            self.sample_idxs = list(range(len(self.dataset)))
        else:
            if self.use_cache and os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as p_f:
                    self.sample_idxs = pickle.load(p_f)
                logger.info(f"Loaded cache for {self.name}_{self.data_split}_{self.split_mode} from {self.cache_path}")
            else:
                self.sample_idxs = []

                file_idxs = open(f'common/cache/DexYCB/valid.txt', 'r').readlines()
                file_idxs = [s.strip() for s in file_idxs]

                for i, sample in enumerate(etqdm(self.dataset)):
                    if not self.use_left_hand and sample["mano_side"] == "left":
                        continue
                    if (self.filter_invisible_hand or self.filter_no_contact) and np.all(self.get_joints_2d(i) == -1.0):
                        continue

                    if self.filter_no_contact and (cdist(self.get_obj_verts_transf(i), self.get_joints_3d(i)).min() *
                                                   1000.0 > self.filter_thresh):
                        continue

                    if self.get_image_path(i)[14:] not in file_idxs:
                        continue

                    self.sample_idxs.append(i)

        logger.info(f"loaded {len(self.sample_idxs)} valid samples")

    def load_obj_mesh(self):
        self.obj_raw_meshes = {}
        self.obj_name = {}
        for obj_idx, obj_file in self.dataset.obj_file.items():
            obj_file = obj_file.replace("models", "models_resample_mashlab").replace('DexYCB/', '')
            obj_file = obj_file.replace("textured_simple.obj", "textured_simple_2000.obj")
            verts, faces, _ = pytorch3d.io.load_obj(obj_file)
            # obj_mesh = trimesh.load(obj_file, process=False)
            obj_mesh = mesh(vertices=verts.numpy(), faces=faces.verts_idx.int().numpy())
            assert obj_mesh.vertices.shape[0] == 1000
            obj_name = obj_file.split('/')[-2]
            self.obj_name[obj_idx] = obj_name
            self.obj_raw_meshes[obj_idx] = obj_mesh

    def __len__(self):
        return len(self.sample_idxs)

    def get_sample_idxs(self) -> List[int]:
        return self.sample_idxs

    # @lru_cache(maxsize=None)
    def get_label(self, label_file: str):
        return np.load(label_file)

    def get_cam_intr(self, idx):
        sample = self.dataset[idx]
        return np.array(
            [
                [sample["intrinsics"]["fx"], 0.0, sample["intrinsics"]["ppx"]],
                [0.0, sample["intrinsics"]["fy"], sample["intrinsics"]["ppy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def get_center_scale_wrt_bbox(self, idx):
        if self.require_full_image:
            full_width, full_height = self.raw_size[0], self.raw_size[1]
            center = np.array((full_width / 2, full_height / 2))
            scale = full_width
            return center, scale

        if self.crop_model == "hand_obj":
            joints2d = self.get_joints_2d(idx)  # (21, 2)
            corners_2d = self.get_corners_2d(idx)  # (8, 2)
            all2d = np.concatenate([joints2d, corners_2d], axis=0)  # (29, 2)
            center = HOdata.get_annot_center(all2d)
            scale = HOdata.get_annot_scale(all2d)
            return center, scale
        elif self.crop_model == "hand":
            joints2d = self.get_joints_2d(idx)  # (21, 2)
            center = HOdata.get_annot_center(joints2d)
            scale = HOdata.get_annot_scale(joints2d)
            return center, scale
        else:
            raise NotImplementedError()

    def get_corners_vis(self, idx):
        if self.data_split not in ["train", "trainval"]:
            corners_vis = np.ones(self.ncorners)
        else:
            corners_2d = self.get_corners_2d(idx)
            corners_vis = ((corners_2d[:, 0] >= 0) &
                           (corners_2d[:, 0] < self.raw_size[0])) & ((corners_2d[:, 1] >= 0) &
                                                                     (corners_2d[:, 1] < self.raw_size[1]))

        return corners_vis.astype(np.float32)

    def get_corners_2d(self, idx):
        corners_3d = self.get_corners_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return HOdata.persp_project(corners_3d, cam_intr)

    def get_corners_3d(self, idx):
        transf = self.get_obj_transf(idx)
        R, t = transf[:3, :3], transf[:3, [3]]
        corners_can = self.get_corners_can(idx)
        corners = (R @ corners_can.T + t).T
        return corners

    def get_obj_name(self, idx):
        sample = self.dataset[idx]
        grasp_ycb_idx = sample["ycb_ids"][sample["ycb_grasp_ind"]]
        obj_name = self.obj_name[grasp_ycb_idx]
        return obj_name

    def get_corners_can(self, idx):
        sample = self.dataset[idx]
        grasp_ycb_idx = sample["ycb_ids"][sample["ycb_grasp_ind"]]
        obj_mesh = self.obj_raw_meshes[grasp_ycb_idx]
        # NOTE: verts_can = verts - bbox_center
        _, offset, _ = transform.center_vert_bbox(obj_mesh.vertices, scale=False)  # !! CENTERED HERE
        corners = trimesh.bounds.corners(trimesh.Trimesh(obj_mesh.vertices, obj_mesh.faces).bounds)
        corners = corners - offset
        return np.asfarray(corners, dtype=np.float32)

    def get_hand_faces(self, idx):
        sample = self.dataset[idx]
        mano_layer = self.dexycb_mano_left if sample["mano_side"] == "left" else self.dexycb_mano_right
        faces = np.array(mano_layer.th_faces).astype(np.long)
        return faces

    def get_hand_verts_2d(self, idx):
        verts_3d = self.get_hand_verts_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return HOdata.persp_project(verts_3d, cam_intr)
    
    def get_random_hand_verts_3d(self, idx):
        sample = self.dataset[idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        pose_m = torch.from_numpy(label["pose_m"]) + torch.randn(label["pose_m"].shape)
        shape = (torch.tensor(sample["mano_betas"]) + torch.randn(10)).unsqueeze(0)
        mano_layer = self.dexycb_mano_left if sample["mano_side"] == "left" else self.dexycb_mano_right
        mano_out: MANOOutput = mano_layer(pose_m[:, :48], shape)
        hand_verts = mano_out.verts + pose_m[:, 48:]
        return hand_verts.squeeze(0).numpy().astype(np.float32)

    def get_hand_verts_3d(self, idx):
        sample = self.dataset[idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        pose_m = torch.from_numpy(label["pose_m"])
        shape = torch.tensor(sample["mano_betas"]).unsqueeze(0)
        mano_layer = self.dexycb_mano_left if sample["mano_side"] == "left" else self.dexycb_mano_right
        mano_out: MANOOutput = mano_layer(pose_m[:, :48], shape)
        hand_verts = mano_out.verts + pose_m[:, 48:]
        return hand_verts.squeeze(0).numpy().astype(np.float32)
    
    def get_hand_verts_3d_can(self, idx):
        """
        This just give the hand_verts without adding the root
        """
        sample = self.dataset[idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        pose_m = torch.from_numpy(label["pose_m"])
        shape = torch.tensor(sample["mano_betas"]).unsqueeze(0)
        mano_layer = self.dexycb_mano_left if sample["mano_side"] == "left" else self.dexycb_mano_right
        mano_out: MANOOutput = mano_layer(pose_m[:, :48], shape)
        hand_verts = mano_out.verts
        return hand_verts.squeeze(0).numpy().astype(np.float32)
    
    def get_random_hand_pose_and_joints(self, idx):
        sample = self.dataset[idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        pose_m = torch.from_numpy(label["pose_m"]) # this pose now in pca
        trans = pose_m[:, 48:51]
        pose = pose_m[:, 0:48]
        shape = torch.tensor(sample["mano_betas"]).unsqueeze(0)
        mano_layer = self.standard_mano_layer_pca
        th_hand_pose_coeffs = pose[:, mano_layer.rot:mano_layer.rot +mano_layer.ncomps]
        th_full_hand_pose = th_hand_pose_coeffs.mm(mano_layer.th_selected_comps)
        th_full_pose = torch.cat([ # full_pose in axis_angle
            pose[:, :mano_layer.rot],
            mano_layer.th_hands_mean + th_full_hand_pose
            ], 1)
        random_quat = random_quaternions(1)
        random_pose = quaternion_to_axis_angle(random_quat).reshape(3,)
        random_pose = torch.zeros(3).reshape(3,)

        th_full_pose_random = th_full_pose.clone()
        p = np.random.rand(1)[0]
        if p < 0.5:
            th_full_pose_random[:,:3] = random_pose

        joints_3d_random = (self.standard_mano_layer_axis(th_full_pose_random,shape,trans)[1][0] / 1000.).numpy()
        if p >= 0.5:
            j3d = self.get_joints_3d(idx)
            assert np.isclose(joints_3d_random, j3d, atol=1e-2).all(), f"{joints_3d_random} {j3d}" # note this may differ less than in 1cm
        
        return joints_3d_random, th_full_pose_random
    
    def get_mano_shape(self, idx):
        sample = self.dataset[idx]
        shape = torch.tensor(sample["mano_betas"]).reshape(10,)
        return shape
    
    def get_hand_trans(self, idx):
        sample = self.dataset[idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        pose_m = torch.from_numpy(label["pose_m"]) # this pose now in pca
        trans = pose_m[:, 48:51]
        return trans

    def get_hand_pose_m(self, idx):
        # get GT mano pose, in Euler angle format
        sample = self.dataset[idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        pose_m = torch.from_numpy(label["pose_m"]) # this pose now in pca
        trans = pose_m[:, 48:51]
        pose = pose_m[:, 0:48]
        mano_layer = self.standard_mano_layer_pca
        th_hand_pose_coeffs = pose[:, mano_layer.rot:mano_layer.rot +mano_layer.ncomps]
        th_full_hand_pose = th_hand_pose_coeffs.mm(mano_layer.th_selected_comps)

        th_full_pose = torch.cat([ # full_pose in axis_angle
            pose[:, :mano_layer.rot],
            mano_layer.th_hands_mean + th_full_hand_pose
            ], 1)
        
        if np.random.rand(1)[0] < 0.01: # randomly debug the result
            shape = torch.tensor(sample["mano_betas"]).unsqueeze(0)
            vert, _, _ = mano_layer(pose, shape, trans)
            vert_ho3d, j3d, _ = self.standard_mano_layer_axis(th_full_pose,shape,trans)
            assert (vert_ho3d-vert).abs().max() < 0.0001
            j3d_np = j3d.numpy().reshape(21, 3) / 1000
            assert np.abs(j3d_np - self.get_joints_3d(idx)).max() < 0.01, f"{np.abs(j3d_np - self.get_joints_3d(idx)).max()}"

        return th_full_pose

    def get_bone_scale(self, idx):
        joints_3d = self.get_joints_3d(idx)
        bone_len = batch_ref_bone_len(np.expand_dims(joints_3d, axis=0)).squeeze(0)
        return bone_len.astype(np.float32)

    def get_image(self, idx):
        img_path = self.get_image_path(idx)
        img = Image.open(img_path).convert("RGB")
        return img

    def get_image_path(self, idx):
        sample = self.dataset[idx]
        return sample["color_file"]

    def get_joints_2d(self, idx):
        sample = self.dataset[idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        return label["joint_2d"].squeeze(0)

    def get_joints_3d(self, idx):
        sample = self.dataset[idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        return label["joint_3d"].squeeze(0)

    def get_obj_faces(self, idx):
        sample = self.dataset[idx]
        grasp_ycb_idx = sample["ycb_ids"][sample["ycb_grasp_ind"]]
        obj_mesh = self.obj_raw_meshes[grasp_ycb_idx]
        faces = np.array(obj_mesh.faces).astype(np.long)
        # pad the face so that it becomes of shape 2008
        n_rows_to_pad = 2008 - faces.shape[0]
        # print(n_rows_to_pad)
        faces = np.pad(faces, ((0, n_rows_to_pad), (0, 0)), 'constant')
        return faces

    def get_obj_idx(self, idx):
        sample = self.dataset[idx]
        grasp_ycb_idx = sample["ycb_ids"][sample["ycb_grasp_ind"]]
        return grasp_ycb_idx

    def get_obj_onek(self, idx):
        sample = self.dataset[idx]
        grasp_ycb_idx = sample["ycb_ids"][sample["ycb_grasp_ind"]]
        onek = torch.zeros(21)
        onek[grasp_ycb_idx - 1] = 1 # since grasp_ycb_idx start from 1
        return onek
        
    def get_obj_transf(self, idx):
        sample = self.dataset[idx]
        label = self.get_label(sample["label_file"])
        transf = label["pose_y"][sample["ycb_grasp_ind"]]
        grasp_ycb_idx = sample["ycb_ids"][sample["ycb_grasp_ind"]]
        obj_mesh = self.obj_raw_meshes[grasp_ycb_idx]
        _, offset, _ = transform.center_vert_bbox(obj_mesh.vertices, scale=False)  # !! CENTERED HERE
        R, t = transf[:3, :3], transf[:, 3:]
        new_t = R @ offset.reshape(3, 1) + t
        new_transf = np.concatenate(
            [np.concatenate([R, new_t], axis=1),
             np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)])
        return new_transf.astype(np.float32)

    # * deprecated
    def _get_raw_obj_transf(self, idx):
        sample = self.dataset[idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        transf = label["pose_y"][sample["ycb_grasp_ind"]]
        transf = np.concatenate([transf, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)])
        return transf

    def get_obj_verts_2d(self, idx):
        verts_3d = self.get_obj_verts_transf(idx)
        cam_intr = self.get_cam_intr(idx)
        return HOdata.persp_project(verts_3d, cam_intr)

    def get_obj_verts_can(self, idx):
        sample = self.dataset[idx]
        grasp_ycb_idx = sample["ycb_ids"][sample["ycb_grasp_ind"]]
        obj_mesh = self.obj_raw_meshes[grasp_ycb_idx]
        # NOTE: verts_can = verts - bbox_center
        verts_can, obj_cantrans, obj_canscale = transform.center_vert_bbox(np.asfarray(obj_mesh.vertices,
                                                                                       dtype=np.float32),
                                                                           scale=False)  # !! CENTERED HERE
        return verts_can, obj_cantrans, obj_canscale

    # * deprecated
    def _get_raw_obj_verts(self, idx):
        sample = self.dataset[idx]
        grasp_ycb_idx = sample["ycb_ids"][sample["ycb_grasp_ind"]]

        obj_mesh = trimesh.load(self.dataset.obj_file[grasp_ycb_idx], process=False)
        return np.array(obj_mesh.vertices).astype(np.float32)

    def get_obj_verts_transf(self, idx):
        # * deprecated
        # transf = self._get_raw_obj_transf(idx)
        # R, t = transf[:3, :3], transf[:3, [3]]
        # verts_can = self._get_raw_obj_verts(idx)
        # raw_verts = (R @ verts_can.T + t).T

        transf = self.get_obj_transf(idx)
        R, t = transf[:3, :3], transf[:3, [3]]
        # this verts_can are offset by bbox center, so the get_obj_transf cancel the application
        # makes that the final result is still within the camera coordinate.
        verts_can, _, _ = self.get_obj_verts_can(idx)
        verts = (R @ verts_can.T + t).T

        return verts

    def get_sample_identifier(self, idx):
        res = f"{self.name}__{self.cache_identifier_raw}__{idx}"
        return res

    def obj_load_driver(self) -> Tuple[List[str], List[trimesh.base.Trimesh], List[np.ndarray]]:
        obj_names = []
        obj_meshes = []
        obj_corners_can = []
        for idx, obj_mesh in self.obj_raw_meshes.items():
            obj_name = CONST.YCB_IDX2CLASSES[idx]
            obj_names.append(obj_name)

            # ===== meshes can >>>>>>
            omesh = deepcopy(obj_mesh)
            verts_can, bbox_center, bbox_scale = transform.center_vert_bbox(omesh.vertices, scale=False)
            omesh.vertices = verts_can
            obj_meshes.append(omesh)

            # ===== corners can >>>>>
            corners = trimesh.bounds.corners(obj_mesh.bounds)
            corners_can = (corners - bbox_center) / bbox_scale
            obj_corners_can.append(corners_can)
        return (obj_names, obj_meshes, obj_corners_can)

    def get_sides(self, idx):
        sample = self.dataset[idx]
        return sample["mano_side"]
