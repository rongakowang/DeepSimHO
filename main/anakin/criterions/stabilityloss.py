
import torch
import sys
sys.path.append('..')
import random
from typing import Dict
from anakin.opt import cfg
from anakin.datasets.hoquery import Queries
from anakin.utils.builder import LOSS
from pytorch3d.transforms import matrix_to_quaternion, random_rotations
from kaolin.ops.mesh import check_sign, index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from .criterion import TensorLoss
from model.mujoco_mesh_simulator import MuJoCoMeshSimulatorFast

def calculate_sdf(hand_verts, obj_verts, hand_faces, obj_faces):
    batch_size = hand_verts.shape[0]

    hand_sample_obj_mesh_pd_ = []

    for i in range(batch_size):
        obj_face = obj_faces[i]
        for k in range(obj_face.shape[0]-1, 0, -1):
            if not obj_face[k].eq(0).all():
                break
        obj_face = obj_face[:k+1, :]
        face_vertices = index_vertices_by_faces(obj_verts[i].unsqueeze(0).contiguous(), obj_face)
        distance, _, _ = point_to_mesh_distance(hand_verts[i].unsqueeze(0).contiguous(), face_vertices)
        sign = check_sign(obj_verts[i].unsqueeze(0).contiguous(), obj_face, hand_verts[i].unsqueeze(0).contiguous()).int()
        sign[sign == 0] = -1
        hand_sample_obj_mesh_pd_.append(distance * sign.int())

    ho_sdf = torch.stack(hand_sample_obj_mesh_pd_)

    object_sample_hand_mesh_pd_ = []

    for i in range(batch_size):
        hand_face = hand_faces[i]
        face_vertices = index_vertices_by_faces(hand_verts[i].unsqueeze(0).contiguous(), hand_face)
        distance, _, _ = point_to_mesh_distance(obj_verts[i].unsqueeze(0).contiguous(), face_vertices)
        sign = check_sign(hand_verts[i].unsqueeze(0).contiguous(), hand_face, obj_verts[i].unsqueeze(0).contiguous()).int()
        sign[sign == 0] = -1
        object_sample_hand_mesh_pd_.append(distance * sign.int())

    oh_sdf = torch.stack(object_sample_hand_mesh_pd_)

    sdf_dist = torch.cat([ho_sdf.squeeze(1), oh_sdf.squeeze(1)], dim=1) # 778 + 1000 shape

    return sdf_dist
    

@LOSS.register_module
class MLPMeshMixRegPretrainLoss(TensorLoss):
    """
    This loss mixes all discriminator losses in one epoch
    """
    def __init__(self, **cfg_local):
        super().__init__() # hand must be disabled

    def prepare_discriminator_data(self, preds, targs):

        ### for simulator ###
        hand_verts_abs_all = []
        obj_trans_all = []
        obj_rot_all = []
        obj_verts_abs_all = []
        obj_type_all = []
        obj_name_all = []
        root_all = []
        hand_faces_all = []
        obj_faces_all = []

        ############### gt ###################
        batch_size = preds['obj_center'].shape[0]
        obj_gt_transf = targs[Queries.OBJ_TRANSF].cuda().clone()
        obj_trans = obj_gt_transf[:, :3, 3].reshape(-1, 3)
        obj_rot_matrix = obj_gt_transf[:, :3, :3]
        hand_verts_abs = targs[Queries.HAND_VERTS_3D].cuda().clone()
        
        obj_rot = matrix_to_quaternion(obj_rot_matrix)

        mesh_can = targs[Queries.OBJ_VERTS_CAN].cuda()
        obj_mesh_ = obj_rot_matrix.bmm(mesh_can.float().transpose(1, 2)).transpose(1, 2)
        obj_verts = obj_mesh_ + obj_trans.unsqueeze(1)

        hand_verts_abs_all.append(hand_verts_abs.detach().clone())
        obj_trans_all.append(obj_trans.detach().clone())
        obj_rot_all.append(obj_rot.detach().clone())
        obj_verts_abs_all.append(obj_verts.detach().clone())
        obj_type_all.append(targs['obj_one_k'].reshape(-1, 21).cuda())
        obj_name_all.extend(targs['obj_name'])
        root_all.append(preds["root_joint"])
        hand_faces_all.append(targs[Queries.HAND_FACES].cuda())
        obj_faces_all.append(targs[Queries.OBJ_FACES].cuda())

        ############### gt, random ###################
        batch_size = preds['obj_center'].shape[0]
        obj_gt_transf = targs[Queries.OBJ_TRANSF].cuda().clone()
        obj_trans = obj_gt_transf[:, :3, 3].reshape(-1, 3) + torch.randn(batch_size, 3).cuda() / 20. # 50 % + free fall
        random_rot = random_rotations(batch_size).cuda()
        obj_rot_matrix = obj_gt_transf[:, :3, :3]
        obj_rot_matrix = torch.bmm(random_rot, obj_rot_matrix)
        hand_verts_abs = targs[Queries.HAND_VERTS_3D].cuda().clone()
        
        obj_rot = matrix_to_quaternion(obj_rot_matrix)

        mesh_can = targs[Queries.OBJ_VERTS_CAN].cuda()
        obj_mesh_ = obj_rot_matrix.bmm(mesh_can.float().transpose(1, 2)).transpose(1, 2)
        obj_verts = obj_mesh_ + obj_trans.unsqueeze(1)

        hand_verts_abs_all.append(hand_verts_abs.detach().clone())
        obj_trans_all.append(obj_trans.detach().clone())
        obj_rot_all.append(obj_rot.detach().clone())
        obj_verts_abs_all.append(obj_verts.detach().clone())
        obj_type_all.append(targs['obj_one_k'].reshape(-1, 21).cuda())
        obj_name_all.extend(targs['obj_name'])
        root_all.append(preds["root_joint"])
        hand_faces_all.append(targs[Queries.HAND_FACES].cuda())
        obj_faces_all.append(targs[Queries.OBJ_FACES].cuda())

        ############### pred, no partial ###################
        obj_trans = preds['obj_center'].reshape(-1, 3)
        obj_rot = matrix_to_quaternion(preds["obj_pred_rot"])
        obj_verts = preds["obj_verts_3d_abs"]
        hand_verts_abs = preds["hand_verts_3d_abs"]

        hand_verts_abs_all.append(hand_verts_abs.detach().clone())
        obj_trans_all.append(obj_trans.detach().clone())
        obj_rot_all.append(obj_rot.detach().clone())
        obj_verts_abs_all.append(obj_verts.detach().clone())
        obj_type_all.append(targs['obj_one_k'].reshape(-1, 21).cuda())
        obj_name_all.extend(targs['obj_name'])
        root_all.append(preds["root_joint"])
        hand_faces_all.append(targs[Queries.HAND_FACES].cuda())
        obj_faces_all.append(targs[Queries.OBJ_FACES].cuda())

        ############### merge ##################################
        hand_verts_abs_all = torch.cat(hand_verts_abs_all, dim=0)
        obj_trans_all = torch.cat(obj_trans_all, dim=0)
        obj_rot_all = torch.cat(obj_rot_all, dim=0)
        obj_verts_abs_all = torch.cat(obj_verts_abs_all, dim=0)
        obj_type_all = torch.cat(obj_type_all, dim=0)
        root_all = torch.cat(root_all, dim=0)
        hand_faces_all = torch.cat(hand_faces_all, dim=0)
        obj_faces_all = torch.cat(obj_faces_all, dim=0)

        ############## shuffle #################################
        batch_size_all = len(obj_name_all)
        idxs = list(range(batch_size_all))
        random.shuffle(idxs)

        hand_verts_abs_all = hand_verts_abs_all[idxs, :, :]
        obj_trans_all = obj_trans_all[idxs, :]
        obj_rot_all = obj_rot_all[idxs, :]
        obj_verts_abs_all = obj_verts_abs_all[idxs, :, :]
        obj_type_all = obj_type_all[idxs, :]
        root_all = root_all[idxs, :]
        hand_faces_all = hand_faces_all[idxs, :, :]
        obj_faces_all = obj_faces_all[idxs, :, :]
        obj_name_all_shuffle = []
        for k in idxs:
            obj_name_all_shuffle.append(obj_name_all[k])
        
        return hand_verts_abs_all, obj_trans_all, obj_rot_all, obj_verts_abs_all, obj_type_all, obj_faces_all, hand_faces_all, root_all, obj_name_all_shuffle, idxs
        

    def __call__(self, preds: Dict, targs: Dict, **kwargs):
        ### inputs ###
        final_loss, losses = super().__call__(preds, targs, **kwargs) # this final loss is just 0, and losses is just empty dict
        
        ### prepare data
        net = preds['discriminator']
        camera_dir = 'camera' if cfg['DATASET']['TRAIN']['TYPE'] == 'DexYCB' else 'OpenGL'

        hand_verts_abs, obj_trans, obj_rot, obj_verts_abs, _, obj_faces, hand_faces, root, obj_name, idxs = \
                    self.prepare_discriminator_data(preds, targs)

        with torch.no_grad():
            final_state = MuJoCoMeshSimulatorFast.batched_simulate(obj_name, hand_verts_abs, obj_rot, obj_trans, camera_dir)
            gt_displacement = torch.norm(final_state[:, :3] - obj_trans[:, :3].clone().detach().cuda(), p=2, dim=-1)
            gt_displacement = torch.clamp(gt_displacement, min=-0.2, max=0.2).reshape(-1, 1)

        hand_verts = hand_verts_abs - root.unsqueeze(1)
        obj_verts = obj_verts_abs - root.unsqueeze(1)

        sdf_distance = calculate_sdf(hand_verts, obj_verts, hand_faces, obj_faces.cuda())

        global_feat_train_discriminator = torch.cat([hand_verts.clone().detach().reshape(-1, 778 * 3),
                                                obj_verts.clone().detach().reshape(-1, 1000 * 3)], dim=1).cuda()

        pred_final_state = net(global_feat_train_discriminator, sdf_distance.detach().clone())
        discriminator_loss = torch.norm(gt_displacement - pred_final_state, p=2, dim=-1)
        losses['discriminator_loss'] = discriminator_loss.mean() # discriminator_loss is independent

        ############# For stability loss ################
        obj_verts_stb_abs = preds["obj_verts_3d_abs"]
        obj_verts_stb = obj_verts_stb_abs - preds['root_joint'].unsqueeze(1)
        hand_verts_stb_abs = preds["hand_verts_3d_abs"]
        hand_verts_stb = hand_verts_stb_abs - preds['root_joint'].unsqueeze(1)
        is_valid = targs[Queries.IS_VALID].int()

        sdf_stb = calculate_sdf(hand_verts_stb, obj_verts_stb, targs[Queries.HAND_FACES].cuda(), targs[Queries.OBJ_FACES].cuda())

        global_feat_train_generator = torch.cat([hand_verts_stb.reshape(-1, 778 * 3),
                                                                obj_verts_stb.reshape(-1, 1000 * 3)], dim=1).cuda()
            
        pred_final_state_trace = net(global_feat_train_generator, sdf_stb)

        stability_loss = pred_final_state_trace
        index_raw = torch.ones(stability_loss.shape[0]).cuda()

        if 'gt_displacement' in locals():
            for i in range(stability_loss.shape[0]):
                if stability_loss[i] > 0.01 and gt_displacement[idxs[i]] < 0.01 or stability_loss[i] > 0.01 and gt_displacement[idxs[i]] < 0.01:
                    index_raw[i] = 0

        index_raw = index_raw * is_valid.cuda()
        stability_loss = stability_loss * index_raw.reshape(-1, 1)
        N = index_raw.sum() if index_raw.sum() > 0 else 1
        assert stability_loss.shape[1] == 1

        losses['stability_loss'] = cfg["PHYSICS_MLP"]["STABILITY_WEIGHT"] * stability_loss.sum() / N # this loss goes to all losses
        
        return final_loss, losses