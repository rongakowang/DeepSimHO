import sys
import torch
sys.path.append('..')
from anakin.datasets.hoquery import Queries
from anakin.metrics.metric import AverageMeter, Metric
from anakin.utils.builder import METRIC
from anakin.utils.logger import logger
from pytorch3d.transforms import matrix_to_quaternion
from anakin.opt import cfg
from typing import Dict
from model.mujoco_mesh_simulator import MuJoCoMeshSimulatorFast
from kaolin.ops.mesh import check_sign, index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance

def calculate_sdf(hand_verts, obj_verts, hand_faces, obj_faces):
    # copy implementation from stabilityloss.calculate_sdf to avoid module conflict
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

@METRIC.register_module
class DiscriminatorMetric(Metric):
    # metrics for DeepSim used in ablation study
    def __init__(self, **cfg) -> None:
        """
        Physics Metric for SR, SM, PD, CP
        """
        super(DiscriminatorMetric, self).__init__()
        self.val_keys_list = ['D-SD', 'D-SR']
        self.avg_meters: Dict[str, AverageMeter] = {}
        for key in self.val_keys_list:
            self.avg_meters[key] = AverageMeter()
        self.reset()

    def reset(self):
        for _, meter in self.avg_meters.items():
            meter.reset()

    def get_measures(self, **kwargs) -> Dict[str, float]:
        measures = {}
        for key in self.val_keys_list:
            avg = (self.avg_meters[key]).avg
            measures[f"{key}"] = avg

        return measures

    def __str__(self):
        return " | ".join([f"{key}: {self.avg_meters[key].avg:6.4f}" for key in self.val_keys_list])


    def feed(self, preds: Dict, targs: Dict, **kwargs):

        hand_verts = preds["hand_verts_3d_abs"] - preds["root_joint"].unsqueeze(1) # (B, 778, 3)
        obj_verts = preds["obj_verts_3d_abs"] - preds["root_joint"].unsqueeze(1) # (B, 1000, 3)

        net = preds['discriminator']

        with torch.no_grad():

            global_feat_train_discriminator = torch.cat([hand_verts.clone().detach().reshape(-1, 778 * 3),
                                    obj_verts.clone().detach().reshape(-1, 1000 * 3)], dim=1).cuda()

            sdf_distance = calculate_sdf(hand_verts, obj_verts, targs[Queries.HAND_FACES].cuda(), targs[Queries.OBJ_FACES].cuda())
            pred_final_state = net(global_feat_train_discriminator, sdf_distance.detach())

            if pred_final_state.shape[-1] == 3:
                sd = torch.norm(pred_final_state.reshape(-1, 3), p=2, dim=-1) # take one more norm
            else:
                sd = pred_final_state.flatten()

            sr = (sd <= cfg["PHYSICS_MLP"]["SR_THRESHOLD"]).int()

        batch_size = sd.shape[0]

        self.avg_meters['D-SD'].update(sd.sum().item(), n=batch_size)
        self.avg_meters['D-SR'].update(sr.sum().item(), n=batch_size)

@METRIC.register_module
class MeshPhysicsMetric(Metric): # physics metric for prediction
    def __init__(self, **cfg) -> None:
        #Physics Metric for SR, SM, PD, CP
        super(MeshPhysicsMetric, self).__init__()
        self.val_keys_list = ['CP', 'PD', 'SD', 'SR']
        self.avg_meters: Dict[str, AverageMeter] = {}
        for key in self.val_keys_list:
            self.avg_meters[key] = AverageMeter()
        self.use_gt = cfg.get("USE_GT", False) # evaluate GT stability
        logger.info(f"use gt mode for physics metrics: {self.use_gt}")
        self.reset()

    def reset(self):
        for _, meter in self.avg_meters.items():
            meter.reset()

    def feed(self, preds: Dict, targs: Dict, **kwargs):
        if self.use_gt:
            obj_trans = targs["obj_transf"][:,:3, 3].cpu().detach() 
            # only supports for quaternion
            obj_rot = matrix_to_quaternion(targs["obj_transf"][:, :3, :3]).cpu().detach()
            hand_verts = targs["hand_verts_3d_abs"].cpu().detach()
        else:
            obj_trans = preds['obj_center'].reshape(-1, 3).cpu().detach() 
            obj_rot = matrix_to_quaternion(preds["obj_pred_rot"]).cpu().detach() 
            hand_verts = preds["hand_verts_3d_abs"].cpu().detach()
            obj_verts = preds['obj_verts_3d_abs'].cpu().detach() 
            
        camera_dir = 'camera' if cfg['DATASET']['TRAIN']['TYPE'] == 'DexYCB' else 'OpenGL'
        obj_name = targs['obj_name']
        batch_size = len(obj_name)

        with torch.no_grad():
            final_state = MuJoCoMeshSimulatorFast.batched_simulate(obj_name, hand_verts, obj_rot, obj_trans, camera_dir)

            # must be sum stability
            sd = torch.norm(final_state[:,:3] - obj_trans.cuda(), p=2, dim=-1) # (B, 1)
            sr = (sd <= cfg["PHYSICS_MLP"]["SR_THRESHOLD"]).int() # (B, 1)

            b_cn, _ = MuJoCoMeshSimulatorFast.batched_get_contact(obj_name, hand_verts, obj_rot, obj_trans, camera_dir)
            b_cp = (b_cn > 0).int()

            hand_to_obj_sdf = calculate_sdf(hand_verts.cuda(), obj_verts.cuda(), targs[Queries.HAND_FACES].cuda(),
                   targs[Queries.OBJ_FACES].cuda())[:, :778]
            b_pd = []
            for k in hand_to_obj_sdf:
                b_pd.append(torch.abs(k[k < 0].min()))
            b_pd = torch.stack(b_pd).cuda()
                         
        self.avg_meters['CP'].update(b_cp.sum().item(), n=batch_size)
        self.avg_meters['PD'].update(b_pd.sum().item(), n=batch_size)
        self.avg_meters['SD'].update(sd.sum().item(), n=batch_size)
        self.avg_meters['SR'].update(sr.sum().item(), n=batch_size)

        
    def get_measures(self, **kwargs) -> Dict[str, float]:
        measures = {}
        for key in self.val_keys_list:
            avg = (self.avg_meters[key]).avg
            measures[f"{key}"] = avg

        return measures

    def __str__(self):
        return " | ".join([f"{key}: {self.avg_meters[key].avg:6.4f}" for key in self.val_keys_list])



    

