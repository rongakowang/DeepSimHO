import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import jax.numpy as npj
from jax import grad, jit
from jax.experimental import optimizers

from .manolayer import ManoLayer
from .checkpoints import CheckpointIO
from . import utils
from .model import IKNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

mano_layer = ManoLayer(
    center_idx=9,
    side="right",
    mano_root="assets/mano_v1_2",
    use_pca=False,
    flat_hand_mean=True,
)
mano_layer = jit(mano_layer)


@jit
def hm_to_kp2d(hm):
    b, c, w, h = hm.shape
    hm = hm.reshape(b, c, -1)
    hm = hm / npj.sum(hm, -1, keepdims=True)
    coord_map_x = npj.tile(npj.arange(0, w).reshape(-1, 1), (1, h))
    coord_map_y = npj.tile(npj.arange(0, h).reshape(1, -1), (w, 1))
    coord_map_x = coord_map_x.reshape(1, 1, -1)
    coord_map_y = coord_map_y.reshape(1, 1, -1)
    x = npj.sum(coord_map_x * hm, -1, keepdims=True)
    y = npj.sum(coord_map_y * hm, -1, keepdims=True)
    kp_2d = npj.concatenate((y, x), axis=-1)
    return kp_2d


@jit
def geo(joint):
    idx_a = npj.array([1, 5, 9, 13, 17])
    idx_b = npj.array([2, 6, 10, 14, 18])
    idx_c = npj.array([3, 7, 11, 15, 19])
    idx_d = npj.array([4, 8, 12, 16, 20])
    p_a = joint[:, idx_a, :]
    p_b = joint[:, idx_b, :]
    p_c = joint[:, idx_c, :]
    p_d = joint[:, idx_d, :]
    v_ab = p_a - p_b  # (B, 5, 3)
    v_bc = p_b - p_c  # (B, 5, 3)
    v_cd = p_c - p_d  # (B, 5, 3)
    loss_1 = npj.abs(npj.sum(npj.cross(v_ab, v_bc, -1) * v_cd, -1)).mean()
    loss_2 = -npj.clip(npj.sum(npj.cross(v_ab, v_bc, -1) * npj.cross(v_bc, v_cd, -1)), -npj.inf, 0).mean()
    loss = 10000 * loss_1 + 100000 * loss_2

    return loss


@jit
def residuals(input_list, so3_init, beta_init, joint_root, joint):
    so3 = input_list["so3"]
    beta = input_list["beta"]
    bone = input_list["bone"]
    so3 = so3[npj.newaxis, ...]
    beta = beta[npj.newaxis, ...]
    _, joint_mano, _ = mano_layer(pose_coeffs=so3, betas=beta)
    bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :], axis=1, keepdims=True)
    bone_pred = bone_pred[:, npj.newaxis, ...]
    reg = (so3 - so3_init) ** 2
    reg_beta = np.sum((beta - beta_init) ** 2)
    joint_mano = joint_mano / bone_pred
    errkp = np.mean((joint_mano - joint) ** 2)
    joint_mano = joint_mano * bone + joint_root
    geo_reg = geo(joint_mano)
    err = 0.01 * reg.mean() + 0.01 * reg_beta.mean() + 1 * errkp.mean() + 100 * geo_reg.mean()
    return err


@jit
def mano_de(params, joint_root, bone):
    so3 = params["so3"]
    beta = params["beta"]
    verts_mano, joint_mano, _ = mano_layer(pose_coeffs=so3[npj.newaxis, ...], betas=beta[npj.newaxis, ...])
    
    bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :], axis=1, keepdims=True)
    bone_pred = bone_pred[:, npj.newaxis, ...]
    # verts_mano = verts_mano / bone_pred
    # verts_mano = verts_mano * bone + joint_root
    verts_mano += joint_root
    joint_mano = joint_mano / bone_pred
    joint_mano = joint_mano * bone + joint_root
    v = verts_mano[0]
    j = joint_mano[0]
    return v, j

@jit
def mano_de2(params, param, joint_root, bone):
    so3 = params["so3"]
    beta = param["beta"]
    verts_mano, joint_mano, _ = mano_layer(pose_coeffs=so3[npj.newaxis, ...], betas=beta[npj.newaxis, ...])

    bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :], axis=1, keepdims=True)
    bone_pred = bone_pred[:, npj.newaxis, ...]
    verts_mano = verts_mano / bone_pred
    verts_mano = verts_mano * bone + joint_root
    joint_mano = joint_mano / bone_pred
    joint_mano = joint_mano * bone + joint_root
    v = verts_mano[0]
    j = joint_mano[0]
    return v, j


@jit
def mano_de_j(so3, beta):
    _, joint_mano, _ = mano_layer(pose_coeffs=so3[npj.newaxis, ...], betas=beta[npj.newaxis, ...])

    bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :], axis=1, keepdims=True)
    bone_pred = bone_pred[:, npj.newaxis, ...]
    joint_mano = joint_mano / bone_pred
    j = joint_mano[0]
    return j


# TODO: REWRAP this class for consistent interface, also change the signature and way of calling in submit_epoch_pass inheritance tree
# ! important
class FittingUnit(object):
    def __init__(self, aquire_gt=False, reload_prefix="assets/postprocess", hand_face_path="assets/postprocess/hand_close.npy"):
        super().__init__()

        self.reload_prefix = reload_prefix
        self.iknet = IKNet()
        self.iknet.cuda()

        # reload if prefix is not None
        if self.reload_prefix is not None:
            self.reload() # reloading at different position

        # initialize other stuff
        self.face = np.loadtxt(hand_face_path).astype(np.int32)
        self.renderer = utils.MeshRenderer(self.face, img_size=224)

        # model in eval
        self.iknet.eval()

        self.gr = jit(grad(residuals))
        self.lr = 0.03
        opt_init, opt_update, get_params = optimizers.adam(self.lr, b1=0.5, b2=0.5)
        self.opt_init = jit(opt_init)
        self.opt_update = jit(opt_update)
        self.get_params = jit(get_params)

        self.draw = False
        self.aquire_gt = aquire_gt

    def reload(self):
        cio = CheckpointIO(self.reload_prefix, model=self.iknet)
        cio.load("iknet.pt")

    def __call__(self, inp, pred_joints): # pred in abs joints
        joint_tensor = pred_joints
        batch_size = joint_tensor.shape[0]

        # first batch over iknet
        joint_root = joint_tensor[:, 9, :].unsqueeze(1)
        joint_ = joint_tensor - joint_root
        bone_pred = torch.zeros((batch_size, 1)).to(device)
        bone_pred += torch.norm(joint_[:, 0, :] - joint_[:, 9, :], dim=1, keepdim=True)  # Tensor[B, 1]
        bone_pred = bone_pred.unsqueeze(1)  # (B,1,1)
        _joint_ = joint_ / bone_pred
        so3 = self.iknet.forward(_joint_)['pred_mano_pose'].view(-1, 16 * 3)

        bone = bone_pred.cpu().numpy()
        joint_root = joint_root.cpu().numpy()
        so3 = so3.detach().cpu().float().numpy()
        _joint_ = _joint_.cpu().float().numpy()

        v_list = []
        j_list = []
        s_list = []
        # frame_list = []
        for batch_id in range(batch_size):
            so3_this = so3[batch_id]
            beta_this = np.zeros((10,))
            joint_this = _joint_[batch_id]

            so3_this = npj.array(so3_this)
            beta_this = npj.array(beta_this)
            bone_this = npj.array(bone[batch_id : batch_id + 1, ...])
            joint_root_this = npj.array(joint_root[batch_id : batch_id + 1, ...])
            so3_init = so3
            beta_init = beta_this

            params = {"so3": so3_this, "beta": beta_this, "bone": bone_this}
            opt_state = self.opt_init(params)
            n = 0
            while n < 20:
                n = n + 1
                params = self.get_params(opt_state)
                grads = self.gr(params, so3_init, beta_init, joint_root_this, joint_this)
                opt_state = self.opt_update(n, grads, opt_state)
            params = self.get_params(opt_state)

            if self.aquire_gt:
                s_list.append(torch.from_numpy(np.array(params['so3'])))
                v, j = mano_de(params, joint_root_this, bone_this)
                j_list.append(torch.from_numpy(np.array(j)))
                v_list.append(torch.from_numpy(np.array(v)))
            else:
                v, j = mano_de(params, joint_root_this, bone_this)
                v = np.array(v)
                j = np.array(j)

                v_list.append(v)
                j_list.append(j)

        if self.aquire_gt:
            return torch.stack(s_list, dim=0).to(pred_joints.device), torch.stack(v_list, dim=0).to(pred_joints.device), torch.stack(j_list, dim=0).to(pred_joints.device)
        else:
            return v_list, j_list