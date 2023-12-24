import torch.nn as nn
import torch.nn.functional as F
import torch.nn
from anakin.opt import cfg
from anakin.postprocess.iknet import utils
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_euler_angles

class IKNet(nn.Module): # this IKNet was pretrained on A-MANO, need to replace with standard MANO
    def __init__(
        self,
        njoints=21,
        hidden_size_pose=[256, 512, 1024, 1024, 512, 256],
        output_format = 'angle_axis',
    ):
        super(IKNet, self).__init__()
        self.njoints = njoints
        in_neurons = 3 * njoints
        out_neurons = 16 * 4  # 16 quats
        neurons = [in_neurons] + hidden_size_pose
        self.output_format = output_format

        invk_layers = []
        for layer_idx, (inps, outs) in enumerate(zip(neurons[:-1], neurons[1:])):
            invk_layers.append(nn.Linear(inps, outs))
            invk_layers.append(nn.BatchNorm1d(outs))
            invk_layers.append(nn.ReLU())

        invk_layers.append(nn.Linear(neurons[-1], out_neurons))

        self.invk_layers = nn.Sequential(*invk_layers)

    def forward(self, inputs):
        joint = inputs
        joint = joint.contiguous().view(-1, self.njoints * 3)
        quat = self.invk_layers(joint)
        quat = quat.view(-1, 16, 4)
        quat = utils.normalize_quaternion(quat)
        if self.output_format == 'angle_axis':
            so3 = utils.quaternion_to_angle_axis(quat).contiguous()
            so3 = so3.view(-1, 1, 16 * 3) # Both targets and predicts should be B, 1, 48
        elif self.output_format == 'euler':
            so3 = quaternion_to_matrix(quat)
            so3 = matrix_to_euler_angles(so3, convention='XYZ')
            so3 = so3.view(-1, 16, 3)

        pred = {'pred_mano_pose': so3, 'pred_mano_quat': quat}
        return pred