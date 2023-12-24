from typing import Dict, Tuple

import torch
from anakin.datasets.hoquery import Queries
from anakin.utils.builder import LOSS
from anakin.utils.logger import logger

from .criterion import TensorLoss

@LOSS.register_module
class IKLoss(TensorLoss):
    def __init__(self, **cfg):
        super().__init__()

    def __call__(self, preds: Dict, targs: Dict, **kwargs):
        final_loss, losses = super().__call__(preds, targs, **kwargs)
        pred_pose = preds['pred_mano_pose']
        gt_pose = targs[Queries.HAND_POSE_M]
        ik_loss = torch.square(pred_pose - gt_pose).mean() # L2 square
        final_loss += ik_loss # in this stage, there should only be 1 loss
        losses['ik_loss'] = ik_loss
        losses[self.output_key] = final_loss
        return final_loss, losses
