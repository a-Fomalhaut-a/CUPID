import torch
import torch.nn as nn
from torch import Tensor

from config_train import data_info


class CUPIDLoss(nn.Module):
    def __init__(self) -> None:
        super(CUPIDLoss, self).__init__()
        self.L_l1 = nn.L1Loss(reduction='mean')

    def forward(
            self,
            mean: Tensor,
            otheroutput,
            target: Tensor,
            target2: Tensor,
            enfea: Tensor,
            defea: Tensor,
            T1=1e0,
            T2=5e-4,
            T11=0.1,
    ):
        l1a = self.L_l1(mean, target)
        l1b = self.L_l1(defea, enfea)
        l1 = l1a - T11 * l1b

        sigma = otheroutput[0]
        sigma_now = torch.exp(-sigma)

        smean = torch.softmax(mean, dim=1)
        target2 = torch.nn.functional.one_hot(target2, num_classes=data_info['lab_dim'])
        l2a = ((smean - target2) ** 2) * sigma_now / 2
        l2b = sigma / 2
        l2 = (l2a + l2b).mean()

        loss = T1 * l1 + T2 * l2
        return loss, [l1, l2]
