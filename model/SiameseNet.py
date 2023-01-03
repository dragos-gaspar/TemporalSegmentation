import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNet(nn.Module):
    def __init__(self, backbone):

        super(SiameseNet, self).__init__()

        self.backbone = backbone

    def forward(self, img1, img2):

        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        output = ((feat1-feat2)**2).sum()

        return output
