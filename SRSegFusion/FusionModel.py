import torch
import torch.nn.functional as F
from math import exp
import numpy as np
from torch import nn
from RCAN2D import *
from AUNet import *

class FusionModel(nn.Module):
    def __init__(self, rcan_pretrained_path, num_seg_classes=2, rcan_scale=2):
        super().__init__()

        self.rcan = RCAN2D(input_channels=1)
        self.rcan.load_state_dict(torch.load(rcan_pretrained_path, map_location='cpu'))
        print(f"成功加载RCAN预训练权重：{rcan_pretrained_path}")

        self.attention_unet = AttentionUNet(
            n_channels=1,
            n_classes=num_seg_classes,
            bilinear=False
        )

    def forward(self, x):
        rcan_out = self.rcan(x)  # 输出：(B, 1, H*scale, W*scale)
        seg_out = self.attention_unet(rcan_out)  # 输入：(B, 1, H*scale, W*scale)
        return seg_out, rcan_out