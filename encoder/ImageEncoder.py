import torch
import torch.nn as nn
import math
import time
from thop import profile
from thop import clever_format
from torchinfo import summary
from backbone.vision.ImageEncoder import *
from neck.spp import *
from backbone.conv_utils.normal_conv import *
from backbone.conv_utils.ghost_conv import *
from backbone.attention_modules.shuffle_attention import *
from backbone.vision.mobilevit_modules.mobilevit import mobilevit_xxs, mobilevit_xs, mobilevit_s
from backbone.vision.edgenext_modules.model import edgenext_xx_small, edgenext_x_small, edgenext_small
from backbone.vision.edgevit_modules.edgevit import edgevit_xxs, edgevit_xs, edgevit_s
from backbone.vision.repvit_modules.repvit import repvit_m1, repvit_m2, repvit_m3
from backbone.vision.poolformer_modules.poolformer import poolformer_S0, poolformer_S1, poolformer_S2


image_encoder_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}


class ImageEncoder(nn.Module):
    def __init__(self, resolution=416, phi='S0', backbone='ef'):
        super().__init__()

        self.phi = phi
        self.channel_widths = image_encoder_width[phi]
        self.resolution = resolution

        if phi == 'S0':
            if backbone == 'ef':
                self.backbone = image_encoder_s0(resolution=resolution)
            elif backbone == 'mv':
                self.backbone = mobilevit_xxs(resolution=resolution)
            elif backbone == 'en':
                self.backbone = edgenext_xx_small()
            elif backbone == 'ev':
                self.backbone = edgevit_xxs(resolution=resolution)
            elif backbone == 'rv':
                self.backbone = repvit_m1()
            elif backbone == 'pf':
                self.backbone = poolformer_S0()

        elif phi == 'S1':
            if backbone == 'ef':
                self.backbone = image_encoder_s1(resolution=resolution)
            elif backbone == 'mv':
                self.backbone = mobilevit_xs(resolution=resolution)
            elif backbone == 'en':
                self.backbone = edgenext_x_small()
            elif backbone == 'ev':
                self.backbone = edgevit_xs(resolution=resolution)
            elif backbone == 'rv':
                self.backbone = repvit_m2()
            elif backbone == 'pf':
                self.backbone = poolformer_S1()

        elif phi == 'S2':
            if backbone == 'ef':
                self.backbone = image_encoder_s2(resolution=resolution)
            elif backbone == 'mv':
                self.backbone = mobilevit_s(resolution=resolution)
            elif backbone == 'en':
                self.backbone = edgenext_small()
            elif backbone == 'ev':
                self.backbone = edgevit_s(resolution=resolution)
            elif backbone == 'rv':
                self.backbone = repvit_m3()
            elif backbone == 'pf':
                self.backbone = poolformer_S2()

        elif phi == 'L':
            self.backbone = image_encoder_l(resolution=resolution)
            print("Only EfficientFormer V2 supports L size model.")              
    
    def forward(self, input):
        map_stage1, map_stage2, map_stage3, map_stage4, map_stage5 = self.backbone(input)

        return (map_stage1, map_stage2, map_stage3, map_stage4, map_stage5)


if __name__ == "__main__":
    input = torch.ones([1, 3, 320, 320])

    model = ImageEncoder(resolution=320, phi='S0', backbone='mv')

    output_features = model(input)

    for each in output_features:
        print(each.shape)