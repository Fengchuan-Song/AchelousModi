import torch
import torch.nn as nn
from .ImageEncoder import ImageEncoder
from .LocalEntropy import LocalEntropyGating
from .AdaptiveSoftGating import AdaptiveSoftGating
from .RadarEncoder import RCNet
from backbone.conv_utils.dcn import DeformableConv2d

image_encoder_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}

class RadarConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, first_calculator='pool'):
        super(RadarConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if first_calculator == 'conv':
            self.initial_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=kernel_size // 2)
        elif first_calculator == 'pool':
            self.initial_conv = nn.AvgPool2d(3, stride=1, padding=1)

        self.deformable_conv = DeformableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                                stride=stride, padding=3 // 2)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.deformable_conv(x)
        return x


class RCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super(RCBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.radar_conv = RadarConv(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1)

        self.weight_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                      padding=0)

        self.norm = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU(inplace=True)

        if down is False:
            self.weight_conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                          padding=0)
        else:
            self.weight_conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                          padding=1)

    def forward(self, x):
        x_res = x
        x = self.radar_conv(x)
        x = self.weight_conv1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = x_res + x
        x = self.weight_conv2(x)

        return x


class ImRaEncoder(nn.Module):
    def __init__(self, resolution, phi, backbone, radar_channels):
        super().__init__()

        self.image_encoder = ImageEncoder(resolution=resolution, phi=phi, backbone=backbone)
        # self.radar_encoder = RCNet(in_channels=radar_channels, phi=phi)
        self.entropy_gating = LocalEntropyGating()

        stage_blocks = []
        for i in range(4):
            if i == 0:
                stage_blocks.append(RCBlock(in_channels=radar_channels,
                                            out_channels=image_encoder_width[phi][i] // 4, down=True))
                stage_blocks.append(RCBlock(in_channels=image_encoder_width[phi][i] // 4,
                                            out_channels=image_encoder_width[phi][i] // 4, down=True))
            else:
                stage_blocks.append(RCBlock(in_channels=image_encoder_width[phi][i-1] // 4,
                                            out_channels=image_encoder_width[phi][i-1] // 4, down=False))
                stage_blocks.append(RCBlock(in_channels=image_encoder_width[phi][i-1] // 4,
                                            out_channels=image_encoder_width[phi][i] // 4, down=True))

        self.rc_blocks = nn.ModuleList(stage_blocks)

        self.gate1 = AdaptiveSoftGating(in_channels=3)
        self.gate2 = AdaptiveSoftGating(in_channels=image_encoder_width[phi][0])
        self.gate3 = AdaptiveSoftGating(in_channels=image_encoder_width[phi][-1])

        self.radar_aggre1 = nn.Conv2d(in_channels=radar_channels * 2, out_channels=radar_channels, kernel_size=1)
        self.radar_aggre2 = nn.Conv2d(in_channels=image_encoder_width[phi][0] * 2, out_channels=image_encoder_width[phi][0], kernel_size=1)
        self.radar_aggre3 = nn.Conv2d(in_channels=image_encoder_width[phi][3] * 2, out_channels=image_encoder_width[phi][3], kernel_size=1)

        self.radar_batchnorm1 = nn.BatchNorm2d(num_features=radar_channels)
        self.radar_batchnorm2 = nn.BatchNorm2d(num_features=image_encoder_width[phi][0])
        self.radar_batchnorm3 = nn.BatchNorm2d(num_features=image_encoder_width[phi][3])

        self.activate = nn.ReLU(inplace=True)


    def forward(self, input_image, input_radar):
        # image encoder
        image_features = self.image_encoder(input_image)
        image_feature1, image_feature2, image_feature3, image_feature4, image_feature5 = image_features

        # radar encoder
        # input: 4 x H x W --> soft gate 1
        gate1 = self.gate1(input_image)
        # input_radar = input_radar + gate1 * input_radar
        input_radar = self.radar_aggre1(torch.cat([input_radar, gate1 * input_radar], dim=1))
        input_radar = self.radar_batchnorm1(input_radar)
        input_radar = self.activate(input_radar)
        # block-0 --> feat 1: image_encoder_width[phi][0] // 4 x H/2 x W/2
        radar_feature1 = self.rc_blocks[0](input_radar)
        # block-1 --> feat 2: image_encoder_width[phi][0] // 4 x H/4 x W/4 --> soft gate 2
        radar_feature2 = self.rc_blocks[1](radar_feature1)
        gate2 = self.gate2(image_feature2)
        radar_feature2 = self.radar_aggre2(torch.cat([radar_feature2, gate2 * radar_feature2]))
        radar_feature2 = self.radar_batchnorm2(radar_feature2)
        radar_feature2 = self.activate(radar_feature2)
        # block-2 + block-3 --> feat 3: image_encoder_width[phi][1] // 4 x H/4 x W/4
        # radar_feature3 = self.rc_blocks[2](radar_feature2 + gate2 * radar_feature2)
        radar_feature3 = self.rc_blocks[2](radar_feature2)
        radar_feature3 = self.rc_blocks[3](radar_feature3)
        # block-4 + block-5 --> feat 4: image_encoder_width[phi][2] // 4 x H/4 x W/4
        radar_feature4 = self.rc_blocks[4](radar_feature3)
        radar_feature4 = self.rc_blocks[5](radar_feature4)
        # block-6 + block-7 --> feat 5: image_encoder_width[phi][3] // 4 x H/4 x W/4 --> soft gate 3
        radar_feature5 = self.rc_blocks[6](radar_feature4)
        radar_feature5 = self.rc_blocks[7](radar_feature5)
        gate3 = self.gate3(image_feature5)
        # radar_feature5 = radar_feature5 + gate3 * radar_feature5
        radar_feature5 = self.radar_aggre3(torch.cat([radar_feature5, gate3 * radar_feature5], dim=1))
        radar_feature5 = self.radar_batchnorm3(radar_feature5)
        radar_feature5 = self.activate(radar_feature5)

        return image_features, (radar_feature1, radar_feature2, radar_feature3, radar_feature4, radar_feature5)