import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class AdaptiveSoftGating(nn.Module):
    def __init__(self, in_channels, window_size=7):
        super(AdaptiveSoftGating, self).__init__()

        if in_channels == 3:
            self.channels = in_channels

            self.avg_pool = nn.AvgPool2d(kernel_size=window_size, stride=1, padding=window_size // 2)

            # Sobel 滤波器
            sobel_x = torch.tensor([[-1., 0., 1.],
                                    [-2., 0., 2.],
                                    [-1., 0., 1.]]).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1., -2., -1.],
                                    [ 0.,  0.,  0.],
                                    [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
            
            self.register_buffer('kernel_x', sobel_x)
            self.register_buffer('kernel_y', sobel_y)
        else:
            self.kernel_size = 3
            self.channels = in_channels // 4

            self.offset_conv = nn.Conv2d(in_channels=in_channels, out_channels=2*self.kernel_size*self.kernel_size, 
                                         kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
            self.mask_conv = nn.Conv2d(in_channels=in_channels, out_channels=self.kernel_size*self.kernel_size, 
                                       kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
            # 只调整特征相关性，不聚合特征
            self.dcn = DeformConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=self.kernel_size, 
                                    padding=self.kernel_size//2, stride=1)
            
            self.aggrate = nn.Conv2d(in_channels=in_channels, out_channels=self.channels, kernel_size=1)

        self.conv = nn.Conv2d(in_channels=self.channels, out_channels=1, kernel_size=3, padding=1,
                              padding_mode='reflect')
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: Region Mask [B, 1, H, W]
        """
        # 灰度化提取感官特征
        if x.shape[1] == 3:
            x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

            # 局部信息熵（局部方差近似）
            mu = self.avg_pool(x_gray)
            mu_sq = self.avg_pool(x_gray**2)
            var = (mu_sq - mu**2).clamp(min=1e-6)
            std = torch.sqrt(var)
            std_min = std.view(std.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
            std_max = std.view(std.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
            std_norm = (std - std_min) / (std_max - std_min + 1e-6)

            # 垂直、水平方向梯度场
            # grad_x = F.conv2d(x_gray, self.kernel_x, padding=1).abs()
            # grad_y = F.conv2d(x_gray, self.kernel_y, padding=1).abs()
            grad_x = F.conv2d(x_gray, self.kernel_x, padding=1)
            grad_y = F.conv2d(x_gray, self.kernel_y, padding=1)

            max_grad_x = torch.abs(grad_x).view(grad_x.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
            max_grad_y = torch.abs(grad_y).view(grad_y.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)

            grad_x_norm = grad_x / (max_grad_x + 1e-6)
            grad_y_norm = grad_y / (max_grad_y + 1e-6)

            temp_feature = torch.cat([std_norm, grad_x_norm, grad_y_norm], dim=1)

        else:
            offsets = self.offset_conv(x)
            masks = self.sigmoid(self.mask_conv(x))

            dcn_out = self.dcn(x, offsets, mask=masks)

            temp_feature = self.aggrate(dcn_out)
        
        feature = self.conv(temp_feature)

        mask = self.sigmoid(feature)

        return mask

if __name__ == "__main__":
    import cv2
    
    # image = cv2.imread('H:/dataset/WaterScenes/image/00001.jpg', cv2.IMREAD_GRAYSCALE)

    # input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()

    input = torch.rand([5, 3, 320, 320], requires_grad=False)
    
    module = AdaptiveSoftGating(in_channels=3)

    output = module(input)
    print(output.shape)

    # cv2.imshow('result', output)
    # cv2.waitKey(0)