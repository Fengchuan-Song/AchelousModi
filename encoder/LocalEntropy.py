import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalEntropyGating(nn.Module):
    def __init__(self, window_size=21, threshold=0.2):
        super(LocalEntropyGating, self).__init__()
        self.window_size = window_size
        self.threshold = threshold
        # 定义一个平均池化层来计算局部均值 E[X]
        self.avg_pool = nn.AvgPool2d(
            kernel_size=window_size, 
            stride=1, 
            padding=window_size // 2
        )
        self.max_pool = nn.MaxPool2d(
            kernel_size=window_size,
            stride=1,
            padding=window_size//2
        )

    def forward(self, x):
        """
        x: 图像特征图 [B, C, H, W]
        returns: 门控权重 Mask [B, 1, H, W]
        """
        # 1. 转换为单通道灰度感官特征 (如果是彩色图)
        if x.shape[1] == 3:
            x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            x_gray = torch.mean(x, dim=1, keepdim=True)

        # 2. 计算 E[X]
        mu = self.avg_pool(x_gray)
        
        # 3. 计算 E[X^2]
        mu_sq = self.avg_pool(x_gray**2)
        
        # 4. 计算局部方差 Var = E[X^2] - (E[X])^2
        # 使用 clamp 保证数值稳定性，避免微小的负数导致开方出现 NaN
        var = (mu_sq - mu**2).clamp(min=1e-6)
        std = torch.sqrt(var)
        
        # 5. 归一化到 [0, 1]
        std_norm = (std - std.min()) / (std.max() - std.min() + 1e-6)
        
        # 6. 生成门控 Mask (使用 Sigmoid 软化阈值)
        # 这里的 10.0 是增益系数，用于让 Mask 更加“非黑即白”
        # mask = torch.sigmoid(10.0 * std_norm)
        mask = torch.sigmoid(10.0 * (std_norm - self.threshold))
        # std_norm[std_norm>self.threshold] = 1
        # std_norm[std_norm<self.threshold] = 0
        
        return mask


if __name__ == "__main__":
    import cv2
    
    image = cv2.imread('H:/dataset/WaterScenes/image/00015.jpg', cv2.IMREAD_GRAYSCALE)

    input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    
    module = LocalEntropyGating()

    output = module.forward(input).numpy()[0][0]

    cv2.imshow('result', output)
    cv2.waitKey(0)