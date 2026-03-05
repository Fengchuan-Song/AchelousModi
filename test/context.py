import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转为灰度
img = cv2.imread('H:/dataset/WaterScenes/image/01339.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel 边缘检测
# 参数：图像, 输出深度, x方向导数阶数, y方向导数阶数, 核大小(必须是1,3,5或7)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # 水平边缘
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # 垂直边缘

# 计算梯度幅值
sobel_combined = np.sqrt(sobelx**2 + sobely**2)
sobel_combined = np.uint8(sobel_combined)

# 或者使用 cv2.addWeighted 合并
sobel_abs = cv2.addWeighted(np.abs(sobelx), 0.5, np.abs(sobely), 0.5, 0)

# 显示结果
plt.figure(figsize=(10, 4))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(np.abs(sobelx), cmap='gray'), plt.title('Sobel X')
plt.subplot(133), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel Combined')
plt.tight_layout()
plt.show()