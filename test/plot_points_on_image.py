import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # 用于显示进度条

# --- 1. 路径设置 ---
image_folder = 'H:/dataset/WaterScenes/image'      # 原始图像文件夹
csv_folder = 'H:/dataset/WaterScenes/radar/radar_origin'      # 存放 00001.csv 等文件的文件夹
output_folder = 'H:/dataset/WaterScenes/ploted_image'    # 结果保存路径

# 如果输出目录不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- 2. 遍历图像文件夹 ---
# 假设文件名是一一对应的，例如 00001.jpg 对应 00001.csv
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])

print(f"开始处理，共计 {len(image_files)} 张图像...")

for img_name in tqdm(image_files):
    # 构建对应的文件名
    base_name = os.path.splitext(img_name)[0]
    csv_name = base_name + '.csv'
    
    img_path = os.path.join(image_folder, img_name)
    csv_path = os.path.join(csv_folder, csv_name)
    
    # 检查 CSV 文件是否存在
    if not os.path.exists(csv_path):
        continue

    # --- 3. 读取数据 ---
    img = cv2.imread(img_path)
    if img is None: continue
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    df = pd.read_csv(csv_path)

    # 过滤掉图像范围外的点 (u, v)
    mask = (df['u'] >= 0) & (df['u'] < w) & (df['v'] >= 0) & (df['v'] < h)
    valid_points = df[mask]

    # --- 4. 绘图 (使用 Matplotlib 离屏渲染) ---
    fig = plt.figure(figsize=(16, 9), dpi=100)
    plt.imshow(img_rgb)
    
    # 绘制点云，按距离(range)上色
    sc = plt.scatter(valid_points['u'], valid_points['v'], 
                     c=valid_points['range'], s=10, 
                     cmap='jet', vmin=0, vmax=80) # vmin/max 固定量程使颜色统一
    
    plt.axis('off')
    
    # --- 5. 保存结果 ---
    save_path = os.path.join(output_folder, img_name)
    # bbox_inches='tight' 移除白边
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    # 释放内存，防止批量处理时内存溢出
    plt.close(fig)

print(f"处理完成！结果已保存至: {output_folder}")