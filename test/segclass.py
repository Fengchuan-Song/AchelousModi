import cv2
from PIL import Image
import numpy as np

image_path = r'H:\dataset\WaterScenes\semantic\SegmentationClass/00001.png'

mask = np.asarray(Image.open(image_path))
print(mask.shape)
print(mask[-1][-1])
print(mask.max())