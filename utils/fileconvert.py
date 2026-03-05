import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


def convert_voc(image_list, xml_dir, image_root, class_list, output_dir):
    # 加载类别映射表
    with open(class_list, 'r', encoding='utf-8') as f:
        classes = []
        for line in f.readlines():
            stripped_line = line.strip()
            if stripped_line:
                classes.append(stripped_line)
    class_to_idx = {}
    for index, name in enumerate(classes):
        class_to_idx[name] = index

    # 读取图像列表
    with open(image_list, 'r', encoding='utf-8') as f:
        image_paths = []
        for line in f.readlines():
            stripped_line = line.strip()
            if stripped_line:
                image_paths.append(stripped_line[2:])
    
    results = []

    for image_path in tqdm(image_paths):
        name_entire = os.path.basename(image_path)
        name = name_entire.split('.')[0]
        xml_file = os.path.join(xml_dir, f"{name}.xml")

        image_path_entire = os.path.join(image_root, image_path)

        tree = ET.parse(xml_file)
        root = tree.getroot()

        bboxes = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_to_idx:
                continue

            class_id = class_to_idx[class_name]
            xml_box = obj.find('bndbox')

            b = (
                int(float(xml_box.find('xmin').text)),
                int(float(xml_box.find('ymin').text)),
                int(float(xml_box.find('xmax').text)),
                int(float(xml_box.find('ymax').text))
            )
            # 格式化为 x_min,y_min,x_max,y_max,class_id
            bboxes.append(f"{b[0]},{b[1]},{b[2]},{b[3]},{class_id}")

        if bboxes:
            line = f"{image_path_entire} {' '.join(bboxes)}"
            results.append(line)
    
    with open(output_dir, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))



if __name__ == "__main__":
    config = {
        "image_list": "H:/dataset/WaterScenes/val.txt",                                # 你的图片列表文件
        "xml_dir": "H:/dataset/WaterScenes/detection/xml/",      
        "image_root": "/data_ssd/datasets/WaterScenes/",                 # 图片根目录
        "class_list": "../model_data/waterscenes_benchmark.txt",                # 类别列表
        "output_dir": "H:/dataset/WaterScenes/MIPC/2007_val.txt"                  # 输出文件路径
    }

    convert_voc(**config)