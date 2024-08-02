# 数据目录
import os
import shutil
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

METAINFO = {
    "classes": (
        "unlabelled",
        "asphalt",
        "dirt",
        "mud",
        "water",
        "gravel",
        "other-terrain",
        "tree-trunk",
        "tree-foliage",
        "bush",
        "fence",
        "structure",
        "pole",
        "vehicle",
        "rock",
        "log",
        "other-object",
        "sky",
        "grass",
    ),
    "palette": [
        (0, 0, 0),
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (230, 25, 75),
        (0, 128, 128),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (250, 190, 190),
        (0, 0, 128),
        (128, 128, 128),
    ],
    "cidx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
}

image_dir = 'C:/Users/Jerry/PycharmProjects9491/9517groupproject ++/WildScenes2d/K-01/image'
mask_dir = 'C:/Users/Jerry/PycharmProjects9491/9517groupproject ++/WildScenes2d/K-01/indexLabel'
output_image_dir = './WildScenes2d/K-01_Sample/images'
output_mask_dir = './WildScenes2d/K-01_Sample/masks'

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# 检查目录是否存在
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Image directory not found: {image_dir}")

if not os.path.exists(mask_dir):
    raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

# 获取所有图像和掩码文件名
all_images = sorted(os.listdir(image_dir))
all_masks = sorted(os.listdir(mask_dir))
print(f"Total number of images: {len(all_images)}")

# 检查图像和掩码文件是否对应
assert len(all_images) == len(all_masks)

# 类别总数
num_classes = 19

# 计算类别分布函数
def compute_class_presence(mask_path, num_classes):
    mask = np.array(Image.open(mask_path))
    class_presence = np.isin(np.arange(num_classes), mask)
    return class_presence

# 计算每个掩码的类别存在
print("Computing class presence for each mask...")
class_presences = []
for mask in tqdm(all_masks, desc="Processing masks"):
    class_presences.append(compute_class_presence(os.path.join(mask_dir, mask), num_classes))

# 将每个图像的类别信息转换为唯一的标签
unique_class_combinations, unique_indices = np.unique(class_presences, axis=0, return_inverse=True)

# 将图像文件名和掩码文件名与类别标签结合
data = pd.DataFrame({
    'image': all_images,
    'mask': all_masks,
    'label': unique_indices
})

# 使用全部数据
sampled_data = data

# 打印结果
print(f"Total number of images: {len(sampled_data)}")
print(f"Total number of masks: {len(sampled_data)}")

# 复制所有图像和掩码到输出目录
print("Copying images and masks to output directory...")
for _, row in tqdm(sampled_data.iterrows(), total=len(sampled_data), desc="Copying files"):
    shutil.copy(os.path.join(image_dir, row['image']), os.path.join(output_image_dir, row['image']))
    shutil.copy(os.path.join(mask_dir, row['mask']), os.path.join(output_mask_dir, row['mask']))

print("All images and masks have been copied to the output directory.")