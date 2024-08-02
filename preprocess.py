import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 参数设置
image_dir = './WildScenes2d/K-01_Sample/images'
mask_dir = './WildScenes2d/K-01_Sample/masks'
output_dir = './WildScenes2d/K-01_Sample/cache'
img_size = 640  # YOLO v8 默认输入大小
num_classes = 19

os.makedirs(output_dir, exist_ok=True)


def resize_with_aspect_ratio(image, target_size):
    """等比缩放图像，并在必要时进行填充"""
    iw, ih = image.size
    scale = min(target_size / iw, target_size / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.LANCZOS)
    new_image = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    new_image.paste(image, ((target_size - nw) // 2, (target_size - nh) // 2))

    return new_image


def preprocess_image(img_path, target_size):
    img = Image.open(img_path).convert('RGB')
    img = resize_with_aspect_ratio(img, target_size)
    img = np.array(img) / 255.0  # 归一化
    return img


def preprocess_mask(mask_path, target_size):
    mask = Image.open(mask_path)
    mask = resize_with_aspect_ratio(mask, target_size)
    mask = np.array(mask)
    return mask


# 获取文件列表
image_files = sorted(os.listdir(image_dir))
mask_files = sorted(os.listdir(mask_dir))

preprocessed_images = []
preprocessed_masks = []

print("预处理图像和掩码...")
for img_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files)):
    img_path = os.path.join(image_dir, img_file)
    mask_path = os.path.join(mask_dir, mask_file)

    img = preprocess_image(img_path, img_size)
    mask = preprocess_mask(mask_path, img_size)

    preprocessed_images.append(img)
    preprocessed_masks.append(mask)

preprocessed_images = np.array(preprocessed_images)
preprocessed_masks = np.array(preprocessed_masks)

# 打印独特的类别
unique_categories = np.unique(preprocessed_masks)
print(f"Unique categories in masks: {unique_categories.tolist()}")

# 保存预处理后的数据
print("保存预处理后的数据...")
np.save(os.path.join(output_dir, 'images.npy'), preprocessed_images)
np.save(os.path.join(output_dir, 'masks.npy'), preprocessed_masks)

# 划分数据集
print("划分数据集...")
X_train, X_val, y_train, y_val = train_test_split(preprocessed_images, preprocessed_masks, test_size=0.2,
                                                  random_state=42)

# 保存数据集划分结果
np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
np.save(os.path.join(output_dir, 'y_val.npy'), y_val)

print("预处理和数据划分完成")
print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")
print(f"图像形状: {X_train[0].shape}")
print(f"掩码形状: {y_train[0].shape}")