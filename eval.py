import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import get_model  # 从model.py导入get_model函数
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_iou(pred, target, num_classes, smooth=1e-6):
    ious = []
    for cls in range(num_classes):
        pred_class = pred == cls
        target_class = target == cls
        intersection = (pred_class & target_class).sum().float()
        union = (pred_class | target_class).sum().float()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou.item())
    return ious


def compute_miou(model, dataloader, device, num_classes):
    model.eval()
    total_ious = np.zeros(num_classes)
    num_samples = 0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Computing mIoU"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            preds = torch.argmax(outputs, dim=1)

            ious = compute_iou(preds, masks, num_classes)
            total_ious += np.array(ious) * images.size(0)
            num_samples += images.size(0)

    mean_ious = total_ious / num_samples
    miou = np.mean(mean_ious)

    return miou, mean_ious


def compute_accuracy(y_true, y_pred):
    correct = torch.sum(y_true == y_pred).float()
    total = y_true.numel()
    return (correct / total).item()


# 参数设置
output_dir = r'./WildScenes2d/K-01_Sample/cache'
num_classes = 19

# 加载数据
X_test = np.load(os.path.join(output_dir, 'X_val.npy'))
y_test = np.load(os.path.join(output_dir, 'y_val.npy'))

# 转换为Tensor
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
y_test = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 检查是否可以使用CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 模型实例化并加载权重
model = get_model(in_channels=3, out_channels=num_classes).to(device)
model.load_state_dict(torch.load('best_yolo_model.pth', map_location=device))
model.eval()

# 计算 mIoU
miou, class_ious = compute_miou(model, test_loader, device, num_classes)

# 计算准确度
all_preds = []
all_labels = []
with torch.no_grad():
    for imgs, masks in tqdm(test_loader, desc="Computing accuracy"):
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)

        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        preds = torch.argmax(outputs, dim=1)

        all_preds.append(preds)
        all_labels.append(masks)

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
accuracy = compute_accuracy(all_labels, all_preds)

# 类别名称
class_names = ['_background_', 'bush', 'dirt', 'fence', 'grass', 'gravel', 'log', 'mud', 'other-object',
               'other-terrain', 'rock', 'sky', 'structure', 'tree-foliage', 'tree-trunk', 'water']

# 确保类别名称数量正确
if len(class_names) != num_classes:
    class_names = [f'Class_{i}' for i in range(num_classes)]

# 按IoU分数排序类别
sorted_indices = np.argsort(class_ious)[::-1]
sorted_class_names = [class_names[i] for i in sorted_indices]
sorted_iou_scores = [class_ious[i] for i in sorted_indices]

# 绘制IoU分数
plt.figure(figsize=(12, 8))
bars = plt.barh(sorted_class_names, sorted_iou_scores, height=0.6)
plt.title(f'mIoU = {miou:.2%}', fontsize=16)
plt.xlabel('Intersection over Union', fontsize=12)
plt.xlim(0, 1)
plt.gca().invert_yaxis()  # 反转y轴，使最高IoU在顶部

# 在条形上添加数值标签
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
             ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('iou_scores.png', dpi=300, bbox_inches='tight')
plt.close()

# 打印结果
print(f'mIoU: {miou:.4f}')
print(f'Accuracy: {accuracy:.4f}')
for class_name, iou in zip(sorted_class_names, sorted_iou_scores):
    print(f'{class_name}: {iou:.4f}')