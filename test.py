import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import get_model
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

METAINFO = {
    "classes": (
        "unlabelled", "asphalt", "dirt", "mud", "water", "gravel", "other-terrain",
        "tree-trunk", "tree-foliage", "bush", "fence", "structure", "pole", "vehicle",
        "rock", "log", "other-object", "sky", "grass",
    ),
    "palette": [
        (0, 0, 0), (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
        (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (230, 25, 75),
        (0, 128, 128), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
        (128, 128, 0), (250, 190, 190), (0, 0, 128), (128, 128, 128),
    ],
}


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


def predict_with_sliding_window(model, image, window_size, stride):
    _, h, w = image.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')

    output = torch.zeros((num_classes, h + pad_h, w + pad_w), device=image.device)
    count = torch.zeros((1, h + pad_h, w + pad_w), device=image.device)

    for i in range(0, h + pad_h - window_size + 1, stride):
        for j in range(0, w + pad_w - window_size + 1, stride):
            patch = image_padded[:, i:i + window_size, j:j + window_size].unsqueeze(0)
            with torch.no_grad():
                pred = model(patch)
            output[:, i:i + window_size, j:j + window_size] += pred.squeeze()
            count[:, i:i + window_size, j:j + window_size] += 1

    output = output / count
    return output[:, :h, :w]


def visualize_segmentation(image, true_mask, pred_mask, class_names, num_classes):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 原图
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # 真实分割图
    color_mask = np.zeros((*true_mask.shape, 3), dtype=np.float32)
    for i in range(num_classes):
        if i < len(METAINFO["palette"]):
            color_mask[true_mask == i] = np.array(METAINFO["palette"][i]) / 255.0
        else:
            # 对于超出范围的类别,归类到"other-object"(索引16)
            color_mask[true_mask == i] = np.array(METAINFO["palette"][16]) / 255.0
    ax2.imshow(color_mask)
    ax2.set_title('True Segmentation')
    ax2.axis('off')

    # 预测分割图
    color_mask = np.zeros((*pred_mask.shape, 3), dtype=np.float32)
    for i in range(num_classes):
        if i < len(METAINFO["palette"]):
            color_mask[pred_mask == i] = np.array(METAINFO["palette"][i]) / 255.0
        else:
            # 对于超出范围的类别,归类到"other-object"(索引16)
            color_mask[pred_mask == i] = np.array(METAINFO["palette"][16]) / 255.0
    ax3.imshow(color_mask)
    ax3.set_title('Predicted Segmentation')
    ax3.axis('off')

    # 添加颜色图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(color) / 255.0) for color in
               METAINFO["palette"][:num_classes]]
    plt.figlegend(handles, class_names, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.1))

    plt.tight_layout()
    plt.savefig('segmentation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


# 参数设置
output_dir = r'./WildScenes2d/K-01_Sample/cache'
num_classes = 19
batch_size = 4  # 减小batch size以适应更高的分辨率

# 加载数据
X_test = np.load(os.path.join(output_dir, 'X_val.npy'))
y_test = np.load(os.path.join(output_dir, 'y_val.npy'))

# 转换为Tensor
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
y_test = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 检查是否可以使用CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 模型实例化并加载权重
model = get_model(in_channels=3, out_channels=num_classes)
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
        outputs = predict_with_sliding_window(model, imgs, window_size=256, stride=128)

        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        preds = torch.argmax(outputs, dim=1)

        all_preds.append(preds)
        all_labels.append(masks)

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
accuracy = compute_accuracy(all_labels, all_preds)

# 类别名称
class_names = METAINFO["classes"][:num_classes]

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
plt.gca().invert_yaxis()

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

# 可视化一些预测结果
num_samples_to_visualize = 5
for i in range(num_samples_to_visualize):
    idx = np.random.randint(0, len(X_test))
    image = X_test[idx].permute(1, 2, 0).cpu().numpy()
    true_mask = y_test[idx].cpu().numpy()

    with torch.no_grad():
        pred = model(X_test[idx:idx + 1].to(device))
        pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

    visualize_segmentation(image, true_mask, pred_mask, class_names, num_classes)
    plt.savefig(f'segmentation_comparison_{i + 1}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Evaluation completed. Results saved as 'iou_scores.png' and 'segmentation_comparison_*.png'")