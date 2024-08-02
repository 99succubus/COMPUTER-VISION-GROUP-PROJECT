import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from model import get_model
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_miou(outputs, targets, num_classes):
    outputs = outputs.argmax(dim=1)
    if targets.dim() == 4:
        targets = targets.squeeze(1)
    iou_list = []
    for cls in range(num_classes):
        pred_inds = outputs == cls
        target_inds = targets == cls
        intersection = (pred_inds & target_inds).float().sum()
        union = (pred_inds | target_inds).float().sum()
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_list.append(iou.item())
    return np.mean(iou_list)

output_dir = './WildScenes2d/K-01_Sample/cache'
batch_size = 8
epochs = 1000
initial_lr = 5e-5
num_classes = 19
weight_decay = 1e-4
T_max = epochs
min_lr = 1e-7

if not os.path.exists(output_dir):
    raise FileNotFoundError(f"Cache directory not found: {output_dir}")

logger.info(f"Files in {output_dir}:")
logger.info(os.listdir(output_dir))

try:
    X_train = np.load(os.path.join(output_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(output_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(output_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(output_dir, 'y_val.npy'))
except FileNotFoundError as e:
    logger.error(f"Error loading data: {e}")
    raise

def preprocess_labels(labels, num_classes):
    if labels.ndim == 4:
        labels = labels[..., 0]
    return np.clip(labels, 0, num_classes - 1)

y_train = preprocess_labels(y_train, num_classes)
y_val = preprocess_labels(y_val, num_classes)

X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

model = get_model(in_channels=3, out_channels=num_classes).to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr)

train_losses, val_losses, train_mious, val_mious, lr_history = [], [], [], [], []
best_val_loss, best_val_miou = float('inf'), 0
best_model_path = 'best_yolo_model.pth'

early_stopping_patience = 30
no_improve_epochs = 0

try:
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_miou = 0.0

        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}')
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            try:
                outputs = model(imgs)
                if outputs is None:
                    logger.error("Model returned None. Skipping this batch.")
                    continue
            except Exception as e:
                logger.error(f"Error during forward pass: {e}")
                logger.error(f"Input shape: {imgs.shape}")
                continue

            loss = criterion(outputs, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
            epoch_miou += compute_miou(outputs, masks, num_classes) * imgs.size(0)
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        pbar.close()

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        avg_train_miou = epoch_miou / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        train_mious.append(avg_train_miou)

        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                outputs = model(imgs)
                if outputs is None:
                    logger.error("Model returned None during validation. Skipping this batch.")
                    continue

                batch_loss = criterion(outputs, masks)
                batch_miou = compute_miou(outputs, masks, num_classes)

                val_loss += batch_loss.item() * imgs.size(0)
                val_miou += batch_miou * imgs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_miou = val_miou / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        val_mious.append(avg_val_miou)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lr_history.append(current_lr)

        logger.info(f'Epoch {epoch + 1}/{epochs}:')
        logger.info(f'  Train Loss: {avg_train_loss:.4f}, Train mIoU: {avg_train_miou:.4f}')
        logger.info(f'  Val Loss: {avg_val_loss:.4f}, Val mIoU: {avg_val_miou:.4f}')
        logger.info(f'  Current Learning Rate: {current_lr:.6f}')

        improved = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            improved = True
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            improved = True

        if improved:
            torch.save(model.state_dict(), best_model_path)
            logger.info(f'  Saved best model with validation loss: {best_val_loss:.4f} and mIoU: {best_val_miou:.4f}')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break

        logger.info(f'  No improvement for {no_improve_epochs} epochs')

except Exception as e:
    logger.error(f"Unexpected error during training: {e}")
    import traceback
    logger.error(traceback.format_exc())

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(train_mious, label='Train mIoU')
plt.plot(val_mious, label='Validation mIoU')
plt.title('Training and Validation mIoU')
plt.xlabel('Epochs')
plt.ylabel('mIoU')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(lr_history)
plt.title('Learning Rate Schedule')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

logger.info("Training completed. Best model saved as 'best_yolo_model.pth'")
logger.info("Training history plot saved as 'training_history.png'")

def evaluate_model(model, test_loader, device, num_classes):
    model.eval()
    total_miou = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)

            batch_miou = compute_miou(outputs, masks, num_classes)

            total_miou += batch_miou * imgs.size(0)
            total_samples += imgs.size(0)

    avg_miou = total_miou / total_samples
    return avg_miou

best_model = get_model(in_channels=3, out_channels=num_classes).to(device)
best_model.load_state_dict(torch.load(best_model_path))

test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_miou = evaluate_model(best_model, test_loader, device, num_classes)

logger.info(f"Test mIoU: {test_miou:.4f}")