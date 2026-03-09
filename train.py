"""
train.py — Train the LungCancerCNN on your dataset.

Expects dataset folder structure:
    data/
      train/
        normal/
        stage1/
        stage2/
        stage3/
        stage4/
      val/
        normal/
        stage1/
        stage2/
        stage3/
        stage4/

Run:
    python train.py --data_dir ./data --epochs 30 --batch_size 16
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# ─── Argument Parser ──────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Lung Cancer CNN")
parser.add_argument("--data_dir",   default="./data",   help="Path to dataset root")
parser.add_argument("--epochs",     type=int, default=30)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--output",     default="model_weights.pth")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ─── Data Transforms ──────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─── Datasets ─────────────────────────────────────────────
train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), train_transform)
val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"),   val_transform)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"[INFO] Classes: {train_ds.classes}")
print(f"[INFO] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

# ─── Model ────────────────────────────────────────────────
from app import LungCancerCNN

model = LungCancerCNN(num_classes=5).to(DEVICE)

# Freeze backbone initially, train head only
for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.backbone.fc.parameters():
    param.requires_grad = True

# ─── Loss, Optimizer, Scheduler ───────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

# ─── Training Loop ────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(1, args.epochs + 1):

    # ── Unfreeze backbone after epoch 5 ──
    if epoch == 6:
        print("[INFO] Unfreezing backbone for fine-tuning...")
        for param in model.backbone.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - 5)

    # ── Train ──
    model.train()
    train_loss, train_correct = 0.0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * imgs.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_loss /= len(train_ds)
    train_acc   = train_correct / len(train_ds) * 100

    # ── Validate ──
    model.eval()
    val_loss, val_correct = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss    += loss.item() * imgs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_ds)
    val_acc   = val_correct / len(val_ds) * 100
    scheduler.step()

    print(f"Epoch [{epoch:02d}/{args.epochs}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    # ── Save best model ──
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), args.output)
        print(f"  ✅ Saved best model → {args.output} (Val Acc: {val_acc:.2f}%)")

print(f"\n[DONE] Best Validation Accuracy: {best_val_acc:.2f}%")

# ─── Final Report ─────────────────────────────────────────
model.load_state_dict(torch.load(args.output, map_location=DEVICE))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=train_ds.classes))
