import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# Ensure project root is on sys.path so package imports work when running this script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Package imports
from workshop_detection.ssdlite_mobilenetv2.model import ssdlite_mobilenetv2
from workshop_detection.ssdlite_mobilenetv2.anchors import generate_ssd_priors, cxcywh_to_xyxy
from workshop_detection.ssdlite_mobilenetv2.losses import MultiBoxLoss
from workshop_detection.dataset import DetectionDataset
import numpy as np
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import subprocess
import logging

# --- Настройки ---
NUM_CLASSES = 2  # фон + мяч (labels: 1..)
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = (320, 320)


def collate_fn(batch):
    # batch: list of samples {'img': HxWx3 float numpy, 'annot': Nx5 numpy}
    images = []
    targets = []
    for sample in batch:
        img = sample['img']
        # resize if needed
        img_t = transforms.functional.to_tensor(img).float()
        images.append(img_t)
        ann = sample['annot']
        if isinstance(ann, np.ndarray):
            # ann columns: xmin_abs, ymin_abs, xmax_abs, ymax_abs, class
            boxes = torch.tensor(ann[:, :4], dtype=torch.float32)
            labels = torch.tensor(ann[:, 4], dtype=torch.long)
        else:
            boxes = torch.tensor([], dtype=torch.float32).view(0,4)
            labels = torch.tensor([], dtype=torch.long)
        targets.append({'boxes': boxes, 'labels': labels})
    images = torch.stack([transforms.functional.resize(img, IMAGE_SIZE) for img in images])
    # normalize (ImageNet)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    images = (images - mean) / std
    return images, targets


# --- Датасет и загрузчики ---
dataset_dir = os.path.join(project_root, 'workshop_detection')
train_json = os.path.join(dataset_dir, 'train.json')
val_json = os.path.join(dataset_dir, 'val.json')
if not (os.path.exists(train_json) and os.path.exists(val_json)):
    logging.info('train.json or val.json not found — creating dummy dataset')
    # run the helper to generate dummy json files
    helper = os.path.join(dataset_dir, 'make_dummy_dataset.py')
    subprocess.run([sys.executable, helper], check=True)

train_dataset = DetectionDataset(train_json)
val_dataset = DetectionDataset(val_json)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --- Модель ---
model = ssdlite_mobilenetv2(num_classes=NUM_CLASSES, pretrained_backbone=True)
model.to(DEVICE)

# --- Priors и Loss ---
feature_shapes = model.get_feature_map_shapes(IMAGE_SIZE)
priors = generate_ssd_priors(feature_shapes, image_size=IMAGE_SIZE)
criterion = MultiBoxLoss(priors, iou_threshold=0.5)

# --- Оптимизатор и AMP ---
optimizer = optim.AdamW(model.parameters(), lr=LR)
scaler = GradScaler()


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for images, targets in train_loader:
        images = images.to(DEVICE)
        # targets: list of dicts with boxes in absolute coords -> normalize to [0,1]
        normalized_targets = []
        for t in targets:
            boxes = t['boxes']
            if boxes.numel() == 0:
                normalized_targets.append({'boxes': torch.zeros((0,4), device=DEVICE), 'labels': torch.zeros((0,), dtype=torch.long, device=DEVICE)})
                continue
            h, w = IMAGE_SIZE
            # assume boxes are given in pixels; convert to normalized xyxy
            boxes_norm = boxes.clone()
            boxes_norm[:, 0] /= w
            boxes_norm[:, 2] /= w
            boxes_norm[:, 1] /= h
            boxes_norm[:, 3] /= h
            normalized_targets.append({'boxes': boxes_norm.to(DEVICE), 'labels': t['labels'].to(DEVICE)})

        optimizer.zero_grad()
        with autocast():
            loc_preds, conf_preds = model(images)
            loss = criterion((loc_preds, conf_preds), normalized_targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss:.4f}")
    torch.save(model.state_dict(), f'ssdlite_epoch{epoch+1}.pth')

print("Training finished")
