import os
from pathlib import Path
p = Path.cwd().resolve()
while p != p.parent and not (p / 'requirements.txt').exists() and not (p / 'data').exists():
    p = p.parent

import sys
sys.path.insert(0, str(p / "src"))

print(f"Current working directory: {sys.path[0]}")

#check which device is available if multiple GPUs are available, list which ones are free
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    print("Available CUDA devices:")
    for i in range(torch.cuda.device_count()):
        print(f"  {i}: {torch.cuda.get_device_name(i)}")
    print(f"Using GPU: {torch.cuda.current_device()}")

from data import build_dataloaders

train_loader0, val_loader0, test_loader0, meta0 = build_dataloaders(data_root=os.path.join(p, 'data'), size=128, batch_size=64, augmentationTier=0)
train_loader1, val_loader1, test_loader1, meta1 = build_dataloaders(data_root=os.path.join(p, 'data'), size=128, batch_size=64, augmentationTier=1)
train_loader2, val_loader2, test_loader2, meta2 = build_dataloaders(data_root=os.path.join(p, 'data'), size=128, batch_size=64, augmentationTier=2)
train_loader3, val_loader3, test_loader3, meta3 = build_dataloaders(data_root=os.path.join(p, 'data'), size=128, batch_size=64, augmentationTier=3)
model_out = os.path.join(p, 'experiments')

print("----------Finished loading data----------")

from train import train_x_epoch
import torch
from model import build_model
model = build_model(in_channels=3, num_classes=10, name="mobilenet_v2").to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

val_losses0, val_accuracies0, train_losses0 = train_x_epoch(
    model, train_loader0, criterion=criterion, optimizer=optimizer, epochs=10, scheduler=scheduler, val_loader=val_loader0, seed=67, augTier=0
)
val_losses1, val_accuracies1, train_losses1 = train_x_epoch(
    model, train_loader1, criterion=criterion, optimizer=optimizer, epochs=10, scheduler=scheduler, val_loader=val_loader1, seed=67, augTier=1
)
val_losses2, val_accuracies2, train_losses2 = train_x_epoch(
    model, train_loader2, criterion=criterion, optimizer=optimizer, epochs=10, scheduler=scheduler, val_loader=val_loader2, seed=67, augTier=2
)
val_losses3, val_accuracies3, train_losses3 = train_x_epoch(
    model, train_loader3, criterion=criterion, optimizer=optimizer, epochs=10, scheduler=scheduler, val_loader=val_loader3, seed=67, augTier=3
)