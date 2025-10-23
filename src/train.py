import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
from src.utils import ensure_dataset_exists
import yaml

from src.utils import seed_everything
from src.eval import evaluate

def train_x_epoch(model, dataloader, criterion, optimizer, epochs, scheduler=None,
                  val_loader=None, seed=67, augTier=0,
                  ckpt_out_dir="experiments/checkpoints", log_out_dir="experiments/logs"):
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_losses = []
    val_accuracies = []
    train_losses = []
    best_val_acc = -1.0
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"Epoch {epoch+1}/{epochs}  lr={optimizer.param_groups[0]['lr']:.6f}")
        model.train()

        total_loss = 0.0
        correct = 0
        samples = 0

        for i, data in enumerate(dataloader, 0):
            print(f"  Batch {i+1}/{len(dataloader)}", end='\r')
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(1) == labels).sum().item()
            samples += batch_size

        train_loss = total_loss / samples
        train_acc = 100 * correct / samples

        val_loss, val_acc = None, None
        if val_loader:
            print("Validation Results:")
            # ensure evaluation also runs on device
            model.to(device)
            val_loss, val_acc = evaluate(model, val_loader, criterion)


        if scheduler:
            scheduler.step()

        is_best = False
        if val_acc is not None:
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                checkpoint_model(model, optimizer, epoch, is_best, augTier, out_dir=ckpt_out_dir)
        else:
            # fallback to train_acc if no val set
            is_best = train_acc > best_val_acc
            if is_best:
                best_val_acc = train_acc
                checkpoint_model(model, optimizer, epoch, is_best, augTier, out_dir=ckpt_out_dir)

        metrics = { # TODO TEST IF WORK LATER WHEN HOME
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss if val_loss is not None else "",
            "val_acc": val_acc if val_acc is not None else "",
            "lr": optimizer.param_groups[0]['lr'],
        }
        log_metrics(metrics, augTier, out_dir=log_out_dir)


        epoch_train_loss = total_loss / max(1, samples)
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    print('Finished Training')
    return val_losses, val_accuracies, train_losses

def checkpoint_model(model, optimizer, epoch, is_best=False, augTier=0, out_dir="experiments/checkpoints"):
    # Save last and (optionally) best model checkpoints into out_dir
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"model_epoch_best_Tier{augTier}.pth")
    last_path = os.path.join(out_dir, f"model_epoch_last_Tier{augTier}.pth")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, last_path)

    if is_best:
        torch.save(checkpoint, best_path)
        print(f"✅ Saved best model: {best_path}")

def log_metrics(metrics_dict, augTier=0, out_dir="experiments/logs"):
    os.makedirs(out_dir, exist_ok=True)
    logfile = os.path.join(out_dir, f"training_log_Tier{augTier}.csv")
    mode = "w" if int(metrics_dict.get("epoch", 0)) == 1 else "a"
    with open(logfile, mode=mode, newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:  # file is empty, write header
            writer.writerow(list(metrics_dict.keys()))
        writer.writerow(list(metrics_dict.values()))

def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def main(config_path="configs/baseline.yaml"):

    ensure_dataset_exists(target_dir="./data")

    cfg = load_config(config_path)
    from src.data import build_dataloaders
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    ckpt_cfg = cfg.get("checkpoints", {"out_dir": "experiments/checkpoints"})
    log_cfg = cfg.get("logging", {"out_dir": "experiments/logs"})

    # Prepare output dirs
    os.makedirs(ckpt_cfg["out_dir"], exist_ok=True)
    os.makedirs(log_cfg["out_dir"], exist_ok=True)

    # Build dataloaders
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        data_root=data_cfg["data_root"],
        size=data_cfg["size"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        seed=cfg["seed"],
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
        augmentationTier=data_cfg["augmentationTier"],
    )

    num_classes = meta["num_classes"]

    # Initialize model, loss function, optimizer
    from src.model import build_model
    model = build_model(
        name=model_cfg.get("name", "mobilenet_v2"),
        num_classes=model_cfg.get("num_classes", num_classes),
    )
    criterion = nn.CrossEntropyLoss()
    if train_cfg["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=train_cfg["learning_rate"],
                               weight_decay=train_cfg["weight_decay"])
    elif train_cfg["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=train_cfg["learning_rate"],
                              momentum=0.9,
                              weight_decay=train_cfg["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg['optimizer']}")

    scheduler = None
    if train_cfg.get("use_scheduler", False):
        if train_cfg["scheduler"]["name"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=train_cfg["scheduler"]["T_max"]
            )

    # Train the model
    print("Starting training...")
    train_x_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=train_cfg["epochs"],
        scheduler=scheduler,
        val_loader=val_loader,
        seed=cfg["seed"],
        augTier=data_cfg["augmentationTier"],
        ckpt_out_dir=ckpt_cfg["out_dir"],
        log_out_dir=log_cfg["out_dir"],
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train model from YAML config")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    main(args.config)