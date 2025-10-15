import torch
import torch.nn as nn
import torch.optim as optim
import csv

from utils import seed_everything
from eval import evaluate

def train_x_epoch(model, dataloader, criterion, optimizer, epochs, scheduler=None, val_loader=None, seed=67, augTier=0):
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

            # Print every 20% of an epoch
            #if i % (len(dataloader) // 5) == 0:
            #    avg_so_far = total_loss / samples
            #    acc_so_far = 100 * correct / samples
            #    percent = int((i + 1) / len(dataloader) * 100)
            #    print(f'{percent}% - Batch {i+1}/{len(dataloader)}  loss: {avg_so_far:.4f}  acc: {acc_so_far:.2f}%  lr: {optimizer.param_groups[0]["lr"]:.6f}')

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
                checkpoint_model(model, optimizer, epoch, is_best, augTier)
        else:
            # fallback to train_acc if no val set
            is_best = train_acc > best_val_acc
            if is_best:
                best_val_acc = train_acc
                checkpoint_model(model, optimizer, epoch, is_best, augTier)

        metrics = { # TODO TEST IF WORK LATER WHEN HOME
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss if val_loss is not None else "",
            "val_acc": val_acc if val_acc is not None else "",
            "lr": optimizer.param_groups[0]['lr'],
        }
        log_metrics(metrics, augTier)


        epoch_train_loss = total_loss / max(1, samples)
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    print('Finished Training')
    return val_losses, val_accuracies, train_losses

def checkpoint_model(model, optimizer, epoch, is_best=False, augTier=0):
    # if model is best so far, save as best
    best_path = f"../experiments/checkpoints/model_epoch_best_Tier{augTier}.pth"
    last_path = f"../experiments/checkpoints/model_epoch_last_Tier{augTier}.pth"
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, last_path)

    if is_best:
        torch.save(checkpoint, best_path)
        print(f"Model checkpoint saved to {best_path}")

def log_metrics(metrics_dict, augTier=0):
    logfile = f"../experiments/logs/training_log_Tier{augTier}.csv"
    with open(logfile, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:  # file is empty, write header
            writer.writerow(metrics_dict.keys())
        writer.writerow(metrics_dict.values())

def main():
    from data import build_dataloaders
    from model import SmallCNN

    # Hyperparameters
    data_root = "data"  # Update this path
    batch_size = 64
    num_workers = 8
    learning_rate = 3e-3
    num_epochs = 10

    # Build dataloaders
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentationTier=0
    )

    # Initialize model, loss function, optimizer
    model = SmallCNN(in_channels=3, num_classes=len(meta["class_names"]))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)
    scheduler = None

    # Train the model
    print("Starting training...")
    val_losses, val_accuracies, train_losses = train_x_epoch(model, train_loader, criterion, optimizer, num_epochs, scheduler, val_loader, seed=67)

if __name__ == "__main__":
    main()