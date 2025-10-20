import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

def seed_everything(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic behavior for cuda (may be slower)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

def plot(val_losses, val_accuracies, train_losses, dest=None):

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 3))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set(title='Training and Validation Loss', xlabel='Epochs', ylabel='Loss')
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.set(title='Validation Accuracy', xlabel='Epochs', ylabel='Accuracy (%)')
    ax2.legend()

    # collect valid numeric accuracies
    try:
        accs = np.array([float(a) for a in val_accuracies if a not in (None, "")])
        accs = accs[np.isfinite(accs)]
    except Exception:
        accs = np.array([])

    if accs.size:
        best = accs.max()
        n = min(20, accs.size)
        mean_last = accs[-n:].mean()
        ax2.text(0.02, 0.95, f"Best: {best:.2f}%\nMean(last {n}): {mean_last:.2f}%",
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    plt.tight_layout()
    if dest:
        plt.savefig(dest)
    plt.show()
    plt.close()

def plot_from_csv(source, dest):
    df = pd.read_csv(source)
    val_losses = df['val_loss'].tolist()
    val_accuracies = df['val_acc'].tolist()
    train_losses = df['train_loss'].tolist()
    plot(val_losses, val_accuracies, train_losses, dest=dest)