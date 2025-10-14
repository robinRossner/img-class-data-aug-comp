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

def plot(val_losses, val_accuracies, train_losses, dest=None):

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

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