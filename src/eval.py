import torch
import os

def load_model_checkpoint(model, checkpoint_path):
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        print("Checkpoint format not recognized.")
        return None
    print(f"Loaded model weights from {checkpoint_path}")
    return checkpoint

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0

    device = next(model.parameters()).device
    model.to(device)
    with torch.no_grad():
        for input, labels in dataloader:
            input, labels = input.to(device), labels.to(device)
            outputs = model(input)
            loss = criterion(outputs, labels)
        
            total_loss += loss.item() * input.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            n += labels.size(0)


    avg_loss = total_loss / n
    accuracy = 100 * correct / n
    print(f'Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy


def evaluateClassAccuracy(model, loader, criterion, num_classes=10):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    # Initialize counters for per-class accuracy
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    device = next(model.parameters()).device
    model.to(device)
    criterion.to(device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss) * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            # Update per-class counters
            equal = (preds == y)
            for i in range(len(y)):
                label = int(y[i].item())
                class_correct[label] += int(equal[i].item())
                class_total[label] += 1

    # Calculate per-class accuracies
    class_accuracies = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 for i in range(num_classes)]
    print(f'Overall Accuracy: {100 * correct / total:.2f}%')
    print("Per-Class Accuracies:")
    for i in range(num_classes):
        print(f'Class {i}: {100 * class_accuracies[i]:.2f}%')
    return class_accuracies