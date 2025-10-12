from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os, random, csv
from PIL import Image

def build_dataloaders(
    data_root: str,
    size: int = 224,
    batch_size: int = 10,
    num_workers: int = 7,
    seed: int = 67,
    val_split: float = 0.1,
    test_split: float = 0.1,
):
    assert os.path.isdir(data_root), f"Cant find data path: {data_root}"

    samples, class_to_idx, class_names = discover_samples(data_root, limit_per_class=False)

    # # Make sure data exists DEBUGGING
    assert samples, "No Img found in the dataset root."


    train, val, test = split_samples(samples, class_names, val_split, test_split, seed)
    

    save_split_csv("experiments/splits", train, val, test)

    # Create transforms
    transforms_dict = make_transforms(size, seed)

    # Datasets
    train_ds = FileListDataset(train)
    val_ds = FileListDataset(val)
    test_ds = FileListDataset(test)

    train_ds.transform = transforms_dict["train"]
    val_ds.transform = transforms_dict["val"]
    test_ds.transform = transforms_dict["test"]

    print(f"Discovered {len(samples)} samples in {len(class_names)} classes.")
    print(f"Train/val/test split: {len(train)}/{len(val)}/{len(test)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meta = {
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "split_counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "size": size,
        "num_classes": len(class_names),
    }

    return train_loader, val_loader, test_loader, meta


def discover_samples(data_root: str, limit_per_class: bool = False):
    samples = []
    class_to_idx = {}
    class_names = []

    for class_name in sorted(os.listdir(data_root)):
        class_path = os.path.join(data_root, class_name)
        if not os.path.isdir(class_path):
            continue

        class_idx = len(class_to_idx)
        class_to_idx[class_name] = class_idx
        class_names.append(class_name)


        all_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpeg'))
        ]

        # Limit samples per class for quicker testing/debugging
        if limit_per_class:
            all_files = all_files[:200]

        for fname in all_files:
            filepath = os.path.join(class_path, fname)
            samples.append((filepath, class_idx))

    return samples, class_to_idx, class_names

def split_samples(samples, class_names, val_split: float, test_split: float, seed: int):
    random.seed(seed)

    train_samples = []
    val_samples = []
    test_samples = []

    class_to_samples = {class_name: [] for class_name in class_names}
    for filepath, class_idx in samples:
        class_name = class_names[class_idx]
        class_to_samples[class_name].append((filepath, class_idx))

    for class_name, items in class_to_samples.items():
        random.shuffle(items)

        num_val = int(len(items) * val_split)
        num_test = int(len(items) * test_split)

        train, val, test = items[num_val + num_test:], items[:num_val], items[num_val:num_val + num_test]
        train_samples += train
        val_samples += val
        test_samples += test

    assert len(train_samples) + len(val_samples) + len(test_samples) == len(samples), "Split mismatch: some samples lost or duplicated!"

    return train_samples, val_samples, test_samples

def save_split_csv(out_dir: str, train, val, test):
    os.makedirs(out_dir, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        csv_path = os.path.join(out_dir, f"{split_name}_split.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label", "class_name"])

            for filepath, label in split_data:
                class_name = os.path.basename(os.path.dirname(filepath))
                writer.writerow([filepath, label, class_name])
        print(f"Saved {len(split_data)} samples to {csv_path}")

def load_split_csv(csv_path: str):
    # For future use
    samples = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            filepath, label, class_name = row
            samples.append((filepath, int(label)))
    return samples

def make_transforms(size: int, seed: int):

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    random.seed(seed)

    baseTransform = transforms.Compose([
        transforms.Resize((int(size + 32), int(size + 32))),
        transforms.RandomResizedCrop(int(size)), # TODO: Replace with RandomResizedCrop
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return {
        "train": baseTransform,
        "val": baseTransform,
        "test": baseTransform,
    }
    
class FileListDataset(Dataset):
    def __init__(self, file_list):
        self.items = file_list
        self.transform = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filepath, label = self.items[idx]
        try:
            img = Image.open(filepath).convert("RGB")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            img = Image.new("RGB", (224, 224), (0, 0, 0))


        if self.transform:
            img = self.transform(img)

        return img, label


def main():
    # Example usage
    data_root = "data"
    build_dataloaders(data_root)

if __name__ == "__main__":
    main()