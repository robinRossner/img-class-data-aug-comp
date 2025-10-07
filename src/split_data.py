#!/usr/bin/env python3
"""Create stratified train/val/test splits from a directory of class subfolders.

Outputs CSV files with columns: image_path,label,class_name for each split.

Usage: python src/split_data.py --data-dir ./data/104x128 --out-dir ./data_splits --train 0.7 --val 0.15 --test 0.15 --seed 42
"""
import argparse
import csv
import os
import random
from pathlib import Path
from collections import defaultdict


def discover_images(data_dir: Path):
    classes = []
    class_to_files = {}
    for entry in sorted(data_dir.iterdir()):
        if entry.is_dir():
            classes.append(entry.name)
            files = [p for p in sorted(entry.iterdir()) if p.is_file()]
            class_to_files[entry.name] = files
    return classes, class_to_files


def stratified_split(class_to_files, train_ratio, val_ratio, test_ratio, seed=None, min_per_split=1):
    if seed is not None:
        random.seed(seed)

    splits = {'train': [], 'val': [], 'test': []}

    for class_name, files in class_to_files.items():
        n = len(files)
        if n == 0:
            continue
        indices = list(range(n))
        random.shuffle(indices)

        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        n_test = n - n_train - n_val

        # Fix rounding issues by adjusting test size
        if n_test < 0:
            n_test = 0
            if n_val > 0:
                n_val = n - n_train

        # Ensure minimum samples per split when possible
        # If class too small, put at least one sample into train
        if n < (min_per_split * 3):
            # fallback: 80/20 split to train/test if very small
            n_train = max(1, n - 1)
            n_val = 0
            n_test = n - n_train

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val: n_train + n_val + n_test]

        for i in train_idx:
            splits['train'].append(files[i])
        for i in val_idx:
            splits['val'].append(files[i])
        for i in test_idx:
            splits['test'].append(files[i])

    return splits


def write_csv(split_list, out_path: Path, class_to_label):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label', 'class_name'])
        for p in split_list:
            class_name = p.parent.name
            label = class_to_label[class_name]
            writer.writerow([str(p), label, class_name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True, help='Path to folder containing class subfolders')
    parser.add_argument('--out-dir', type=Path, required=True, help='Output folder for CSVs')
    parser.add_argument('--train', type=float, default=0.7)
    parser.add_argument('--val', type=float, default=0.15)
    parser.add_argument('--test', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-per-split', type=int, default=1, help='Minimum items per split per class when possible')
    args = parser.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        parser.error('train+val+test must sum to 1.0')

    classes, class_to_files = discover_images(args.data_dir)
    class_to_label = {c: i for i, c in enumerate(classes)}

    # Print discovered classes and counts
    print(f'Found {len(classes)} classes')
    for c in classes:
        print(f'  {c}: {len(class_to_files.get(c, []))} images')

    splits = stratified_split(class_to_files, args.train, args.val, args.test, seed=args.seed, min_per_split=args.min_per_split)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(splits['train'], args.out_dir / 'train.csv', class_to_label)
    write_csv(splits['val'], args.out_dir / 'val.csv', class_to_label)
    write_csv(splits['test'], args.out_dir / 'test.csv', class_to_label)

    # Also write class mapping
    with open(args.out_dir / 'classes.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_name', 'label'])
        for c, l in class_to_label.items():
            writer.writerow([c, l])

    print('Wrote CSVs to', args.out_dir)


if __name__ == '__main__':
    main()
