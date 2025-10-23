import os
import argparse
import yaml

import torch

from eval import load_model_checkpoint, evaluate, evaluateClassAccuracy


def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def derive_checkpoint_path(ckpt_arg, cfg, augTier):
    # If user provided explicit path, use it
    if ckpt_arg:
        return ckpt_arg

    ckpt_cfg = cfg.get("checkpoints", {})
    out_dir = ckpt_cfg.get("out_dir", "experiments/checkpoints")
    template = ckpt_cfg.get("save_best_template", None) or ckpt_cfg.get("save_last_template", None)
    if template is None:
        return None
    fname = template.format(augmentationTier=augTier)
    return os.path.join(out_dir, fname)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model using YAML config and checkpoint")
    parser.add_argument("--config", type=str, default="experiments/configs/baseline.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit checkpoint file to load (overrides template)")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val",
                        help="Which split to evaluate: val or test")
    parser.add_argument("--class-acc", action="store_true",
                        help="Also compute per-class accuracies")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Build dataloaders and metadata using existing data builder
    from data import build_dataloaders
    data_cfg = cfg["data"]
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        data_root=data_cfg["data_root"],
        size=data_cfg["size"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        seed=cfg.get("seed", 67),
        val_split=data_cfg.get("val_split", 0.1),
        test_split=data_cfg.get("test_split", 0.1),
        augmentationTier=data_cfg.get("augmentationTier", 0),
    )

    num_classes = meta.get("num_classes", cfg.get("model", {}).get("num_classes", 10))
    class_names = meta.get("class_names", None)

    # Build model
    from model import build_model
    model_cfg = cfg.get("model", {})
    model = build_model(
        name=model_cfg.get("name", "mobilenet_v2"),
        num_classes=model_cfg.get("num_classes", num_classes),
    )

    # Determine checkpoint path
    augTier = data_cfg.get("augmentationTier", 0)
    checkpoint_path = derive_checkpoint_path(args.checkpoint, cfg, augTier)
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Provide --checkpoint or ensure YAML checkpoint templates/out_dir are correct.")
        return

    # Load checkpoint
    ckpt = load_model_checkpoint(model, checkpoint_path)
    if ckpt is None:
        print("Failed to load checkpoint; aborting.")
        return

    # Pick loader
    loader = val_loader if args.split == "val" else test_loader
    if loader is None:
        print(f"No loader available for split: {args.split}")
        return

    # Criterion
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()

    # Run evaluation
    print(f"Evaluating checkpoint: {checkpoint_path} on split: {args.split}")
    avg_loss, acc = evaluate(model, loader, criterion)

    if args.class_acc:
        class_accs = evaluateClassAccuracy(model, loader, criterion, num_classes=num_classes)
        if class_names and len(class_names) == len(class_accs):
            print("Per-class accuracies:")
            for name, a in zip(class_names, class_accs):
                print(f"{name}: {100*a:.2f}%")
        else:
            print("Per-class accuracies (by index):")
            for i, a in enumerate(class_accs):
                print(f"Class {i}: {100*a:.2f}%")


if __name__ == "__main__":
    main()
