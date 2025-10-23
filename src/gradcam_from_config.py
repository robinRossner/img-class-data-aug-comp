import os
import argparse
import yaml

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from gradcam import grad_cam
from eval import load_model_checkpoint


def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def derive_checkpoint_path(ckpt_arg, cfg, augTier):
    if ckpt_arg:
        return ckpt_arg
    ckpt_cfg = cfg.get("checkpoints", {})
    out_dir = ckpt_cfg.get("out_dir", "experiments/checkpoints")
    template = ckpt_cfg.get("save_best_template", None) or ckpt_cfg.get("save_last_template", None)
    if template is None:
        return None
    fname = template.format(augmentationTier=augTier)
    return os.path.join(out_dir, fname)


def find_target_layer(model, prefer_conv_only=True):
    # Try to find last nn.Conv2d layer in model
    import torch.nn as nn
    target = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            target = module
    return target


def image_from_tensor(tensor):
    # tensor shape (C,H,W), assume in [0,1] or normalized; try to unnormalize if possible
    arr = tensor.detach().cpu().numpy()
    # If normalized with mean/std, we don't know; just clip
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    if arr.shape[0] == 1:
        pil = Image.fromarray(arr[0], mode='L')
    else:
        pil = Image.fromarray(np.transpose(arr, (1, 2, 0)))
    return pil


def overlay_cam_on_image(img_pil, cam, alpha=0.5, cmap='jet'):
    cam_img = plt.get_cmap(cmap)(cam)[:, :, :3]
    cam_img = (cam_img * 255).astype(np.uint8)
    cam_pil = Image.fromarray(cam_img).resize(img_pil.size, resample=Image.BILINEAR)
    overlay = Image.blend(img_pil.convert('RGBA'), cam_pil.convert('RGBA'), alpha=alpha)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Run Grad-CAM using YAML config and checkpoint")
    parser.add_argument("--config", type=str, default="experiments/configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--index", type=int, default=0, help="Index of sample in chosen split to visualize")
    parser.add_argument("--target-layer", type=str, default=None, help="Dotted attribute path to target layer on model (e.g. features.18) if you want to override auto-detect")
    parser.add_argument("--out", type=str, default="gradcam_overlay.png")
    args = parser.parse_args()

    cfg = load_config(args.config)

    from data import build_dataloaders
    data_cfg = cfg["data"]
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        data_root=data_cfg["data_root"],
        size=data_cfg["size"],
        batch_size=1,
        num_workers=0,
        seed=cfg.get("seed", 67),
        val_split=data_cfg.get("val_split", 0.1),
        test_split=data_cfg.get("test_split", 0.1),
        augmentationTier=data_cfg.get("augmentationTier", 0),
    )

    num_classes = meta.get("num_classes", cfg.get("model", {}).get("num_classes", 10))

    # Build model
    from model import build_model
    model_cfg = cfg.get("model", {})
    model = build_model(
        name=model_cfg.get("name", "mobilenet_v2"),
        num_classes=model_cfg.get("num_classes", num_classes),
    )

    augTier = data_cfg.get("augmentationTier", 0)
    checkpoint_path = derive_checkpoint_path(args.checkpoint, cfg, augTier)
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Provide --checkpoint or ensure YAML checkpoint templates/out_dir are correct.")
        return

    load_model_checkpoint(model, checkpoint_path)

    loader = val_loader if args.split == "val" else test_loader
    if loader is None:
        print(f"No loader available for split: {args.split}")
        return

    # Get sample
    # loader yields (input, label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i == args.index:
                img_tensor = x.squeeze(0).to(device)
                label = int(y.item())
                break
        else:
            print(f"Index {args.index} out of range for split {args.split}")
            return

    # Choose target layer
    target_layer = None
    if args.target_layer:
        # Resolve dotted path
        parts = args.target_layer.split('.')
        target_layer = model
        for p in parts:
            if hasattr(target_layer, p):
                target_layer = getattr(target_layer, p)
            else:
                try:
                    # allow numeric indexing into sequential
                    idx = int(p)
                    target_layer = target_layer[idx]
                except Exception:
                    target_layer = None
                    break

    if target_layer is None:
        target_layer = find_target_layer(model)

    if target_layer is None:
        print("Could not locate a convolutional target layer for Grad-CAM")
        return

    cam = grad_cam(model, img_tensor, target_layer, class_idx=label)

    # Prepare image and overlay
    img_pil = image_from_tensor(img_tensor.cpu())
    # cam is (H,W) normalized 0..1
    overlay = overlay_cam_on_image(img_pil, cam, alpha=0.5)
    overlay.save(args.out)
    print(f"Saved Grad-CAM overlay to {args.out}")


if __name__ == "__main__":
    main()
