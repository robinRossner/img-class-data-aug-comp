# Flower Classification (10 classes) — CNN + Augmentation

Goal: Train a baseline CNN / ResNet on a 10-class flower dataset and compare data augmentations and Grad-CAM visualizations.

[![python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]() [![pytorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]() [![license](https://img.shields.io/badge/License-MIT-green.svg)]()

## TL;DR
Train a baseline CNN / ResNet on a 10-class flower dataset, then compare data augmentations and visualize model focus with Grad-CAM.

---

## Dataset
Source: Kaggle — “Flower Classification (5 classes / V2 etc.)”, but this repo uses **10 classes** organized as `root/<class_name>/*.jpg`.

**Classes (from `experiments/data_meta.json`):**
Not available: experiments/data_meta.json not found in the repo. Run the data discovery/split script (see src/data.py) to generate experiments/data_meta.json; the class list will be autofilled here afterwards.

**Split sizes (from `experiments/splits/*.csv`):**
Splits not found — please run the split generation script to create CSVs under experiments/splits/. See "Data pipeline (Week 2)" for details.

**Normalization**
- Image size: **224×224**
- Mode: imagenet
- Mean / Std: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]

> Note: Raw images are converted to RGB if needed; non-RGB/corrupt files are skipped with a warning.

---

## Repository structure


.
├─ src/
│ ├─ data.py # datasets, transforms, loaders, split helpers
│ ├─ utils.py # seeding, image grid, misc
├─ experiments/
│ ├─ splits/
│ │ ├─ train.csv
│ │ ├─ val.csv
│ │ └─ test.csv
│ └─ data_meta.json # class names/mapping, counts, mean/std
├─ notebooks/
│ └─ 01_data_sanity.ipynb
├─ plots/
├─ requirements.txt
└─ README.md


---

## Environment & setup
```bash
# 1) Create env (edit python version if needed)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) Optional: verify Torch CUDA
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

Data pipeline (Week 2

What’s implemented:

Class discovery from folder names (frozen to class_to_idx in data_meta.json).

Stratified train/val/test split (seeded) — CSVs saved in experiments/splits/.

Deterministic transforms (no augmentation yet):

Train: Resize→CenterCrop (or Resize to 224), ToTensor, Normalize.

Val/Test: deterministic Resize/CenterCrop, ToTensor, Normalize.

DataLoaders with shuffle=True (train), pin_memory and num_workers set.

How to use the loaders in training (preview):

from src.data import build_dataloaders
train_loader, val_loader, test_loader, meta = build_dataloaders(
    data_root="</path/to/your/dataset/root>",
    img_size=224, batch_size=64, num_workers=4, seed=42,
    val_ratio=0.10, test_ratio=0.10, normalize_mode="<!-- AUTOFILL -->"
)

Reproducibility

Fixed seed: 42 (changeable in CLI/config).

Frozen splits: we train/evaluate against experiments/splits/*.csv.

Class mapping: stored in experiments/data_meta.json and used everywhere.

To regenerate the same splits, keep the same CSVs and seed.

Quick sanity checks

Open notebooks/01_data_sanity.ipynb and run:

- Visual grid of a training batch (labels vs. images).
- Shapes and value ranges after normalization.
- Or run a small CLI in src/data.py (if present) to preview a batch and counts.

Per-split counts (autofilled):
Splits not found — experiments/splits/*.csv are missing. Run the split creation flow in src/data.py to generate train/val/test CSVs.

Per-class counts (autofilled):
Not available — splits are required to compute per-class counts.

Example batch (saved path if available):
No sample image found in plots/; if you save the notebook output, place it as plots/sample_train_batch.png to have it linked here.

Roadmap & status

 Week 1 — Repo scaffolding & env

 Week 2 — Data pipeline (this README)

 Week 3 — Baseline model & training loop

 Week 4 — Augmentation experiments

 Week 5 — Grad-CAM visualizations

 Week 6 — Results, conclusions, polish

License & citation

License: MIT (see LICENSE).

If you use this repo, please cite or link back. Dataset credit to the Kaggle authors.