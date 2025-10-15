# Flower Classification (10 classes) — CNN + Augmentation

Goal: Train a baseline CNN / ResNet on a 10-class flower dataset and compare data augmentations and Grad-CAM visualizations.

[![python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]() [![pytorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]() [![license](https://img.shields.io/badge/License-MIT-green.svg)]()

## TL;DR
Train a CNN on a 10-class flower dataset, then compare data augmentations and visualize model focus with Grad-CAM.

---

## Dataset
Source: Kaggle — “Flower Classification (10 classes / V2 etc.)”, organized as `root/<class_name>/*.jpg`.

**Split sizes (from `experiments/splits/*.csv`):**  
Discovered 21,710 images across 10 classes.  
Current split (train / val / test): **17,380 / 2,165 / 2,165** (≈80/10/10).

**Normalization**
- Image size: **224×224**
- Mode: imagenet
- Mean / Std: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]

> Note: Raw images are converted to RGB if needed; non-RGB/corrupt files are skipped with a warning.

---

## Repository structure

.  
├─ src/                       # (versioned) core code  
│  ├─ data.py                 # dataset discovery, splits, transforms, loaders  
│  ├─ model.py                # SmallCNN + model factory  
│  ├─ train.py                # training loop, checkpointing, CSV logging  
│  ├─ eval.py                 # evaluation helpers, checkpoint loading  
│  └─ utils.py                # seeding, plotting helpers  
├─ experiments/               # (runtime outputs)  
│  ├─ splits/                 # generated CSVs (recommended to commit for reproducibility)  
│  │  ├─ train.csv  
│  │  ├─ val.csv  
│  │  └─ test.csv  
│  ├─ checkpoints/            # model checkpoints (.pth) -> keep out of git (gitignored)  
│  └─ logs/                   # per-run CSV logs -> commit small summaries only  
├─ notebooks/                 # analysis & sanity-check notebooks  
│  ├─ 01_data_sanity.ipynb  
│  ├─ 02_model_sanity.ipynb  
│  └─ sanity_check_full.ipynb  
├─ plots/                     # generated plots (usually ignored)  
├─ requirements.txt           # (versioned) dependency list  
└─ README.md                  # this file

---

Reproducibility

- Default seed: 67 (changeable via function args or config).
- Splits are persisted to `experiments/splits/*.csv` so runs are reproducible using the same CSVs.

Quick sanity checks

- Open `notebooks/sanity_check.ipynb` and run the first two cells to import model/data and run a short train loop.

---

## Roadmap (short)

- Week 1 — Repo scaffolding & env ✅
- Week 2 — Data pipeline ✅
- Week 3 — Baseline model & training loop ✅
- Week 4 — Data augmentation experiments ⏳
- Week 5 — Grad-CAM visualizations ⏳
- Week 6 — Results, conclusions, polish ⏳

License & citation

License: MIT (see LICENSE).

If you use this repo, please cite or link back. Dataset credit to the Kaggle authors.