# 🟢 Image Classification with Data Augmentation

## 📘 Overview
This project trains a Convolutional Neural Network (CNN) to classify images from small datasets (e.g. **CIFAR-10** or **Oxford Flowers102**) and compares baseline training with various **data-augmentation** techniques such as flipping, cropping, and color jittering.  
The goal is to study how augmentation affects accuracy and visualize model understanding using **Grad-CAM**.

---

## ⚙️ Tech Stack
- Python 3.10+
- PyTorch / torchvision  
- matplotlib  
- tqdm  
- PyYAML  
- torchmetrics  

---

## 📂 Project Structure
```
image-class-data-aug-class/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ src/
│  ├─ data.py          # dataset loading & transforms
│  ├─ model.py         # CNN or ResNet-18 architecture
│  ├─ train.py         # training loop, logging
│  ├─ eval.py          # evaluation & metrics
│  ├─ gradcam.py       # Grad-CAM visualizations
│  └─ utils.py         # helpers (seed, plotting, configs)
├─ experiments/
│  ├─ configs/         # YAML configs for runs
│  └─ logs/            # checkpoints, csv/tensorboard logs
├─ notebooks/          # exploration, sanity checks
└─ plots/              # training curves & Grad-CAM images
```

---

## 🚀 Getting Started

### 1️⃣ Setup
```bash
git clone <your-repo-url>
cd image-class-data-aug-class
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2️⃣ Environment Dependencies
Typical `requirements.txt` contents:
```
torch
torchvision
matplotlib
tqdm
pyyaml
torchmetrics
```

### 3️⃣ Git Setup
```bash
git init
git add .
git commit -m "Initial project structure setup"
git branch -M main
git remote add origin <repo-url>
git push -u origin main
```

---

## 🧠 Week 1 Goals
✅ Create and verify project structure  
✅ Add `.gitignore` and `requirements.txt`  
✅ Set up virtual environment  
✅ Draft this `README.md`  
✅ Initialize Git and push to GitHub  
✅ Plan dataset & baseline model choice  

---

## 🧩 Next Steps (Week 2 Preview)
- Implement `data.py` to load CIFAR-10 (train/val/test splits)
- Add basic transforms (ToTensor + Normalize)
- Visualize samples in a notebook for sanity check

---

## 📄 License
MIT License © 2025 Robin Rossner

---

## 🙌 Acknowledgments
- Dataset: TBD
- Model architectures: TBD
- Grad-CAM reference: TBD