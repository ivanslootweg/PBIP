# PBIP: Prototype-Based Image Prompting for Weakly Supervised Histopathological Image Segmentation

**Official PyTorch implementation**

> **Prototype-Based Image Prompting for Weakly Supervised Histopathological Image Segmentation**  
> CVPR 2025

---

## Quick Start

### **Automated Pipeline (Recommended)**
```bash
bash pipeline.sh --config work_dirs/custom_wsi_template.yaml --gpu 0
```

This runs feature extraction → clustering → training in sequence.

### **Manual Steps**
```bash
# 1. Extract patch features
python3 features/extract_patch_features.py --config work_dirs/custom_wsi_template.yaml --gpu 0  # Uses encoder from config

# 2. Cluster features into prototypes
python3 features/k_mean_cos_per_class.py --config work_dirs/custom_wsi_template.yaml

# 3. Train patch-level refinement
python3 train_patch_refinement.py --config work_dirs/custom_wsi_template.yaml --gpu 0
```

---

## 📚 Documentation

**Start here:** → **[docs/](docs/)**

| Document | Purpose |
|----------|---------|
| [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | Complete training guide (config, data prep, troubleshooting) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and component details |
| [docs/SETUP.md](docs/SETUP.md) | Installation and environment setup |
| [docs/DEPRECATED.md](docs/DEPRECATED.md) | Migration guide from old pixel-level pipeline |

---

## 🏗️ Current Pipeline: Feature-Based Patch Refinement

The pipeline extracts patch features using foundation models and refines them with prototypes:

```
WSI patches
    ↓
Encoder (Virchow2 / DinoV3)
    ↓
Precomputed Patch Features (n_patches × feature_dim)
    ↓
K-means Clustering → Prototype Bank
    ↓
  Prototype Matching: Softmax(S_match)
    ↓
Patch-Level Segmentation
```

### Key Features
- ✅ **Fast training:** Precomputed features, no on-the-fly extraction
- ✅ **Flexible encoders:** Virchow2 (pathology-specific, recommended), DinoV3
- ✅ **Prototype-guided:** Leverages prototype matching for refinement
- ✅ **Weakly supervised:** Uses image-level labels only (no pseudo-labels)
- ✅ **Memory efficient:** Features cached, not WSI files

---

## 💾 Required Data Format

### Directory Structure
```
project/
├── patch_features/           # Precomputed encoder features
│   ├── wsi_1.pt             # (n_patches, feature_dim)
│   └── ...
├── global_features/          # Optional slide-level features
│   ├── wsi_1.pt             # (feature_dim,)
│   └── ...
├── split.csv                 # Train/val split
└── labels.csv                # WSI-level class labels
```

### Configuration
```yaml
dataset:
  use_feature_based_training: true
  patch_features_dir: /path/to/patch_features/
  global_feature_dir: /path/to/global_features/  # optional
  split_csv: /path/to/split.csv
  labels_csv: /path/to/labels.csv
  
model:
  patch_encoder: virchow2  # or dinov3
```

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for detailed config reference.

---

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
python3 test_pipeline.py --config work_dirs/custom_wsi_template.yaml
```

See [docs/SETUP.md](docs/SETUP.md) for detailed installation.

### 2. Data Preparation
```bash
# Extract features using configured encoder
python3 features/extract_patch_features.py --config your_config.yaml --gpu 0

# (Optional) Generate global features
python3 scripts/compute_global_features.py --config your_config.yaml

# Create split and labels CSVs
```

### 3. Train Model
```bash
bash pipeline.sh --config your_config.yaml --gpu 0
```

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for comprehensive guide.

---

## 📊 Encoder Comparison

| Encoder | Dimension | Source | Best For |
|---------|-----------|--------|----------|
| **Virchow2** | 1280 | PAIGE-AI | Pathology ✓ Recommended |
| **DinoV3** | 1024 | Meta | General vision |

---

## 🔧 Configuration

### Essential Settings
```yaml
model:
  patch_encoder: virchow2         # Encoder choice
  label_feature_path: ...         # Prototype bank path
  
dataset:
  use_feature_based_training: true
  patch_features_dir: ...         # ← REQUIRED
  global_feature_dir: ...         # ← OPTIONAL
  num_classes: 2
  
train:
  samples_per_gpu: 5              # Batch size
  epoch: 10
```

### Full Configuration Reference
See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md#configuration)

---

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU)

### Setup

#### Using requirements.txt
```bash

# Create virtual environment
conda create -n pbip python=3.8
conda activate pbip

# Install exact dependencies (recommended for reproducibility)
pip install -r requirements.txt
```

## 📊 Dataset

This project uses the **BCSS (Breast Cancer Semantic Segmentation)** dataset with 5 tissue classes:

| Class | Description | Color |
|-------|-------------|-------|
| TUM | Tumor | 🔴 Red |
| STR | Stroma | 🟢 Green |
| LYM | Lymphocyte | 🔵 Blue |
| NEC | Necrosis | 🟣 Purple |
| BACK | Background | ⚪ White |

### Data Structure
```
data/
├── BCSS-WSSS/
│   ├── train/
│   │   └── *.png  # Training images with class labels in filename
│   ├── test/
│   │   ├── img/   # Test images
│   │   └── mask/  # Ground truth masks
│   └── valid/
│       ├── img/   # Validation images
│       └── mask/  # Ground truth masks
```

## 🚀 Quick Start

### 🔄 Feature-Based Training Pipeline (Recommended)

The current implementation uses a **feature-based patch refinement approach** that works with precomputed patch features instead of raw images:

#### 1. Extract Patch Features

Choose your encoder (Virchow2 recommended for pathology):
```bash
# Extract features with configured encoder (Virchow2 or DinoV3)
python features/extract_patch_features.py --config ./work_dirs/custom/config.yaml --gpu 0
```

Configure your `config.yaml`:
```yaml
model:
  patch_encoder: virchow2  # or dinov3

dataset:
  patch_features_dir: /path/to/encoder_features/
  global_feature_dir: /path/to/global_features/
  use_feature_based_training: true
```

#### 2. Train Patch Refinement Model

```bash
# Train with prototype-guided attention refinement
python train_patch_refinement.py --config ./work_dirs/custom/config.yaml --gpu 0
```

### 📚 Documentation

- **[FEATURE_BASED_CONFIG_GUIDE.md](FEATURE_BASED_CONFIG_GUIDE.md)** - Configuration reference and examples
- **[REPOSITORY_CLEANUP_SUMMARY.md](REPOSITORY_CLEANUP_SUMMARY.md)** - Migration guide from pixel-level to feature-based approach

### ⚠️ Legacy Approach (Deprecated)

The old pixel-level training approach (multi-scale CAM generation) is deprecated but still available:
```bash
# NOT RECOMMENDED - use train_patch_refinement.py instead
python train_stage_1_features.py --config ./work_dirs/custom/config.yaml --gpu 0
```

See [REPOSITORY_CLEANUP_SUMMARY.md](REPOSITORY_CLEANUP_SUMMARY.md) for migration details.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```bibtex
@inproceedings{pbip2025,
  title={Prototype-Based Image Prompting for Weakly Supervised Histopathological Image Segmentation},
  author={Qingchen Tang and Lei Fan and Maurice Pagnucco and Yang Song},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

---

⭐ **Star this repo if you find it helpful!** ⭐ 
