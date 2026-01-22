# 🗺️ Visual Guide to New Training Pipeline

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PBIP FEATURE-BASED TRAINING                      │
│                                                                     │
│  [Step 0]      [Step 1]        [Step 2]           [Step 3]         │
│  PATCHES       FEATURES        PROTOTYPES         TRAINING          │
│                                                                     │
│  Raw WSI    Extract Patches  Extract Features  K-Means Cluster     │
│      ↓            ↓                ↓                ↓               │
│   [.tif]  ──→  [.h5]  ──→  virchow2/dinov3  ──→  Prototype Bank   │
│           extract_     extract_patch_        k_mean_cos_           │
│           patches.py   features.py           per_class.py          │
│                                                                     │
│           coords/metadata  features_for_     label_feature_        │
│                           prototype_clusters/ prototypes.pkl       │
│                           [.pt files]        [pickle file]         │
│                                                                     │
│                                          ↓                         │
│                              Train Patch Refinement Model          │
│                              train_patch_refinement.py             │
│                                          ↓                         │
│                              PrototypeGuidedAttention              │
│                                (refines predictions)               │
│                                          ↓                         │
│                                  Patch-level Segmentation          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Organization

```
📁 PBIP/
│
├─ 🎯 train_patch_refinement.py              ← NEW: Run this to train
│  (Simplified training script for feature-based pipeline)
│
├─ 📁 datasets/
│  ├─ feature_dataset.py                     ← Active: Feature loading
│  ├─ patch_dataset.py                       ← Deprecated: Raises error
│  └─ wsi_dataset.py                         ← Deprecated: Marked notice
│
├─ 📁 utils/
│  ├─ prototype_guided_attention.py          ← Active: Prototype matching
│  ├─ encoders.py                            ← Active: Encoder factory
│  ├─ pseudo_labels.py                       ← Active: Weak labels
│  ├─ fgbg_feature.py                        ← Deprecated: Marked notice
│  └─ trainutils.py                          ← Modified: Routing logic
│
├─ 📁 features/
│  ├─ extract_patch_features.py              ← Active: Feature extraction
│  └─ k_mean_cos_per_class.py                ← Active: Clustering
│
├─ 📁 work_dirs/
│  └─ custom_wsi_template.yaml               ← Modified: New config
│
├─ 📚 Documentation (NEW - 8 files):
│  ├─ DOCUMENTATION_INDEX.md                 ← START HERE
│  ├─ GETTING_STARTED.md                     ← For new users
│  ├─ FEATURE_BASED_CONFIG_GUIDE.md           ← For configuration
│  ├─ QUICK_START_CHECKLIST.md                ← For verification
│  ├─ REPOSITORY_CLEANUP_SUMMARY.md           ← For migration
│  ├─ CLEANUP_COMPLETION_REPORT.md            ← For overview
│  ├─ COMPLETE_CHANGELOG.md                   ← For details
│  └─ README.md (Modified)                    ← Project info
│
└─ ✓ Setup complete
```

---

## Data Flow

### Old Approach ❌
```
┌────────────────┐
│   Raw WSI      │
│   (.tif file)  │
└────────┬───────┘
         │
         ↓
┌────────────────────────────────────┐
│ Extract Patches On-the-Fly         │
│ (slow, memory-intensive)           │
└────────┬───────────────────────────┘
         │
         ↓
┌────────────────────────────────────┐
│ MedCLIP Encoding                   │
│ (per-batch encoding)               │
└────────┬───────────────────────────┘
         │
         ↓
┌────────────────────────────────────┐
│ ClsNetwork (Multi-scale)           │
│ 4 CAM outputs per image            │
│ Dense pixel-level predictions      │
└────────┬───────────────────────────┘
         │
         ↓
┌────────────────────────────────────┐
│ Pixel-Level Segmentation Mask      │
│ (Requires pixel-level GT)          │
└────────────────────────────────────┘

❌ Problems:
   - Slow patch extraction
   - High memory usage
   - Complex architecture
   - Requires pixel-level labels
```

### New Approach ✅
```
┌────────────────┐
│   Raw WSI      │
│   (.tif file)  │
└────────┬───────┘
         │
         ↓
┌────────────────────────────────────┐
│ STEP 0: Extract Patches            │
│ features/extract_patches.py        │
│ → Saves patch coordinates to .h5   │
└────────┬───────────────────────────┘
         │
         ↓
┌────────────────────────────────────┐
│ STEP 1: Extract Features           │
│ features/extract_patch_features.py │
│ → Virchow2/DinoV3 encoding         │
│ → Save to .pt files                │
└────────┬───────────────────────────┘
         │
         ↓
┌────────────────────────────────────┐
│ STEP 2: K-Means Clustering         │
│ features/k_mean_cos_per_class.py   │
│ → Generate prototype bank          │
│ → label_feature_prototypes.pkl     │
└────────┬───────────────────────────┘
         │
         ↓
┌────────────────────────────────────┐
│ STEP 3: Train Refinement Model     │
│ train_patch_refinement.py          │
│ → PrototypeGuidedAttention         │
│ → Refine predictions               │
└────────┬───────────────────────────┘
         │
         ↓
┌────────────────────────────────────┐
│ Patch-Level Segmentation           │
│ (Per-patch class predictions)      │
└────────────────────────────────────┘

✅ Advantages:
   - Clear 4-step pipeline
   - Efficient prototype-guided training
   - Works with patch coordinates
   - No MIL attention scores needed
```

---

## Getting Started Roadmap

```
START
  │
  ├─→ You are a NEW user
  │   └─→ Read DOCUMENTATION_INDEX.md
  │       └─→ Follow GETTING_STARTED.md
  │           └─→ Use QUICK_START_CHECKLIST.md
  │               └─→ Run: python train_patch_refinement.py
  │
  ├─→ You need CONFIGURATION help
  │   └─→ Read FEATURE_BASED_CONFIG_GUIDE.md
  │       └─→ Update work_dirs/custom_wsi_template.yaml
  │           └─→ Run: python train_patch_refinement.py
  │
  ├─→ You're MIGRATING from old approach
  │   └─→ Read REPOSITORY_CLEANUP_SUMMARY.md
  │       └─→ Compare old vs. new config
  │           └─→ Update your config files
  │               └─→ Rerun with new script
  │
  ├─→ You want to understand CHANGES
  │   └─→ Read CLEANUP_COMPLETION_REPORT.md
  │       └─→ Check COMPLETE_CHANGELOG.md
  │           └─→ Review specific file changes
  │
  └─→ You're a DEVELOPER
      └─→ Read REPOSITORY_CLEANUP_SUMMARY.md
          └─→ Review COMPLETE_CHANGELOG.md
              └─→ Examine train_patch_refinement.py
                  └─→ Extend/modify as needed

TRAIN
  │
  └─→ python train_patch_refinement.py --config work_dirs/custom/config.yaml --gpu 0
```

---

## Configuration Template

### Minimal Config (Copy & Customize)
```yaml
# work_dirs/my_project/config.yaml

model:
  patch_encoder: virchow2          # Choose: virchow2, dinov3
  label_feature_path: ${features.save_dir}/${features.label_feature_pkl}

work_dir: /data/my_project/PBIP
run_uid: null                       # Auto-filled

dataset:
  num_classes: 2
  class_order: [benign, tumor]
  
  # REQUIRED: Paths for data
  data_root_dir: /data/my_project/wsi_files/
  split_csv: /data/my_project/splits.csv
  labels_csv: /data/my_project/labels.csv

features:
  # REQUIRED: Where to save/load features
  save_dir: /data/my_project/features_for_prototype_clusters/
  label_feature_pkl: label_feature_prototypes.pkl
  
  # Patch extraction settings
  patch_size: 256
  overlap: 0.5

train:
  samples_per_gpu: 5
  epoch: 10
  pretrained: true

optimizer:
  type: AdamW
  learning_rate: 0.00001
  betas: [0.9, 0.999]
  weight_decay: 0.003
```

---

##Navigate to project directory
cd /path/to/PBIP

# OPTION 1: Run entire pipeline with one command
bash pipeline.sh --config work_dirs/custom_wsi_template.yaml --gpu 0

# OPTION 2: Run steps manually
# Step 0: Extract patch coordinates
python features/extract_patches.py --config work_dirs/custom/config.yaml

# Step 1: Extract patch features
python features/extract_patch_features.py --config work_dirs/custom/config.yaml

# Step 2: Generate prototype bank
python features/k_mean_cos_per_class.py --config work_dirs/custom/config.yaml

# Step 3: Train refinement model
python train_patch_refinement.py --config work_dirs/custom/config.yaml --gpu 0

# Verify setup with diagnostic (optional)
python tatch Extraction
- Extract patch coordinates from WSI files
- Stored as `.h5` files with coordinates and metadata
- Output: `coords/` directory with patch coordinate files

### 🔹 Precomputed Features
- Patch embeddings computed offline using Virchow2 or DinoV3
- Stored as `.pt` files in `features_for_prototype_clusters/`
- Shape: `(num_patches_in_wsi, feature_dim)`
  - Virchow2: 1280-dim
  - DinoV3: 1024-dim

### 🔹 Prototype Bank
- K-means clustering of patch features per class
- Generates representative prototypes for each class
- Stored as `label_feature_prototypes.pkl`
- Used for prototype-guided attention refinement

### 🔹 Prototype Matching
- Compare patch features to prototype bank
- PrototypeGuidedAttention module refines predictions
- Combines cosine similarity with learnable attention
- No external MIL attention scores needed

### 🔹 FeatureWSIDataset
- Loads precomputed features from `.pt` files
- Uses prototype bank for training
- Returns: `(wsi_name, patch_features, class_label
- Contains key `'patch_logits'` with attention scores
- Shape: `(num_patches,)` for binary, `(num_patches, num_classes)` for multi-class

### 🔹 Prototype Matching
- Compare patch features to prototype bank
- Refine predictions virchow2` | Feature encoder (virchow2, dinov3) |
| `samples_per_gpu` | 5 | Batch size |
| `epoch` | 10 | Training epochs |
| `learning_rate` | 0.00001 | Optimizer learning rate |
| `patch_size` | 256 | Size of extracted patches |
| `overlap` | 0.5 | Overlap ratio for patch extraction
- Returns: `(wsi_name, patch_features, class_label, attention_scores)`

---

## Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_encoder` | `medclip` | Feature encoder (medclip, virchow2, dinov3) |
| `samples_per_gpu` | 5 | Batch size |
| `epoch` | 10 | Training epochs |
| `learning_rate` | 0.00001 | Optimizer learning rate |
| `binary_mode` | `true` | Binary vs multi-class attention |
| `prototype_label_required_classes` | `[1]` | Which classes need pseudo-labels |

---

## Success Indicators

✅ Training is working if you see:
- Datasets loaded successfully
- Model initialized on GPU
- Training loss decreasing each epoch
- Validation loss decreasing
- Checkpoint saved: `ckpt_dir/best_model.pth`

❌ Something is wrong if you see:
- Import errors
- Feature files not found
- Pseudo-label files not found
- Out of memory errors
- Increasing training loss

---

## Next Steps After Training

1. **Evaluate on test set** - Use checkpoint to compute metrics
2. **Analyze results** - Visualize patch segmentations
3. **Fine-tune hyperparameters** - Try different settings
4. **Experiment with encoders** - Compare medclip vs virchow2 vs dinov3

---

## Documentation Quick Links

| Need | Link |
|------|------|
| New to PBIP | [GETTING_STARTED.md](GETTING_STARTED.md) |
| Setup help | [QUICK_START_CHECKLIST.md](QUICK_START_CHECKLIST.md) |
| Config reference | [FEATURE_BASED_CONFIG_GUIDE.md](FEATURE_BASED_CONFIG_GUIDE.md) |
| Migration guide | [REPOSITORY_CLEANUP_SUMMARY.md](REPOSITORY_CLEANUP_SUMMARY.md) |
| See all changes | [COMPLETE_CHANGELOG.md](COMPLETE_CHANGELOG.md) |
| Navigation | [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) |

---

**Ready to start?** → [GETTING_STARTED.md](GETTING_STARTED.md)
