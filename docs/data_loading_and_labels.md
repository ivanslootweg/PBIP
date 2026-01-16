# Data Loading & Label Selection

How data is discovered, validated, and labeled for PBIP.

## Original PBIP Data Loading

### Inputs
- **Split CSV**: columns `train`, `val`, `test` with filenames (can be full paths or basenames)
- **Labels CSV**: `image_name` + one or more label columns (binary or multi-class)
- **WSI Directory**: `.tif/.tiff` slides
- **Coordinates Directory**: `.npy` or `.txt` files (slide2vec or plain x,y)
- **Optional GT Masks**: for val/test pixel supervision

### Split Ingestion
- File: `features/extract_patches.py` and `datasets/wsi_dataset.py`
- For each split column, filenames are read; missing entries are skipped
- Filenames are normalized to basenames (without extension) for lookup

### Label Resolution
- `labels_csv` is loaded into a dict keyed by basename (no extension)
- Supports:
  - Single-column binary/multiclass (value or space/comma-delimited list)
  - Multi-column one-hot (each class column is 0/1)
- During training split load, samples without labels are skipped and reported

### Coordinate & Asset Checks
- For train: require WSI + coordinates + label
- For val/test: require WSI + coordinates + GT mask + label
- Missing assets are counted and reported per reason (no_wsi, no_coords, no_label, no_gt)

### Patch Sampling (train)
- Per-class file lists built from labels
- For each class: shuffle files (seeded) and take up to `num_wsis_per_class`
- Per WSI: randomly sample up to `num_per_wsi` coordinates without replacement
- Saved as slide2vec npy with UID

### GT Mask Usage (val/test)
- Masks loaded from `dataset.gt_dir` with `mask_suffix`
- Required for evaluation and thumbnail generation

### Caching & Reproducibility
- Seeds applied for deterministic sampling
- UID encodes sampling params + file hash
- Downstream features reuse UID for alignment

---

## WSI-Level Adaptation: Pseudo-Label Integration

### Additional Input: Pseudo-Labels

When using the WSI adaptation with pseudo-label driven patch selection:

- **Pseudo-Label Files**: `.pt` files with attention scores from MIL model
- **Location**: `dataset.pseudo_label_dir` (from config)
- **Format**: PyTorch tensors saved via `torch.save(scores, 'wsi_name.pt')`
- **Naming**: One `.pt` file per WSI matching the basename

### Pseudo-Label Format

#### Binary Mode (default)
```python
# Single attention channel per patch
scores = torch.tensor([0.1, 0.8, 0.3, 0.2, ...])  # shape: (num_patches,)
torch.save(scores, 'slide_001.pt')

# Interpretation:
# - Low values (< 0.5) → class 0 patches
# - High values (> 0.5) → class 1 patches
```

#### Multi-Class Mode
```python
# Multiple attention scores per patch (one per class)
scores = torch.tensor([
    [0.1, 0.8, 0.1],    # Patch 0: class 1 most likely
    [0.8, 0.1, 0.1],    # Patch 1: class 0 most likely
    [0.3, 0.3, 0.4],    # Patch 2: class 2 most likely
    ...
])  # shape: (num_patches, num_classes)
torch.save(scores, 'slide_001.pt')
```

### Enhanced Dataset Loading

When `use_pseudo_labels: true`, the dataset loader:

1. **Loads Pseudo-Labels**
   - Uses `PseudoLabelLoader` from `utils/pseudo_labels.py`
   - Validates binary/multi-class mode consistency
   - Detects and reports missing or malformed files

2. **Selects High-Confidence Patches**
   - Uses `PatchSelector` with configured strategy
   - Applies confidence threshold (default: top 15% by percentile)
   - Per-WSI selection respects `pseudo_label_min_patches` constraint

3. **Reports Statistics**
   - Number of selected patches per WSI
   - Per-class distribution
   - Quality metrics (mean, std, min, max attention)

4. **Enables Filtering**
   - `dataset.get_high_confidence_patches(wsi_name)` returns selected indices
   - `dataset.get_wsi_pseudo_scores(wsi_name)` returns attention scores
   - Features extracted only from selected patches

### Configuration for Pseudo-Labels

```yaml
dataset:
  # Enable pseudo-label driven patch selection
  use_pseudo_labels: true
  
  # Path to pseudo-label .pt files
  pseudo_label_dir: /path/to/attention_scores
  
  # Binary (single channel) or multi-class (multiple channels)
  pseudo_label_binary_mode: true
  
  # Patch selection strategy
  # Options: 'percentile' (recommended), 'threshold', 'entropy', 'margin'
  pseudo_label_selection_strategy: percentile
  
  # Selection threshold (meaning depends on strategy)
  # For percentile: 0.85 = select top 15% of patches
  # For threshold: absolute score cutoff
  pseudo_label_confidence_threshold: 0.85
  
  # Minimum patches per WSI that must be selected
  # Prevents prototype corruption from low-quality WSIs
  pseudo_label_min_patches: 5
  
  # Whether to print statistics during initialization
  pseudo_label_analyze: true
```

### Selection Strategies Explained

#### Percentile (Recommended)
- **What it does**: Selects patches in top k% by confidence score
- **Threshold interpretation**: 0.85 = top 15% of patches
- **Best for**: Balanced datasets where quality is relatively uniform
- **Example**: If WSI has 1000 patches and top 15% have high attention → select 150 patches

#### Threshold
- **What it does**: All patches with score ≥ threshold
- **Threshold interpretation**: Absolute value (e.g., 0.5)
- **Best for**: Well-calibrated attention scores with clear decision boundary
- **Example**: If threshold is 0.5 and you have scores [0.2, 0.7, 0.3, 0.8] → select [0.7, 0.8]

#### Entropy
- **What it does**: Selects patches with lowest entropy (most confident predictions)
- **Threshold interpretation**: Entropy cutoff (lower = more confident)
- **Best for**: When you need extreme confidence, can tolerate very few selected patches
- **Example**: High entropy = uncertain; low entropy = highly confident decision

#### Margin
- **What it does**: Selects patches with largest margin between top-2 class predictions
- **Threshold interpretation**: Minimum margin value
- **Best for**: Multi-class classification, want discriminative samples
- **Example**: p=[0.8, 0.1, 0.1] has margin 0.7; p=[0.4, 0.35, 0.25] has margin 0.05

### Data Requirements for WSI Adaptation

```
project_dir/
├── attention_scores/               ← NEW: Pseudo-label .pt files
│   ├── slide_001.pt               ← (num_patches,) or (num_patches, num_classes)
│   ├── slide_002.pt
│   └── ...
├── wsi_images/
│   ├── slide_001.tif
│   ├── slide_002.tif
│   └── ...
├── coordinates/
│   ├── slide_001.npy
│   ├── slide_002.npy
│   └── ...
├── splits.csv                     ← train, val, test columns
├── labels.csv                     ← image-level WSI labels
└── gt_masks/                      ← (optional) pixel-level annotations
    ├── slide_001.tif
    └── ...
```

### Validation & Quality Assurance

During dataset initialization with pseudo-labels:

1. **File Validation**
   - Check all .pt files exist
   - Validate tensor shapes
   - Report missing files with WSI names

2. **Format Consistency**
   - Verify binary/multi-class mode matches data
   - Check class count matches configuration
   - Report dimension mismatches

3. **Quality Metrics**
   - Compute statistics (mean, std, min, max) per class
   - Identify outlier WSIs with unusual distributions
   - Report WSIs with insufficient high-confidence patches

4. **Reporting**
   - Summary of loaded vs skipped WSIs
   - Per-class patch selection statistics
   - Warnings for edge cases

---

## Integration with Original PBIP Workflow

The WSI adaptation **does not** require changes to existing data loading infrastructure:

- All original `.npy`/`.txt` coordinate loading still works
- All original label loading still works
- All original validation still works
- Pseudo-label loading is **optional** and **additive**

To use without pseudo-labels, simply set `use_pseudo_labels: false` or omit the pseudo-label config section.
