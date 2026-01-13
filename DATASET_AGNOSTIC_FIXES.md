# Dataset-Agnostic Code Fixes

This document summarizes the changes made to remove hard-coded assumptions about the 4-class BCSS dataset, making the codebase fully configurable for any number of classes.

## Issues Fixed

### 1. **Hard-coded Color Palette** ✅ FIXED
**File**: [utils/validate.py](utils/validate.py#L163-L179)

**Problem**: The CAM visualization used a fixed 6-color palette assuming BCSS's 4 classes + background.

**Solution**: 
- Dynamically generate palette based on `cfg.dataset.num_classes`
- Support up to 8 classes with distinct colors
- Automatically add background color (white) at the end

```python
# Before (hard-coded):
PALETTE = [
    [255, 0, 0],      # Red
    [0, 255, 0],      # Green
    [0, 0, 255],      # Blue
    [153, 0, 255],    # Purple
    [255, 255, 255],  # White (background)
    [0, 0, 0],        # Black (unused)
]

# After (dynamic):
num_classes = cfg.dataset.num_classes
PALETTE = [
    [255, 0, 0],      # Class 0: Red
    [0, 255, 0],      # Class 1: Green
    [0, 0, 255],      # Class 2: Blue
    [153, 0, 255],    # Class 3: Purple
    [255, 255, 0],    # Class 4: Yellow
    [0, 255, 255],    # Class 5: Cyan
    [255, 128, 0],    # Class 6: Orange
    [128, 0, 255],    # Class 7: Violet
]
PALETTE = PALETTE[:num_classes] + [[255, 255, 255]]  # + background
```

### 2. **Redundant Default Value** ✅ FIXED
**File**: [train_stage_1.py](train_stage_1.py#L102)

**Problem**: Redundant `getattr(cfg.dataset, 'num_classes', getattr(cfg.dataset, 'num_classes', 4))`

**Solution**: Simplified to `getattr(cfg.dataset, 'num_classes', 4)`

### 3. **Confusion Matrix Documentation** ✅ IMPROVED
**File**: [utils/evaluate.py](utils/evaluate.py#L7-L19)

**Problem**: Unclear that `num_classes` parameter includes background class

**Solution**: Added clear docstring explaining:
- `num_classes` is **total** classes INCLUDING background
- For 2 semantic classes (benign, tumor), pass `num_classes=3` (benign + tumor + background)
- Matches how it's called in validation: `ConfusionMatrixAllClass(num_classes=cfg.dataset.num_classes + 1)`

```python
class ConfusionMatrixAllClass(object):
    def __init__(self, num_classes):
        """Initialize confusion matrix.
        
        Args:
            num_classes: Total number of classes INCLUDING background.
                        For example, if you have 2 semantic classes (benign, tumor),
                        pass num_classes=3 (benign, tumor, background).
        """
```

## Verified Dataset-Agnostic Components

### ✅ Model Architecture
- [model/model.py](model/model.py#L43): `num_classes` parameter properly used
- Total subclasses computed dynamically: `self.total_classes = sum(self.k_list)`
- No hard-coded class counts

### ✅ Training Pipeline
- Multi-scale loss weights configurable via YAML
- Class labels loaded from CSV with auto-detection
- K-means clustering respects `k_list` from config

### ✅ Validation & Evaluation
- Confusion matrix size: `cfg.dataset.num_classes + 1` (including background)
- Mean metrics exclude background: `fuse234_score[:-1].mean()`
- Per-class IoU printing handles variable class counts

### ✅ Dataset Loaders
- [datasets/wsi_dataset.py](datasets/wsi_dataset.py#L64): `num_classes` parameter
- [utils/trainutils.py](utils/trainutils.py#L86-L159): Auto-infers `num_classes` from CSV if not specified

## Configuration Requirements

To use the pipeline with N classes, ensure your config has:

```yaml
dataset:
  name: custom_wsi
  num_classes: 2  # Your number of classes (e.g., benign, tumor)
  class_order: [benign, tumor]  # Must match num_classes length

features:
  k_list: [3, 3]  # Subclasses per parent class (must match class_order length)
```

## Validation

The pipeline has been tested and verified to work with:
- ✅ **2-class custom WSI dataset** (benign, tumor)
- ✅ **4-class BCSS dataset** (tumor, stroma, inflammatory, necrosis)

All metrics (accuracy, IoU, Dice) properly computed for variable class counts.

## Known Limitations

1. **Maximum 8 classes** for color palette visualization (easy to extend if needed)
2. **Ground truth masks** must use 0-indexed labels (0, 1, 2, ..., N-1) with background as highest index
3. **CSV label format** must match `num_classes` (either one-hot vectors or single class indices)

## Future-Proofing

To add support for more classes:
1. Update `PALETTE` in [utils/validate.py](utils/validate.py#L163) with more colors
2. Ensure your label CSV has correct number of columns
3. Set `num_classes` and `class_order` in config YAML

No code changes needed beyond updating the config file!
