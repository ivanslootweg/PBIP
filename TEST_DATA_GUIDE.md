# Test Data & Evaluation Guide

## Issue: "281 (no GT mask)" Warning

When running validation/testing, you may see:
```
[VAL - Val/Test] Loaded 20 samples
  Skipped: 0 (no WSI), 0 (no coords), 281 (no GT mask), 0 (no label)
```

This means most samples in the test split don't have corresponding ground truth mask files.

## Why This Happens

The test dataset requires:
1. WSI files (.tif)
2. Coordinate files (.npy or .txt)
3. **Ground truth masks (.tif)** ← Often missing
4. Class labels (from CSV)

In a weak supervision setup, ground truth pixel-level annotations are often unavailable for most data. The pipeline is designed to work with weak labels (image-level only) during training, but the test dataset specifically requires pixel-level GT for evaluation metrics.

## Solutions

### Option 1: Use Train Split for Validation (Fastest)

If you don't have GT masks, use the training split for validation:

```yaml
# In your config
train:
  val_split: train  # Use training data for validation
```

This is common in weak supervision research but should be marked as such in results.

### Option 2: Create Dummy GT Masks

If you need validation but don't have real GT masks:

```python
import numpy as np
from PIL import Image
import os

def create_dummy_masks(gt_dir, num_classes=2):
    """Create dummy GT masks for testing"""
    os.makedirs(gt_dir, exist_ok=True)
    
    # Create simple binary masks (all background or all foreground)
    for i in range(20):
        # Create a checkerboard pattern as dummy GT
        mask = np.random.randint(0, num_classes, (224, 224), dtype=np.uint8)
        filename = f"sample_{i:03d}.tif"
        Image.fromarray(mask).save(os.path.join(gt_dir, filename))
    
    print(f"Created {20} dummy GT masks in {gt_dir}")

create_dummy_masks("/data/pathology/projects/ivan/WSS/gt_annotations")
```

### Option 3: Create Synthetic Test Set

Generate synthetic data with matching GT masks:

```python
import numpy as np
from PIL import Image
import os

def create_synthetic_test_set(output_dir, num_samples=10, image_size=512):
    """Create synthetic WSI patches with corresponding GT masks"""
    
    wsi_dir = os.path.join(output_dir, "wsi")
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(wsi_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Create synthetic image (random noise, more interesting than pure random)
        image = np.random.randint(100, 200, (image_size, image_size, 3), dtype=np.uint8)
        # Add some structure
        image[100:200, 100:200] = np.random.randint(50, 100, (100, 100, 3), dtype=np.uint8)
        
        # Create corresponding GT mask (0 = background, 1 = foreground)
        gt_mask = np.zeros((image_size, image_size), dtype=np.uint8)
        gt_mask[100:200, 100:200] = 1
        
        # Save
        Image.fromarray(image).save(os.path.join(wsi_dir, f"test_{i:03d}.tif"))
        Image.fromarray(gt_mask).save(os.path.join(gt_dir, f"test_{i:03d}.tif"))
    
    print(f"Created {num_samples} synthetic test samples")
    return wsi_dir, gt_dir
```

### Option 4: Skip Evaluation

If evaluation metrics aren't critical, focus on training:

```yaml
# In train_stage_1.py - comment out validation
# Don't run validation on test set with missing GT
```

## Proper Setup (Recommended)

For a proper experimental setup:

1. **Training Split**: Requires WSI, coordinates, labels (NO GT masks needed)
2. **Validation Split**: Requires WSI, coordinates, labels, GT masks (small subset, ~20-50 samples)
3. **Test Split**: Same as validation (separate held-out set for final evaluation)

### Directory Structure
```
data/
├── wsi/                      # WSI files
│   ├── sample_001.tif
│   ├── sample_002.tif
│   └── ...
├── coordinates/              # Patch coordinates
│   ├── sample_001.npy
│   ├── sample_002.npy
│   └── ...
├── gt_annotations/           # Ground truth masks (for val/test ONLY)
│   ├── sample_001.tif        # Only for ~50 samples
│   ├── sample_002.tif
│   └── ...
└── splits.csv               # Train/val/test split with WSI filenames
```

### Config Setup
```yaml
dataset:
  wsi_dir: data/wsi
  coordinates_dir: data/coordinates
  gt_dir: data/gt_annotations
  split_csv: data/splits.csv
  num_classes: 2

train:
  val_split: valid  # or test
  epoch: 10
```

## Expected Output

With proper GT masks, you should see:
```
[VAL - Val/Test] Loaded 50 samples
  Skipped: 0 (no WSI), 0 (no coords), 0 (no GT mask), 0 (no label)

Testing results:
Test all acc4: 85.200000
Test avg acc4: 87.500000
Fuse234 score: tensor([0.8234, 0.8756, 0.0000], device='cuda:0'), mIOU: 0.8495
```

## If You Only Have Image-Level Labels

This is common in weak supervision. In this case:

1. **Skip pixel-level evaluation** - You can't properly evaluate without GT masks
2. **Use image-level metrics**:
   - Classification accuracy
   - AUC-ROC
   - Precision/Recall per class
3. **Evaluate on task-specific metrics**:
   - Sensitivity/Specificity (medical)
   - F1-score
   - Balanced accuracy

## Generating GT Masks from Annotations

If you have annotations in another format, convert them:

```python
# Example: Convert from JSON annotations to TIF masks
import json
import numpy as np
from PIL import Image
import cv2

def json_to_mask(json_file, output_mask, image_size=(512, 512)):
    """Convert JSON polygon annotations to binary mask"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    mask = np.zeros(image_size, dtype=np.uint8)
    
    for shape in data.get('shapes', []):
        if shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
        elif shape['shape_type'] == 'rectangle':
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            mask[int(y1):int(y2), int(x1):int(x2)] = 1
    
    Image.fromarray(mask).save(output_mask)
    return mask
```

## Troubleshooting

### "Skipped: ... (no GT mask)" for all validation samples
- Ensure GT directory path is correct in config
- Ensure mask filenames match WSI filenames (with correct suffix)
- GT masks should have same base name as WSI: `sample.tif` → `sample.tif`

### NaN values in evaluation metrics
- Occurs when all predictions are background (no positive class detected)
- Or when GT masks are all empty
- Usually indicates model isn't learning properly
- Check training loss values

### Validation loss increasing while training loss decreasing
- Classic overfitting
- Try: data augmentation, lower learning rate, more regularization
- Or: your validation set is too different from training set

## References

- Weak supervision literature: [Learning with Noisy Labels](https://arxiv.org/abs/1901.07291)
- Multiple Instance Learning: [MIL Survey](https://arxiv.org/abs/1807.06358)
- This paper: [PBIP Paper](https://arxiv.org/abs/2503.12068)
