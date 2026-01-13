# Code Deduplication Summary

This document summarizes the code duplication that has been identified and refactored in the PBIP codebase.

## Created: `utils/common.py`

A new centralized utility module containing commonly duplicated functions:

### 1. **Coordinate Loading** ✅ IMPLEMENTED
**Duplicated in**: 
- `datasets/wsi_dataset.py` (2 copies in different classes)
- `datasets/custom_virchow.py` (2 copies in different classes)  
- `features/extract_prototypes_from_gt.py`

**Solution**: Single `load_coordinates()` function that:
- Handles both `.npy` and `.txt` formats
- Supports structured arrays (slide2vec format)
- Optional max_patches sampling
- Returns consistent (N, 2) int32 array

### 2. **Patch Extraction** ✅ IMPLEMENTED  
**Duplicated in**:
- `datasets/wsi_dataset.py` (4 methods across 2 classes)
- `datasets/custom_virchow.py` (4 methods across 2 classes)
- `features/extract_prototypes_from_gt.py`
- `features/excract_medclip_proces.py`

**Solution**: Three unified functions:
- `extract_patch_numpy()` - Extract from numpy array with padding
- `extract_patch_openslide()` - Extract using OpenSlide
- `extract_patch()` - Wrapper that chooses method

### 3. **Multi-scale Merge Operations** ✅ IMPLEMENTED
**Duplicated in**:
- `train_stage_1.py` (4 repeated calls to merge_to_parent_predictions)
- `utils/validate.py` (8 repeated calls across 2 functions)

**Solution**:
- `merge_multiscale_predictions()` - Merge all 4 cls outputs at once
- `merge_multiscale_cams()` - Merge all 4 CAM outputs at once

**Before**:
```python
cls1 = merge_to_parent_predictions(cls1, k_list, method='mean')
cls2 = merge_to_parent_predictions(cls2, k_list, method='mean')
cls3 = merge_to_parent_predictions(cls3, k_list, method='mean')
cls4 = merge_to_parent_predictions(cls4, k_list, method='mean')
```

**After**:
```python
cls1, cls2, cls3, cls4 = merge_multiscale_predictions(
    [cls1, cls2, cls3, cls4], k_list, method='mean')
```

### 4. **Multi-scale Loss Computation** ✅ IMPLEMENTED
**Duplicated in**:
- `train_stage_1.py`
- `utils/validate.py`

**Solution**: `compute_multiscale_loss()` function

**Before**:
```python
loss_scale1 = loss_function(cls1_merge, cls_labels)
loss_scale2 = loss_function(cls2_merge, cls_labels)
loss_scale3 = loss_function(cls3_merge, cls_labels)
loss_scale4 = loss_function(cls4_merge, cls_labels)

cls_loss = (cfg.train.scale1_weight * loss_scale1 + 
           cfg.train.scale2_weight * loss_scale2 + 
           cfg.train.scale3_weight * loss_scale3 + 
           cfg.train.scale4_weight * loss_scale4)
```

**After**:
```python
weights = (cfg.train.scale1_weight, cfg.train.scale2_weight,
          cfg.train.scale3_weight, cfg.train.scale4_weight)
cls_loss = compute_multiscale_loss(
    [cls1_merge, cls2_merge, cls3_merge, cls4_merge], 
    cls_labels, loss_function, weights)
```

### 5. **Color Palette Generation** ✅ IMPLEMENTED
**Hard-coded in**:
- `utils/validate.py` (15 lines of palette definition)

**Solution**: `get_color_palette()` function
- Supports up to 10 classes (easily extensible)
- Optional background color
- Single source of truth

**Before**:
```python
num_classes = cfg.dataset.num_classes
PALETTE = [
    [255, 0, 0],      # Red
    [0, 255, 0],      # Green
    # ... 8 more lines
]
PALETTE = PALETTE[:num_classes] + [[255, 255, 255]]
```

**After**:
```python
PALETTE = get_color_palette(cfg.dataset.num_classes, include_background=True)
```

## Refactored Files

### ✅ `utils/validate.py`
- **Imports**: Added common utilities
- **validate()**: Uses merge helpers and compute_multiscale_loss
- **validate() TTA loop**: Uses merge helpers (16 lines → 4 lines)
- **generate_cam()**: Uses merge helpers and get_color_palette
- **Lines saved**: ~50 lines

### ✅ `train_stage_1.py`
- **Imports**: Added common utilities  
- **Training loop**: Uses merge helpers and compute_multiscale_loss
- **Lines saved**: ~20 lines

### 🔧 `datasets/wsi_dataset.py` (Partially refactored)
- **Imports**: Added common utilities
- **CustomWSIPatchTrainingDataset**: Can remove _load_coordinates, _extract_patch_numpy, _extract_patch_openslide methods
- **CustomWSIPatchTestDataset**: Can remove same 3 methods
- **Potential lines saved**: ~90 lines

### 🔧 `datasets/custom_virchow.py` (Not yet refactored)
- Same opportunities as wsi_dataset.py
- **Potential lines saved**: ~90 lines

### 🔧 `features/extract_prototypes_from_gt.py` (Not yet refactored)
- Can replace load_coordinates() function
- Can replace extract_patch() function  
- **Potential lines saved**: ~40 lines

### 🔧 `features/excract_medclip_proces.py` (Not yet refactored)
- Can replace extract_patch() function
- **Potential lines saved**: ~20 lines

## Benefits

### Code Quality
- ✅ **Single source of truth** for common operations
- ✅ **Consistent behavior** across all uses
- ✅ **Easier to test** - centralized functions
- ✅ **Easier to maintain** - fix bugs in one place
- ✅ **Better documentation** - docstrings in one place

### Metrics
- **Lines of code reduced**: ~70 lines already, ~240 total potential
- **Files improved**: 2/6 so far
- **Duplicated methods eliminated**: 5/15 so far

## Remaining Work

To complete the deduplication:

1. **wsi_dataset.py** (Both classes):
   ```python
   # Replace methods with:
   coords = load_coordinates(coord_path, self.coordinates_suffix, self.max_patches)
   patch = extract_patch_numpy(wsi, x, y, self.patch_size)  # or
   patch = extract_patch_openslide(wsi_path, x, y, self.patch_size)
   ```

2. **custom_virchow.py** (Both classes):
   - Same changes as wsi_dataset.py

3. **extract_prototypes_from_gt.py**:
   ```python
   from utils.common import load_coordinates, extract_patch
   # Remove local load_coordinates() and extract_patch() functions
   ```

4. **excract_medclip_proces.py**:
   ```python
   from utils.common import extract_patch
   # Remove local extract_patch() function
   ```

## Testing Recommendations

After completing refactoring:

1. **Unit tests** for `utils/common.py` functions
2. **Integration tests** to ensure datasets still load correctly
3. **End-to-end test** of full training pipeline
4. **Verify** CAM visualization colors match expected palette

## Future Improvements

Consider centralizing:
- CSV label loading (duplicated in trainutils.py and extract_prototypes_from_gt.py)
- WSI format conversion logic (GRAY2RGB, RGBA2RGB conversions)
- Patch padding logic (currently slightly different in some places)
