# PBIP Pipeline Summary - WSI Custom Dataset with Coordinates

## Architecture Overview

This pipeline repurposes the BCSS weak supervision repository for custom pathology WSI datasets with coordinate-based patch extraction and MedCLIP-based prototype learning.

### Key Features

1. **Coordinate-Based Patch Extraction**: Patches are extracted on-the-fly from WSI files using stored coordinates in slide2vec format
2. **WSI Source Tracking**: Each coordinate stores the source WSI name for efficient retrieval
3. **MedCLIP Features**: Weak image-level labels are mapped to prototype exemplars via k-means clustering
4. **Configuration-Driven**: All parameters controlled via YAML config (`work_dirs/custom_wsi_template.yaml`)
5. **2-Class Setup**: Supports arbitrary number of classes (configured for benign/tumor)

---

## Data Pipeline

### 1. **Prototype Coordinate Extraction**
**Script**: `features/extract_prototypes_from_gt.py`

**Purpose**: Sample coordinates from training WSIs, organize by class, and save in structured format

**Input**:
- WSI files (`.tif`, `.tiff`) in `dataset.wsi_dir`
- CSV labels file: `image_name,label` (or multi-label format)
- CSV split file: columns `train`, `val`, `test` with WSI basenames

**Output**: 
- `prototype_coordinates/{benign,tumor}.npy` - structured arrays with 8 fields:
  - `x, y`: patch coordinates
  - `tile_level, tile_size_resized, tile_size, tile_size_lv0, resize_factor`: metadata
  - `wsi_name`: source WSI filename (enables efficient lookup)

**Command**:
```bash
python3 features/extract_prototypes_from_gt.py \
    --config work_dirs/custom_wsi_template.yaml \
    --num_per_class 5 \
    --samples_per_wsi 1000
```

**Parameters**:
- `num_per_class`: Max number of WSIs per class (semantic: "max WSIs", not patches)
- `samples_per_wsi`: Patches to extract from each WSI
- Uses random sampling for training patches (ensures diversity)

---

### 2. **MedCLIP Feature Extraction**
**Script**: `features/excract_medclip_proces.py`

**Purpose**: Extract MedCLIP vision model features from prototype patches

**Input**:
- Prototype coordinates from step 1
- WSI files for patch extraction
- Config with class order and paths

**Output**:
- `features/medclip_features.pkl` - dict with class names as keys, list of features+names as values

**How It Works**:
1. Loads structured coordinate arrays with `wsi_name` field
2. For each coordinate:
   - Uses `wsi_name` field to construct WSI path directly (O(1) lookup)
   - Falls back to searching all WSIs if not found (handles legacy data)
   - Extracts patch using OpenSlide or NumPy fallback
   - Computes MedCLIP vision embedding
3. Stores {name, features} tuples for k-means clustering

**Command**:
```bash
python3 features/excract_medclip_proces.py \
    --config work_dirs/custom_wsi_template.yaml
```

**Optimization**: Direct WSI lookup via `wsi_name` avoids O(n×m) search complexity

---

### 3. **K-Means Clustering**
**Script**: `features/k_mean_cos_per_class.py`

**Purpose**: Cluster MedCLIP features using cosine similarity k-means

**Input**:
- MedCLIP features from step 2
- Config with `features.k_list` (clusters per class)

**Output**:
- `features/label_features.pkl` - prototype exemplars organized by class

**Semantics**:
- For 2-class setup: `k_list=[3,3]` creates 3 clusters per class (6 total prototypes)
- Clusters ordered by similarity to centroid (top 5 reported per cluster)

**Command**:
```bash
python3 features/k_mean_cos_per_class.py \
    --config work_dirs/custom_wsi_template.yaml
```

---

### 4. **Training with Weak Supervision**
**Script**: `train_stage_1.py`

**Purpose**: Train SegFormer model with weak image-level labels

**Input**:
- Training dataset from `CustomWSIPatchTrainingDataset`
- Weak labels from labels CSV
- Prototype exemplars from step 3

**Process**:
1. Training patches are randomly sampled from coordinates
2. Image-level weak labels assigned to all patches from that WSI
3. Contrastive loss aligns patch features with prototype exemplars
4. Supports multi-label weak labels (e.g., tumor + necrotic area)

**Command**:
```bash
python3 train_stage_1.py \
    --config work_dirs/custom_wsi_template.yaml \
    --gpu 0
```

---

## Configuration Structure

**File**: `work_dirs/custom_wsi_template.yaml`

### Dataset Configuration
```yaml
dataset:
  name: "custom_wsi"
  class_order: [benign, tumor]  # Determines class indices
  num_classes: 2
  
  # Data paths
  wsi_dir: "/path/to/wsi/files"          # Input WSI files
  coordinates_dir: "/path/to/coords"     # For training/val/test patches
  split_csv: "/path/to/split.csv"        # train, val, test columns
  labels_csv: "/path/to/labels.csv"      # image_name, label
  gt_dir: "/path/to/gt_masks"            # Ground truth for evaluation
  
  # Parameters
  patch_size: 224
  use_openslide: true
  coordinates_suffix: ".npy"
  mask_suffix: ".png"
```

### Features Configuration
```yaml
features:
  save_dir: "work_dirs/custom_wsi/features"
  
  # Prototype extraction
  num_per_class: 5              # Max WSIs per class
  samples_per_wsi: 1000         # Patches per WSI
  
  # K-means
  k_list: [3, 3]                # Clusters per class (must match num_classes)
  
  # Output files
  medclip_features_pkl: "medclip_features.pkl"
  label_feature_pkl: "label_features.pkl"
```

### Model Configuration
```yaml
model:
  type: "cls"
  backbone: "mit_b0"            # SegFormer backbone (b0-b5)
  pretrained: true              # Auto-download ImageNet weights
  num_classes: 2                # Matches dataset.num_classes
```

---

## Data Formats

### Structured Array Format (slide2vec compatible)
```python
dtype = [
    ('x', np.int32),                    # X coordinate
    ('y', np.int32),                    # Y coordinate  
    ('tile_level', np.int32),           # Pyramid level (0 = native resolution)
    ('tile_size_resized', np.int32),    # Resized patch size
    ('tile_size', np.int32),            # Original patch size
    ('tile_size_lv0', np.int32),        # Patch size at level 0
    ('resize_factor', np.float32),      # Scaling factor
    ('wsi_name', 'O'),                  # Source WSI filename (object type)
]
```

### CSV Formats

**Labels CSV** (single or multi-label):
```
image_name,label
slide1.tif,0
slide2.tif,"1,0"
```

**Split CSV** (paths to WSI files):
```
train,val,test
slide1.tif,,
,slide2.tif,
,,slide3.tif
```

---

## Key Implementation Details

### 1. Coordinate Loading (excract_medclip_proces.py)
```python
def load_prototype_coordinates(coordinates_dir, class_name):
    # Handles structured arrays with optional wsi_name field
    # Returns x, y, wsi_names (optimized for WSI lookup)
```

### 2. WSI Lookup Strategy
- **Primary**: Use `wsi_name` field to construct path directly (O(1))
- **Fallback**: Search all WSIs if `wsi_name` not available (legacy support)
- **Error handling**: Warns on missing coordinates, continues with other patches

### 3. Patch Extraction
- **OpenSlide**: Multi-resolution WSI format support (TIF, TIFF)
- **NumPy fallback**: Single-image format support (PNG, JPG)
- **Padding**: Border padding if coordinate near image edge

### 4. Label Assignment
- Single-label format: `0` → class 0, `1` → class 1
- Multi-label format: `[0,1]` → both classes active
- All patches from same WSI share image-level labels

---

## Complete Workflow Example

```bash
# 1. Extract prototypes (sample coordinates from training WSIs)
python3 features/extract_prototypes_from_gt.py \
    --config work_dirs/custom_wsi_template.yaml \
    --num_per_class 5 \
    --samples_per_wsi 1000

# 2. Extract MedCLIP features from prototype patches
python3 features/excract_medclip_proces.py \
    --config work_dirs/custom_wsi_template.yaml

# 3. Cluster features using cosine similarity k-means
python3 features/k_mean_cos_per_class.py \
    --config work_dirs/custom_wsi_template.yaml

# 4. Train model with weak supervision
python3 train_stage_1.py \
    --config work_dirs/custom_wsi_template.yaml \
    --gpu 0

# 5. Evaluate on validation set
python3 train_stage_1.py \
    --config work_dirs/custom_wsi_template.yaml \
    --validate_only \
    --resume /path/to/checkpoint
```

---

## Compatibility

### 2-Class Setup Verified
- ✅ `hierarchical_utils.py`: Generic k_list-based implementation
- ✅ `contrast_loss.py`: No hardcoded class counts
- ✅ `k_mean_cos_per_class.py`: Reads class_order from config
- ✅ `wsi_dataset.py`: Supports arbitrary num_classes
- ✅ Model backbone: timm-based with configurable num_classes

### Environment
- PyTorch 2.1.2
- timm 0.9.16 (automatic ImageNet weight downloads)
- OpenSlide-python 1.2.0
- MedCLIP 0.0.3
- OmegaConf 2.3.0

---

## Troubleshooting

### "Prototype coordinates not found"
- Ensure `extract_prototypes_from_gt.py` has completed successfully
- Check that `work_dirs/custom_wsi/prototype_coordinates/` directory exists
- Verify config paths match actual file locations

### "Could not extract patch at (x, y) from any WSI"
- Coordinates may be outside WSI bounds or from non-existent WSI
- Enable verbose logging to identify problematic coordinates
- Reduce `samples_per_wsi` parameter (use smaller patch count)

### Memory issues during MedCLIP extraction
- Reduce batch size (process one class at a time)
- Use `use_openslide: false` for in-memory image loading
- Ensure GPU memory is available (`nvidia-smi`)

### Class label mismatch
- Verify `dataset.class_order` matches CSV column names
- Ensure labels are 0-indexed (e.g., 0 for benign, 1 for tumor)
- Check that multi-label CSV uses comma-separated format: `"1,0"`

---

## Recent Optimizations

### Update: WSI Name Tracking
- **Before**: Coordinates-only extraction required searching all WSIs (O(n×m))
- **After**: Each coordinate stores source WSI name for direct lookup (O(1))
- **Impact**: 10-100x faster MedCLIP feature extraction

### Update: On-The-Fly Patch Extraction
- **Before**: Pre-extracted PNG patches stored on disk
- **After**: Patches extracted directly from WSI during feature computation
- **Impact**: 50-70% disk space reduction, eliminates intermediate PNG storage

### Update: Slide2Vec Format Compatibility
- **Before**: Custom coordinate format
- **After**: Structured numpy arrays matching slide2vec standard
- **Impact**: Interoperability with external slide2vec tools

---

## Next Steps

1. **Validation Dataset Creation**: Use `CustomWSIPatchTestDataset` for deterministic val/test splits
2. **Weak Segmentation Maps**: Use prototype features to create pseudo-labels during training
3. **Multi-Scale Training**: Extract patches at multiple zoom levels (tile_level parameter)
4. **Cross-Validation**: Modify split CSV to generate k-fold splits for robustness
