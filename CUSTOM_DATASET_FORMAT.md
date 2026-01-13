# Custom WSI Dataset Format

This guide explains how to prepare your data for training with the custom WSI + coordinates dataset.

## Directory Structure

```
data/
├── wsi/
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
├── coordinates/
│   ├── image1.npy  (or .txt)
│   ├── image2.npy
│   └── ...
├── ground_truth/
│   ├── image1.png  (validation/test only)
│   ├── image2.png
│   └── ...
├── split.csv
└── labels.csv
```

## File Formats

### 1. WSI Files (wsi_dir)
- **Format**: GeoTIFF (.tif) or any format readable by `scikit-image.io.imread`
- **Content**: Full whole slide images
- **Filename**: Should match base name used in split_csv and coordinates_dir
- **Color space**: RGB or will be automatically converted to RGB

### 2. Coordinates Files (coordinates_dir)
Patch coordinates for extracting regions from WSI files.

#### Option A: NumPy format (.npy)
```python
import numpy as np
# Shape: (n_patches, 2) for [x, y] coordinates
coords = np.array([
    [100, 150],
    [200, 250],
    [300, 350],
    # ... more coordinates
])
np.save('image1.npy', coords)
```

#### Option B: Text format (.txt)
```
100 150
200 250
300 350
...
```
Space or tab-separated x, y coordinates.

**Coordinate System**:
- Origin (0, 0) is top-left
- x = horizontal position (column)
- y = vertical position (row)
- Patch of size 224×224 is extracted centered at each coordinate

### 3. Split CSV (split_csv)
CSV file defining train/val/test splits.

**Format**: columns: `train, val, test` (one of these can be empty for a given row)

**Example** (split.csv):
```csv
train,val,test
image1.tif,image2.tif,image3.tif
image4.tif,image5.tif,image6.tif
image7.tif,,image8.tif
```

Each row contains one filename per split. Use empty cells for splits a file doesn't belong to.

**Alternative format** (column per file):
```csv
filename,split
image1.tif,train
image2.tif,val
image3.tif,test
```

### 4. Labels CSV (labels_csv)
Image-level class labels.

**Format**: columns: `image_name, label1, label2, label3, ...`

**Example** (labels.csv):
```csv
image_name,tumor,stroma,lymphocyte,necrosis
image1.tif,1,1,0,0
image2.tif,0,1,1,0
image3.tif,1,0,0,1
image4.tif,0,0,1,1
```

**Or with comma-separated labels**:
```csv
image_name,labels
image1.tif,"1,1,0,0"
image2.tif,"0,1,1,0"
```

### 5. Ground Truth Masks (gt_dir)
Pixel-level segmentation masks for validation/test sets.

- **Format**: PNG files (.png)
- **Content**: Each pixel value corresponds to a class index (0, 1, 2, 3, ...)
- **Filename**: Should match base name in split_csv
- **Expected only for**: val and test splits (not needed for train split)

**Example**:
```python
from PIL import Image
import numpy as np

# Create mask (H × W with class indices)
mask = np.array([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [2, 2, 3, 3],
    [2, 2, 3, 3],
], dtype=np.uint8)

Image.fromarray(mask).save('image1.png')
```

## Configuration File

Create a YAML config file with paths to your data:

```yaml
dataset:
  name: custom_wsi
  wsi_dir: ./data/wsi
  coordinates_dir: ./data/coordinates
  split_csv: ./data/split.csv
  labels_csv: ./data/labels.csv
  gt_dir: ./data/ground_truth
  num_classes: 4
  patch_size: 224
  coordinates_suffix: .npy  # or .txt
```

See `work_dirs/custom_wsi_template.yaml` for a complete example.

## Usage

```bash
# Training
python train_stage_1.py --config ./work_dirs/custom/config.yaml --gpu 0
```

## Key Parameters

- **patch_size** (default: 224): Size of patches extracted around coordinates
- **max_patches** (default: None): Limit patches per WSI. None = use all available
- **coordinates_suffix** (.npy or .txt): File format of coordinate files
- **num_classes**: Number of tissue classes in your labels

## Notes

- Patches are extracted centered at each coordinate
- Boundaries are handled by padding with white (255) if patch extends beyond WSI
- Images are automatically converted to RGB if grayscale or RGBA
- Features are normalized using BCSS normalization stats (can be customized)
