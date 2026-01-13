# Complete Data Flow Documentation for PBIP Custom WSI Pipeline

## Overview

This document describes the complete data flow through the PBIP pipeline for weakly-supervised histopathological image segmentation using custom WSI + coordinate data, including training and inference stages.

---

## Stage 0: Prototype Coordinate Extraction
**File**: `features/extract_prototypes_from_gt.py`

### Input
```
Split CSV (train/val/test columns)
  ↓
Labels CSV (image_name, label)
  ↓
WSI Directory (.tif files)
  ↓
Coordinates Directory (.npy/.txt files)
```

### Process
```
1. Load Config (paths, class_order, patch_size)
2. Parse Split CSV → Extract train column
3. Parse Labels CSV → Build image_name → label dictionary
4. Match files to classes using label values
5. Group files by class (benign, tumor, etc)
6. For each class:
   a. Sample up to num_per_class WSIs
   b. For each WSI:
      - Load coordinate file (.npy or .txt)
      - Randomly sample up to samples_per_wsi coordinates
      - Track source WSI name with each coordinate
   c. Save coordinates in slide2vec format (structured .npy)
```

### Output
```
prototype_coordinates/
├── benign.npy (structured array: x, y, tile_level, tile_size, wsi_name, ...)
└── tumor.npy  (structured array: x, y, tile_level, tile_size, wsi_name, ...)
```

### Key Features
- **Efficient**: Stores coordinates + metadata, patches extracted on-the-fly
- **Traceable**: WSI name embedded in each coordinate for O(1) lookup
- **Flexible**: Handles both simple (N,2) arrays and slide2vec structured formats
- **Diverse**: Random sampling across multiple WSIs per class

---

## Stage 1: MedCLIP Feature Extraction
**File**: `features/excract_medclip_proces.py`

### Input
```
prototype_coordinates/benign.npy, tumor.npy
  ↓
WSI Directory (.tif files)
  ↓
Config (class_order, feature_save_dir, medclip_features_pkl)
```

### Process
```
1. Load prototype coordinates (structured arrays)
2. Parse x, y, tile_level, wsi_name from each coordinate
3. For each coordinate:
   a. Construct WSI path using wsi_name field
   b. Open WSI with OpenSlide (or skimage fallback)
   c. Read region at (x, y) → 224×224 RGB patch
   d. Pass through MedCLIP vision model
   e. Generate 512-D feature vector
4. Aggregate features per class
5. Save all features with metadata
```

### Output
```
image_features/
└── medclip_exemplars.pkl
    └── Contains:
        - features: (N, 512) tensor of embeddings
        - class_names: list of class labels
        - wsi_sources: which WSI each patch came from
```

### Key Features
- **Efficient WSI access**: Uses embedded wsi_name to avoid directory scanning
- **On-the-fly extraction**: No intermediate PNG files
- **Multi-resolution support**: Handles tile_level metadata
- **Fallback**: Works with OpenSlide or standard image I/O

---

## Stage 2: K-Means Clustering → Prototype Generation
**File**: `features/k_mean_cos_per_class.py`

### Input
```
medclip_exemplars.pkl (all embeddings per class)
  ↓
Config (class_order, k_list, save_dir, label_feature_pkl)
```

### Process
```
1. Load embeddings grouped by class
2. For each class:
   a. Apply K-means clustering (k=3 per class from config)
   b. Extract cluster centers (3 typical patterns per class)
3. Combine all cluster centers into single prototype tensor
4. Create metadata:
   - k_list: [3, 3, ...] subclasses per parent class
   - class_order: ['benign', 'tumor', ...]
   - cumsum_k: cumulative index offsets
5. Save in slide2vec format with full path info
```

### Output
```
image_features/
└── label_fea_pro.pkl
    └── Contains:
        - features: (6, 512) tensor (3 benign + 3 tumor prototypes)
        - k_list: [3, 3]
        - class_order: ['benign', 'tumor']
        - cumsum_k: [0, 3, 6]
```

### Key Features
- **Subclass discovery**: K-means finds natural variation within each class
- **Compact representation**: 6 prototypes vs hundreds of exemplars
- **Config-driven**: k_list customizable per use case
- **Multi-class ready**: Handles arbitrary number of classes

---

## Stage 3: Training with Weak Supervision
**File**: `train_stage_1.py`

### Input
```
label_fea_pro.pkl (prototypes)
  ↓
prototype_coordinates/ (for validation debugging)
  ↓
Split CSV (train/val/test columns)
  ↓
Labels CSV (image_name, label)
  ↓
WSI Directory (.tif files)
  ↓
Full Coordinates Directory (.npy/.txt files)
  ↓
GT masks (val/test only) - OPTIONAL
  ↓
Config (model, training hyperparams)
```

### Architecture

```
Input: 224×224 RGB Patch
    ↓
SegFormer Backbone (timm mit_b0-b5)
    ├─ Layer 1 (Stride 4)
    ├─ Layer 2 (Stride 8)
    ├─ Layer 3 (Stride 16)
    └─ Layer 4 (Stride 32)
    ↓
Multi-scale Feature Extraction
    ├─ F1: [B, 64, 56, 56]
    ├─ F2: [B, 128, 28, 28]
    ├─ F3: [B, 256, 14, 14]
    └─ F4: [B, 512, 7, 7]
    ↓
Prototype Matching (Cosine Similarity)
    ├─ F1 vs Prototypes → [B, 6, 56, 56]
    ├─ F2 vs Prototypes → [B, 6, 28, 28]
    ├─ F3 vs Prototypes → [B, 6, 14, 14]
    └─ F4 vs Prototypes → [B, 6, 7, 7]
    ↓
CAM Generation @ Each Scale
    ├─ CAM1: [B, 2, 56, 56]
    ├─ CAM2: [B, 2, 28, 28]
    ├─ CAM3: [B, 2, 14, 14]
    └─ CAM4: [B, 2, 7, 7]
    ↓
Classification Logits (4 scales)
    ├─ L1: [B, 2]
    ├─ L2: [B, 2]
    ├─ L3: [B, 2]
    └─ L4: [B, 2]
```

### Training Process

```
For each epoch:
  For each batch in training set:
    1. Random Coordinate Sampling
       - Select random coordinate from each training WSI
       
    2. Patch Extraction
       - OpenSlide region read at (x, y)
       - Resize to 224×224
       
    3. Data Augmentation (Albumentations)
       - Normalize: apply MEAN/STD
       - HorizontalFlip: p=0.5
       - VerticalFlip: p=0.5
       - RandomRotate90: random 90° rotations
       - ToTensor: convert to torch tensor
       
    4. Forward Pass
       - SegFormer backbone → multi-scale features
       - Prototype matching → cosine similarity
       - Classification logits at each scale
       - CAM generation at each scale
       
    5. Feature Extraction
       - Extract FG (foreground) regions from CAMs
       - Extract BG (background) regions from CAMs
       - Get MedCLIP features for FG and BG patches
       
    6. Loss Computation
       
       a) Classification Loss (multi-scale)
          L_cls = 0.0*BCE(L1, label) +
                  0.1*BCE(L2, label) +
                  1.0*BCE(L3, label) +
                  1.0*BCE(L4, label)
          
       b) Contrastive Loss
          L_fg_contra = InfoNCE(
              query=FG_features,
              positive=FG_prototypes,
              negatives=BG_prototypes
          )
          L_bg_contra = InfoNCE(
              query=BG_features,
              positive=BG_prototypes,
              negatives=FG_prototypes
          )
          L_contra = 0.1*L_fg_contra + 0.1*L_bg_contra
       
       c) CAM Regularization Loss
          L_cam_smooth = Total Variation of CAMs
          L_cam_sparsity = L1 norm of CAMs
          L_cam = 0.1*L_cam_smooth + 0.01*L_cam_sparsity
       
       Total Loss = L_cls + L_contra + L_cam
       
    7. Backward Pass
       - Compute gradients via backpropagation
       - Update model weights
       
    8. Logging
       - Track losses in TensorBoard
       - Log batch-level metrics

  Validation (every N batches or end of epoch)
    1. Load val split coordinates
    2. Deterministic patch extraction (first coord per WSI)
    3. Forward pass (no augmentation, batch norm eval mode)
    4. Generate segmentation masks from CAMs
    5. Crop GT mask to patch region
    6. Compute metrics: IoU, Dice, Precision, Recall
    7. Save best checkpoint based on val IoU
```

### Output
```
work_dirs/custom_wsi/
├── checkpoints/
│   ├── latest.pth (latest epoch weights)
│   ├── best.pth (best validation IoU weights)
│   └── epoch_*.pth (periodic checkpoints)
├── runs/
│   └── tensorboard logs (losses, metrics over time)
└── logs.txt (training history)
```

### Key Features
- **Weak supervision**: Only image-level labels needed (no pixel masks for training)
- **Multi-scale learning**: Leverages 4-level feature pyramid
- **Automatic pseudo-labels**: CAMs generate pixel-wise supervision
- **Contrastive learning**: Separates foreground from background
- **Efficient training**: Random coordinate sampling from unlimited patches
- **Reproducible**: Deterministic validation metrics

### Configuration
```yaml
training:
  epochs: 8
  batch_size: 10
  optimizer:
    name: adamw
    lr: 0.001
    weight_decay: 0.0005
  scheduler:
    type: cosine
    warmup_epochs: 1
  loss_weights:
    scale_weights: [0.0, 0.1, 1.0, 1.0]  # Per-scale classification
    contrastive: 0.1
    cam_smooth: 0.1
    cam_sparsity: 0.01
  validation_interval: 100  # validate every N batches
  checkpoint_interval: 1000
```

---

## Stage 4: Evaluation & Validation
**File**: `utils/validate.py`

### Purpose
Evaluate trained model on validation/test split with ground truth masks

### Input
```
Trained Model (best.pth)
  ↓
Test Split Coordinates (.npy files)
  ↓
WSI Directory (.tif files)
  ↓
Ground Truth Masks (per WSI)
  ↓
Config (class_order, num_classes)
```

### Process
```
1. Load trained model weights
2. Set model to evaluation mode (no dropout, batch norm frozen)
3. For each test WSI:
   a. Load first coordinate (deterministic)
   b. Extract patch from WSI
   c. Forward pass → get logits & CAMs
   d. Load corresponding GT mask
   e. Crop GT mask to match patch region
   f. Convert CAMs to binary segmentation (argmax)
   g. Compute metrics:
      - IoU (Intersection over Union)
      - Dice (Sørensen–Dice coefficient)
      - Precision
      - Recall
      - F1-score
   h. Accumulate per-patch metrics
   
4. Aggregate metrics:
   - Macro-average: mean across all patches
   - Weighted-average: weighted by patch sizes
   - Per-class metrics for each class
   
5. Generate evaluation report
```

### Output
```
results/
├── metrics.json
│   ├── val_iou: 0.85
│   ├── val_dice: 0.92
│   ├── val_precision: 0.89
│   ├── val_recall: 0.87
│   └── per_class_metrics: {...}
│
├── confusion_matrix.csv
│   └── Predicted class distribution vs ground truth
│
├── prediction_samples/
│   ├── sample_1_input.png
│   ├── sample_1_gt.png
│   └── sample_1_pred.png
│
└── evaluation_report.txt
    └── Detailed metrics breakdown
```

### Metrics Definition

**IoU (Intersection over Union)**:
```
IoU = TP / (TP + FP + FN)
where TP = True Positives (correct predictions)
      FP = False Positives (predicted positive, actually negative)
      FN = False Negatives (predicted negative, actually positive)
```

**Dice Coefficient**:
```
Dice = 2*TP / (2*TP + FP + FN)
Similar to F1-score, ranges from 0-1
```

---

## Stage 5: Inference on New WSIs
**File**: `features/inference.py` (or custom script)

### Purpose
Apply trained model to new WSIs without ground truth masks

### Input
```
Trained Model (best.pth)
  ↓
New WSI Directory (.tif files)
  ↓
New Coordinates Directory (.npy files)
  ↓
Config (class_order, num_classes, patch_size)
```

### Process
```
1. Load trained model + set to eval mode
2. For each new WSI:
   a. Load coordinate file
   b. For each coordinate (x, y):
      - Extract 224×224 patch via OpenSlide
      - Forward pass → logits, CAMs, features
      - Convert CAMs to segmentation mask
      - Apply CRF (Conditional Random Field) optional:
        * Refine mask with edge-aware smoothing
      
   c. Upsample mask back to full resolution
      - Scale from 224×224 to original patch size
      - Adjust coordinates if using multiple levels
   
   d. Assemble patches into full WSI segmentation
      - Handle overlaps with averaging or voting
      - Apply post-processing:
        * Morphological closing/opening
        * Connected component filtering
        * Boundary refinement
   
   e. Save segmentation:
      - PNG mask (discrete labels)
      - NPZ features (continuous confidence maps)
      - JSON metadata (timing, version, hyperparams)
```

### Output
```
predictions/
├── wsi_1/
│   ├── segmentation_mask.png (class indices)
│   ├── confidence_maps.npz
│   │   ├── class_0: [H, W] probability map
│   │   └── class_1: [H, W] probability map
│   ├── visualization.png (RGB overlay)
│   └── metadata.json
│
├── wsi_2/
│   └── ...
│
└── batch_results.csv
    ├── wsi_name, num_patches, processing_time, model_version
    └── ...
```

### Configuration
```yaml
inference:
  batch_size: 32  # process multiple patches in parallel
  use_crf: true
  use_tta: false  # Test-Time Augmentation
  overlap_ratio: 0.5  # for tiling large WSIs
  post_processing:
    morphological_op: closing  # closing, opening, or none
    kernel_size: 3
    min_component_size: 100  # pixels
    edge_refinement: true
```

### Post-Processing Details

**Morphological Operations**:
```
Closing (Dilation then Erosion):
  - Fills small holes in foreground
  - Preserves foreground boundaries
  
Opening (Erosion then Dilation):
  - Removes small noise/foreground
  - Preserves background boundaries
```

**Connected Component Analysis**:
```
1. Label connected regions
2. Filter by size (remove components < min_component_size)
3. Relabel with sequential IDs
4. Update mask with valid components only
```

**Boundary Refinement**:
```
1. Compute distance transform from boundary
2. Apply Gaussian smoothing near boundary
3. Re-threshold with lower threshold at boundaries
4. Smooth transition at edges
```

---

## Complete End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ INPUTS: Split CSV + Labels CSV + WSI Dir + Coordinates Dir              │
└──────────────────┬──────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 0: Extract Prototype Coordinates                                   │
│ python3 features/extract_prototypes_from_gt.py --config config.yaml      │
│ Output: prototype_coordinates/{benign,tumor}.npy                         │
└──────────────────┬──────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 1: MedCLIP Feature Extraction                                      │
│ python3 features/excract_medclip_proces.py --config config.yaml          │
│ Output: image_features/medclip_exemplars.pkl                             │
└──────────────────┬──────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 2: K-Means Clustering → Prototypes                                 │
│ python3 features/k_mean_cos_per_class.py --config config.yaml            │
│ Output: image_features/label_fea_pro.pkl                                 │
└──────────────────┬──────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Training with Weak Supervision                                  │
│ python3 train_stage_1.py --config config.yaml --gpu 0                    │
│ Inputs: Prototypes + Training WSIs + Labels                              │
│ Output: checkpoints/{latest,best}.pth + TensorBoard logs                 │
│                                                                           │
│ Training Loop (8 epochs):                                                │
│   • Random coordinate sampling per WSI                                   │
│   • Patch extraction via OpenSlide                                       │
│   • Multi-scale feature extraction                                       │
│   • Prototype matching (cosine similarity)                               │
│   • CAM generation at 4 scales                                           │
│   • Loss computation (classification + contrastive + regularization)     │
│   • Gradient update                                                      │
│   • Validation every N batches                                           │
└──────────────────┬──────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 4: Validation/Evaluation                                           │
│ python3 utils/validate.py --config config.yaml --weights best.pth        │
│ Inputs: Test WSIs + Ground Truth Masks                                   │
│ Output: metrics.json, confusion_matrix.csv, prediction_samples/          │
│                                                                           │
│ Validation Process:                                                      │
│   • Deterministic patch extraction (first coordinate per WSI)            │
│   • Forward pass (no augmentation)                                       │
│   • Generate segmentation masks from CAMs                                │
│   • Compare with ground truth                                            │
│   • Compute: IoU, Dice, Precision, Recall, F1                            │
│   • Generate report                                                      │
└──────────────────┬──────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 5: Inference on New WSIs                                           │
│ python3 features/inference.py --config config.yaml --weights best.pth    │
│ Inputs: New WSI Directory + New Coordinates                              │
│ Output: predictions/ with masks, confidence maps, visualizations         │
│                                                                           │
│ Inference Process:                                                       │
│   • Load all coordinates for new WSI                                     │
│   • Extract patches in batches                                           │
│   • Batch forward pass                                                   │
│   • Generate segmentation masks                                          │
│   • Assemble patches into full WSI prediction                            │
│   • Apply post-processing (morphology, CRF)                              │
│   • Save masks + confidence maps + visualization                         │
└──────────────────┬──────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ OUTPUTS:                                                                 │
│  • Trained Model Weights                                                 │
│  • Validation Metrics & Report                                           │
│  • WSI-level Segmentation Masks                                          │
│  • Confidence Maps                                                       │
│  • Visualizations                                                        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Command Reference

```bash
# Stage 0: Extract Prototype Coordinates
python3 features/extract_prototypes_from_gt.py \
    --config work_dirs/custom_wsi_template.yaml \
    --num_per_class 5 \
    --samples_per_wsi 1000

# Stage 1: Extract MedCLIP Features
python3 features/excract_medclip_proces.py \
    --config work_dirs/custom_wsi_template.yaml

# Stage 2: K-Means Clustering
python3 features/k_mean_cos_per_class.py \
    --config work_dirs/custom_wsi_template.yaml

# Stage 3: Training
python3 train_stage_1.py \
    --config work_dirs/custom_wsi_template.yaml \
    --gpu 0

# Stage 4: Validation
python3 utils/validate.py \
    --config work_dirs/custom_wsi_template.yaml \
    --weights work_dirs/custom_wsi/checkpoints/best.pth

# Stage 5: Inference
python3 features/inference.py \
    --config work_dirs/custom_wsi_template.yaml \
    --weights work_dirs/custom_wsi/checkpoints/best.pth \
    --input_dir /path/to/new/wsi \
    --output_dir ./predictions
```

---

## Data Format Specifications

### Split CSV
```csv
train,val,test
DD_S03_P000551,DD_S03_P000447,DD_S03_P000367
DD_S03_P000431,DD_S03_P000449,DD_S03_P000369
```

### Labels CSV
```csv
image_name,label
DD_S03_P000551,0
DD_S03_P000431,1
```

### Prototype Coordinates (Output from Stage 0)
```python
# Structured numpy array
dtype = [
    ('x', np.int32),              # X coordinate
    ('y', np.int32),              # Y coordinate
    ('tile_level', np.int32),     # Resolution level
    ('tile_size_resized', np.int32),
    ('tile_size', np.int32),
    ('tile_size_lv0', np.int32),
    ('resize_factor', np.float32),
    ('wsi_name', 'O'),            # Source WSI filename
]
```

### Prediction Output (Stage 5)
```json
{
  "wsi_name": "DD_S03_P000551.tif",
  "model_version": "best_epoch_6",
  "num_patches": 1024,
  "processing_time_sec": 45.2,
  "class_order": ["benign", "tumor"],
  "segmentation_mask": "segmentation_mask.png",
  "confidence_maps": "confidence_maps.npz",
  "post_processing": {
    "morphological_op": "closing",
    "kernel_size": 3,
    "min_component_size": 100
  },
  "timestamp": "2026-01-12T14:30:00Z"
}
```

---

## Key Design Decisions

### 1. **Random Coordinate Sampling (Training) vs Deterministic (Validation)**
- **Why**: Avoids overfitting to specific patches, provides diversity
- **Benefit**: Train on millions of virtual patches from limited WSIs
- **Trade-off**: Non-deterministic (set seed for reproducibility)

### 2. **Multi-Scale Architecture**
- **Why**: Captures features at different resolutions
- **Benefit**: Better CAM quality across scales
- **Trade-off**: Increased memory/computation

### 3. **Weak Supervision (Image-level labels)**
- **Why**: Much cheaper to annotate than pixel-level masks
- **Benefit**: Reduces annotation burden significantly
- **Trade-off**: Requires clever loss design (CAMs, contrastive)

### 4. **MedCLIP Prototypes**
- **Why**: Leverage pre-trained vision-language model
- **Benefit**: Better feature representations than random initialization
- **Trade-off**: Additional preprocessing stage

### 5. **K-Means Subclasses**
- **Why**: Capture intra-class variation
- **Benefit**: More robust prototypes
- **Trade-off**: More parameters to learn

---

## Performance Metrics

### Typical Execution Times
- **Stage 0**: ~1 min (5 WSIs × 1000 coords)
- **Stage 1**: ~5-10 min (10K embeddings)
- **Stage 2**: ~1 min (K-means)
- **Stage 3**: ~30 min per epoch (1000 batches × 10 batch_size on GPU)
- **Stage 4**: ~5 min (full validation set)
- **Stage 5**: ~15 min per new WSI (depends on size and overlap)

### Memory Requirements
- **GPU**: 8GB minimum (4GB with smaller batch size)
- **CPU RAM**: 16GB recommended (for loading WSIs)
- **Storage**: 
  - Raw WSIs: varies (typically 1-10 GB per WSI)
  - Coordinates: ~100 KB per 1000 patches
  - Embeddings: ~200 MB per 10K patches
  - Trained model: ~500 MB (SegFormer backbone)
  - Predictions: ~50 MB per WSI (mask + confidence maps)

### Computational Complexity
- **Stage 0**: O(N × M) - N classes, M coords per WSI (linear)
- **Stage 1**: O(K × 512 × F) - K coords, F forward passes
- **Stage 2**: O(N × K_i²) - K_i clusters per class
- **Stage 3**: O(E × B × L) - E epochs, B batches, L loss computation
- **Stage 4**: O(T × F) - T test WSIs, F forward passes
- **Stage 5**: O(W × P × F) - W new WSIs, P patches, F forward passes

---

## Troubleshooting

### Stage 0: No coordinates extracted
- Check: WSI files exist in `wsi_dir`
- Check: Coordinate files exist in `coordinates_dir`
- Check: Image names in split_csv match files (without extension)
- Check: Label values are valid (0, 1, etc)

### Stage 1: "wsi_name not found"
- Cause: Running with coordinates from old Stage 0
- Fix: Re-run Stage 0 to regenerate coordinates with wsi_name

### Stage 3: Training loss not decreasing
- Check: Learning rate too high/low
- Check: Prototypes are reasonable (verify Stage 2 output)
- Check: Labels are correct (verify Stage 0 label parsing)
- Try: Reduce learning rate, increase epochs

### Stage 3: Out of Memory
- Solution: Reduce batch_size in config
- Solution: Use smaller model backbone (mit_b0 vs mit_b5)
- Solution: Reduce num_classes if multi-class setup

### Stage 4: Validation metrics very low
- Check: GT masks alignment with patch coordinates
- Check: Class imbalance (metrics weighted?)
- Try: Visualize predictions vs GT (save_samples)

### Stage 5: Inference very slow
- Check: batch_size setting (increase if memory allows)
- Check: Post-processing complexity (disable CRF if not needed)
- Try: Use smaller model or lower resolution

---

## Next Steps & Extensions

### 1. **Multi-Scale Training**
- Extract patches at multiple zoom levels (tile_level)
- Train multi-resolution model
- Fuse predictions across scales

### 2. **Test-Time Augmentation (TTA)**
- Apply multiple augmentations per patch
- Average predictions across augmentations
- Improves robustness

### 3. **Cross-Validation**
- Generate k-fold splits from split_csv
- Train k separate models
- Ensemble predictions

### 4. **Active Learning**
- Identify uncertain predictions (low confidence)
- Prioritize for manual annotation
- Retrain with updated labels

### 5. **Fine-Tuning**
- Start from best.pth
- Train with new labels/data
- Use lower learning rate for convergence

### 6. **Model Ensembling**
- Train multiple models with different seeds
- Average predictions for robustness
- Improves metrics by 2-5% typically

---

## References

### Papers
- **SegFormer**: Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
- **MedCLIP**: Boecking et al., "Making the Vision Transformer Effective for Cross-Modal Retrieval in Medical Imaging"
- **PBIP**: Tang et al., "Prototype-based Instance Segmentation for Pathology Image"
- **InfoNCE**: Oord et al., "Representation Learning with Contrastive Predictive Coding"

### Tools
- **OpenSlide**: Efficient WSI reading https://openslide.org/
- **Albumentations**: Image augmentation https://albumentations.ai/
- **PyTorch**: Deep learning framework https://pytorch.org/
- **timm**: PyTorch Image Models https://github.com/huggingface/pytorch-image-models
