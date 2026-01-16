# Data Flow

## Original PBIP Pipeline

End-to-end path from raw assets to predictions for patch-level segmentation:

1) **Inputs**
- Whole-slide images (WSI) and coordinate files (.npy/.txt) per slide
- Split CSV with train/val/test filenames
- Labels CSV with image-level labels (multi-class or binary)
- Optional GT masks for val/test (pixel labels)

2) **Patch Coordinate Selection (train)**
- Script: `features/extract_patches.py`
- For each class, sample up to `num_wsis_per_class` WSIs
- From each WSI, randomly sample up to `num_per_wsi` coordinates (seeded)
- Save slide2vec-formatted coordinate npy per class

3) **Prototype Feature Extraction (MedCLIP)**
- Script: `features/excract_medclip_proces.py`
- Extracts patch embeddings using MedCLIP encoder
- Saves features for prototype building

4) **Label Feature Clustering (k-means)**
- Script: `features/k_mean_cos_per_class.py`
- Clusters features per class using k_list
- Creates label prototypes

5) **Training (Stage 1)**
- Script: `train_stage_1.py`
- Loads patch encoder (medclip/virchow2/dinov3)
- Trains classifier with CAM generation
- Saves checkpoints and predictions

6) **Outputs**
- Checkpoints, predictions, logs, cached features

---

## WSI-Level Adaptation with Pseudo-Labels

### Selection-then-Prompting Strategy

This adaptation enables PBIP to handle WSI classification where only one label exists for an entire slide, but many patches are non-representative tissue (stroma, necrosis, etc.).

**Core Innovation**: Use pseudo-labels (attention scores from MIL model) to identify high-confidence patches, then build prototypes from only these representative patches.

### Processing Pipeline

```
WSI Input (Image + Coordinates + Attention Scores)
    ↓
Pseudo-Label Loading & Validation
    • Load .pt files with attention scores
    • Support binary (single channel) and multi-class modes
    • Validate format and shape
    ↓
High-Confidence Patch Selection
    • Strategy: percentile (top 15%), threshold, entropy, or margin
    • Filter to patches above confidence threshold
    • Min/max patch constraints per WSI
    ↓
Dataset Layer with Pseudo-Label Filtering
    • CustomWSIPatchTrainingDatasetWithPseudoLabels
    • Pre-computes selected patch indices
    • Reports quality statistics
    ↓
Feature Extraction
    • Encode selected patches only (MedCLIP/Virchow/DINOv3)
    • 512-1280 dimensional feature vectors
    ↓
Prototype Bank Initialization
    • Cluster high-confidence features via k-means
    • Per-class prototypes
    • L2-normalized for cosine similarity
    ↓
WSI Classification
    • Compute patch-prototype similarity
    • Attention-weighted aggregation
    • MLP classification head
    ↓
(Optional) Multi-Stage Refinement
    • E-step: Classify all WSIs
    • M-step: Update prototypes from correct predictions
    • Iterate until convergence
    ↓
Final Predictions
```

### Key Configuration Parameters

```yaml
dataset:
  use_pseudo_labels: true
  pseudo_label_dir: /path/to/attention_scores  # .pt files
  pseudo_label_binary_mode: true  # true for single channel, false for multi-class
  pseudo_label_selection_strategy: percentile  # Options: percentile, threshold, entropy, margin
  pseudo_label_confidence_threshold: 0.85  # Top 15% if percentile, or absolute threshold
  pseudo_label_min_patches: 5  # Minimum selected patches per WSI
  pseudo_label_analyze: true  # Print statistics
```

### Supported Selection Strategies

| Strategy | Use Case | Threshold Meaning |
|----------|----------|-------------------|
| **percentile** | Balanced data | Top k% (0.85 = top 15%) |
| **threshold** | Calibrated scores | Absolute value (0.5 = score ≥ 0.5) |
| **entropy** | Confident predictions | Entropy cutoff (lower = more confident) |
| **margin** | Multi-class, discriminative | Margin between top-2 classes |

### Data Format

Each WSI requires:
1. **Image**: `slide_001.tif` (WSI file)
2. **Coordinates**: `slide_001.npy` (patch locations)
3. **Attention Scores**: `slide_001.pt` (pseudo-labels from MIL)

Attention scores format:
- **Binary**: `torch.tensor([0.1, 0.8, 0.3, ...])` shape `(num_patches,)`
- **Multi-class**: `torch.tensor([[0.1, 0.8, 0.1], ...])` shape `(num_patches, num_classes)`

---

## Comparison: Original vs Adapted PBIP

| Aspect | Original PBIP | WSI Adaptation |
|--------|---------------|----------------|
| **Input** | Single ROI patch (224×224) | Bag of patches from WSI (100-1000s) |
| **Patch Labeling** | All patches labeled by image label | Only patches with high attention scores |
| **Prototype Quality** | Assumes ~80% class purity | Filters to >85% confidence |
| **Aggregation** | CAM-based segmentation | Similarity-weighted attention pooling |
| **Refinement** | Two-stage (CAM generation + Segmentation) | Multi-stage EM loop |
| **Use Case** | Patch-level tissue segmentation | WSI-level tissue classification |

---

## New Modules for WSI Adaptation

See [WSI-level Changes](#wsi-level-adaptation-guide) for implementation details.

### `utils/pseudo_labels.py`
- Load and validate attention scores
- Select high-confidence patches
- Analyze pseudo-label quality
- Per-class selection

### `utils/prototype_guided_attention.py`
- Prototype Bank management
- Similarity-based attention computation
- WSI classifier with prototype guidance
- Interpretable patch importance

### `utils/prototype_refinement.py`
- Single-stage prototype updates
- Multi-stage EM refinement loop
- Convergence detection
- History tracking

### Enhanced Dataset
- `CustomWSIPatchTrainingDatasetWithPseudoLabels` in `datasets/wsi_dataset.py`
- Automatic pseudo-label loading and filtering
