# WSI-Level Adaptation for PBIP

## Quick Summary

This adaptation enables PBIP to work with **Whole Slide Image (WSI) classification** by integrating it with **Multiple Instance Learning (MIL)** using **pseudo-label driven patch selection**.

### Key Innovation
Instead of using all patches to build prototypes (which dilutes them with non-representative patches), we:
1. **Load pseudo-labels**: Attention scores from a pretrained MIL model for each patch
2. **Select high-confidence patches**: Only patches scoring in top percentile (e.g., 85th percentile)
3. **Build clean prototypes**: Initialize prototype bank from only these representative patches
4. **Classify WSIs**: Use similarity-based attention weighting instead of standard pooling
5. **Refine iteratively**: Update prototypes using correctly classified WSIs (optional)

---

## What Changed?

### New Modules

1. **`utils/pseudo_labels.py`** - Pseudo-label management
   - Load `.pt` files with attention scores
   - Support binary and multi-class modes
   - Multiple selection strategies (percentile, threshold, entropy, margin)

2. **`utils/prototype_guided_attention.py`** - MIL-based classifier
   - Prototype Bank: stores class prototypes
   - PrototypeGuidedAttention: computes attention via similarity
   - WSIClassifierWithPrototypes: end-to-end classifier

3. **`utils/prototype_refinement.py`** - Iterative refinement
   - PrototypeBankRefinement: single-stage updates
   - MultiStageRefinementPipeline: EM loop for multi-stage refinement

4. **Enhanced Dataset** - `datasets/wsi_dataset.py`
   - New class: `CustomWSIPatchTrainingDatasetWithPseudoLabels`
   - Automatically loads and filters high-confidence patches

### Updated Files

- **`work_dirs/custom_wsi_template.yaml`** - Added pseudo-label configuration options
- **`docs/WSI_ADAPTATION_GUIDE.md`** - Comprehensive guide with examples
- **`examples/wsi_training_with_prototypes.py`** - End-to-end training example

---

## Quick Start

### 1. Configure Your Setup

Edit `work_dirs/custom_wsi_template.yaml`:

```yaml
dataset:
  use_pseudo_labels: true
  pseudo_label_dir: /path/to/attention_scores  # Directory with .pt files
  pseudo_label_binary_mode: true               # or false for multi-class
  pseudo_label_selection_strategy: percentile  # 'percentile', 'threshold', etc.
  pseudo_label_confidence_threshold: 0.85      # Top 15% of patches
  pseudo_label_min_patches: 5                  # Minimum per WSI
  pseudo_label_analyze: true                   # Print statistics
```

### 2. Prepare Attention Scores

Generate `.pt` files using a pretrained MIL model:

```python
import torch

# For each WSI, get attention scores from your MIL model
wsi_name = 'slide_001'
attention_scores = mil_model.forward(patches)  # (N_patches,) or (N_patches, num_classes)

# Save as .pt file
torch.save(attention_scores, f'attention_scores/{wsi_name}.pt')
```

### 3. Train With Pseudo-Labels

```python
from datasets.wsi_dataset import CustomWSIPatchTrainingDatasetWithPseudoLabels

# Dataset automatically loads pseudo-labels and filters patches
dataset = CustomWSIPatchTrainingDatasetWithPseudoLabels(
    wsi_dir="...",
    coordinates_dir="...",
    split_csv="...",
    use_pseudo_labels=True,
    pseudo_label_dir="attention_scores",
    # ... other args
)

# High-confidence patches are pre-selected automatically
high_conf = dataset.get_high_confidence_patches('slide_001')
```

### 4. Initialize Prototype Bank

```python
from utils.prototype_guided_attention import create_wsi_classifier_with_prototypes
from utils.prototype_refinement import MultiStageRefinementPipeline

# Create classifier with prototype bank
classifier, prototype_bank = create_wsi_classifier_with_prototypes(
    feature_dim=512,
    num_classes=2,
)

# Initialize from high-confidence patches
pipeline = MultiStageRefinementPipeline(prototype_bank, classifier, num_classes=2)
pipeline.initialize_prototypes_from_patches(initial_patches_by_class)
```

---

## Data Format

### Attention Score Files (.pt)

**Binary mode** (single attention channel):
```python
# One score per patch: low → class 0, high → class 1
scores = torch.randn(1024)  # 1024 patches
torch.save(scores, 'slide_001.pt')
```

**Multi-class mode** (one score per class):
```python
# Multiple scores per patch
scores = torch.randn(1024, 3)  # 1024 patches, 3 classes
torch.save(scores, 'slide_001.pt')
```

### Directory Structure

```
project/
├── attention_scores/           # Pseudo-label directory
│   ├── slide_001.pt
│   ├── slide_002.pt
│   └── ...
├── wsi_images/
├── coordinates/
└── splits.csv
```

---

## Selection Strategies

| Strategy | Use Case | Formula |
|----------|----------|---------|
| `percentile` | Balanced data, know approximate quality | Keep top k% by score |
| `threshold` | Calibrated scores, clear boundary | Keep score ≥ threshold |
| `entropy` | Confident predictions only | Keep entropy < threshold |
| `margin` | Multi-class, discriminative samples | Keep margin > threshold |

---

## Configuration Options

### Pseudo-Label Settings

```yaml
dataset:
  # Enable pseudo-label driven patch selection
  use_pseudo_labels: true
  
  # Path to pseudo-label .pt files
  pseudo_label_dir: /data/attention_scores
  
  # Binary (single channel) vs multi-class (multiple channels)
  pseudo_label_binary_mode: true
  
  # Patch selection: 'percentile', 'threshold', 'entropy', 'margin'
  pseudo_label_selection_strategy: percentile
  
  # Confidence threshold (0-1)
  # For percentile: 0.85 = top 15%
  pseudo_label_confidence_threshold: 0.85
  
  # Minimum selected patches per WSI
  pseudo_label_min_patches: 5
  
  # Print pseudo-label statistics
  pseudo_label_analyze: true
```

### Refinement Settings

```yaml
train:
  # Number of EM iterations
  refinement_iterations: 5
  
  # Number of prototypes per class
  n_clusters_per_class: 5
```

---

## Typical Workflow

```
1. Train MIL model on WSIs
   └─> Get attention scores for each patch coordinate

2. Save attention scores as .pt files
   └─> One file per WSI: (num_patches,) or (num_patches, num_classes)

3. Configure dataset with pseudo-label settings
   └─> Set pseudo_label_dir, selection_strategy, threshold

4. Create dataset with pseudo-labels
   └─> CustomWSIPatchTrainingDatasetWithPseudoLabels
   └─> Automatically loads scores and selects high-confidence patches

5. Extract features from high-confidence patches
   └─> Use MedCLIP or other encoder on selected patches

6. Initialize prototypes from selected features
   └─> Cluster and add to PrototypeBank

7. Train WSI classifier with prototype-guided attention
   └─> Uses similarity to prototypes for attention weighting

8. Optional: Refine prototypes iteratively
   └─> Collect patches from correctly classified WSIs
   └─> Re-cluster to update prototypes
```

---

## Debugging

### Check Pseudo-Label Quality

```python
from utils.pseudo_labels import PseudoLabelAnalyzer, PseudoLabelLoader

loader = PseudoLabelLoader('attention_scores', binary_mode=True)
analyzer = PseudoLabelAnalyzer(loader)
stats = analyzer.analyze_all_wsis()

print(stats['global_stats'])
# Output: {'mean': 0.52, 'std': 0.30, 'min': -0.1, 'max': 1.0}
```

### Visualize Attention Weights

```python
logits, info = classifier(patch_features)
attention_weights = info['attention_weights']

# Top-10 attended patches
top_indices = torch.argsort(attention_weights, descending=True)[:10]
print(f"Most important patches: {top_indices}")
```

### Monitor Prototype Evolution

```python
for class_id in range(num_classes):
    prototypes = prototype_bank.get_class_prototypes(class_id)
    print(f"Class {class_id}: {len(prototypes)} prototypes")
```

---

## Comparison: Standard PBIP vs WSI Adaptation

| Feature | Standard PBIP | WSI Adaptation |
|---------|---------------|----------------|
| **Input** | Single ROI (~500×500) | Bag of patches (~100-1000) |
| **Patch Labeling** | Image label for all patches | Attention scores per patch |
| **Prototype Selection** | All patches in image | Top k% by attention score |
| **Prototype Purity** | Assumes ~100% | Filtered to >85% confidence |
| **Aggregation** | CAM segmentation | Similarity-weighted pooling |
| **Refinement** | Two-stage (CAM, Seg) | Multi-stage EM loop |
| **Use Case** | Tissue segmentation | WSI classification |

---

## References & Further Reading

### Key Papers
- **PBIP (Original)**: Tang et al., CVPR 2025 - Prototype-Based Image Prompting
- **MIL for WSI**: Ilse et al., 2018 - Attention-based Deep Multiple Instance Learning
- **Weakly Supervised**: Oquab et al., 2024 - DINOv2 self-supervised learning

### Related Documents
- [WSI_ADAPTATION_GUIDE.md](WSI_ADAPTATION_GUIDE.md) - Detailed technical guide
- [examples/wsi_training_with_prototypes.py](../examples/wsi_training_with_prototypes.py) - Training example

---

## Support for Multi-Class Classification

### Binary vs Multi-Class Modes

**Binary (default)**:
- One attention score per patch
- Interprets as: low → background, high → foreground
- Configuration: `pseudo_label_binary_mode: true`

**Multi-class**:
- Multiple attention scores per patch (one per class)
- Each score represents confidence for that class
- Configuration: `pseudo_label_binary_mode: false`

### Handling Multi-Class Pseudo-Labels

```python
# Shape: (num_patches, num_classes)
scores = torch.randn(1024, 5)  # 1024 patches, 5 classes
torch.save(scores, 'slide_001.pt')

# Configuration
pseudo_label_binary_mode: false
num_classes: 5
```

### Per-Class Patch Selection

```python
from utils.pseudo_labels import PatchSelector

selector = PatchSelector(num_classes=5, selection_strategy='percentile')

# Select high-confidence patches for each class
per_class_selection = selector.select_patches_per_class(
    scores,
    confidence_threshold=0.8,
    return_indices=True
)

# per_class_selection: Dict[int, array] mapping class_id -> patch_indices
```

---

## Troubleshooting

### Problem: "Too many insufficient patches"
**Cause**: Threshold too high, attention scores not well-calibrated
**Solution**: 
```yaml
# Lower threshold
pseudo_label_confidence_threshold: 0.75

# Or change strategy
pseudo_label_selection_strategy: threshold
```

### Problem: "Feature dimension mismatch"
**Cause**: Pseudo-label tensor shape doesn't match configuration
**Solution**: Check `.pt` file format
```python
scores = torch.load('slide.pt')
print(scores.shape)  # Should be (num_patches,) or (num_patches, num_classes)
```

### Problem: "Prototypes not improving classification"
**Cause**: Patches not representative enough, clustering params suboptimal
**Solution**:
```yaml
# Use more/fewer prototypes
n_clusters_per_class: 10  # or 3

# Lower confidence threshold to include more patches
pseudo_label_confidence_threshold: 0.7
```

---

## Citation

If using this WSI adaptation, please cite:

```bibtex
@inproceedings{tang2025pbip,
  title={Prototype-Based Image Prompting for Weakly Supervised Histopathological Image Segmentation},
  author={Tang, ...},
  booktitle={CVPR},
  year={2025}
}
```

And mention the WSI-level modifications in your methods.

