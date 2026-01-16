# WSI-Level PBIP Adaptation Guide

## Overview

This guide explains the adaptations made to PBIP (Prototype-Based Image Prompting) to support **Whole Slide Image (WSI) classification** using **pseudo-label driven patch selection**.

The key innovation is the **"Selection-then-Prompting" strategy**: instead of using all patches to build prototypes (which leads to "poison" from non-representative patches), we only use high-confidence patches from attention scores to construct a clean, representative prototype bank.

---

## Problem Statement

**Original PBIP**: Designed for weakly supervised **semantic segmentation** at the ROI/patch level
- Assumes image-level labels directly represent patch-level content
- Works well when ~80% of a labeled "Tumor" image actually contains tumor tissue

**WSI Reality**: Whole slide images have one label for gigapixel images
- A "Tumor" WSI might be 90% healthy stroma, inflammation, and necrosis
- Only 10% contains actual tumor tissue
- Using all patches would create corrupted prototypes

**Solution**: Use attention scores (pseudo-labels from a pretrained MIL model) to identify representative patches, then build prototypes only from high-confidence patches.

---

## Architecture Components

### 1. **Pseudo-Label Loading** (`utils/pseudo_labels.py`)

Loads attention scores from `.pt` files for each WSI coordinate.

**Supported Modes**:
- **Binary**: Single attention channel (low = class 0, high = class 1)
- **Multi-class**: Multiple channels, one per class

**Classes**:
- `PseudoLabelLoader`: Loads and validates pseudo-label files
- `PatchSelector`: Selects high-confidence patches using multiple strategies
  - `percentile`: Top-k% patches by confidence
  - `threshold`: Patches above absolute score threshold
  - `entropy`: Patches with lowest entropy
  - `margin`: Patches with largest margin between top-2 classes
- `PseudoLabelAnalyzer`: Analyzes quality across dataset

**Example Usage**:
```python
from utils.pseudo_labels import PseudoLabelLoader, PatchSelector

# Load pseudo-labels
loader = PseudoLabelLoader(
    pseudo_label_dir="/path/to/attention_scores",
    binary_mode=True,
    num_classes=2
)

# Select high-confidence patches
selector = PatchSelector(
    num_classes=2,
    selection_strategy='percentile'
)

scores = loader.load_wsi_scores('slide_001')
selected_mask = selector.select_patches(scores, confidence_threshold=0.85)
selected_indices = np.where(selected_mask)[0]
```

### 2. **Pseudo-Label Enhanced Dataset** (`datasets/wsi_dataset.py`)

Extended training dataset that integrates pseudo-label filtering.

**Class**: `CustomWSIPatchTrainingDatasetWithPseudoLabels`
- Extends `CustomWSIPatchTrainingDataset`
- Automatically loads pseudo-labels during initialization
- Pre-computes high-confidence patch indices
- Provides access to pseudo-label scores for each WSI

**Features**:
- Automatic initialization and validation of pseudo-labels
- Quality analysis and statistics reporting
- Handling of insufficient patches with warnings
- Per-class patch selection

**Example Usage**:
```python
from datasets.wsi_dataset import CustomWSIPatchTrainingDatasetWithPseudoLabels

dataset = CustomWSIPatchTrainingDatasetWithPseudoLabels(
    wsi_dir="/path/to/wsis",
    coordinates_dir="/path/to/coordinates",
    split_csv="/path/to/splits.csv",
    use_pseudo_labels=True,
    pseudo_label_dir="/path/to/attention_scores",
    pseudo_label_binary_mode=True,
    pseudo_label_selection_strategy='percentile',
    pseudo_label_confidence_threshold=0.85,
    pseudo_label_min_patches=5,
)

# Access high-confidence patches
high_conf_indices = dataset.get_high_confidence_patches('slide_001')
pseudo_scores = dataset.get_wsi_pseudo_scores('slide_001')
```

### 3. **Prototype-Guided Attention** (`utils/prototype_guided_attention.py`)

Replaces standard MIL pooling with similarity-based attention using prototypes.

**Classes**:
- `PrototypeBank`: Stores and manages class prototypes
- `PrototypeGuidedAttention`: Computes patch attention using prototype similarity
- `WSIClassifierWithPrototypes`: End-to-end WSI classifier with prototype guidance

**Key Idea**: 
Instead of learning attention weights from scratch, a patch's importance is determined by its similarity to class prototypes. This provides interpretability: "This patch is important because it matches the learned prototype of class X."

**Attention Types**:
- `cosine_sim`: Use max similarity to class prototypes as weight
- `softmax`: Softmax over prototype similarities
- `mean`: Average similarity across prototypes

**Example Usage**:
```python
from utils.prototype_guided_attention import (
    PrototypeBank, PrototypeGuidedAttention, WSIClassifierWithPrototypes
)

# Create components
prototype_bank = PrototypeBank(
    num_classes=2,
    feature_dim=512,  # MedCLIP dimension
    max_prototypes_per_class=100
)

classifier = WSIClassifierWithPrototypes(
    feature_dim=512,
    num_classes=2,
    prototype_bank=prototype_bank,
    attention_type='cosine_sim'
)

# Initialize prototypes from high-confidence patches
for class_id, class_features in initial_prototypes.items():
    prototype_bank.add_prototypes(class_id, class_features)

# Classify WSI
patch_features = torch.randn(100, 512)  # 100 patches, 512-dim features
logits, info = classifier(patch_features)

# Inspect attention weights
attention_weights = info['attention_weights']  # (100,) for single class
similarities = info['similarities']  # Patch-to-prototype similarities
```

### 4. **Prototype Bank Refinement** (`utils/prototype_refinement.py`)

Implements iterative EM-style refinement of prototypes.

**Classes**:
- `PrototypeBankRefinement`: Single-stage refinement
- `MultiStageRefinementPipeline`: Multi-stage EM loop

**Algorithm**:
```
Stage 0 (Initialization):
  - Cluster initial high-confidence patches for each class
  - Initialize prototype bank with cluster centers

Repeat for max_iterations:
  E-step (Expectation):
    - Classify all WSIs using current prototypes
    - Compute classification accuracy
  
  M-step (Maximization):
    - Collect patches from correctly classified WSIs
    - Re-cluster to update prototypes
    - Check convergence
```

**Example Usage**:
```python
from utils.prototype_refinement import MultiStageRefinementPipeline

pipeline = MultiStageRefinementPipeline(
    prototype_bank=prototype_bank,
    classifier=classifier,
    num_classes=2,
    max_refinement_iterations=5,
    convergence_threshold=0.01
)

# Initialize prototypes
initial_patches = {
    0: torch.randn(500, 512),  # Class 0 patches
    1: torch.randn(500, 512),  # Class 1 patches
}
pipeline.initialize_prototypes_from_patches(initial_patches)

# Run EM refinement
em_results = pipeline.run_em_refinement_loop(
    wsi_dataloader=train_loader,
    wsi_labels=wsi_labels,
    device='cuda'
)
```

---

## Configuration

### YAML Configuration (`work_dirs/custom_wsi_template.yaml`)

```yaml
dataset:
  # ... existing settings ...
  
  # Pseudo-label settings
  use_pseudo_labels: true
  pseudo_label_dir: /path/to/attention_scores
  
  # Binary vs multi-class mode
  pseudo_label_binary_mode: true  # true for binary, false for multi-class
  
  # Patch selection strategy
  # Options: 'percentile', 'threshold', 'entropy', 'margin'
  pseudo_label_selection_strategy: percentile
  
  # Confidence threshold for patch selection
  # For 'percentile': 0.85 means top 15% of patches
  pseudo_label_confidence_threshold: 0.85
  
  # Minimum patches per WSI to be used for prototype construction
  pseudo_label_min_patches: 5
  
  # Analyze pseudo-label quality during initialization
  pseudo_label_analyze: true
```

---

## Expected Data Structure

### Input Files

```
project_dir/
├── attention_scores/  # Pseudo-label scores from MIL model
│   ├── slide_001.pt   # Shape: (1024,) for binary or (1024, 2) for 2 classes
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
└── splits.csv         # With columns: train, val, test
```

### .pt File Format

The pseudo-label `.pt` file contains attention scores for each patch coordinate in a WSI.

**Binary mode** (single channel):
```python
# Shape: (num_patches,)
# Values: [0, 1] or [-1, 1] or [0, ∞]
# Low values → class 0, High values → class 1
scores = torch.tensor([0.1, 0.8, 0.3, 0.9, ...])
torch.save(scores, 'slide_001.pt')
```

**Multi-class mode** (multiple channels):
```python
# Shape: (num_patches, num_classes)
# Each row represents attention scores for that patch across classes
scores = torch.tensor([
    [0.1, 0.8, 0.1],    # Patch 0: class 1 most likely
    [0.8, 0.1, 0.1],    # Patch 1: class 0 most likely
    [0.3, 0.3, 0.4],    # Patch 2: class 2 most likely
    ...
])
torch.save(scores, 'slide_001.pt')
```

---

## Workflow

### Step 1: Prepare Pseudo-Labels

Train a standard MIL model (e.g., ABMIL, TransMIL) to get attention scores:

```python
# Pseudo-code for generating attention scores
for wsi_name, wsi_data in dataset:
    patches = extract_patches(wsi_data)  # (N_patches, 512)
    
    # Forward through MIL model
    attention_scores = mil_model.get_attention_scores(patches)  # (N_patches,) or (N_patches, num_classes)
    
    # Save scores
    torch.save(attention_scores, f'attention_scores/{wsi_name}.pt')
```

### Step 2: Analyze Pseudo-Labels

Check quality and distribution:

```python
from utils.pseudo_labels import PseudoLabelLoader, PseudoLabelAnalyzer

loader = PseudoLabelLoader('attention_scores', binary_mode=True, num_classes=2)
analyzer = PseudoLabelAnalyzer(loader)
stats = analyzer.analyze_all_wsis()

print(f"Global mean: {stats['global_stats']['mean']:.4f}")
print(f"Global std: {stats['global_stats']['std']:.4f}")
```

### Step 3: Create Dataset with Pseudo-Labels

```python
dataset = CustomWSIPatchTrainingDatasetWithPseudoLabels(
    wsi_dir="...",
    coordinates_dir="...",
    split_csv="...",
    use_pseudo_labels=True,
    pseudo_label_dir="attention_scores",
    pseudo_label_binary_mode=True,
    pseudo_label_selection_strategy='percentile',
    pseudo_label_confidence_threshold=0.85,
)

# Dataset initialization automatically:
# - Loads pseudo-labels
# - Selects high-confidence patches
# - Reports statistics
```

### Step 4: Initialize Prototypes

Extract high-confidence patches and initialize prototype bank:

```python
from utils.pseudo_labels import PatchSelector

selector = PatchSelector(num_classes=2, selection_strategy='percentile')

# Collect high-confidence patches for each class
class_patches = {0: [], 1: []}
for wsi_name in dataset.filenames:
    pseudo_scores = dataset.get_wsi_pseudo_scores(wsi_name)
    high_conf_indices = dataset.get_high_confidence_patches(wsi_name)
    
    wsi_class = wsi_labels[wsi_name]
    class_patches[wsi_class].extend(high_conf_indices)

# Initialize prototypes via clustering
pipeline.initialize_prototypes_from_patches(class_patches)
```

### Step 5: Train WSI Classifier

Train the WSI classifier using prototype-guided attention:

```python
classifier = WSIClassifierWithPrototypes(
    feature_dim=512,
    num_classes=2,
    prototype_bank=prototype_bank
)

# Training loop
for epoch in range(num_epochs):
    for patch_features, wsi_labels in train_loader:
        logits, _ = classifier(patch_features)
        loss = criterion(logits, wsi_labels)
        loss.backward()
        optimizer.step()
```

### Step 6: Iterative Refinement (Optional)

Refine prototypes using correctly classified WSIs:

```python
pipeline = MultiStageRefinementPipeline(
    prototype_bank=prototype_bank,
    classifier=classifier,
    num_classes=2,
    max_refinement_iterations=5
)

# EM loop
em_results = pipeline.run_em_refinement_loop(
    wsi_dataloader=validation_loader,
    wsi_labels=validation_labels
)
```

---

## Selection Strategies Explained

### 1. Percentile
**Best for**: Balanced datasets, when you know approximate class balance
```
Select top k% of patches by attention score
If threshold=0.85, keep patches in [85th percentile, 100th percentile]
```
Example: For binary classification with uniform class distribution, use 85th percentile

### 2. Threshold
**Best for**: Known decision boundary, calibrated scores
```
Select all patches with score >= threshold
If threshold=0.5, keep all patches with attention >= 0.5
```
Useful when attention scores are in [0, 1] and well-calibrated

### 3. Entropy
**Best for**: Confident predictions only
```
Select patches where prediction entropy < entropy_threshold
Low entropy = high confidence = should be used for prototypes
```
Ensures only most certain patches contribute to prototypes

### 4. Margin
**Best for**: Multi-class classification, discriminative samples
```
Select patches where margin between top-2 classes > threshold
Margin = p(top_class) - p(second_class)
```
Only highly discriminative samples used for prototypes

---

## Interpretation & Debugging

### Understanding Attention Weights

After classification, inspect what the model attended to:

```python
logits, info = classifier(patch_features)
attention_weights = info['attention_weights']  # Which patches matter?
similarities = info['similarities']            # How similar to prototypes?

# Top 5 attended patches
top_indices = torch.argsort(attention_weights, descending=True)[:5]
print(f"Most attended patches: {top_indices}")

# Save attention visualization
visualize_attention_on_wsi(attention_weights, coordinates, output_path)
```

### Prototype Quality

Monitor prototype bank evolution:

```python
for class_id in range(num_classes):
    prototypes = prototype_bank.get_class_prototypes(class_id)
    print(f"Class {class_id}: {len(prototypes)} prototypes")
```

### Pseudo-Label Quality

Analyze distribution of attention scores:

```python
analyzer = PseudoLabelAnalyzer(loader)
stats = analyzer.analyze_all_wsis()

for class_id in range(num_classes):
    class_mean = stats['global_stats'][f'class_{class_id}']['mean']
    class_std = stats['global_stats'][f'class_{class_id}']['std']
    print(f"Class {class_id}: μ={class_mean:.4f}, σ={class_std:.4f}")
```

---

## Advantages Over Standard PBIP

| Aspect | Original PBIP | WSI Adaptation |
|--------|---------------|----------------|
| **Input** | Single ROI image (224×224) | Bag of patches from WSI |
| **Prototype Source** | All patches in image | Only high-confidence patches |
| **Prototype Purity** | Assumes ~100% class purity | Filters to >85% confidence |
| **Aggregation** | CAM-based segmentation | Similarity-based attention weighting |
| **Refinement** | Single stage | Multi-stage EM loop |
| **Interpretability** | Prototype matching + CAM | Prototype matching + attention |

---

## Common Issues & Solutions

### Issue 1: Too Few High-Confidence Patches
```
Warning: 500 WSIs have insufficient high-confidence patches
```
**Solution**: Lower `pseudo_label_confidence_threshold`
```yaml
# From 0.85 to 0.75
pseudo_label_confidence_threshold: 0.75
```

### Issue 2: Pseudo-Label File Format Mismatch
```
ValueError: Expected 2 classes but got 1 in multi-class mode
```
**Solution**: Check your `.pt` file format matches the configuration
```python
# If binary mode expected (single channel)
scores = torch.randn(1000)  # (num_patches,)
torch.save(scores, 'slide.pt')

# If multi-class (multiple channels)
scores = torch.randn(1000, 3)  # (num_patches, num_classes)
torch.save(scores, 'slide.pt')
```

### Issue 3: Poor Classification Performance
**Causes**:
1. Attention scores not well-calibrated (try different threshold)
2. Too aggressive filtering (increase min_patches threshold)
3. Prototypes not representative (increase n_clusters_per_class)

**Solutions**:
```yaml
# Try lower confidence threshold
pseudo_label_confidence_threshold: 0.75

# Allow more patches
pseudo_label_min_patches: 3

# Use higher number of clusters
n_clusters_per_class: 10
```

---

## References

- **Original PBIP**: Tang et al., CVPR 2025
- **WSI Adaptation Strategy**: Bridges PBIP with MIL-based WSI classification
- **Pseudo-Label Concept**: Inspired by weakly supervised learning literature

---

## Citation

If you use this WSI adaptation in your research, please cite:

```bibtex
@inproceedings{tang2025pbip,
  title={Prototype-Based Image Prompting for Weakly Supervised Histopathological Image Segmentation},
  author={Tang, ...},
  booktitle={CVPR},
  year={2025}
}
```

And acknowledge the WSI adaptation modifications in your methods section.

