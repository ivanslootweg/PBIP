# Next Steps: WSI-Level Classification with PBIP

This guide covers how to transition from patch-level PBIP to WSI-level classification using pseudo-label driven patch selection.

## Part 1: Prepare Pseudo-Labels from MIL Model

### Objective
Generate per-patch attention scores from a pretrained Multiple Instance Learning (MIL) model to identify high-confidence tumor/normal patches.

### Step 1: Generate Attention Scores
1. Use a pretrained MIL model (e.g., CLAM, TransMIL) trained on your WSI dataset
2. Run inference on the training set to get attention weights per patch
3. Save attention scores as `.pt` files named by WSI basename

**Output format (binary classification)**:
```python
import torch

# For each WSI, create a tensor of shape (num_patches,)
attention_scores = torch.randn(1000)  # Example: 1000 patches
torch.save(attention_scores, 'path/to/attention_scores/slide_001.pt')
```

**Output format (multi-class classification)**:
```python
# For multi-class, create shape (num_patches, num_classes)
attention_scores = torch.randn(1000, 3)  # Example: 1000 patches, 3 classes
torch.save(attention_scores, 'path/to/attention_scores/slide_001.pt')
```

### Step 2: Organize Attention Score Directory
```
attention_scores/
├── slide_001.pt
├── slide_002.pt
├── slide_003.pt
└── ...
```

Each `.pt` file should contain PyTorch tensor with per-patch attention weights.

### Step 3: Validate Attention Scores
```python
from utils.pseudo_labels import PseudoLabelLoader

loader = PseudoLabelLoader(
    pseudo_label_dir='path/to/attention_scores',
    binary_mode=True  # or False for multi-class
)

# Check for missing or malformed files
loader.validate_format()

# Get statistics
stats = loader.get_statistics()
print(f"Loaded {stats['num_wsis']} WSIs")
print(f"Mean attention: {stats['mean_attention']:.3f}")
```

---

## Part 2: Select High-Confidence Patches

### Objective
Use attention scores to select patches that represent each class well, filtering out noisy/ambiguous patches.

### Step 1: Configure Selection Strategy

Update your configuration YAML:
```yaml
dataset:
  use_pseudo_labels: true
  pseudo_label_dir: path/to/attention_scores
  pseudo_label_binary_mode: true
  
  # Choose one of: percentile, threshold, entropy, margin
  pseudo_label_selection_strategy: percentile
  
  # For percentile: 0.85 = top 15% of patches
  # For threshold: absolute score cutoff (e.g., 0.5)
  # For entropy: entropy cutoff (lower = more confident)
  # For margin: minimum margin between top-2 classes
  pseudo_label_confidence_threshold: 0.85
  
  # Minimum patches to select per WSI (prevents degenerate cases)
  pseudo_label_min_patches: 5
```

### Step 2: Load Dataset with Pseudo-Labels

```python
from datasets.wsi_dataset import CustomWSIPatchTrainingDatasetWithPseudoLabels

dataset = CustomWSIPatchTrainingDatasetWithPseudoLabels(
    split_csv='splits.csv',
    labels_csv='labels.csv',
    wsi_dir='wsi_images/',
    pseudo_label_dir='attention_scores/',
    pseudo_label_binary_mode=True,
    pseudo_label_selection_strategy='percentile',
    pseudo_label_confidence_threshold=0.85,
    pseudo_label_analyze=True,  # Print statistics
)

# The dataset now filters patches automatically
print(f"Dataset initialized with {len(dataset)} WSIs")
```

### Step 3: Understand Selection Results

The dataset prints per-WSI statistics:
```
PseudoLabelLoader: 
  Loaded 100 WSIs
  Class 0: 50 WSIs, avg selected 120 patches/WSI
  Class 1: 50 WSIs, avg selected 140 patches/WSI
  Total selected: 13000 patches out of 80000 (16.3%)
```

---

## Part 3: Train PBIP with Selected Patches

### Objective
Train the PBIP model using only high-confidence patches, then optionally refine with EM loop.

### Step 1: Initialize Prototypes from Selected Patches

```python
from utils.prototype_guided_attention import PrototypeBank
from sklearn.cluster import KMeans
import torch

# Aggregate high-confidence features per class
high_conf_features_per_class = {
    0: [],  # Class 0 features
    1: [],  # Class 1 features
}

# Collect features from high-confidence patches
for wsi_name in dataset.wsi_names:
    high_conf_indices = dataset.get_high_confidence_patches(wsi_name)
    features = extract_features(wsi_name, indices=high_conf_indices)
    
    wsi_label = dataset.get_label(wsi_name)
    high_conf_features_per_class[wsi_label].append(features)

# Stack and k-means
for class_idx in range(num_classes):
    class_features = np.vstack(high_conf_features_per_class[class_idx])  # (N, feature_dim)
    kmeans = KMeans(n_clusters=10).fit(class_features)
    prototypes = torch.from_numpy(kmeans.cluster_centers_).float()
    
    prototype_bank.add_prototypes(class_idx, prototypes)
```

### Step 2: (Optional) Refine Prototypes with EM Loop

```python
from utils.prototype_refinement import MultiStageRefinementPipeline

refiner = MultiStageRefinementPipeline(
    num_classes=2,
    num_prototypes_per_class=10,
    max_stages=3,
    convergence_threshold=0.01,
)

refined_bank = refiner.run_em_refinement_loop(
    dataset=dataset,
    initial_bank=prototype_bank,
    feature_extractor=feature_backbone,
    device='cuda',
)
```

### Step 3: Train Classifier

```python
from utils.prototype_guided_attention import WSIClassifierWithPrototypes
from torch.utils.data import DataLoader
import torch.nn as nn

classifier = WSIClassifierWithPrototypes(
    prototype_bank=prototype_bank,
    feature_dim=512,
    num_classes=2,
    mlp_hidden_dims=[256, 128],
)
classifier.to(device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
dataloader = DataLoader(dataset, batch_size=32)

for epoch in range(num_epochs):
    for patches, labels in dataloader:
        patches = patches.to(device)
        labels = labels.to(device)
        
        logits = classifier(patches)
        loss = nn.functional.cross_entropy(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Part 4: Visualize and Interpret

### Objective
Understand which patches the model considers important via prototype-based attention.

### Step 1: Get Attention Scores

```python
from utils.prototype_guided_attention import PrototypeGuidedAttention

attention_module = PrototypeGuidedAttention(
    prototype_bank=prototype_bank,
    num_classes=2,
)

# Forward pass with attention output
patch_features = ...  # (N, 512)
logits, attention_scores = attention_module(patch_features, return_attention=True)

# attention_scores shape: (N, num_classes)
# High value = patch similar to class prototype
```

### Step 2: Visualize Patches

```python
import matplotlib.pyplot as plt
import numpy as np

# Find patches most similar to each class
for class_idx in range(num_classes):
    class_attn = attention_scores[:, class_idx]
    top_indices = np.argsort(class_attn)[-10:]  # Top 10 patches
    
    # Load and display these patches
    for idx in top_indices:
        patch_img = load_patch_image(wsi_name, coordinates[idx])
        # Display in grid
```

### Step 3: Analyze Prototype Interpretability

```python
from utils.pseudo_labels import PseudoLabelAnalyzer

analyzer = PseudoLabelAnalyzer()

# Compare attention distribution
print("Pseudo-label statistics:")
for class_idx in range(num_classes):
    stats = analyzer.get_per_class_stats(class_idx)
    print(f"Class {class_idx}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
```

---

## Troubleshooting & Tips

### Issue: Too few patches selected
- **Cause**: Selection threshold too strict or MIL model confidence low
- **Solution**: Lower `pseudo_label_confidence_threshold` or switch strategy to `threshold` with lower cutoff

### Issue: Prototypes collapse (all similar)
- **Cause**: Too many similar patches selected
- **Solution**: Increase `pseudo_label_confidence_threshold` to select more diverse patches

### Issue: Training loss doesn't decrease
- **Cause**: Selected patches too noisy or prototypes not well-initialized
- **Solution**: Enable EM refinement (`max_stages > 1`) or check MIL model quality

### Issue: Memory too high
- **Cause**: Processing too many patches per WSI
- **Solution**: Increase `pseudo_label_confidence_threshold` to reduce number of selected patches

---

## Key Configuration Parameters

```yaml
dataset:
  # Pseudo-label settings
  use_pseudo_labels: true
  pseudo_label_dir: /path/to/attention_scores
  pseudo_label_binary_mode: true
  pseudo_label_selection_strategy: percentile
  pseudo_label_confidence_threshold: 0.85
  pseudo_label_min_patches: 5
  pseudo_label_analyze: true

model:
  # Prototype settings
  num_prototypes_per_class: 10
  head_hidden_dims: [256, 128]
  
  # Feature settings
  encoder_type: medclip
  encoder_checkpoint: /path/to/checkpoint

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 1e-4
```

---

## Original PBIP Training (Reference)

For comparison, the original Stage 1 training uses CAMs as pseudo-labels for Deeplab-v2. See the [PBIP paper (arXiv:2503.12068)](https://arxiv.org/abs/2503.12068) for details on that approach.
