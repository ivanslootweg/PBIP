# Training Components

High-level overview of the trainable components and their interaction.

## Original PBIP Training Architecture

### 1. Feature Backbone (frozen)
- **Type**: Pre-trained encoder (MedCLIP, Virchow2, DINOv3, etc.)
- **Role**: Extract patch-level feature embeddings
- **Configuration**: `model.encoder_type`, `model.encoder_checkpoint`
- **Input**: Patch images (256×256)
- **Output**: Feature vectors (512-dim for MedCLIP, 1280-dim for Virchow2)
- **During training**: Features are pre-extracted and cached; backbone is not trained

### 2. Prototype Bank
- **Type**: Learnable per-class prototype vectors
- **Initialization**: K-means clustering on training features
- **Role**: Class-specific prototypes for similarity-based matching
- **Configuration**: `model.num_prototypes_per_class`
- **Dimensions**: (num_classes × prototypes_per_class) × feature_dim
- **During training**: Optimized via gradient descent
- **Storage**: Saved in checkpoint with optimizer state

### 3. Patch Classification Head
- **Type**: MLP
- **Input**: Patch feature + patch-prototype similarities
- **Output**: Per-class logits (num_classes)
- **Hidden layers**: Configurable in `model.head_hidden_dims`
- **During training**: Learns via supervised loss
- **Typical**: 2-3 layer MLP with ReLU, dropout

### 4. Segmentation Head (for segmentation tasks)
- **Type**: Upsampling decoder
- **Input**: Patch embeddings (from backbone)
- **Output**: Per-pixel predictions
- **Architecture**: Transposed convolutions + skip connections
- **During training**: Learns spatial relationships within patches

## WSI-Level Adaptation: New Training Components

### Enhanced Architecture for WSI Classification

When using pseudo-label driven patch selection, the training pipeline gains three new interpretable components:

```
WSI-Level Workflow:
┌─────────────────┐
│ MIL Pretraining │  ← Pretrained attention weights per patch
└────────┬────────┘
         │ .pt files (pseudo-labels)
         ▼
┌─────────────────────────────┐
│ Patch Selection via Attention │  ← PatchSelector (4 strategies)
└────────┬────────────────────┘
         │ Selected high-confidence patches
         ▼
┌──────────────────────────────┐
│ Prototype Bank Initialization │  ← K-means on selected patches
└────────┬─────────────────────┘
         │ Per-class prototypes
         ▼
┌──────────────────────────┐
│ PBIP Training (original) │  ← Prototype-based classification
└────────┬─────────────────┘
         │ Predictions
         ▼
┌──────────────────────────┐
│ Iterative Refinement     │  ← EM loop to filter noisy labels
└──────────────────────────┘
```

### New Component 1: PseudoLabelLoader (`utils/pseudo_labels.py`)

**Purpose**: Load and validate attention scores from pre-trained MIL model

**Key Methods**:
- `load_wsi_scores(wsi_name)`: Load `.pt` file for single WSI
- `load_all_scores()`: Load all `.pt` files for dataset
- `validate_format()`: Check binary/multi-class consistency
- `get_statistics()`: Compute per-class mean, std, min, max

**Data Handling**:
```python
# Binary mode: single attention channel
scores = torch.tensor([0.1, 0.8, 0.3, 0.2, ...])  # shape (num_patches,)

# Multi-class mode: multiple channels
scores = torch.tensor([
    [0.1, 0.8, 0.1],   # Patch 0: attention per class
    [0.8, 0.1, 0.1],   # Patch 1
    ...
])  # shape (num_patches, num_classes)
```

**Error Handling**:
- Missing files → logged and skipped
- Shape mismatches → validation error with details
- Malformed tensors → informative exception

### New Component 2: PatchSelector (`utils/pseudo_labels.py`)

**Purpose**: Select high-confidence patches from attention scores using configurable strategies

**Selection Strategies**:

1. **Percentile** (recommended)
   - Select patches in top k% by score
   - Threshold e.g. 0.85 = select top 15% of patches
   - Good for: balanced datasets
   - ```python
     selected_indices = selector.select_patches(scores, strategy='percentile', threshold=0.85)
     # Returns indices of top 15% highest-scoring patches
     ```

2. **Threshold** (absolute cutoff)
   - Select patches with score ≥ threshold
   - Threshold e.g. 0.5 = select all patches ≥ 0.5
   - Good for: well-calibrated scores
   - ```python
     selected_indices = selector.select_patches(scores, strategy='threshold', threshold=0.5)
     # Returns indices of patches with score ≥ 0.5
     ```

3. **Entropy** (confidence-based)
   - Select patches with lowest entropy (most confident)
   - Threshold e.g. 0.3 = maximum entropy allowed
   - Good for: extreme confidence requirements
   - ```python
     selected_indices = selector.select_patches(scores, strategy='entropy', threshold=0.3)
     # Returns indices with entropy ≤ 0.3 (most confident)
     ```

4. **Margin** (discriminability)
   - Select patches with largest margin between top-2 classes
   - Threshold e.g. 0.2 = minimum margin
   - Good for: multi-class, discriminative samples
   - ```python
     selected_indices = selector.select_patches(scores, strategy='margin', threshold=0.2)
     # Returns indices where margin ≥ 0.2
     ```

**Configuration**:
```yaml
dataset:
  pseudo_label_selection_strategy: percentile      # or threshold, entropy, margin
  pseudo_label_confidence_threshold: 0.85          # strategy-specific
  pseudo_label_min_patches: 5                      # minimum selected per WSI
```

### New Component 3: PrototypeBank (`utils/prototype_guided_attention.py`)

**Purpose**: Store and manage interpretable per-class prototype vectors

**Initialization**:
```python
from utils.prototype_guided_attention import PrototypeBank

# From pre-selected high-confidence patches
bank = PrototypeBank(
    num_classes=2,
    num_prototypes_per_class=10,
    feature_dim=512,  # MedCLIP dimension
)

# Initialize with k-means on selected patches per class
for class_idx in range(num_classes):
    class_features = features[high_conf_indices[class_idx]]  # (N, 512)
    kmeans = KMeans(n_clusters=10).fit(class_features)
    bank.add_prototypes(class_idx, torch.from_numpy(kmeans.cluster_centers_))
```

**Prototype Representation**:
- L2-normalized vectors in feature space
- One per-class prototype bank (e.g., 10 prototypes for class 0, 10 for class 1)
- Learned during training via prototype refinement

**Usage in Classifier**:
```python
# Compute similarities between patch and prototypes
similarities = torch.nn.functional.cosine_similarity(
    features.unsqueeze(1),           # (batch, 1, 512)
    bank.prototypes[class_idx],      # (num_prototypes, 512)
    dim=2
)  # (batch, num_prototypes)

# Pool similarities to class-level scores
class_score = similarities.max(dim=1)[0]  # (batch,)
```

**Storage & Loading**:
```python
# Save with checkpoint
torch.save(bank.state_dict(), 'prototype_bank.pt')

# Load later
bank.load_state_dict(torch.load('prototype_bank.pt'))
```

### New Component 4: PrototypeGuidedAttention (`utils/prototype_guided_attention.py`)

**Purpose**: Interpretable attention mechanism based on prototype similarity

**Architecture**:
```
Patch Features (N, 512)
    ↓
[Similarity to Prototypes] ← (Interpretable: "similarity to class concept")
    ↓
Attention Scores (N, num_classes)
    ↓
[Max-pooling per class]
    ↓
WSI-level Scores (num_classes,)
    ↓
[MLP Head]
    ↓
Logits (num_classes)
```

**Forward Pass**:
```python
from utils.prototype_guided_attention import PrototypeGuidedAttention

attention = PrototypeGuidedAttention(
    prototype_bank=bank,
    num_classes=2,
    aggregate_method='max',  # or 'mean', 'learnable'
    mlp_hidden_dims=[256, 128],
)

# Compute attention-based WSI predictions
logits = attention(patch_features, return_attention=False)

# Or get interpretable attention scores
logits, attention_scores = attention(patch_features, return_attention=True)
# attention_scores shape: (num_patches, num_classes) → which patches contribute most?
```

**Key Advantage**: Attention scores show **which patches are similar to which class prototypes**, enabling visualization and interpretation of model decisions.

### New Component 5: MultiStageRefinementPipeline (`utils/prototype_refinement.py`)

**Purpose**: Iteratively refine prototypes and re-rank selected patches using EM-style loop

**Refinement Strategy**:
1. **Stage 0 (Initialization)**: Use selected patches from PatchSelector
2. **E-step (Expectation)**: Classify all patches using current prototypes; identify confident predictions
3. **M-step (Maximization)**: Update prototypes using newly confident patches
4. **Convergence Check**: Stop when prototype changes < threshold or max iterations reached

**Usage**:
```python
from utils.prototype_refinement import MultiStageRefinementPipeline

pipeline = MultiStageRefinementPipeline(
    num_classes=2,
    num_prototypes_per_class=10,
    max_stages=3,           # E-M iterations
    convergence_threshold=0.01,
    min_samples_per_class=5,
)

# Run refinement on dataset
refined_bank = pipeline.run_em_refinement_loop(
    dataset=train_dataset,
    initial_bank=prototype_bank,
    feature_extractor=feature_backbone,
    device='cuda',
)
```

**Stage Flow**:
```
Stage 0 (Init)
    ↓ high-conf patches from selector
    ↓
Stage 1 E-step: Classify with current prototypes
    ↓ find new high-conf patches
    ↓
Stage 1 M-step: Update prototypes from new patches
    ↓
Stage 2 E-step: Re-classify...
    ↓ (converges when proto changes small)
    ↓
Final refined prototypes
```

**Benefits**:
- Iteratively improves prototype quality
- Adapts to local dataset characteristics
- Converges when changes stabilize
- Reduces impact of poor initial pseudo-labels

### New Component 6: WSIClassifierWithPrototypes (`utils/prototype_guided_attention.py`)

**Purpose**: End-to-end WSI classification combining all components

**Architecture**:
```python
from utils.prototype_guided_attention import WSIClassifierWithPrototypes

classifier = WSIClassifierWithPrototypes(
    prototype_bank=bank,
    feature_dim=512,
    num_classes=2,
    mlp_hidden_dims=[256, 128],
    aggregate_method='max',
    dropout=0.5,
)

# Forward pass (training or inference)
logits = classifier(patch_features)
```

**Training Mode**:
```python
# Compute loss
ce_loss = torch.nn.functional.cross_entropy(logits, labels)

# Backward pass updates both MLP and (if unfrozen) prototypes
loss.backward()
optimizer.step()
```

**Inference Mode**:
```python
classifier.eval()
with torch.no_grad():
    logits = classifier(patch_features)
    predictions = logits.argmax(dim=-1)
```

### Enhanced Dataset with Pseudo-Labels

**File**: `datasets/wsi_dataset.py`

**New Class**: `CustomWSIPatchTrainingDatasetWithPseudoLabels`

**Features**:
- Loads pseudo-label `.pt` files on initialization
- Pre-computes high-confidence patch indices per WSI
- Reports statistics (selected %, per-class distribution)
- Provides methods:
  - `get_high_confidence_patches(wsi_name)` → numpy array of indices
  - `get_wsi_pseudo_scores(wsi_name)` → attention scores for WSI

**Usage**:
```python
from datasets.wsi_dataset import CustomWSIPatchTrainingDatasetWithPseudoLabels

dataset = CustomWSIPatchTrainingDatasetWithPseudoLabels(
    split_csv='path/to/splits.csv',
    labels_csv='path/to/labels.csv',
    wsi_dir='path/to/wsi_images/',
    pseudo_label_dir='path/to/attention_scores/',
    pseudo_label_binary_mode=True,
    pseudo_label_selection_strategy='percentile',
    pseudo_label_confidence_threshold=0.85,
)

# High-confidence indices are pre-computed
high_conf_idx = dataset.get_high_confidence_patches('slide_001')
print(f"Selected {len(high_conf_idx)} of {len(dataset)} patches")
```

---

## Training Workflow Comparison

### Original PBIP Training
```
1. Extract patches from coordinates
2. Pre-compute features (MedCLIP, etc.)
3. Initialize prototypes via k-means on all training features
4. Train MLP head with CE loss
5. Validate on held-out patches
```

### WSI-Level Adapted Training
```
1. [NEW] Load pseudo-labels (attention scores)
2. [NEW] Select high-confidence patches per WSI
3. Extract patches from selected coordinates only
4. Pre-compute features
5. [NEW] Re-initialize prototypes via k-means on selected features
6. [NEW] Optionally refine prototypes via EM loop
7. Train MLP head (+ optionally prototypes if unfrozen)
8. Validate on held-out patches
```

### Key Differences
| Aspect | Original | WSI-Adapted |
|--------|----------|------------|
| **Patch source** | All patches from coordinates | High-confidence patches only |
| **Prototype init** | K-means on all training features | K-means on selected features |
| **Refinement** | Direct training | Optional EM-loop refinement |
| **Computational cost** | Feature extraction for all patches | Feature extraction for selected patches only (less) |
| **Interpretability** | Prototype similarities | Prototype + Attention scores (more interpretable) |

---

## Integration with Training Loop

To use these components in your training script:

```python
from torch.utils.data import DataLoader
from datasets.wsi_dataset import CustomWSIPatchTrainingDatasetWithPseudoLabels
from utils.pseudo_labels import PseudoLabelLoader, PatchSelector
from utils.prototype_guided_attention import WSIClassifierWithPrototypes, PrototypeBank
from utils.prototype_refinement import MultiStageRefinementPipeline
import torch

# 1. Load dataset with pseudo-labels
dataset = CustomWSIPatchTrainingDatasetWithPseudoLabels(...)

# 2. Build prototype bank from high-confidence patches
high_conf_features = ... # aggregate from dataset
prototype_bank = PrototypeBank(...)
# Initialize with k-means on selected features

# 3. Optionally refine with EM loop
refiner = MultiStageRefinementPipeline(...)
prototype_bank = refiner.run_em_refinement_loop(...)

# 4. Build classifier
classifier = WSIClassifierWithPrototypes(prototype_bank=prototype_bank, ...)
classifier.to(device)

# 5. Standard training loop
optimizer = torch.optim.Adam(classifier.parameters())
dataloader = DataLoader(dataset, batch_size=32)

for epoch in range(num_epochs):
    for patches, labels in dataloader:
        patches, labels = patches.to(device), labels.to(device)
        
        # Forward
        logits = classifier(patches)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

All components integrate seamlessly with PyTorch training loops while providing improved interpretability via prototypes and attention mechanisms.
