# Architecture & Design

## System Overview

```
┌─────────────────┐
│  WSI Patches    │
│  (224×224)      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Encoder (Virchow2/DinoV3)  │
│  - Pathology-specific pretrained     │
│  - Frozen during training            │
└────────┬────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│  Patch Features      │
│  n_patches × d_feat  │
│  (precomputed)       │
└────────┬─────────────┘
         │
         ├─────────────────────────────┐
         │                             │
         ▼                             ▼
    K-means                  MIL Model
    Clustering              (Attention)
         │                             │
         ▼                             ▼
    Prototype Bank      Pseudo-labels
    (k × nk features)   (Attention Scores)
         │                             │
         └────────────┬────────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │ Prototype Matching      │
          │ S_match = cosine sim    │
          └────────┬────────────────┘
                   │
                   ▼
          ┌────────────────────────────┐
          │ Combine with Attention     │
          │ Final = Softmax(S + A_raw) │
          └────────┬───────────────────┘
                   │
                   ▼
          ┌────────────────────────────┐
          │ Patch-level Predictions    │
          │ (Refined Segmentation)     │
          └────────────────────────────┘
```

---

## Component Details

### 1. Encoders

#### Virchow2 (Recommended)
- **Organization:** PAIGE-AI
- **Dimension:** 1280
- **Training:** Pathology-specific pretraining on histology slides
- **Strengths:** Best for histopathology, robust to stain variations
- **Import:** `from utils.encoders import Virchow2Wrapper`

#### DinoV3
- **Organization:** Meta AI
- **Dimension:** 1024
- **Training:** Self-supervised learning (no labels needed)
- **Strengths:** General-purpose, works across modalities
- **Import:** `from utils.encoders import DinoV3`

### 2. Feature Extraction

**Module:** `features/extract_patch_features.py`

Features are pre-extracted once and reused:
- Saves time (no extraction during training)
- Enables reproducibility
- Allows experimenting with different training strategies
- Supports offline preprocessing

**Input:** WSI patches (RGB images)
**Output:** Patch features (n_patches, d_feat)
**Storage:** `.pt` or `.npy` or `.pkl` files

### 3. Prototype Bank

**Module:** `features/k_mean_cos_per_class.py`

K-means clustering creates a prototype bank:
- **Input:** All extracted patch features per class
- **Process:** K-means with K=6, 5 nearest neighbors per cluster
- **Output:** Label features (K*Nk, d_feat) where:
  - K=6 (subclasses per parent class)
  - Nk=5 (representative images per subclass)
- **Total:** 60 prototype features per class (2 classes = 120 total)

### 4. Pseudo-Labels

**Module:** `utils/pseudo_labels.py`

Weak labels from MIL model:
- **Input:** Attention scores from WSI-level classifier
- **Format:** Per-patch confidence scores (0-1 for binary)
- **Usage:** Select high-confidence patches
- **Selection:** Percentile-based (e.g., top 0.5%)

### 5. Prototype Matching

**Module:** `utils/prototype_guided_attention.py`

Refines pseudo-labels using prototype similarity:

```
For each patch i:
  1. f_i = patch feature vector
  2. S_match(f_i) = cosine_sim(f_i, prototype_bank)  → (num_classes,)
  3. A_raw_i = raw attention score from MIL
  4. Final_i = Softmax(S_match(f_i) + A_raw_i)
```

**Benefits:**
- Leverages prototype bank knowledge
- Combines weak supervision with feature similarity
- Smooth transition from coarse (MIL) to fine (prototype) predictions

---

## Training Pipeline

### Step 1: Feature Extraction
```
Input: WSI patches + encoder
Output: patch_features/{wsi_name}.pt (n_patches × 1280)
Time: ~2-5 min per WSI on GPU
```

### Step 2: K-means Clustering
```
Input: All patch features per class
Output: label_features_{run_uid}.pkl (60 × 1280)
Time: ~5-10 min
```

### Step 3: Validation
```
Input: Config + extracted features
Output: Diagnostic report
Time: <1 min
```

### Step 4: Training
```
Input: 
  - patch_features/
  - weak_labels/
  - label_features
  
Process:
  - Load features batch-wise
  - Compute prototype matching scores
  - Combine with attention scores
  - Train lightweight refinement head
  
Output: Model checkpoint
Time: ~10-30 min for 10 epochs
```

---

## Data Flow

### Training Data Loading
```python
# datasets/feature_dataset.py
FeatureWSIDataset:
  1. Load split.csv → train/val WSI names
  2. For each WSI:
     - Load features from patch_features_dir/{wsi_name}.pt
     - Load attention from pseudo_label_dir/{wsi_name}.pt
     - Load class label from labels_csv
     - Min-max scale per WSI
  3. Return: (wsi_name, features, attention, class_label)
```

### Feature Scaling
```python
# Per-WSI min-max normalization
attention_scaled = (attention - attention.min()) / (attention.max() - attention.min() + 1e-6)
```

Benefits:
- Normalizes across different scale ranges
- Prevents numerical instability
- Improves gradient flow

---

## Configuration Hierarchy

```yaml
model:
  patch_encoder: virchow2  # ← Determines encoder
  ↓ (loaded by EncoderFactory)
  
dataset:
  patch_features_dir: /path/...  # ← Feature location
  pseudo_label_dir: /path/...     # ← Attention scores
  ↓ (loaded by get_custom_dataset)
  
train:
  samples_per_gpu: 5              # ← Batch size
  epoch: 10                       # ← Training duration
```

---

## Model Architecture

### Current Implementation
- **Backbone:** MixTransformer (mit_b1)
- **Feature projection:** Adaptive layers (n_ratio=0.5)
- **Prototype matching:** Cosine similarity

### Input/Output
```
Input:  patch_features (batch, 1280)
        ↓
Label feature projection:
  dim_1280 → 64 → dim_feature_i (multi-scale)
        ↓
Output: Classification logits for each scale
```

### Why This Works
1. **Frozen encoders:** Reuse pretrained knowledge
2. **Lightweight training:** Only projection layers updated
3. **Multi-scale:** Captures hierarchical features
4. **Prototype guidance:** Leverages clustering

---

## Performance Characteristics

### Memory
- **Per-WSI features:** ~100-500 MB (depends on patch count)
- **Batch loading:** Features loaded on-demand
- **GPU memory:** 4-8 GB sufficient for batch_size=5

### Speed
- **Feature extraction:** 2-5 min/WSI (GPU)
- **K-means:** 5-10 min (CPU)
- **Training epoch:** 1-3 min (GPU, batch_size=5)
- **Inference:** <1s per WSI (CPU)

### Scalability
- **WSIs:** Tested up to 1000s
- **Patches/WSI:** 100-10000
- **Classes:** 2-10 (tested)
- **Feature dim:** Any (adapter layers handle)

---

## Extension Points

### Add Custom Encoder
```python
# 1. Implement in utils/encoders.py
class MyEncoder(nn.Module):
    def __init__(self):
        self.model = load_pretrained()
    
    def forward(self, x):
        return self.model(x)  # (batch, feature_dim)

# 2. Register in EncoderFactory
```

### Add Custom Loss
```python
# 1. Implement loss function
def prototype_matching_loss(predictions, targets, prototypes):
    # ...

# 2. Use in training loop
loss = prototype_matching_loss(logits, labels, prototype_bank)
```

### Add Custom Dataset
```python
# 1. Implement in datasets/
class CustomDataset(Dataset):
    def __init__(self, ...):
        # Load data
    
    def __getitem__(self, idx):
        # Return sample

# 2. Register in utils/trainutils.py
```

---

## Testing & Validation

### Unit Tests
```bash
python3 -m pytest tests/
```

### Integration Test
```bash
python3 test_pipeline.py --config config.yaml
```

### Manual Validation
```python
from utils.encoders import EncoderFactory
encoder = EncoderFactory.create_encoder('virchow2')
features = encoder(image)  # (1, 1280)
```

---

## Future Improvements

1. **Distributed training:** Multi-GPU support
2. **Curriculum learning:** Progressive prototype difficulty
3. **Online prototype updates:** Refine prototypes during training
4. **Ensemble methods:** Combine multiple encoders
5. **Uncertainty quantification:** Confidence scores per patch

