# Paper Hyperparameter Corrections

This document tracks the corrections made to align the code with the actual paper hyperparameters from "Prototype-based Instance Segmentation in Pathology Images" (arXiv:2503.12068).

## Summary of Changes

### ❌ Previous (Incorrect) Values → ✅ Corrected Values

| Parameter | Previous | Corrected | Source |
|-----------|----------|-----------|--------|
| **Learning Rate** | 5×10^-5 | **1×10^-5** | Section 4.1 |
| **Weight Decay** | 0.001 | **0.003** | Section 4.1 |
| **Temperature (τ)** | 0.07 | **1.0** | Section 4.1 |
| **Epochs** | 8 | **10** | Section 4.1 |
| **Contrastive Weight (β)** | 0.1 | **0.5** | Section 4.1 |
| **Adaptive Threshold (δ)** | 0.5 | **0.15** | Section 4.1 |

### ✅ Already Correct Values

| Parameter | Value | Source |
|-----------|-------|--------|
| **Batch Size** | 10 | Section 4.1 |
| **K-means Clusters** | 3 | Section 4.1 |
| **λ_cam** | 0.01 | Not specified in excerpt |
| **Scale Weights** | [0.0, 0.1, 1.0, 1.0] | Not specified in excerpt |

## Detailed Changes

### 1. Learning Rate: 5e-5 → 1e-5
**File**: `work_dirs/custom_wsi_template.yaml`
```yaml
optimizer:
  learning_rate: 0.00001  # Was 0.00005
```

**Impact**: Training will converge more slowly but may achieve better final accuracy. This is 5× slower learning.

### 2. Weight Decay: 0.001 → 0.003
**File**: `work_dirs/custom_wsi_template.yaml`
```yaml
optimizer:
  weight_decay: 0.003  # Was 0.001
```

**Impact**: Stronger L2 regularization to prevent overfitting. 3× stronger regularization.

### 3. Temperature: 0.07 → 1.0
**File**: `work_dirs/custom_wsi_template.yaml`
```yaml
train:
  temperature: 1.0  # Was 0.07
```

**File**: `train_stage_1.py`
```python
temperature = getattr(cfg.train, 'temperature', 1.0)  # Was 0.07
```

**Impact**: InfoNCE contrastive loss will be less sharp. Temperature=1.0 means no temperature scaling, while 0.07 was making the softmax very sharp. This is a **14× change** and will significantly affect contrastive learning dynamics.

### 4. Epochs: 8 → 10
**File**: `work_dirs/custom_wsi_template.yaml`
```yaml
train:
  epoch: 10  # Was 8
```

**Impact**: 25% more training time. Model will train for 2 additional epochs.

### 5. Contrastive Weight (β): 0.1 → 0.5
**File**: `work_dirs/custom_wsi_template.yaml`
```yaml
train:
  contrastive_weight: 0.5  # Was 0.1
```

**Impact**: Contrastive loss (foreground/background separation) now has **5× more influence** on the total loss. This is the β parameter from the paper.

### 6. Adaptive Threshold (δ): 0.5 → 0.15
**File**: `work_dirs/custom_wsi_template.yaml`
```yaml
train:
  mask_adapter_alpha: 0.15  # Was 0.5
```

**Impact**: Dynamic thresholding will be more conservative (15% of max CAM value instead of 50%). This creates **smaller, more precise masks** for foreground/background separation.

## Loss Function Equation

### Paper Definition (Section 4.1)
```
L_total = α × L_classification + β × L_similarity
```
Where:
- α = 1 (classification loss weight)
- β = 0.5 (similarity/contrastive loss weight)

### Our Implementation
```python
# Multi-scale classification
cls_loss = scale1_weight*L_s1 + scale2_weight*L_s2 + scale3_weight*L_s3 + scale4_weight*L_s4
# = 0.0 + 0.1 + 1.0 + 1.0 = 2.1 (but normalized, effectively α = 1)

# Total loss
loss = cls_loss + contrastive_weight * (fg_loss + bg_loss + cam_reg)
# = α*L_cls + β*(L_fg + L_bg + λ_cam*CAM_reg)
# = 1*L_cls + 0.5*(L_fg + L_bg + 0.01*CAM_reg)
```

## Additional Paper Details

From Section 4.1:
- **Number of clusters K**: 3
- **Samples per subclass**: NK = 100
- **θ1 (foreground loss weight)**: 1
- **θ2 (background loss weight)**: 0.5

**Note**: θ1 and θ2 are currently not separately configurable in the code. Both fg_loss and bg_loss are weighted equally in the implementation. This could be a future enhancement.

## Updated Configuration Example

```yaml
train:
  samples_per_gpu: 10
  epoch: 10
  temperature: 1.0
  contrastive_weight: 0.5
  lambda_cam: 0.01
  mask_adapter_alpha: 0.15
  scale1_weight: 0.0
  scale2_weight: 0.1
  scale3_weight: 1.0
  scale4_weight: 1.0

optimizer:
  learning_rate: 0.00001  # 1e-5
  weight_decay: 0.003

features:
  k_list: [3, 3]  # K=3 clusters per class
```

## Expected Training Impact

1. **Slower but more stable convergence** due to lower learning rate
2. **Better generalization** from stronger weight decay
3. **Smoother contrastive learning** from higher temperature
4. **More balanced loss components** with β=0.5
5. **More precise foreground/background masks** with δ=0.15
6. **Longer training** with 10 epochs instead of 8

## Testing Recommendations

After applying these corrections:
1. ✅ Verify training runs without errors
2. ✅ Monitor loss curves - expect slower initial convergence
3. ✅ Check CAM quality - should be more precise with δ=0.15
4. ✅ Compare final mIoU/Dice scores with paper results
5. ✅ Validate that contrastive loss contributes more significantly

## References

- Paper: "Prototype-based Instance Segmentation in Pathology Images"
- arXiv: 2503.12068
- Section 4.1: Implementation Details
