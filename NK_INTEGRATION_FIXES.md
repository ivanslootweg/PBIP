# Nk Parameter Integration Fixes

## Problem
After implementing dynamic Nk=5 parameter, the training loop failed with shape mismatch:
```
RuntimeError: The size of tensor a (30) must match the size of tensor b (6) at non-singleton dimension 1
```

The issue was that the model now outputs **30 logits** (K × Nk = 6 × 5) for 30-way classification, but all downstream code was written assuming **K=6 logits**.

## Solution
Updated all code paths that handle the expanded feature bank (K×Nk instead of K) to properly merge Nk representatives back to K subclass predictions.

## Files Modified

### 1. **Model Output** (`/model/model.py`)
- **Change**: Now returns `nk` along with `k_list`
- **Line**: 191
- **Impact**: Downstream code can now query the number of representatives per subclass

```python
# Before:
return cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, self.k_list

# After:
return cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, self.k_list, self.nk
```

### 2. **Label Expansion** (`/utils/hierarchical_utils.py`)
- **Function**: `expand_parent_to_subclass_labels()` (lines 94-122)
- **Change**: Now accepts `nk` parameter and expands to K×Nk labels instead of just K
- **Logic**: 
  - Old: Parent labels → K subclass labels
  - New: Parent labels → K×Nk feature labels (Nk copies per subclass)

**Key insight**: Each of the K subclasses has Nk representative features that should all inherit the same parent label.

### 3. **Prediction Merging** (`/utils/hierarchical_utils.py`)
- **Function**: `merge_to_parent_predictions()` (lines 40-93)
- **Change**: Now handles K×Nk→K→parent merging in two steps
- **Two-stage merge**:
  1. Merge Nk features per subclass → K subclass predictions
  2. Merge K subclass predictions → parent predictions
- **Method**: Weighted average or max pooling at each stage

### 4. **CAM Merging** (`/utils/hierarchical_utils.py`)
- **Function**: `merge_subclass_cams_to_parent()` (lines 65-118)
- **Change**: Mirrors prediction merging logic for CAM tensors [B, K×Nk, H, W]
- **Output**: [B, num_parent_classes, H, W]

### 5. **Multi-scale Wrappers** (`/utils/common.py`)
- **Functions**: 
  - `merge_multiscale_predictions()` (lines 175-194)
  - `merge_multiscale_cams()` (lines 197-216)
- **Change**: Added `nk` parameter with default value 1 (backward compatible)
- **Role**: Apply Nk-aware merging to all 4 scales (cls1, cls2, cls3, cls4)

### 6. **Training Loop** (`/train_stage_1.py`)
- **Lines**: 589-597
- **Changes**:
  - Unpack `nk` from model output: `..., k_list, nk = model(inputs)`
  - Pass `nk` to merge functions: `merge_multiscale_predictions(..., nk=nk, ...)`
  - Pass `nk` to label expansion: `expand_parent_to_subclass_labels(..., nk=nk)`

### 7. **Validation Loop** (`/utils/validate.py`)
- **Lines**: 66, 89, 142
- **Changes**:
  - Unpack `nk` from model output at 3 call sites
  - Pass `nk` to merge functions in validation and TTA loops

## Key Architectural Change

**Merging Strategy with Nk:**

```
30 logits (K×Nk features)
    ↓
[Feature 0-4] [Feature 5-9] [Feature 10-14] ... [Feature 25-29]
    ↓           ↓               ↓                    ↓
   (merge)     (merge)        (merge)              (merge)
    ↓           ↓               ↓                    ↓
Subclass 0   Subclass 1    Subclass 2          Subclass 5
    ↓           ↓               ↓                    ↓
[Subclass logits] (K=6 predictions)
    ↓
  (merge)
    ↓
Parent prediction (2 parent classes)
```

## Backward Compatibility

All functions maintain **backward compatibility** with Nk=1 (legacy):
- Default `nk=1` parameter in all functions
- When Nk=1, behavior is identical to pre-Nk code
- Existing checkpoints with `nk=1` continue to work

## Validation

**Feature bank expansion verified:**
- ✅ Config: `nk: 5`
- ✅ K-means output: `Feature shape: torch.Size([30, 512])` (6 × 5)
- ✅ Model loading: `Nk (representatives per subclass): 5`
- ✅ Shape handling: Merging now correctly handles (batch, 30) → (batch, 6) → (batch, 2)

## Next Steps

✅ Code changes complete - all Nk-handling implemented
🔜 Run training to verify end-to-end pipeline works with Nk=5
🔜 Monitor if larger representative bank improves learning

---

**Status**: Ready for training with full Nk integration ✅
