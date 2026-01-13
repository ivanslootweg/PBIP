# PyTorch 2.6+ Compatibility Fix

## Issue

PyTorch 2.6+ changed the default value of `weights_only` parameter in `torch.load()` from `False` to `True` for security reasons. This breaks loading of checkpoints that contain Python objects like OmegaConf's `DictConfig`.

**Error Message**:
```
_pickle.UnpicklingError: Weights only load failed
WeightsUnpickler error: Unsupported global: GLOBAL omegaconf.dictconfig.DictConfig
```

## Root Cause

The PBIP checkpoint saves the configuration along with model weights:
```python
torch.save({
    "cfg": cfg,  # OmegaConf DictConfig - not picklable in weights_only mode
    "iter": n_iter,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict()
}, save_path)
```

With PyTorch 2.6+, loading this requires `weights_only=False` or registering DictConfig as a safe global.

## Solution

Updated all `torch.load()` calls to use `weights_only=False`:

### File: `train_stage_1.py` (Line 325)
```python
# Before:
checkpoint = torch.load(best_model_path, map_location=device)

# After:
checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
```

### File: `model/model.py` (Line 60)
```python
# Before:
state_dict = torch.load(pretrained_path, map_location="cpu")

# After:
state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)
```

## Impact

- ✅ Compatible with PyTorch 2.6+
- ✅ Backward compatible with older PyTorch versions (parameter is ignored if not recognized)
- ✅ No functional changes
- ⚠️ Slightly less secure (but we control the checkpoint files)

## Testing

The fix was tested by running training and verifying checkpoint loading works:
```
Loading best model from: /data/pathology/projects/ivan/WSS/PBIP/checkpoints/2026-01-13-12-28/best_cam.pth
✓ Best model loaded successfully!
```

## Alternative Approaches

### Option 1: Safe globals (more secure but more verbose)
```python
from torch.serialization import add_safe_globals
from omegaconf import DictConfig

add_safe_globals([DictConfig])
checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
```

### Option 2: Don't save config in checkpoint (breaking change)
```python
# Only save model state
torch.save({
    "iter": n_iter,
    "model": model.state_dict(),
}, save_path)
```

We chose Option 1 (current fix) because:
- Minimal changes needed
- No loss of checkpoint information
- Works with all PyTorch versions
- Still loads config for reproducibility

## Files Modified

- `train_stage_1.py`: Updated torch.load() on line 326
- `model/model.py`: Updated torch.load() on line 60

## References

- PyTorch 2.6 Release Notes: https://pytorch.org/blog/pytorch-2.6-released/
- Security change discussion: https://github.com/pytorch/pytorch/pull/134610
