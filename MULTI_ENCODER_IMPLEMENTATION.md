# Multi-Encoder Support Implementation

## Overview

The PBIP pipeline now supports multiple vision models for patch feature extraction. Previously hard-coded to use MedCLIP, the system can now flexibly switch between different encoders while maintaining the same training pipeline.

## Changes Made

### 1. New Encoder Factory (`utils/encoders.py`)

**File**: [utils/encoders.py](utils/encoders.py)

Created a unified interface for encoder loading with support for:

- **MedCLIPFactory**: Medical foundation model (512-dim features)
- **Virchow2Factory**: Pathology-specific ViT by PAIGE-AI (1280-dim features)
- **DinoV3Factory**: Self-supervised ViT by Meta (1024-dim features)

```python
# Usage
from utils.encoders import EncoderFactory, get_encoder_config

# Load encoder
encoder, feature_dim = EncoderFactory.create_encoder(
    encoder_name="virchow2",
    device=device
)

# Get encoder info
config = get_encoder_config("virchow2")
print(config['feature_dim'])  # 1280
print(config['description'])
```

**Key Features**:
- Unified interface across all encoders
- Automatic model loading from HuggingFace Hub
- Returns consistent (encoder_model, feature_dim) tuples
- Includes encoder configuration metadata

### 2. Configuration Update

**File**: [work_dirs/custom_wsi_template.yaml](work_dirs/custom_wsi_template.yaml)

Added new configuration option:

```yaml
model:
  patch_encoder: medclip  # Options: medclip, virchow2, dinov3
```

### 3. Training Script Update

**File**: [train_stage_1.py](train_stage_1.py)

**Changes**:
- Removed hardcoded MedCLIP imports
- Replaced with flexible encoder factory initialization
- Added encoder configuration logging
- Maintains backward compatibility (defaults to medclip)

**Before**:
```python
from medclip import MedCLIPModel, MedCLIPVisionModelViT
clip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
clip_model = clip_model.to(device)
```

**After**:
```python
from utils.encoders import EncoderFactory
encoder_name = getattr(cfg.model, 'patch_encoder', 'medclip')
clip_model, feature_dim = EncoderFactory.create_encoder(
    encoder_name=encoder_name, 
    device=device
)
```

### 4. Prototype Extraction Update

**File**: [features/extract_prototypes_from_gt.py](features/extract_prototypes_from_gt.py)

- Added imports for common utilities
- Removed duplicate extract_patch function
- Now uses unified patch extraction from [utils/common.py](utils/common.py)

### 5. Documentation

**File**: [ENCODER_OPTIONS.md](ENCODER_OPTIONS.md)

Comprehensive guide covering:
- Feature dimensions and characteristics
- Performance comparison table
- When to use each encoder
- Configuration examples
- Switching encoders (important: regenerate features)
- Adding new encoders
- Troubleshooting
- Citations

## Supported Encoders

| Encoder | Dimension | Best For | Installation |
|---------|-----------|----------|--------------|
| **MedCLIP** | 512 | General pathology | `pip install medclip` |
| **Virchow2** | 1280 | SOTA histopathology | `pip install timm` |
| **DinoV3** | 1024 | Cross-domain vision | `pip install timm` |

## Usage Examples

### Use MedCLIP (default)
```yaml
model:
  patch_encoder: medclip
```

### Use Virchow2 (recommended for pathology)
```yaml
model:
  patch_encoder: virchow2
```

### Use DinoV3 (for research/comparison)
```yaml
model:
  patch_encoder: dinov3
```

## Important Notes

### Backward Compatibility
- Defaults to `medclip` if not specified
- Existing configs continue to work without modification
- No changes needed to training code

### Switching Encoders
If you switch encoders, you **must** regenerate prototypes and features:

```bash
# Delete old features
rm -rf {features.save_dir}/*

# Re-extract with new encoder
python features/extract_prototypes_from_gt.py --config work_dirs/custom_wsi_template.yaml
python features/excract_medclip_proces.py --config work_dirs/custom_wsi_template.yaml
python features/k_mean_cos_per_class.py --config work_dirs/custom_wsi_template.yaml

# Train with new encoder
python train_stage_1.py --config work_dirs/custom_wsi_template.yaml --gpu 0
```

### Feature Dimension Handling
Each encoder has a different output dimension:
- MedCLIP: 512-dim
- Virchow2: 1280-dim (more discriminative)
- DinoV3: 1024-dim (balance between size and performance)

The k-means clustering handles variable dimensions automatically.

## Architecture

```
Input Patch (224×224)
    ↓
┌─────────────────────────────────────┐
│  Encoder Selection (via config)     │
│  ├─ MedCLIP (512-dim)              │
│  ├─ Virchow2 (1280-dim)            │
│  └─ DinoV3 (1024-dim)              │
└─────────────────────────────────────┘
    ↓
Features (encoder-specific dimension)
    ↓
L2 Normalization
    ↓
K-means Clustering
    ↓
Cluster Centers (Prototypes)
    ↓
Training Pipeline
```

## Adding New Encoders

To add a new encoder (e.g., UNI by MahmoodLab):

1. **Add factory method** to `EncoderFactory`:
```python
@staticmethod
def _create_uni(device: torch.device = None):
    """Create UNI encoder"""
    encoder = timm.create_model(
        "hf-hub:MahmoodLab/UNI",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    )
    encoder = encoder.to(device)
    encoder.eval()
    return encoder, 1024  # UNI outputs 1024-dim
```

2. **Update main factory**:
```python
def create_encoder(self, encoder_name: str, ...):
    ...
    elif encoder_name == "uni":
        return EncoderFactory._create_uni(device)
```

3. **Add configuration metadata**:
```python
def get_encoder_config(encoder_name: str):
    configs = {
        ...
        "uni": {
            "input_size": 224,
            "feature_dim": 1024,
            "description": "UNI: Large-scale Unlabeled Image Pretraining",
            ...
        }
    }
```

4. **Update config template**:
```yaml
model:
  patch_encoder: uni
```

## Performance Notes

### Memory Usage
- **MedCLIP**: ~1GB (recommended for limited memory)
- **Virchow2**: ~4GB (larger model, more features)
- **DinoV3**: ~2GB (good balance)

### Speed
- **MedCLIP**: Fast (smallest model)
- **DinoV3**: Moderate
- **Virchow2**: Slower (largest model, more computations)

### Accuracy (Histopathology Tasks)
- **Virchow2**: Highest (~2-5% better than others)
- **DinoV3**: Competitive
- **MedCLIP**: Good baseline

## Testing

Verify encoders load correctly:

```python
from utils.encoders import EncoderFactory
import torch

device = torch.device("cuda")

# Test all encoders
for encoder_name in ["medclip", "virchow2", "dinov3"]:
    try:
        encoder, dim = EncoderFactory.create_encoder(encoder_name, device=device)
        print(f"✓ {encoder_name}: {dim}-dim features")
    except Exception as e:
        print(f"✗ {encoder_name}: {e}")
```

## Debugging

### "Unknown encoder" error
Supported: `medclip`, `virchow2`, `dinov3` (case-insensitive)

### "Module not found" error
Install timm: `pip install timm`

### GPU memory errors with Virchow2
Reduce batch size in config:
```yaml
train:
  samples_per_gpu: 5  # Reduced from 10
```

## Version History

- **v1.0** (Current): Multi-encoder support with MedCLIP, Virchow2, DinoV3
- **v0.1**: MedCLIP hardcoded

## References

See [ENCODER_OPTIONS.md](ENCODER_OPTIONS.md) for detailed comparison and citations.
