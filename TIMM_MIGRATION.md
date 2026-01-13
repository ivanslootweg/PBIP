# TIMM Integration

The model has been migrated to use **timm** (PyTorch Image Models) for the backbone encoder instead of the custom SegFormer implementation.

## Benefits

- **Automatic pretrained weights**: timm handles downloading and loading ImageNet pretrained weights automatically
- **No manual checkpoint downloads**: No need to manually download and place pretrained `.pth` files
- **Better maintained**: timm is actively maintained and updated
- **More models available**: Easy to swap to other backbones

## Supported Backbones

The following SegFormer backbones are available via timm:

- `mit_b0` → `segformer_b0_backbone` (32, 64, 160, 256 channels)
- `mit_b1` → `segformer_b1_backbone` (64, 128, 320, 512 channels) **[default]**
- `mit_b2` → `segformer_b2_backbone` (64, 128, 320, 512 channels)
- `mit_b3` → `segformer_b3_backbone` (64, 128, 320, 512 channels)
- `mit_b4` → `segformer_b4_backbone` (64, 128, 320, 512 channels)
- `mit_b5` → `segformer_b5_backbone` (64, 128, 320, 512 channels)

## Configuration

In your config YAML:

```yaml
model:
  backbone:
    config: mit_b1  # or mit_b0, mit_b2, etc.
    stride: [4, 2, 2, 1]
  n_ratio: 0.5

train:
  pretrained: true  # Downloads ImageNet pretrained weights automatically
```

## Changes from Original

1. **Removed dependency** on `model.segform.mix_transformer`
2. **Removed need** for manual pretrained checkpoint files in `./pretrained/`
3. **Added timm** as the backbone provider
4. **Simplified loading**: timm handles all weight initialization

## Usage

No changes needed to your training command:

```bash
python train_stage_1.py --config work_dirs/custom_wsi_template.yaml --gpu 0
```

On first run, timm will automatically download the pretrained weights from Hugging Face Hub.

## Troubleshooting

If you encounter issues:

1. **Check timm version**: Ensure you have `timm>=0.9.0`
   ```bash
   pip install --upgrade timm
   ```

2. **Network issues**: If timm can't download weights, you can set offline mode:
   ```python
   export HF_HUB_OFFLINE=1
   ```
   (But you'll need to download weights manually first)

3. **Check available models**:
   ```python
   import timm
   print(timm.list_models('segformer*'))
   ```
