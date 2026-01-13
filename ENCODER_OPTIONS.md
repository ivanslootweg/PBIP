# Patch Encoder Options

The PBIP pipeline supports multiple vision models for patch feature extraction. Choose the encoder that best fits your use case.

## Supported Encoders

### 1. **MedCLIP** (Default)
- **Dimension**: 512
- **Type**: Medical foundation model (CLIP-based)
- **Best for**: General pathology, datasets with text descriptions
- **Paper**: [Learning Visual Representations from Large-Scale Unlabeled Videos](https://arxiv.org/abs/2112.02624)
- **Installation**: `pip install medclip`

```yaml
model:
  patch_encoder: medclip
```

**Pros**:
- Trained on medical texts and images
- Aligned with medical terminology
- Smaller feature dimension (fast)

**Cons**:
- Less specialized for pathology than Virchow2
- Lower absolute performance on histology tasks

---

### 2. **Virchow2** (Recommended)
- **Dimension**: 1280
- **Type**: Pathology-specific Vision Transformer
- **Best for**: Histopathology analysis, high-performance requirements
- **Organization**: PAIGE-AI (developed for pathology AI)
- **Paper**: [Virchow: A Million-slide Digital Pathology Foundation Model](https://arxiv.org/abs/2404.23228)
- **Installation**: `pip install timm`

```yaml
model:
  patch_encoder: virchow2
```

**Pros**:
- Trained on millions of pathology slides
- State-of-the-art performance on histology tasks
- Larger feature dimension for better discrimination
- Optimized for WSI analysis

**Cons**:
- Larger model (slower inference)
- Requires more GPU memory
- Model weights are large (~4GB)

---

### 3. **DinoV3** (Generic Vision)
- **Dimension**: 1024
- **Type**: Self-supervised Vision Transformer
- **Best for**: General computer vision, cross-domain evaluation
- **Organization**: Meta (Facebook Research)
- **Paper**: [DINOv3: The Power of Self-Supervised Learning with Vision Transformers](https://arxiv.org/abs/2305.08243)
- **Installation**: `pip install timm`

```yaml
model:
  patch_encoder: dinov3
```

**Pros**:
- Generic vision model (works across domains)
- Large feature dimension (1024)
- Good zero-shot performance
- Self-supervised (no task-specific bias)

**Cons**:
- Not specialized for pathology
- May underperform vs. Virchow2 on histology tasks
- Requires careful tuning for pathology

---

## Performance Comparison

| Metric | MedCLIP | Virchow2 | DinoV3 |
|--------|---------|----------|--------|
| Feature Dimension | 512 | 1280 | 1024 |
| Pathology Optimization | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Training Speed | Fast | Slow | Moderate |
| Memory Usage | Low | High | Moderate |
| Model Size | ~1GB | ~4GB | ~2GB |
| Zero-shot Performance | Good | Excellent | Excellent |

---

## Configuration Examples

### Using MedCLIP (default)
```yaml
model:
  patch_encoder: medclip
```

### Using Virchow2 for optimal performance
```yaml
model:
  patch_encoder: virchow2
```

### Using DinoV3 for cross-domain evaluation
```yaml
model:
  patch_encoder: dinov3
```

---

## How Encoders Are Used

1. **Prototype Feature Extraction** (Stage 1):
   - Selected encoder extracts features from prototype patches
   - Features are stored for k-means clustering
   
2. **K-means Clustering** (Stage 2):
   - Cluster center features based on selected encoder output
   - Encoder **must** be the same in training and inference

3. **Training** (Stage 3):
   - Encoder is frozen (not trained)
   - Used only for generating comparison features in contrastive loss

---

## Recommendation

### For Maximum Performance
Use **Virchow2** if:
- You have pathology data (WSIs, H&E slides)
- You can afford additional GPU memory (~4GB for model + 10GB for batch)
- You want state-of-the-art results

### For Speed/Memory Efficiency
Use **MedCLIP** if:
- You have limited GPU memory
- You need fast inference
- You're running on consumer GPUs

### For Research/Cross-Domain
Use **DinoV3** if:
- You want to evaluate on diverse tissue types
- You're doing research comparison with generic vision models
- You want self-supervised learning properties

---

## Switching Encoders

**Important**: If you switch encoders, you **must** regenerate prototypes and features:

1. Delete old features:
   ```bash
   rm -rf /data/pathology/projects/ivan/WSS/PBIP/image_features/*
   ```

2. Extract prototypes with new encoder:
   ```bash
   python features/extract_prototypes_from_gt.py --config work_dirs/custom_wsi_template.yaml
   ```

3. Extract MedCLIP features:
   ```bash
   python features/excract_medclip_proces.py --config work_dirs/custom_wsi_template.yaml
   ```

4. Cluster prototypes:
   ```bash
   python features/k_mean_cos_per_class.py --config work_dirs/custom_wsi_template.yaml
   ```

5. Train with new encoder:
   ```bash
   python train_stage_1.py --config work_dirs/custom_wsi_template.yaml --gpu 0
   ```

---

## Technical Details

### Feature Extraction Process

```
Input Patch (224×224, RGB)
    ↓
Encoder (MedCLIP / Virchow2 / DinoV3)
    ↓
Features (512-1280 dimensional)
    ↓
Normalized (L2 norm)
    ↓
K-means Clustering
    ↓
Cluster Centers (Prototypes)
```

### Encoder Loading

All encoders are loaded from:
- **MedCLIP**: Local installation
- **Virchow2**: Hugging Face Hub (`hf-hub:paige-ai/Virchow2`)
- **DinoV3**: Hugging Face Hub (`hf-hub:timm/vit_large_patch16_dinov3.lvd1689m`)

First download may take time but is cached locally.

---

## Troubleshooting

### "Unknown encoder: xxx"
Supported encoders: `medclip`, `virchow2`, `dinov3` (case-insensitive)

### "Module not found"
Install missing dependencies:
- MedCLIP: `pip install medclip`
- Virchow2/DinoV3: `pip install timm`

### GPU Memory Error with Virchow2
- Reduce batch size: `train.samples_per_gpu: 5` (from 10)
- Use gradient accumulation
- Enable mixed precision training (already enabled)

### Inconsistent Results After Switching Encoders
- You **must** regenerate prototypes and features
- See "Switching Encoders" section above

---

## Adding New Encoders

To add a new encoder (e.g., ViT-Base from timm):

1. Edit `utils/encoders.py`
2. Add new method to `EncoderFactory`:
   ```python
   @staticmethod
   def _create_vitbase(device: torch.device = None):
       encoder = timm.create_model("vit_base_patch16_224", pretrained=True)
       return encoder, 768  # ViT-Base has 768-dim output
   ```
3. Update `create_encoder()` method
4. Update `get_encoder_config()` with metadata
5. Update config YAML with new option

---

## Citation

If you use these encoders in research, please cite:

**MedCLIP**:
```bibtex
@article{wang2023medclip,
  title={MedCLIP: Contrastive Learning from Unlabeled Medical Images and Text},
  author={Wang, Zifeng and others},
  journal={arXiv:2112.02624},
  year={2021}
}
```

**Virchow2**:
```bibtex
@article{dong2024virchow,
  title={Virchow: A Million-slide Digital Pathology Foundation Model},
  author={Dong, Andrew and others},
  journal={arXiv:2404.23228},
  year={2024}
}
```

**DinoV3**:
```bibtex
@article{oquab2023dinov3,
  title={DINOv3: The Power of Self-Supervised Learning with Vision Transformers},
  author={Oquab, Maxime and others},
  journal={arXiv:2305.08243},
  year={2023}
}
```
