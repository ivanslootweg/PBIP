# Next Steps: Train Deeplab-v2 on CAMs

Goal: use Stage 1 CAMs as pseudo-labels to train a Deeplab-v2 vision(-language) model, as described in the paper (arXiv:2503.12068).

## Prerequisites
- Stage 1 finished with high-quality CAMs saved in `output_dirs.pred_dir/*` plus slide/patch metadata.
- Class names and UID used for the run are recorded (for consistent label mapping).
- Access to Deeplab-v2 weights/checkpoints and training code (vision or vision-language variant specified by the paper).
- Adequate GPU budget (paper trains at high resolution; plan for multi-GPU/Distributed Data Parallel).

## Data preparation from CAMs
1) **Run full-slide inference** with the trained Stage 1 checkpoint to generate CAMs for all slides of interest; store confidence maps and logits if available.
2) **Select confident regions**: threshold CAMs per class; optionally apply morphological cleanup (open/close) and remove tiny components below an area cutoff.
3) **Attach coordinates**: keep slide/patch origin, spacing, magnification, and class ID alongside each mask.
4) **Convert to training format** expected by Deeplab-v2 codebase:
   - If segmentation: export binary/instance masks plus polygons; keep class labels.
   - If detection-style supervision: export boxes derived from CAM connected components.
   - If vision-language supervision: pair each region with text prompts/class names per paper.
5) **Create splits**: train/val/test JSON(L) manifests with paths to images, masks/boxes, and metadata. Balance classes if CAM density is uneven.

## Training Deeplab-v2 (high-level checklist)
- **Model init**: load the Deeplab-v2 variant used in the paper (e.g., vision-only vs VLM). Freeze or finetune text tower per paper guidance.
- **Input resolution & patching**: match CAM resolution; decide on tiling vs full-slide crops; enable multi-scale if supported.
- **Augmentations**: flips/rotations/color jitter; avoid heavy geometry that would desync masks.
- **Losses**: use the paper’s objective (likely segmentation/detection losses) and any CAM-specific regularizers; keep class weights consistent with Stage 1.
- **Optimization**: set LR, warmup, batch size, and total steps to mirror the paper; use DDP with gradient accumulation if memory-bound.
- **Validation**: run on held-out slides; report metrics the paper uses (e.g., mIoU, Dice, AUC) for comparability.

## Suggested work plan
1) Export CAM dataset (steps above) and verify a small sample visually.
2) Implement a converter from CAM outputs to the Deeplab-v2 training format (masks/boxes + manifest files).
3) Draft Deeplab-v2 training config mirroring the paper; run a short sanity-training to ensure loss decreases.
4) Full-scale training; log metrics and keep checkpoints.
5) Evaluate on the held-out set and compare to paper baselines; adjust thresholds/augmentations if underperforming.

## Open items (needs paper alignment)
- Exact Deeplab-v2 variant, input resolution, and loss functions.
- Confidence thresholding and post-processing used on CAMs.
- Any text supervision or prompts paired with CAM regions.

Document owners can fill these with specifics after aligning with the paper or released Deeplab-v2 training scripts.
