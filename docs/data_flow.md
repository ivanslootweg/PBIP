# Data Flow

End-to-end path from raw assets to predictions.

1) Inputs
- Whole-slide images (WSI) and coordinate files (.npy/.txt) per slide
- Split CSV with train/val/test filenames
- Labels CSV with image-level labels (multi-class or binary)
- Optional GT masks for val/test (pixel labels)

2) Patch coordinate selection (train)
- Script: features/extract_patches.py
- For each class, sample up to `num_wsis_per_class` WSIs
- From each WSI, randomly sample up to `num_per_wsi` coordinates (seeded)
- Save slide2vec-formatted coordinate npy per class with UID: `{class}_{num_per_wsi}_{num_wsis_per_class}_{hash}.npy`

3) Prototype feature extraction (MedCLIP)
- Script: features/excract_medclip_proces.py
- Loads UID-tagged coordinates; extracts patch embeddings
- Saves features to `{features.save_dir}/medclip_exemplars_{uid}.pkl`

4) Label feature clustering (k-means)
- Script: features/k_mean_cos_per_class.py
- Consumes MedCLIP exemplar file; clusters per class with k_list
- Saves label prototypes to `{features.save_dir}/label_fea_pro_{uid}.pkl`

5) Training (stage 1)
- Script: train_stage_1.py
- Loads patch encoder (medclip/virchow2/dinov3)
- Loads label prototypes (UID-aware lookup)
- Trains classifier + CAM generation, saves ckpts/preds under output_dirs

6) Outputs
- Checkpoints: `output_dirs.ckpt_dir/<timestamp>/best_cam.pth`
- Predictions/CAMs: `output_dirs.pred_dir/*.png`
- Logs: `output_dirs.train_log_dir`
- Cached patch features & thumbnails under `work_dir`
