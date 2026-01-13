# Data Loading & Label Selection

How data is discovered, validated, and labeled for PBIP.

## Inputs
- Split CSV: columns `train`, `val`, `test` with filenames (can be full paths or basenames)
- Labels CSV: `image_name` + one or more label columns (binary or multi-class)
- WSI directory: `.tif/.tiff` slides
- Coordinates directory: `.npy` or `.txt` files (slide2vec or plain x,y)
- Optional GT masks: for val/test pixel supervision

## Split ingestion
- File: `features/extract_patches.py` and `datasets/wsi_dataset.py`
- For each split column, filenames are read; missing entries are skipped
- Filenames are normalized to basenames (without extension) for lookup

## Label resolution
- `labels_csv` is loaded into a dict keyed by basename (no extension)
- Supports:
  - Single-column binary/multiclass (value or space/comma-delimited list)
  - Multi-column one-hot (each class column is 0/1)
- During training split load, samples without labels are skipped and reported

## Coordinate & asset checks
- For train: require WSI + coordinates + label
- For val/test: require WSI + coordinates + GT mask + label
- Missing assets are counted and reported per reason (no_wsi, no_coords, no_label, no_gt)

## Patch sampling (train)
- Per-class file lists built from labels
- For each class: shuffle files (seeded) and take up to `num_wsis_per_class`
- Per WSI: randomly sample up to `num_per_wsi` coordinates without replacement
- Saved as slide2vec npy with UID: `{class}_{num_per_wsi}_{num_wsis_per_class}_{hash}.npy`

## GT mask usage (val/test)
- Masks loaded from `dataset.gt_dir` with `mask_suffix`
- Required for evaluation and thumbnail generation
- For visualisation: Thumbnail sampling finds 10x10 (configurable) patch grids with ≥30% tumor in 5 test slides with positive slide label

## Caching & reproducibility
- Seeds applied in extract_patches (`np.random` + `random`)
- UID encodes sampling params + file hash; downstream features reuse the UID
- MedCLIP and label feature files inherit the same UID to keep artifacts aligned
