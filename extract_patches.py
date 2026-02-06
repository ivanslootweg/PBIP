import argparse
import pandas as pd
import torch
import numpy as np
import openslide
from tqdm import tqdm
import pickle as pkl
import os
import yaml
import h5py
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2 as cv
from utils.common import load_coordinates

def get_transform():
    MEAN = [0.66791496, 0.47791372, 0.70623304]
    STD = [0.1736589, 0.22564577, 0.19820057]
    
    transform = A.Compose([
        A.Normalize(MEAN, STD),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(),
        ToTensorV2(transpose_mask=True),
    ])
    return transform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Output guard: skip if features already exist
    work_dir = config.get('work_dir', './work_dirs')
    experiment_name = config.get('experiment_name', 'default_experiment')
    features_dir = os.path.join(work_dir, experiment_name, 'features')
    os.makedirs(features_dir, exist_ok=True)
    save_path = os.path.join(features_dir, "prototype_features.pkl")
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"Features already exist at {save_path}. Skipping extraction.")
        return

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model = model.to(device)
    model.eval()
    
    transform = get_transform()
    
    # Config/Columns
    col_map = {
        'image_name': 'image_name',
        'wsi_label': 'wsi_label',
        'coords': 'attention_coordinates_path',
        'scores': 'attention_scores_path',
        'split': 'split'
    }
    if 'columns' in config:
        col_map.update(config['columns'])
        
    wsi_dir = config.get('wsi_dir')
    wsi_ext = config.get('wsi_extension', '.tif')

    binary_mode = config.get('binary_mode', True)
    thresholds = config.get('prototype_thresholds', [0.005, 0.1])
    class_names = config.get('class_names', ["Negative", "Positive"])

    # Read CSV
    df = pd.read_csv(config['data_csv_path'])
    if col_map['split'] in df.columns:
        df = df[df[col_map['split']] == 'train'] # Only train set
    
    # Limit slides if configured
    max_slides_per_class = config.get('max_slides_per_class', None)
    if max_slides_per_class:
        # Group by class and sample
        df_limited = []
        for class_idx, class_name in enumerate(class_names):
            class_df = df[df[col_map['wsi_label']] == class_idx]
            if len(class_df) > max_slides_per_class:
                class_df = class_df.sample(n=max_slides_per_class, random_state=42)
            df_limited.append(class_df)
        df = pd.concat(df_limited).reset_index(drop=True)
        print(f"Limited to max {max_slides_per_class} slides per class: {len(df)} total slides")
    
    features_dict = {c: [] for c in class_names}
    
    # Tracking statistics
    stats = {
        'slides_processed': 0,
        'slides_skipped': 0,
        'patches_per_class': {c: 0 for c in class_names},
        'wsis_per_class': {c: 0 for c in class_names}
    }
    
    print(f"\n=== Starting Prototype Extraction ===")
    print(f"Total slides in train split: {len(df)}")
    print(f"Processing slides...\n")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing WSIs"):
        # Path Construction
        fname = str(row[col_map['image_name']])
        
        col_wsi_path = config.get('columns', {}).get('wsi_path', 'wsi_path')
        wsi_path = None
        if col_wsi_path in row and pd.notna(row[col_wsi_path]):
            wsi_path = str(row[col_wsi_path])
            
        if not wsi_path or not os.path.exists(wsi_path):
            if wsi_dir:
                name_part = fname
                if not name_part.lower().endswith(tuple(['.tif', '.svs', '.ndpi', '.mrxs', '.tiff'])):
                    name_part += wsi_ext
                wsi_path = os.path.join(wsi_dir, name_part)
            else:
                if not wsi_path: wsi_path = fname

        if not wsi_path or not os.path.exists(wsi_path):
             stats['slides_skipped'] += 1
             continue

        wsi_label = row[col_map['wsi_label']]
        coords_path = row[col_map['coords']]
        attn_path = row[col_map['scores']]
        
        # Skip if coords or attention files don't exist
        if not os.path.exists(coords_path):
            stats['slides_skipped'] += 1
            continue
            
        if not os.path.exists(attn_path):
            stats['slides_skipped'] += 1
            continue
        
        # Load Coords
        coords = load_coordinates(coords_path)
        if len(coords) == 0:
            stats['slides_skipped'] += 1
            continue
            
        # Load Attn
        try:
            attn_data = torch.load(attn_path, map_location='cpu', weights_only=False)
            attn = None
            if isinstance(attn_data, dict):
                if 'patch_logits' in attn_data:
                    attn = attn_data['patch_logits']
                elif 'logits' in attn_data:
                    attn = attn_data['logits']
                else:
                    for k, v in attn_data.items():
                        if isinstance(v, torch.Tensor) and len(v) == len(coords):
                            attn = v; break
            elif isinstance(attn_data, torch.Tensor):
                attn = attn_data
            elif isinstance(attn_data, np.ndarray):
                attn = attn_data
                
            if isinstance(attn, torch.Tensor): attn = attn.numpy()
            if attn is None:
                stats['slides_skipped'] += 1
                continue
        except Exception as e:
            stats['slides_skipped'] += 1
            continue
        
        # Ensure coords and attention match
        if len(coords) != len(attn):
            min_len = min(len(coords), len(attn))
            coords = coords[:min_len]
            attn = attn[:min_len]
            
        # Select Logic
        selected_indices = []
        target_class_idx = -1 

        if binary_mode:
            # Binary mode: use class-1 scores for both positive and negative selection
            if len(attn.shape) > 1:
                scores_pos = attn[:, 1] if attn.shape[1] > 1 else attn[:, 0]
            else:
                scores_pos = attn

            if wsi_label == 1:  # Positive Slide
                target_class_idx = 1
                k = max(1, int(len(scores_pos) * thresholds[0]))
                selected_indices = np.argsort(scores_pos)[-k:]
                print(f"  [Positive] {fname}: {len(selected_indices)} patches from {len(coords)} (top {thresholds[0]*100:.2f}%)")
            else:  # Negative Slide (hard negatives)
                target_class_idx = 0
                k = max(1, int(len(scores_pos) * thresholds[1]))
                selected_indices = np.argsort(scores_pos)[-k:]
                print(f"  [Negative] {fname}: {len(selected_indices)} patches from {len(coords)} (top {thresholds[1]*100:.1f}%)")
        else:
            # Multiclass mode
            target_class_idx = int(wsi_label)
            if target_class_idx < len(class_names) and len(attn.shape) > 1:
                if target_class_idx == 0:
                    # Class 0: select lowest class-0 scores (hard negatives)
                    scores = attn[:, 0]
                    k = max(1, int(len(scores) * thresholds[1]))
                    selected_indices = np.argsort(scores)[:k]
                else:
                    # Class > 0: select top scores for that class
                    scores = attn[:, target_class_idx]
                    k = max(1, int(len(scores) * thresholds[0]))
                    selected_indices = np.argsort(scores)[-k:]

        # Cap patches per slide if configured
        max_patches_per_slide = config.get('max_patches_per_slide', None)
        if max_patches_per_slide and len(selected_indices) > max_patches_per_slide:
            # Randomly sample to keep runtime bounded
            selected_indices = np.random.choice(
                selected_indices, size=max_patches_per_slide, replace=False
            )
        
        # Extract and Encode
        if len(selected_indices) == 0:
            stats['slides_skipped'] += 1
            continue
            
        try:
            slide = openslide.OpenSlide(wsi_path)
        except Exception as e:
            stats['slides_skipped'] += 1
            continue

        target_class_name = class_names[target_class_idx]
        patches_extracted = 0

        for p_idx in selected_indices:
            if p_idx >= len(coords):
                continue
            c = coords[p_idx]
            x, y = int(c[0]), int(c[1])
            
            try:
                patch = slide.read_region((x, y), 0, (224, 224)).convert("RGB")
                patch_np = np.array(patch)
                
                if transform:
                    patch_t = transform(image=patch_np)["image"]
                    patch_t = (patch_t - patch_t.min()) / (patch_t.max() - patch_t.min() + 1e-8)
                
                with torch.no_grad():
                    features = model.vision_model(patch_t.unsqueeze(0).to(device))
                    features = features.cpu().numpy()
                
                # Store features with WSI-agnostic id or include path?
                # We use basename in visualization.
                base = os.path.basename(fname) # Use original fname from CSV
                features_dict[target_class_name].append({
                    'name': f"{base}_{x}_{y}",
                    'features': features
                })
                patches_extracted += 1
            except Exception as e:
                pass
        
        # Update statistics
        if patches_extracted > 0:
            stats['slides_processed'] += 1
            stats['patches_per_class'][target_class_name] += patches_extracted
            stats['wsis_per_class'][target_class_name] += 1
        else:
            stats['slides_skipped'] += 1
        
    # Save to experiment-specific features directory
    work_dir = config.get('work_dir', './work_dirs')
    experiment_name = config.get('experiment_name', 'default_experiment')
    features_dir = os.path.join(work_dir, experiment_name, 'features')
    os.makedirs(features_dir, exist_ok=True)
    save_path = os.path.join(features_dir, "prototype_features.pkl")
    
    with open(save_path, 'wb') as f:
        pkl.dump(features_dict, f)
    
    # Print final statistics
    print(f"\n=== Extraction Complete ===")
    print(f"Slides processed: {stats['slides_processed']}")
    print(f"Slides skipped: {stats['slides_skipped']}")
    print(f"\nPatches extracted per class:")
    for class_name in class_names:
        n_patches = stats['patches_per_class'][class_name]
        n_wsis = stats['wsis_per_class'][class_name]
        print(f"  {class_name}: {n_wsis} WSIs → {n_patches} patches")
    print(f"\nFeatures saved to: {save_path}")

if __name__ == "__main__":
    main()
