"""
Extract patch coordinates from WSIs and generate thumbnails from test set.

This script:
1. Samples coordinates from training WSIs (organized by class labels)
2. Saves coordinates in slide2vec format for feature extraction
3. Generates 10x10 patch area thumbnails from test set positive class samples
   - Finds areas with ≥30% tumor in ground truth
   - Exports both WSI image and GT mask thumbnails

Usage:
    python features/extract_patches.py --config work_dirs/custom_wsi_template.yaml \
        --num_per_wsi 1000 --num_wsis_per_class 5

This will extract up to 1000 patches per WSI, from up to 5 WSIs per class.
"""

import os
import argparse
import numpy as np
import cv2 as cv
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import random
import csv
import sys
import hashlib

# Import common utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.common import extract_patch_numpy, extract_patch_openslide

try:
    import openslide
    HAS_OPENSLIDE = True
except:
    from skimage import io
    HAS_OPENSLIDE = False

try:
    import wholeslidedata as wsd
    HAS_WSD = True
except:
    HAS_WSD = False


def generate_coordinates_uid(num_per_wsi, num_wsis_per_class, filenames):
    """
    Generate unique identifier for coordinate set.
    
    UID format: {num_per_wsi}_{num_wsis_per_class}_{file_hash}
    where file_hash is first 8 chars of SHA256 hash of sorted filenames
    
    Args:
        num_per_wsi: Max patches per WSI
        num_wsis_per_class: Max WSIs per class
        filenames: List of filenames in the coordinate set
    
    Returns:
        UID string (e.g., "1000_100_a3f7b2c1")
    """
    # Sort filenames for consistency
    sorted_names = sorted(filenames)
    names_str = "|".join(sorted_names)
    
    # Hash the sorted filenames
    file_hash = hashlib.sha256(names_str.encode()).hexdigest()[:8]
    
    uid = f"{num_per_wsi}_{num_wsis_per_class}_{file_hash}"
    return uid


def load_class_labels(labels_csv):
    """Load class labels from CSV."""
    class_labels_dict = {}
    with open(labels_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name_raw = row['image_name'].strip()
            image_key = os.path.splitext(os.path.basename(image_name_raw))[0]
            
            label_cols = [col for col in row.keys() if col != 'image_name']
            if len(label_cols) == 1:
                label_str = row[label_cols[0]].strip()
                if ',' in label_str:
                    labels = [int(x.strip()) for x in label_str.split(',')]
                else:
                    labels = [int(x.strip()) for x in label_str.split()]
            else:
                labels = [int(row[col].strip()) for col in label_cols]
            
            class_labels_dict[image_key] = labels
    
    return class_labels_dict


def load_coordinates(coord_path, coordinates_suffix='.npy'):
    """Load coordinates, handling both structured and simple arrays."""
    if coordinates_suffix == '.npy':
        data = np.load(coord_path, allow_pickle=True)
        # Check if it's a structured array (slide2vec format)
        if isinstance(data, np.ndarray) and data.dtype.names:
            x = data['x']
            y = data['y']
        else:
            # Simple (N, 2) array
            if len(data.shape) == 1:
                x = data[::2]
                y = data[1::2]
            else:
                x = data[:, 0]
                y = data[:, 1]
    else:
        data = np.loadtxt(coord_path)
        if len(data.shape) == 1:
            x = data[::2]
            y = data[1::2]
        else:
            x = data[:, 0]
            y = data[:, 1]
    
    return x, y


def save_coordinates_slide2vec(coords_x, coords_y, patch_size, tile_level, 
                                tile_size_resized, save_path, tile_size=None, 
                                resize_factor=None, wsi_names=None):
    """Save coordinates in slide2vec format (.npy with metadata)."""
    if tile_size is None:
        tile_size = tile_size_resized
    if resize_factor is None:
        resize_factor = np.ones_like(coords_x, dtype=float)
    if wsi_names is None:
        wsi_names = np.array([''] * len(coords_x))
    else:
        wsi_names = np.array(wsi_names, dtype=object)
    
    # Create structured array
    n_coords = len(coords_x)
    data = np.empty(n_coords, dtype=[
        ('x', np.int32),
        ('y', np.int32),
        ('tile_level', np.int32),
        ('tile_size_resized', np.int32),
        ('tile_size', np.int32),
        ('tile_size_lv0', np.int32),
        ('resize_factor', np.float32),
        ('wsi_name', 'O'),  # Object type for strings
    ])
    
    data['x'] = coords_x.astype(np.int32)
    data['y'] = coords_y.astype(np.int32)
    data['tile_level'] = np.full(n_coords, tile_level, dtype=np.int32)
    data['tile_size_resized'] = np.full(n_coords, tile_size_resized, dtype=np.int32)
    data['tile_size'] = np.full(n_coords, tile_size, dtype=np.int32)
    data['tile_size_lv0'] = np.full(n_coords, patch_size, dtype=np.int32)
    data['resize_factor'] = np.full(n_coords, resize_factor, dtype=np.float32)
    data['wsi_name'] = wsi_names
    
    np.save(save_path, data)


def find_tumor_regions(gt_mask, patch_size, grid_size=10, min_tumor_ratio=0.3):
    """
    Find 10x10 patch areas with at least 30% tumor coverage.
    
    Args:
        gt_mask: Ground truth mask (0=background, >0=tumor)
        patch_size: Size of each patch (e.g., 224)
        grid_size: Number of patches in grid (e.g., 10 for 10x10)
        min_tumor_ratio: Minimum ratio of tumor pixels in grid area
    
    Returns:
        List of (x, y) top-left corner coordinates for valid grid areas
    """
    grid_pixel_size = grid_size * patch_size
    h, w = gt_mask.shape[:2]   
    valid_regions = []
    
    # Slide window across mask
    for y in range(0, h - grid_pixel_size + 1, patch_size):  # Step by patch_size for efficiency
        for x in range(0, w - grid_pixel_size + 1, patch_size):
            # Extract grid region
            grid_region = gt_mask[y:y+grid_pixel_size, x:x+grid_pixel_size]
            
            # Calculate tumor ratio
            tumor_pixels = np.sum(grid_region > 0)
            total_pixels = grid_region.size
            tumor_ratio = tumor_pixels / total_pixels
            if tumor_ratio >= min_tumor_ratio:
                valid_regions.append((x, y))
    
    return valid_regions


def generate_test_thumbnails(cfg, num_samples=5):
    """
    Generate thumbnails from random test set positive class samples.
    
    For each sample:
    1. Find 10x10 patch area with ≥30% tumor
    2. Export WSI image thumbnail (100x100 px)
    3. Export GT mask thumbnail (100x100 px)
    """
    wsi_dir = cfg.dataset.wsi_dir
    gt_dir = cfg.dataset.gt_dir
    split_csv = cfg.dataset.split_csv
    labels_csv = cfg.dataset.labels_csv
    patch_size = getattr(cfg.dataset, 'patch_size', 224)
    use_openslide = getattr(cfg.dataset, 'use_openslide', HAS_OPENSLIDE)
    mask_suffix = getattr(cfg.dataset, 'mask_suffix', '.png')
    
    # Output directory
    work_dir = Path(cfg.work_dir)
    thumbnails_dir = work_dir / 'thumbnails' / 'test_samples'
    image_thumb_dir = thumbnails_dir / 'wsi_images'
    mask_thumb_dir = thumbnails_dir / 'gt_masks'
    image_thumb_dir.mkdir(parents=True, exist_ok=True)
    mask_thumb_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating thumbnails for {num_samples} test samples...")
    print(f"Output directory: {thumbnails_dir}")
    
    # Load class labels
    class_labels_dict = load_class_labels(labels_csv)
    
    # Load test split
    test_files = []
    with open(split_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'test' in row and row['test']:
                test_files.append(row['test'].strip())
    
    # Filter for positive class (tumor) - assumes binary classification where class 1 = tumor
    positive_files = []
    for filename in test_files:
        base_key = os.path.splitext(os.path.basename(filename))[0]
        if base_key not in class_labels_dict:
            continue
        
        labels = class_labels_dict[base_key]
        # Check if tumor class (index 1) is positive
        if isinstance(labels, list) and len(labels) > 1 and labels[1] == 1:
            positive_files.append(base_key)
        elif isinstance(labels, list) and len(labels) == 1 and labels[0] == 1:
            positive_files.append(base_key)
    
    print(f"Found {len(positive_files)} positive class test samples")
    
    if len(positive_files) == 0:
        print("No positive test samples found!")
        return
    
    # Randomly select num_samples
    selected_samples = random.sample(positive_files, min(num_samples, len(positive_files)))
    
    generated_count = 0
    for sample_idx, base_key in enumerate(selected_samples):
        wsi_path = os.path.join(wsi_dir, base_key + ".tif")
        mask_path = os.path.join(gt_dir, base_key + mask_suffix)
        
        if not os.path.exists(wsi_path):
            print(f"  Warning: WSI not found for {base_key}")
            continue
        
        if not os.path.exists(mask_path):
            print(f"  Warning: GT mask not found for {base_key}")
            continue
        
        # Load GT mask
        gt_mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"  Warning: Failed to load GT mask for {base_key}")
            continue
        
        # Find tumor regions
        grid_size=4
        tumor_regions = find_tumor_regions(gt_mask, patch_size, grid_size=grid_size, min_tumor_ratio=0.2)

        if len(tumor_regions) == 0:
            print(f"  Warning: No {grid_size} x {grid_size} tumor regions found for {base_key}")
            continue
        
        # Select first valid region (could randomize if desired)
        region_x, region_y = tumor_regions[0]
        grid_pixel_size = 10 * patch_size
        
        # Extract GT mask region
        mask_region = gt_mask[region_y:region_y+grid_pixel_size, 
                             region_x:region_x+grid_pixel_size]
        
        # Save GT mask thumbnail
        # thumb_size = 100
        # mask_thumbnail = cv.resize(mask_region, (thumb_size, thumb_size), 
        #                           interpolation=cv.INTER_NEAREST)
        mask_thumbnail = mask_region
        mask_thumb_path = mask_thumb_dir / f"{base_key}_sample{sample_idx}.png"
        cv.imwrite(str(mask_thumb_path), mask_thumbnail)
        
        # Extract WSI image region
        if use_openslide and HAS_OPENSLIDE:
            slide = openslide.OpenSlide(wsi_path)
            wsi_region = slide.read_region((region_x, region_y), 0, 
                                          (grid_pixel_size, grid_pixel_size))
            slide.close()
            wsi_region = np.array(wsi_region.convert('RGB'))
        else:
            wsi = io.imread(wsi_path)
            if len(wsi.shape) == 2:
                wsi = cv.cvtColor(wsi, cv.COLOR_GRAY2RGB)
            elif wsi.shape[2] == 4:
                wsi = cv.cvtColor(wsi, cv.COLOR_RGBA2RGB)
            
            wsi_region = wsi[region_y:region_y+grid_pixel_size,
                            region_x:region_x+grid_pixel_size]
        
        # Save WSI image thumbnail
        # wsi_thumbnail = cv.resize(wsi_region, (thumb_size, thumb_size),
        wsi_thumbnail = wsi_region
        img_thumb_path = image_thumb_dir / f"{base_key}_sample{sample_idx}.png"
        cv.imwrite(str(img_thumb_path), cv.cvtColor(wsi_thumbnail, cv.COLOR_RGB2BGR))
        
        generated_count += 1
        print(f"  ✓ Generated thumbnails for {base_key} (region at {region_x}, {region_y})")
    
    print(f"\n✓ Generated {generated_count} thumbnail pairs")
    print(f"  WSI images: {image_thumb_dir}")
    print(f"  GT masks: {mask_thumb_dir}")


def extract_patches(cfg, num_per_wsi=1000, num_wsis_per_class=20, seed=42):
    """Extract patch coordinates from WSIs using image-level labels.
    
    Args:
        cfg: OmegaConf configuration
        num_per_wsi: Max number of patches to extract from each WSI
        num_wsis_per_class: Max number of WSIs to process per class
        seed: Random seed for reproducibility
    """
    
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    wsi_dir = cfg.dataset.wsi_dir
    coordinates_dir = cfg.dataset.coordinates_dir
    split_csv = cfg.dataset.split_csv
    labels_csv = cfg.dataset.labels_csv
    patch_size = getattr(cfg.dataset, 'patch_size', 224)
    use_openslide = getattr(cfg.dataset, 'use_openslide', HAS_OPENSLIDE)
    coordinates_suffix = getattr(cfg.dataset, 'coordinates_suffix', '.npy')
    
    # Output directory
    work_dir = Path(cfg.work_dir)
    proto_coords_dir = work_dir / 'prototype_coordinates'
    class_order = list(getattr(cfg.dataset, 'class_order', ['benign', 'tumor']))
    
    proto_coords_dir.mkdir(parents=True, exist_ok=True)
    
    # Load class labels
    print(f"Loading class labels from {labels_csv}")
    class_labels_dict = load_class_labels(labels_csv)
    
    # Load train split (use training data for prototypes)
    train_files = []
    with open(split_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'train' in row and row['train']:
                train_files.append(row['train'].strip())
    
    print(f"Found {len(train_files)} training files")
    
    # Organize files by class
    files_by_class = {i: [] for i in range(len(class_order))}
    for filename in train_files:
        base_key = os.path.splitext(os.path.basename(filename))[0]
        
        if base_key not in class_labels_dict:
            continue
        
        labels = class_labels_dict[base_key]
        
        # Assign to class based on label value
        if isinstance(labels, (int, float)):
            class_idx = int(labels)
            if 0 <= class_idx < len(class_order):
                files_by_class[class_idx].append(base_key)
        elif len(labels) == 1:
            class_idx = int(labels[0])
            if 0 <= class_idx < len(class_order):
                files_by_class[class_idx].append(base_key)
        else:
            # Multi-label: assign to all classes where label==1
            for class_idx, label in enumerate(labels):
                if label == 1 and class_idx < len(class_order):
                    files_by_class[class_idx].append(base_key)
    
    print(f"Files per class: {[len(files_by_class[i]) for i in range(len(class_order))]}")
    for i, class_name in enumerate(class_order):
        print(f"  {class_name}: {len(files_by_class[i])} files")
    
    # Extract patches from WSIs
    wsis_processed_per_class = {cls: 0 for cls in class_order}
    proto_coords_by_class = {cls: {"x": [], "y": [], "wsi": [], "base_keys": []} for cls in class_order}
    
    for class_idx, class_name in enumerate(class_order):
        print(f"\nExtracting {class_name} patches from up to {num_wsis_per_class} WSIs...")
        print(f"  Max {num_per_wsi} patches per WSI")
        
        files = files_by_class[class_idx]
        if not files:
            print(f"  No files labeled as {class_name}")
            continue
        
        # Shuffle for diversity (seed already set at function start)
        random.shuffle(files)
        
        # Create progress bar that tracks WSIs processed
        with tqdm(total=min(num_wsis_per_class, len(files)), desc=f"{class_name} WSIs", unit="wsi") as pbar_wsi:
            for base_key in files:
                if wsis_processed_per_class[class_name] >= num_wsis_per_class:
                    break
                
                # Paths
                wsi_path = os.path.join(wsi_dir, base_key + ".tif")
                coord_path = os.path.join(coordinates_dir, base_key + coordinates_suffix)
                
                if not os.path.exists(wsi_path) or not os.path.exists(coord_path):
                    continue
                
                # Load coordinates
                try:
                    coords_x, coords_y = load_coordinates(coord_path, coordinates_suffix)
                except Exception as e:
                    print(f"  Error loading coordinates from {coord_path}: {e}")
                    continue
                
                if len(coords_x) == 0:
                    continue
                
                # Select up to num_per_wsi coordinates from this WSI (limit per WSI)
                n_samples = min(num_per_wsi, len(coords_x))
                print(f"  Sampling {n_samples} patches from {len(coords_x)}")
                sampled_indices = np.random.choice(len(coords_x), n_samples, replace=False)
                
                # Store coordinates (patches extracted on-the-fly when needed)
                for idx in sampled_indices:
                    cx, cy = int(coords_x[idx]), int(coords_y[idx])
                    proto_coords_by_class[class_name]["x"].append(cx)
                    proto_coords_by_class[class_name]["y"].append(cy)
                    proto_coords_by_class[class_name]["wsi"].append(base_key + ".tif")
                
                # Track base_key for UID generation
                proto_coords_by_class[class_name]["base_keys"].append(base_key)
                
                wsis_processed_per_class[class_name] += 1
                pbar_wsi.update(1)
    
    # Generate UID from parameters and actual filenames used
    print("\nGenerating coordinate identifiers...")
    
    # Save prototype coordinates in slide2vec format for each class
    for class_name in class_order:
        if len(proto_coords_by_class[class_name]["x"]) > 0:
            coords_x = np.array(proto_coords_by_class[class_name]["x"])
            coords_y = np.array(proto_coords_by_class[class_name]["y"])
            wsi_names = proto_coords_by_class[class_name]["wsi"]
            base_keys = proto_coords_by_class[class_name]["base_keys"]
            
            # Generate UID based on parameters and filenames
            uid = generate_coordinates_uid(num_per_wsi, num_wsis_per_class, base_keys)
            
            # Save with UID in filename
            coord_save_path = proto_coords_dir / f"{class_name}_{uid}.npy"
            save_coordinates_slide2vec(coords_x, coords_y, patch_size, 0, patch_size, 
                                      str(coord_save_path), wsi_names=wsi_names)
            print(f"Saved {class_name} prototype coordinates to {coord_save_path}")
            print(f"  UID: {uid}")
    
    print("\n=== Coordinate Extraction Complete ===")
    for class_name in class_order:
        n_coords = len(proto_coords_by_class[class_name]["x"])
        n_wsis = wsis_processed_per_class[class_name]
        print(f"{class_name}: {n_wsis} WSIs → {n_coords} coordinates")
    print(f"\nPrototype coordinates saved to: {proto_coords_dir}")
    print(f"Patches will be extracted on-the-fly during MedCLIP feature extraction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--num_per_wsi', type=int, default=1000, 
                       help='Max number of patches per WSI')
    parser.add_argument('--num_wsis_per_class', type=int, default=20,
                       help='Max number of WSIs to process per class')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--generate_thumbnails', action='store_true',
                       help='Generate test set thumbnails')
    parser.add_argument('--num_thumbnail_samples', type=int, default=5,
                       help='Number of test samples for thumbnail generation')
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    
    # Extract patches from training set
    extract_patches(cfg, args.num_per_wsi, args.num_wsis_per_class, seed=args.seed)
    
    # Generate test thumbnails if requested
    if args.generate_thumbnails:
        generate_test_thumbnails(cfg, args.num_thumbnail_samples)

