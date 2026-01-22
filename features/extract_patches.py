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
import h5py
import torch

# Import common utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.common import extract_patch_numpy, extract_patch_openslide
from utils.pseudo_labels import PseudoLabelLoader, PatchSelector
from utils.pyutils import build_uid_from_config

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


def find_wsi_file(wsi_dir, base_key):
    """Find WSI file with any common extension."""
    wsi_extensions = ['.tif', '.tiff', '.svs', '.ndpi', '.mrxs', '.vms', '.vmu', '.scn', '.bif', '.qptiff']
    
    for ext in wsi_extensions:
        wsi_path = os.path.join(wsi_dir, base_key + ext)
        if os.path.exists(wsi_path):
            return wsi_path
    
    # Try without extension (some systems)
    wsi_path = os.path.join(wsi_dir, base_key)
    if os.path.exists(wsi_path):
        return wsi_path
    
    return None


def find_coordinate_file(coordinates_dir, base_key, coordinates_suffix):
    """Find coordinate file with flexible naming."""
    # Try exact match first
    coord_path = os.path.join(coordinates_dir, base_key + coordinates_suffix)
    if os.path.exists(coord_path):
        return coord_path
    
    # Try with _patches suffix for h5 files
    if coordinates_suffix == '.h5' or coordinates_suffix.endswith('.h5'):
        coord_path = os.path.join(coordinates_dir, base_key + "_patches.h5")
        if os.path.exists(coord_path):
            return coord_path
    
    # Try other common suffixes
    alt_suffixes = ['.npy', '.npz', '.h5', '.hdf5', '_patches.h5', '.txt']
    for suf in alt_suffixes:
        if suf != coordinates_suffix:
            coord_path = os.path.join(coordinates_dir, base_key + suf)
            if os.path.exists(coord_path):
                return coord_path
    
    return None


def _match_coordinates(coords_x, coords_y, pseudo_coords_x, pseudo_coords_y):
    """Find coordinate intersections between main coords and pseudo-label coords.

    Returns two index lists of equal length: indices into the main coordinate
    arrays and the corresponding indices into the pseudo-label coordinates.
    """
    pseudo_map = {}
    for j, (px, py) in enumerate(zip(pseudo_coords_x, pseudo_coords_y)):
        pseudo_map.setdefault((int(px), int(py)), []).append(j)

    idx_main, idx_pseudo = [], []
    for i, (cx, cy) in enumerate(zip(coords_x, coords_y)):
        key = (int(cx), int(cy))
        if key in pseudo_map and pseudo_map[key]:
            j = pseudo_map[key].pop()
            idx_main.append(i)
            idx_pseudo.append(j)
            if not pseudo_map[key]:
                del pseudo_map[key]
    return idx_main, idx_pseudo


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
    """Load coordinates, handling .npy, .txt, and .h5 files."""
    if coordinates_suffix.endswith('.h5'):
        # Load from HDF5 file
        with h5py.File(coord_path, 'r') as f:
            # Try common keys
            if 'coords' in f:
                data = f['coords'][:]
            elif 'coordinates' in f:
                data = f['coordinates'][:]
            else:
                # Use first dataset found
                keys = list(f.keys())
                if len(keys) == 0:
                    raise ValueError(f"No datasets found in {coord_path}")
                data = f[keys[0]][:]
        
        # Extract x, y from (N, 2) array
        if len(data.shape) == 2 and data.shape[1] == 2:
            x = data[:, 0]
            y = data[:, 1]
        else:
            raise ValueError(f"Unexpected shape {data.shape} in h5 file {coord_path}. Expected (N, 2)")
    elif coordinates_suffix.endswith('.npy'):
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
        wsi_path = find_wsi_file(wsi_dir, base_key)
        mask_path = os.path.join(gt_dir, base_key + mask_suffix)
        
        if not wsi_path:
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


def extract_patches(cfg, seed=42):
    """Extract patch coordinates from WSIs using image-level labels.
    
    Args:
        cfg: OmegaConf configuration
        seed: Random seed for reproducibility
    """
    
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Read prototype extraction parameters from config
    n_slides_per_class_for_prototypes = getattr(cfg.features, 'n_slides_per_class_for_prototypes', 20)
    selection_method = getattr(cfg.features, 'selection_method', 'top_attention')
    
    print(f"\n=== Prototype Extraction Configuration ===")
    print(f"Max slides per class: {n_slides_per_class_for_prototypes}")
    print(f"Selection method: {selection_method}")
    print(f"Random seed: {seed}")
    
    wsi_dir = cfg.dataset.wsi_dir
    coordinates_dir = cfg.dataset.coordinates_dir
    split_csv = cfg.dataset.split_csv
    labels_csv = cfg.dataset.labels_csv
    patch_size = getattr(cfg.dataset, 'patch_size', 224)
    use_openslide = getattr(cfg.dataset, 'use_openslide', HAS_OPENSLIDE)
    coordinates_suffix = getattr(cfg.dataset, 'coordinates_suffix', '.npy')
    
    # Optional pseudo-label coordinate files (used when pseudo-label counts mismatch coordinates)
    pseudo_label_coord_dir = getattr(cfg.dataset, 'pseudo_label_coordinates_dir', None)
    pseudo_label_coord_suffix = getattr(cfg.dataset, 'pseudo_label_coordinates_suffix', coordinates_suffix)

    # Fallback directories
    fallback_wsi_dir = getattr(cfg.dataset, 'fallback_wsi_dir', None)
    fallback_coordinates_dir = getattr(cfg.dataset, 'fallback_coordinates_dir', None)
    fallback_coordinates_suffix = getattr(cfg.dataset, 'fallback_coordinates_suffix', coordinates_suffix)
    
    # Pseudo-label configuration
    use_pseudo_labels = getattr(cfg.dataset, 'use_pseudo_labels', False)
    pseudo_label_dir = getattr(cfg.dataset, 'pseudo_label_dir', None)
    binary_mode = getattr(cfg.dataset, 'binary_mode', True)
    prototype_selection_strategy = getattr(cfg.dataset, 'prototype_selection_strategy', 'percentile')
    pseudo_label_confidence_threshold = getattr(cfg.dataset, 'pseudo_label_confidence_threshold', 0.85)
    prototype_min_patches = getattr(cfg.dataset, 'prototype_min_patches', 5)
    
    # Per-class thresholds (optional) - can be list [class0_thresh, class1_thresh, ...] or dict {0: thresh0, 1: thresh1}
    threshold_for_prototype = getattr(cfg.dataset, 'threshold_for_prototype', None)
    # Convert OmegaConf containers to native Python types
    if threshold_for_prototype is not None:
        threshold_for_prototype = OmegaConf.to_container(threshold_for_prototype, resolve=True)
    
    # Classes that require pseudo-labels (e.g., [1] for tumor only, None for all classes)
    # For benign slides, we don't need pseudo-labels as all patches are class 0
    prototype_label_required_classes = getattr(cfg.dataset, 'prototype_label_required_classes', None)
    if prototype_label_required_classes is not None:
        prototype_label_required_classes = OmegaConf.to_container(prototype_label_required_classes, resolve=True)
    
    # Output directory
    work_dir = Path(cfg.work_dir)
    proto_coords_dir = work_dir / 'prototype_coordinates'
    class_order = list(getattr(cfg.dataset, 'class_order', ['benign', 'tumor']))
    
    # Initialize pseudo-label loader if enabled
    pseudo_loader = None
    pseudo_selector = None
    if use_pseudo_labels and pseudo_label_dir:
        print(f"\n=== Attention Score Configuration ===")
        print(f"Attention scores directory: {pseudo_label_dir}")
        print(f"Binary mode: {binary_mode}")
        print(f"Selection strategy: {prototype_selection_strategy}")
        
        # Display threshold configuration
        if threshold_for_prototype is not None:
            print(f"Per-class confidence thresholds:")
            if isinstance(threshold_for_prototype, dict):
                for class_id, thresh in threshold_for_prototype.items():
                    class_name = class_order[class_id] if class_id < len(class_order) else f"class_{class_id}"
                    print(f"  {class_name}: {thresh}")
            elif isinstance(threshold_for_prototype, (list, tuple)):
                for class_id, thresh in enumerate(threshold_for_prototype):
                    class_name = class_order[class_id] if class_id < len(class_order) else f"class_{class_id}"
                    print(f"  {class_name}: {thresh}")
        else:
            print(f"Confidence threshold (global): {pseudo_label_confidence_threshold}")
        
        print(f"Min patches per WSI: {prototype_min_patches}")
        
        if prototype_label_required_classes is not None:
            required_class_names = [class_order[c] if c < len(class_order) else f"class_{c}" for c in prototype_label_required_classes]
            print(f"Attention scores required only for: {required_class_names}")
            other_classes = [i for i in range(len(class_order)) if i not in prototype_label_required_classes]
            if other_classes:
                other_class_names = [class_order[c] if c < len(class_order) else f"class_{c}" for c in other_classes]
                print(f"Random sampling for: {other_class_names}")
        else:
            print(f"Attention scores required for all classes")
        
        try:
            pseudo_loader = PseudoLabelLoader(
                pseudo_label_dir=pseudo_label_dir,
                binary_mode=binary_mode,
                num_classes=len(class_order)
            )
            pseudo_selector = PatchSelector(
                num_classes=len(class_order),
                selection_strategy=prototype_selection_strategy
            )
            print(f"✓ Attention score loader initialized")
        except Exception as e:
            print(f"✗ Error: Failed to initialize attention score loader: {e}")
            print(f"  Make sure attention score files exist in: {pseudo_label_dir}")
            raise
    
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
    
    # Filter out files without pseudo-labels (attention scores) if required
    if use_pseudo_labels and pseudo_label_dir:
        print(f"\nFiltering files based on pseudo-label availability...")
        files_by_class_filtered = {i: [] for i in range(len(class_order))}
        total_excluded = 0
        
        for class_idx, class_name in enumerate(class_order):
            # Check if this class requires pseudo-labels
            class_needs_pseudo_labels = True
            if prototype_label_required_classes is not None:
                class_needs_pseudo_labels = class_idx in prototype_label_required_classes
            
            if not class_needs_pseudo_labels:
                # Keep all files for classes that don't need pseudo-labels
                files_by_class_filtered[class_idx] = files_by_class[class_idx]
                print(f"  {class_name}: Keeping all {len(files_by_class[class_idx])} files (pseudo-labels not required)")
                continue
            
            # Filter files with pseudo-labels
            for base_key in files_by_class[class_idx]:
                # Check if pseudo-label file exists
                pseudo_label_extensions = ['.pt', '.pth', '.npy', '.npz', '.pkl']
                pseudo_label_found = False
                
                for ext in pseudo_label_extensions:
                    pseudo_label_path = os.path.join(pseudo_label_dir, base_key + ext)
                    if os.path.exists(pseudo_label_path):
                        pseudo_label_found = True
                        break
                
                if pseudo_label_found:
                    files_by_class_filtered[class_idx].append(base_key)
                else:
                    total_excluded += 1
            
            excluded_count = len(files_by_class[class_idx]) - len(files_by_class_filtered[class_idx])
            print(f"  {class_name}: {len(files_by_class_filtered[class_idx])}/{len(files_by_class[class_idx])} files (excluded {excluded_count} without pseudo-labels)")
        
        files_by_class = files_by_class_filtered
        if total_excluded > 0:
            print(f"\n  ⚠ Total excluded: {total_excluded} files without pseudo-labels")
    
    # Debug: Show sample files for first class
    if files_by_class.get(0) and len(files_by_class[0]) > 0:
        print(f"\nSample files for {class_order[0]}: {files_by_class[0][:3]}")
    
    # Generate UID early to check if coordinates already exist
    # For UID generation, we need to know which files would be used
    # Shuffle files with same seed for consistency
    files_by_class_for_uid = {}
    for class_idx, class_name in enumerate(class_order):
        files = files_by_class[class_idx].copy()
        # Shuffle with same seed to get deterministic file selection
        random.Random(seed).shuffle(files)
        files_by_class_for_uid[class_idx] = files[:n_slides_per_class_for_prototypes]
    
    # Combine all expected files for UID
    expected_base_keys = []
    for class_idx, class_name in enumerate(class_order):
        expected_base_keys.extend(files_by_class_for_uid[class_idx])
    
    # Generate UID from config (deterministic, only once per run)
    expected_uid = build_uid_from_config(cfg)
    # Never concatenate or interpolate UID into itself; only use this value
    
    # Check if coordinates with this UID already exist
    existing_coords = True
    for class_name in class_order:
        coord_file = proto_coords_dir / f"{class_name}_{expected_uid}.npy"
        if not coord_file.exists():
            existing_coords = False
            break
    
    if existing_coords:
        print(f"\n✓ Coordinates already exist for this configuration!")
        print(f"  UID: {expected_uid}")
        print(f"  Skipping extraction and using existing coordinates:")
        for class_name in class_order:
            coord_file = proto_coords_dir / f"{class_name}_{expected_uid}.npy"
            if coord_file.exists():
                data = np.load(coord_file, allow_pickle=True)
                print(f"    {class_name}: {len(data)} coordinates from {coord_file.name}")
        
        # No longer saving UID to latest_uid.txt
        print(f"  Skipping to next pipeline step...")
        return
    
    print(f"\n=== Extracting New Coordinates ===")
    print(f"Expected UID: {expected_uid}")
    
    # Extract patches from WSIs
    wsis_processed_per_class = {cls: 0 for cls in class_order}
    proto_coords_by_class = {cls: {"x": [], "y": [], "wsi": [], "base_keys": []} for cls in class_order}
    
    for class_idx, class_name in enumerate(class_order):
        print(f"\nProcessing class: {class_name}")
        files = files_by_class[class_idx]
        if not files:
            print(f"  No files labeled as {class_name}")
            continue
        
        # Shuffle for diversity (seed already set at function start)
        random.shuffle(files)
        
        # Debug: Check first few files
        files_found = 0
        files_missing_wsi = 0
        files_missing_coord = 0
        
        print(f"  Checking files (showing first 5):")
        for idx, base_key in enumerate(files[:5]):
            wsi_path = find_wsi_file(wsi_dir, base_key)
            if not wsi_path and fallback_wsi_dir:
                wsi_path = find_wsi_file(fallback_wsi_dir, base_key)

            coord_path = find_coordinate_file(coordinates_dir, base_key, coordinates_suffix)
            if not coord_path and fallback_coordinates_dir:
                coord_path = find_coordinate_file(fallback_coordinates_dir, base_key, fallback_coordinates_suffix)
            
            wsi_exists = wsi_path is not None
            coord_exists = coord_path is not None
            
            status = "✓" if (wsi_exists and coord_exists) else "✗"
            wsi_info = os.path.basename(wsi_path) if wsi_path else "NOT FOUND"
            coord_info = os.path.basename(coord_path) if coord_path else "NOT FOUND"
            if not wsi_exists and fallback_wsi_dir:
                wsi_info += f" (checked fallback: {fallback_wsi_dir})"
            if not coord_exists and fallback_coordinates_dir:
                coord_info += f" (checked fallback: {fallback_coordinates_dir})"
            print(f"    {status} {base_key}:")
            print(f"       WSI: {wsi_info}")
            print(f"       Coord: {coord_info}")
            
            if not wsi_exists:
                files_missing_wsi += 1
            if not coord_exists:
                files_missing_coord += 1
            if wsi_exists and coord_exists:
                files_found += 1
        
        # Quick scan of all files
        total_valid = 0
        for base_key in files:
            wsi_path = find_wsi_file(wsi_dir, base_key)
            if not wsi_path and fallback_wsi_dir:
                wsi_path = find_wsi_file(fallback_wsi_dir, base_key)

            coord_path = find_coordinate_file(coordinates_dir, base_key, coordinates_suffix)
            if not coord_path and fallback_coordinates_dir:
                coord_path = find_coordinate_file(fallback_coordinates_dir, base_key, fallback_coordinates_suffix)
            
            if wsi_path and coord_path:
                total_valid += 1
        
        print(f"  Total files with both WSI and coordinates: {total_valid}/{len(files)}")
        
        if total_valid == 0:
            print(f"  ⚠ WARNING: No valid files found for {class_name}!")
            print(f"    Check that files exist in:")
            print(f"      WSI dir: {wsi_dir}")
            print(f"      Coord dir: {coordinates_dir}")
            continue
        
        # Create progress bar that tracks WSIs processed
        with tqdm(total=min(n_slides_per_class_for_prototypes, len(files)), desc=f"{class_name} WSIs", unit="wsi") as pbar_wsi:
            for base_key in files:
                if wsis_processed_per_class[class_name] >= n_slides_per_class_for_prototypes:
                    break
                
                # Find files with any extension
                wsi_path = find_wsi_file(wsi_dir, base_key)
                if not wsi_path and fallback_wsi_dir:
                    wsi_path = find_wsi_file(fallback_wsi_dir, base_key)

                coord_path = find_coordinate_file(coordinates_dir, base_key, coordinates_suffix)
                if not coord_path and fallback_coordinates_dir:
                    # Use fallback coordinates
                    coord_path = find_coordinate_file(fallback_coordinates_dir, base_key, fallback_coordinates_suffix)
                
                if not wsi_path or not coord_path:
                    continue
                
                # Load coordinates (detect suffix from actual file found)
                try:
                    detected_suffix = os.path.splitext(coord_path)[1]
                    coords_x, coords_y = load_coordinates(coord_path, detected_suffix)
                except Exception as e:
                    print(f"  Error loading coordinates from {coord_path}: {e}")
                    continue
                
                if len(coords_x) == 0:
                    continue
                
                # Determine if this class requires pseudo-labels
                class_needs_pseudo_labels = use_pseudo_labels and pseudo_loader is not None
                if prototype_label_required_classes is not None:
                    # Only use pseudo-labels for specified classes
                    class_needs_pseudo_labels = class_needs_pseudo_labels and (class_idx in prototype_label_required_classes)
                
                # Extract patches using pseudo-labels or random sampling
                if class_needs_pseudo_labels:
                    try:
                        # Load pseudo-label attention scores for this WSI
                        scores = pseudo_loader.load_wsi_scores(base_key)
                        print()
                        # Verify scores match coordinate count, attempt to align if mismatch
                        if len(scores) != len(coords_x):
                            if pseudo_label_coord_dir:
                                pseudo_coord_path = os.path.join(pseudo_label_coord_dir, base_key + pseudo_label_coord_suffix)
                                if os.path.exists(pseudo_coord_path):
                                    try:
                                        pseudo_x, pseudo_y = load_coordinates(pseudo_coord_path, pseudo_label_coord_suffix)
                                        idx_main, idx_pseudo = _match_coordinates(coords_x, coords_y, pseudo_x, pseudo_y)
                                        if len(idx_main) == 0:
                                            print(f"  Warning: No overlapping coordinates between dataset and pseudo-label coords for {base_key}")
                                            print(f"  Skipping this WSI")
                                            continue
                                        # Subset to overlapping coordinates
                                        coords_x = coords_x[idx_main]
                                        coords_y = coords_y[idx_main]
                                        if scores.dim() == 1:
                                            scores = scores[idx_pseudo]
                                        else:
                                            scores = scores[idx_pseudo, :]
                                        print(f"  Aligned pseudo-labels to coordinates: {len(idx_main)} overlaps (coords {len(coords_x)}, pseudo {len(scores)})")
                                    except Exception as e:
                                        print(f"  Warning: Failed to align pseudo-label coords for {base_key}: {e}")
                                        print(f"  Skipping this WSI")
                                        continue
                                else:
                                    print(f"  Warning: Pseudo-label coord file missing for {base_key}: {pseudo_coord_path}")
                                    print(f"  Skipping this WSI")
                                    continue
                            else:
                                print(f"  Warning: Pseudo-label count ({len(scores)}) != coordinate count ({len(coords_x)}) for {base_key}")
                                print(f"  Provide pseudo_label_coordinates_dir to align; skipping this WSI")
                                continue
                        
                        # IMPORTANT: Normalize attention scores per slide (min-max scaling)
                        # This ensures consistent prototype selection across slides
                        if scores.dim() == 1:
                            s_min = scores.min()
                            s_max = scores.max()
                            if (s_max - s_min) > 1e-8:
                                scores = (scores - s_min) / (s_max - s_min)
                            else:
                                scores = torch.zeros_like(scores)
                        elif scores.dim() == 2:
                            # Normalize per class independently
                            mins = scores.min(dim=0, keepdim=True)[0]
                            maxs = scores.max(dim=0, keepdim=True)[0]
                            denom = (maxs - mins)
                            denom[denom == 0] = 1.0
                            scores = (scores - mins) / denom
                        
                        # Binary mode: special handling for negative class (benign)
                        # Negative class needs both high-tumor and low-no-tumor patches for robust prototypes
                        if binary_mode and class_idx == 0:
                            # For negative class (benign):
                            # 1. Select top n-percentile patches with HIGHEST tumor score (dim 1) - these are false positives
                            # 2. Select top n-percentile patches with LOWEST no-tumor score (dim 0) - these are ambiguous
                            # These will be clustered together to form robust negative prototypes
                            
                            if scores.dim() == 1:
                                # Convert binary scores to 2D: [1-score, score]
                                scores_2d = torch.stack([1 - scores, scores], dim=1)
                            else:
                                scores_2d = scores
                            
                            tumor_scores = scores_2d[:, 1].numpy()  # Higher = more tumor-like
                            no_tumor_scores = scores_2d[:, 0].numpy()  # Higher = more benign-like
                            
                            # Threshold for selection
                            threshold = threshold_for_prototype if threshold_for_prototype is not None else pseudo_label_confidence_threshold
                            if isinstance(threshold, dict):
                                thresh_val = threshold.get(class_idx, 0.85)
                            elif isinstance(threshold, (list, tuple)):
                                thresh_val = threshold[class_idx]
                            else:
                                thresh_val = threshold
                            
                            # Select high tumor score patches (false positives for negative class)
                            tumor_threshold = np.quantile(tumor_scores, thresh_val)
                            high_tumor_indices = np.where(tumor_scores >= tumor_threshold)[0]
                            
                            # Select low no-tumor score patches (ambiguous patches)
                            low_notumor_threshold = np.quantile(no_tumor_scores, 1 - thresh_val)
                            low_notumor_indices = np.where(no_tumor_scores <= low_notumor_threshold)[0]
                            
                            # Combine both sets (union)
                            high_conf_indices = np.unique(np.concatenate([high_tumor_indices, low_notumor_indices]))
                            
                            print(f"  Negative class selection: {len(high_tumor_indices)} high-tumor + {len(low_notumor_indices)} low-no-tumor = {len(high_conf_indices)} total")
                        
                        elif binary_mode and class_idx == 1:
                            # For positive class (tumor): select top n-percentile with highest tumor score
                            if scores.dim() == 1:
                                tumor_scores = scores.numpy()
                            else:
                                tumor_scores = scores[:, 1].numpy()
                            
                            threshold = threshold_for_prototype if threshold_for_prototype is not None else pseudo_label_confidence_threshold
                            if isinstance(threshold, dict):
                                thresh_val = threshold.get(class_idx, 0.995)
                            elif isinstance(threshold, (list, tuple)):
                                thresh_val = threshold[class_idx]
                            else:
                                thresh_val = threshold
                            
                            tumor_threshold = np.quantile(tumor_scores, thresh_val)
                            high_conf_indices = np.where(tumor_scores >= tumor_threshold)[0]
                            
                            print(f"  Positive class selection: {len(high_conf_indices)} high-tumor patches")
                        
                        else:
                            # Multi-class mode: select top n-percentile for this class
                            if scores.dim() == 1:
                                class_scores = scores.numpy()
                            else:
                                class_scores = scores[:, class_idx].numpy()
                            
                            threshold = threshold_for_prototype if threshold_for_prototype is not None else pseudo_label_confidence_threshold
                            if isinstance(threshold, dict):
                                thresh_val = threshold.get(class_idx, 0.85)
                            elif isinstance(threshold, (list, tuple)):
                                thresh_val = threshold[class_idx]
                            else:
                                thresh_val = threshold
                            
                            class_threshold = np.quantile(class_scores, thresh_val)
                            high_conf_indices = np.where(class_scores >= class_threshold)[0]
                            
                            print(f"  Class {class_idx} selection: {len(high_conf_indices)} high-confidence patches")
                        
                        # Check minimum patch requirement
                        if len(high_conf_indices) < prototype_min_patches:
                            print(f"  Warning: Only {len(high_conf_indices)} high-conf patches (< {prototype_min_patches} min) for {base_key}")
                            print(f"  Skipping this WSI")
                            continue
                        
                        # Use all high-confidence patches selected by percentile threshold
                        sampled_indices = high_conf_indices
                        
                        print(f"  Selected {len(sampled_indices)} patches from {len(coords_x)} (top {(len(high_conf_indices)/len(coords_x)*100):.1f}%)")
                    except Exception as e:
                        print(f"  Warning: Error loading pseudo-labels for {base_key}: {e}")
                        print(f"  Skipping this WSI")
                        continue
                else:
                    # For classes without pseudo-labels (e.g., benign), use a subset of coordinates
                    # Limit to 10,000 patches per slide to avoid massive coordinate files
                    n_total = len(coords_x)
                    limit = 10000
                    if n_total > limit:
                        sampled_indices = np.random.choice(n_total, limit, replace=False)
                        print(f"  Randomly sampled {limit} patches from {n_total} (no pseudo-label filtering)")
                    else:
                        sampled_indices = np.arange(n_total)
                        print(f"  Using all {n_total} coordinates (no pseudo-label filtering)")
                
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
    
    # Use the same expected_uid that was calculated earlier
    # This ensures consistency between check and save
    uid = expected_uid
    
    # Save prototype coordinates in slide2vec format for each class
    for class_name in class_order:
        if len(proto_coords_by_class[class_name]["x"]) > 0:
            coords_x = np.array(proto_coords_by_class[class_name]["x"])
            coords_y = np.array(proto_coords_by_class[class_name]["y"])
            wsi_names = proto_coords_by_class[class_name]["wsi"]
            
            # Save with UID in filename
            coord_save_path = proto_coords_dir / f"{class_name}_{uid}.npy"
            save_coordinates_slide2vec(coords_x, coords_y, patch_size, 0, patch_size, 
                                      str(coord_save_path), wsi_names=wsi_names)
            print(f"Saved {class_name} prototype coordinates to {coord_save_path}")
            print(f"  UID: {uid}")
    
    # No longer saving UID to latest_uid.txt
    # UID is only used for directory naming and config
    
    print("\n=== Coordinate Extraction Complete ===")
    for class_name in class_order:
        n_coords = len(proto_coords_by_class[class_name]["x"])
        n_wsis = wsis_processed_per_class[class_name]
        print(f"{class_name}: {n_wsis} WSIs → {n_coords} coordinates")
    print(f"\nPrototype coordinates saved to: {proto_coords_dir}")
    print(f"Patches will be extracted on-the-fly during MedCLIP feature extraction.")
    print(f"\nExample outputs will be organized under: {work_dir}/runs/{uid}/")
    # No longer referencing latest_uid.txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract prototype coordinates from WSIs using attention scores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parameters are now read from config file:
  - features.n_slides_per_class_for_prototypes: Max WSIs per class
  - features.selection_method: Patch selection strategy
  - dataset.threshold_for_prototype: Per-class confidence thresholds
"""
    )
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--generate_thumbnails', action='store_true',
                       help='Generate test set thumbnails')
    parser.add_argument('--num_thumbnail_samples', type=int, default=5,
                       help='Number of test samples for thumbnail generation')
    parser.add_argument('--pseudo_label_coordinates_dir', type=str, default=None,
                        help='Directory with coordinates corresponding to attention scores (used to align when counts mismatch)')
    parser.add_argument('--pseudo_label_coordinates_suffix', type=str, default=None,
                        help='File suffix/extension for pseudo-label coordinate files (default: coordinates_suffix)')
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)

    # Allow overriding pseudo-label coordinate paths via CLI
    if args.pseudo_label_coordinates_dir:
        cfg.dataset.pseudo_label_coordinates_dir = args.pseudo_label_coordinates_dir
    if args.pseudo_label_coordinates_suffix:
        cfg.dataset.pseudo_label_coordinates_suffix = args.pseudo_label_coordinates_suffix
    
    # Extract patches from training set (parameters read from config)
    extract_patches(cfg, seed=args.seed)
    
    # Generate test thumbnails if requested
    if args.generate_thumbnails:
        generate_test_thumbnails(cfg, args.num_thumbnail_samples)

