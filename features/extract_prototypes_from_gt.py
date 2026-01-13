"""
Extract prototype coordinates from WSIs for weak supervision learning.

This script samples coordinates from WSIs at the provided locations,
uses image-level class labels to organize them by class, and saves coordinates
in slide2vec format. Patches are extracted on-the-fly during feature extraction.

Usage:
    python features/extract_prototypes_from_gt.py --config work_dirs/custom_wsi_template.yaml \
        --num_per_class 5 --samples_per_wsi 1000

This will extract coordinates from up to 5 WSIs per class, with up to 1000 patches per WSI.
Actual patches are extracted on-the-fly when needed (e.g., during feature extraction).
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


def extract_prototypes(cfg, num_per_class=20, samples_per_wsi=5):
    """Extract prototype patches from WSIs using coordinates.
    
    Args:
        cfg: OmegaConf configuration
        num_per_class: Max number of WSIs to process per class (not patches)
        samples_per_wsi: Number of patches to extract from each WSI
    """
    
    wsi_dir = cfg.dataset.wsi_dir
    coordinates_dir = cfg.dataset.coordinates_dir
    split_csv = cfg.dataset.split_csv
    labels_csv = cfg.dataset.labels_csv
    patch_size = getattr(cfg.dataset, 'patch_size', 224)
    use_openslide = getattr(cfg.dataset, 'use_openslide', HAS_OPENSLIDE)
    coordinates_suffix = getattr(cfg.dataset, 'coordinates_suffix', '.npy')
    
    # Output directory for coordinates only (patches will be extracted on-the-fly)
    proto_coords_dir = Path(cfg.features.save_dir).parent / 'prototype_coordinates'
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
        # If labels is a single value or single-element list: use it directly to assign class
        # If labels is multi-element: use for multi-label assignment (one-hot)
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
    proto_coords_by_class = {cls: {"x": [], "y": [], "wsi": []} for cls in class_order}
    
    for class_idx, class_name in enumerate(class_order):
        print(f"\nExtracting {class_name} patches from up to {num_per_class} WSIs...")
        
        files = files_by_class[class_idx]
        if not files:
            print(f"  No files labeled as {class_name}")
            continue
        
        # Shuffle for diversity
        random.shuffle(files)
        
        # Create progress bar that tracks WSIs processed
        with tqdm(total=min(num_per_class, len(files)), desc=f"{class_name} WSIs", unit="wsi") as pbar_wsi:
            for base_key in files:
                if wsis_processed_per_class[class_name] >= num_per_class:
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
                
                # Select up to samples_per_wsi coordinates from this WSI
                n_samples = min(samples_per_wsi, len(coords_x))
                sampled_indices = np.random.choice(len(coords_x), n_samples, replace=False)
                
                # Just store coordinates (patches will be extracted on-the-fly when needed)
                for idx in sampled_indices:
                    cx, cy = int(coords_x[idx]), int(coords_y[idx])
                    proto_coords_by_class[class_name]["x"].append(cx)
                    proto_coords_by_class[class_name]["y"].append(cy)
                    proto_coords_by_class[class_name]["wsi"].append(base_key)  # Track source WSI
                
                wsis_processed_per_class[class_name] += 1
                pbar_wsi.update(1)
    
    # Save prototype coordinates in slide2vec format for each class
    for class_name in class_order:
        if len(proto_coords_by_class[class_name]["x"]) > 0:
            coords_x = np.array(proto_coords_by_class[class_name]["x"])
            coords_y = np.array(proto_coords_by_class[class_name]["y"])
            wsi_names = proto_coords_by_class[class_name]["wsi"]
            coord_save_path = proto_coords_dir / f"{class_name}.npy"
            save_coordinates_slide2vec(coords_x, coords_y, patch_size, 0, patch_size, 
                                      str(coord_save_path), wsi_names=wsi_names)
            print(f"Saved {class_name} prototype coordinates to {coord_save_path}")
    
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
    parser.add_argument('--num_per_class', type=int, default=20, 
                       help='Max number of WSIs to process per class')
    parser.add_argument('--samples_per_wsi', type=int, default=5,
                       help='Number of patches to extract from each WSI')
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    random.seed(42)
    np.random.seed(42)
    
    extract_prototypes(cfg, args.num_per_class, args.samples_per_wsi)
