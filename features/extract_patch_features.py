"""
Extract Prototype Features from Precomputed Patch Features.

This looks up prototype coordinates and pulls the matching feature vectors
directly from precomputed patch feature files. No WSI reads or re-encoding.
"""

import os
import argparse
import hashlib
import pickle as pkl
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import sys

# Import common utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.pyutils import build_uid_from_config


# Fallbacks to mirror diagnose.py expectations
FEATURE_SUFFIXES = [".pt", ".npy", ".npz", ".pkl", ".h5", "_patches.h5"]
COORD_SUFFIXES = [".npy", ".npz", ".h5", ".hdf5", "_patches.h5"]


def load_prototype_coordinates(coordinates_dir, class_name, expected_uid=None):
    """Load prototype coordinates with strict UID matching."""
    if expected_uid:
        coord_path = os.path.join(coordinates_dir, f"{class_name}_{expected_uid}.npy")
        if not os.path.exists(coord_path):
            raise FileNotFoundError(
                f"Coordinate file not found: {coord_path}\n"
                f"Expected UID '{expected_uid}'. Ensure coordinates are extracted for this config."
            )

        data = np.load(coord_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype.names:
            x = data['x']
            y = data['y']
            wsi_names = data['wsi_name'] if 'wsi_name' in data.dtype.names else None
            return x, y, wsi_names, expected_uid
        else:
            if len(data.shape) == 1:
                return data[::2], data[1::2], None, expected_uid
            else:
                return data[:, 0], data[:, 1], None, expected_uid

    return None, None, None, None


def load_wsi_coordinates(coord_path, coordinates_suffix='.npy'):
    if coordinates_suffix.endswith('.h5'):
        with h5py.File(coord_path, 'r') as f:
            if 'coords' in f:
                data = f['coords'][:]
            elif 'coordinates' in f:
                data = f['coordinates'][:]
            else:
                keys = list(f.keys())
                if len(keys) == 0:
                    raise ValueError(f"No datasets found in {coord_path}")
                data = f[keys[0]][:]

        if len(data.shape) == 2 and data.shape[1] == 2:
            return data[:, 0], data[:, 1]
        else:
            raise ValueError(f"Unexpected shape {data.shape} in h5 file")

    elif coordinates_suffix.endswith('.npy'):
        data = np.load(coord_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype.names:
            return data['x'], data['y']
        else:
            if len(data.shape) == 1:
                return data[::2], data[1::2]
            else:
                return data[:, 0], data[:, 1]
    else:
        data = np.loadtxt(coord_path)
        if len(data.shape) == 1:
            return data[::2], data[1::2]
        else:
            return data[:, 0], data[:, 1]


def load_wsi_features(feature_path):
    """Load precomputed features with robust handling of torch, pickle, npz, and h5."""
    ext = Path(feature_path).suffix.lower()

    def _as_numpy(obj):
        if torch.is_tensor(obj):
            return obj.cpu().numpy()
        return np.array(obj)

    # Torch saves (.pt/.pth or some .pkl) can carry persistent IDs; torch.load handles them.
    if ext in ['.pt', '.pth']:
        data = torch.load(feature_path, map_location='cpu')
    elif ext == '.npz':
        arr = np.load(feature_path)
        data = arr[list(arr.files)[0]]
    elif ext in ['.h5', '.hdf5']:
        with h5py.File(feature_path, 'r') as f:
            key = 'features' if 'features' in f else ('data' if 'data' in f else list(f.keys())[0])
            data = f[key][:]
    else:
        # Try torch.load first to handle persistent IDs even for .pkl
        try:
            data = torch.load(feature_path, map_location='cpu')
        except Exception:
            with open(feature_path, 'rb') as f:
                data = pkl.load(f)

    if isinstance(data, dict):
        if 'features' in data:
            features = data['features']
        elif 'embeddings' in data:
            features = data['embeddings']
        else:
            first_key = list(data.keys())[0]
            features = data[first_key]
    else:
        features = data

    return _as_numpy(features)


def find_feature_file(wsi_base, patch_features_dir, suffixes=None):
    """Find the first existing feature file by trying common suffixes."""
    suffixes = suffixes or FEATURE_SUFFIXES
    for suf in suffixes:
        candidate = os.path.join(patch_features_dir, f"{wsi_base}{suf}")
        if os.path.exists(candidate):
            return candidate
    return None


def find_coord_file(wsi_base, coordinates_dir, coordinates_suffix=None):
    """Find coordinate file allowing multiple suffix and naming variants."""
    candidates = []
    if coordinates_suffix:
        candidates.append(os.path.join(coordinates_dir, f"{wsi_base}{coordinates_suffix}"))
        candidates.append(os.path.join(coordinates_dir, f"{wsi_base}_patches{coordinates_suffix}"))

    for suf in COORD_SUFFIXES:
        candidates.append(os.path.join(coordinates_dir, f"{wsi_base}{suf}"))
        candidates.append(os.path.join(coordinates_dir, f"{wsi_base}_patches{suf}"))

    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


def match_coordinates_to_features(proto_coords_x, proto_coords_y, wsi_coords_x, wsi_coords_y, wsi_features, tolerance=1):
    coord_to_idx = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(wsi_coords_x, wsi_coords_y))}

    matched_features = []
    matched_coords = []

    for px, py in zip(proto_coords_x, proto_coords_y):
        px_int, py_int = int(px), int(py)
        idx = coord_to_idx.get((px_int, py_int))

        if idx is None:
            found = False
            for dx in range(-tolerance, tolerance + 1):
                for dy in range(-tolerance, tolerance + 1):
                    idx = coord_to_idx.get((px_int + dx, py_int + dy))
                    if idx is not None:
                        found = True
                        break
                if found:
                    break

        if idx is not None:
            matched_features.append(wsi_features[idx])
            matched_coords.append((px_int, py_int))
        else:
            print(f"  Warning: No feature found for coordinate ({px_int}, {py_int})")

    return matched_features, matched_coords


def extract_features_from_config(cfg):
    class_order = list(getattr(cfg.dataset, 'class_order', ['benign', 'tumor']))
    patch_features_dir = cfg.dataset.patch_features_dir
    coordinates_dir = cfg.dataset.coordinates_dir
    coordinates_suffix = getattr(cfg.dataset, 'coordinates_suffix', '.npy')
    
    # Fallback directories
    fallback_features_dir = getattr(cfg.dataset, 'fallback_patch_features_dir', None)
    fallback_coords_dir = getattr(cfg.dataset, 'fallback_coordinates_dir', None)
    fallback_coords_suffix = getattr(cfg.dataset, 'fallback_coordinates_suffix', coordinates_suffix)
    
    # Pseudo-label configuration
    pseudo_label_dir = getattr(cfg.dataset, 'pseudo_label_dir', None)
    use_pseudo_labels = getattr(cfg.dataset, 'use_pseudo_labels', False)

    proto_coords_dir = os.path.join(cfg.work_dir, 'prototype_coordinates')

    uid = build_uid_from_config(cfg)
    cfg.run_uid = uid
    print(f"Generated UID from config: {uid}")

    encoder_name = getattr(cfg.model, 'patch_encoder', 'virchow2')

    # Let OmegaConf resolve ${run_uid} interpolation
    save_dir = OmegaConf.to_container(OmegaConf.create({'save_dir': cfg.features.save_dir}), resolve=True)['save_dir']
    output_name = cfg.features.features_for_prototype_clusters

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, output_name)

    if os.path.exists(save_path):
        print(f"\n✓ Features already exist at: {save_path}")
        print("  Skipping feature extraction (delete file to re-extract)")
        return save_path

    print(f"\n{'='*70}")
    print("Extracting Prototype Features from Precomputed Features")
    print(f"{'='*70}")
    print(f"Encoder: {encoder_name}")
    print(f"Patch features directory: {patch_features_dir}")
    print(f"Prototype coordinates directory: {proto_coords_dir}")
    print(f"Output: {save_path}")
    print(f"{'='*70}\n")

    features_dict = {}

    for class_name in class_order:
        features_dict[class_name] = []

        coords_x, coords_y, wsi_names, _ = load_prototype_coordinates(proto_coords_dir, class_name, expected_uid=uid)
        if coords_x is None or len(coords_x) == 0:
            print(f"Warning: No prototype coordinates found for {class_name} — skipping")
            continue

        print(f"\n{class_name.upper()}: {len(coords_x)} prototype coordinates")
        
        # Filter out WSIs without pseudo-labels if required (skip for benign class)
        class_idx = class_order.index(class_name)
        is_benign = (class_idx == 0)
        
        if use_pseudo_labels and pseudo_label_dir and not is_benign:
            print(f"  Checking pseudo-label availability...")
            valid_indices = []
            excluded_wsis = set()
            
            for i, (x, y) in enumerate(zip(coords_x, coords_y)):
                if wsi_names is not None and i < len(wsi_names):
                    wsi_base = Path(wsi_names[i]).stem
                else:
                    continue
                
                # Check if pseudo-label file exists
                pseudo_label_extensions = ['.pt', '.pth', '.npy', '.npz', '.pkl']
                pseudo_label_found = False
                
                for ext in pseudo_label_extensions:
                    pseudo_label_path = os.path.join(pseudo_label_dir, wsi_base + ext)
                    if os.path.exists(pseudo_label_path):
                        pseudo_label_found = True
                        break
                
                if pseudo_label_found:
                    valid_indices.append(i)
                else:
                    excluded_wsis.add(wsi_base)
            
            if excluded_wsis:
                print(f"  ⚠ Excluded {len(excluded_wsis)} WSIs without pseudo-labels: {list(excluded_wsis)[:5]}{'...' if len(excluded_wsis) > 5 else ''}")
                coords_x = coords_x[valid_indices]
                coords_y = coords_y[valid_indices]
                if wsi_names is not None:
                    wsi_names = [wsi_names[i] for i in valid_indices]
                
                if len(coords_x) == 0:
                    print(f"  No coordinates remaining after filtering - skipping {class_name}")
                    continue
                
                print(f"  Remaining: {len(coords_x)} coordinates")
        elif is_benign:
            print(f"  Benign class - pseudo-labels not required (will use zeros during training)")

        coords_by_wsi = {}
        for i, (x, y) in enumerate(zip(coords_x, coords_y)):
            if wsi_names is not None and i < len(wsi_names):
                wsi_base = Path(wsi_names[i]).stem
            else:
                print(f"  Warning: No WSI name for coordinate {i}, skipping")
                continue

            coords_by_wsi.setdefault(wsi_base, {'x': [], 'y': []})
            coords_by_wsi[wsi_base]['x'].append(x)
            coords_by_wsi[wsi_base]['y'].append(y)

        print(f"  Distributed across {len(coords_by_wsi)} WSIs")

        total_matched = 0
        total_failed = 0

        for wsi_base, wsi_coords in tqdm(coords_by_wsi.items(), desc=f"  {class_name} WSIs", ncols=100):
            feature_path = find_feature_file(wsi_base, patch_features_dir)
            if feature_path is None:
                # Try fallback directory
                if fallback_features_dir:
                    feature_path = find_feature_file(wsi_base, fallback_features_dir)
                    if feature_path:
                        print(f"\n  → Using fallback features for {wsi_base}: {os.path.basename(feature_path)}")
                
                if feature_path is None:
                    print(f"\n  Warning: Feature file not found for {wsi_base} in {patch_features_dir}")
                    if fallback_features_dir:
                        print(f"           Also not found in fallback: {fallback_features_dir}")
                    total_failed += len(wsi_coords['x'])
                    continue

            coord_file = find_coord_file(wsi_base, coordinates_dir, coordinates_suffix)
            if coord_file is None:
                # Try fallback directory
                if fallback_coords_dir:
                    coord_file = find_coord_file(wsi_base, fallback_coords_dir, fallback_coords_suffix)
                    if coord_file:
                        print(f"\n  → Using fallback coordinates for {wsi_base}: {os.path.basename(coord_file)}")
                
                if coord_file is None:
                    print(f"\n  Warning: Coordinate file not found for {wsi_base} in {coordinates_dir}")
                    if fallback_coords_dir:
                        print(f"           Also not found in fallback: {fallback_coords_dir}")
                    total_failed += len(wsi_coords['x'])
                    continue

            try:
                wsi_features = load_wsi_features(feature_path)
                wsi_coords_x, wsi_coords_y = load_wsi_coordinates(coord_file, coordinates_suffix)

                matched_features, matched_coords = match_coordinates_to_features(
                    np.array(wsi_coords['x']),
                    np.array(wsi_coords['y']),
                    wsi_coords_x,
                    wsi_coords_y,
                    wsi_features,
                    tolerance=2,
                )

                for feat, (x, y) in zip(matched_features, matched_coords):
                    features_dict[class_name].append({
                        'name': wsi_base,
                        'coords': (int(x), int(y)),
                        'features': torch.from_numpy(feat) if not torch.is_tensor(feat) else feat,
                    })
                    total_matched += 1

                total_failed += len(wsi_coords['x']) - len(matched_features)

            except Exception as e:
                print(f"\n  Error processing {wsi_base}: {e}")
                total_failed += len(wsi_coords['x'])

        print(f"\n  ✓ Successfully matched: {total_matched}/{len(coords_x)} coordinates")
        if total_failed > 0:
            print(f"  ✗ Failed to match: {total_failed} coordinates")

    with open(save_path, 'wb') as f:
        pkl.dump(features_dict, f)

    print(f"\n{'='*70}")
    print(f"✓ Prototype features saved to: {save_path}")
    print(f"{'='*70}")

    print("\nSummary:")
    for class_name in class_order:
        print(f"  {class_name}: {len(features_dict.get(class_name, []))} prototype features extracted")

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract prototype features from precomputed patch features")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    extract_features_from_config(cfg)