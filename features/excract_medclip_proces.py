import os
import torch
from PIL import Image
import pickle as pkl
import numpy as np
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from tqdm import tqdm
import cv2 as cv
from utils.pyutils import set_seed
from albumentations.pytorch import ToTensorV2
import albumentations as A
from omegaconf import OmegaConf
import argparse
import glob

try:
    import openslide
    HAS_OPENSLIDE = True
except:
    from skimage import io as skimage_io
    HAS_OPENSLIDE = False

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

def extract_patch(wsi_path, x, y, patch_size, use_openslide=True):
    """Extract a patch from WSI at coordinates (x, y)."""
    half = patch_size // 2
    
    if use_openslide and HAS_OPENSLIDE:
        slide = openslide.OpenSlide(wsi_path)
        top_left = (max(0, x - half), max(0, y - half))
        region = slide.read_region(top_left, 0, (patch_size, patch_size))
        slide.close()
        patch = np.array(region.convert('RGB'))
    else:
        wsi = skimage_io.imread(wsi_path)
        if len(wsi.shape) == 2:
            wsi = cv.cvtColor(wsi, cv.COLOR_GRAY2RGB)
        elif wsi.shape[2] == 4:
            wsi = cv.cvtColor(wsi, cv.COLOR_RGBA2RGB)
        
        h, w = wsi.shape[:2]
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(w, x1 + patch_size)
        y2 = min(h, y1 + patch_size)
        
        patch = wsi[y1:y2, x1:x2]
        if patch.shape[:2] != (patch_size, patch_size):
            patch = cv.copyMakeBorder(
                patch,
                0, patch_size - patch.shape[0],
                0, patch_size - patch.shape[1],
                cv.BORDER_CONSTANT, value=255
            )
    
    return patch


def load_prototype_coordinates(coordinates_dir, class_name, expected_uid=None):
    """Load prototype coordinates from .npy file.
    
    Args:
        coordinates_dir: Directory containing coordinate files
        class_name: Name of class (e.g., 'benign', 'tumor')
        expected_uid: If provided, only load coordinates with this specific UID
    
    Returns:
        (coords_x, coords_y, wsi_names, uid) tuple
        uid is None for legacy files
    """
    # If we have an expected UID, try that first
    if expected_uid:
        coord_path = os.path.join(coordinates_dir, f"{class_name}_{expected_uid}.npy")
        if os.path.exists(coord_path):
            print(f"  Loading coordinates: {class_name}_{expected_uid}.npy")
            data = np.load(coord_path, allow_pickle=True)
            
            # Handle structured array (slide2vec format)
            if isinstance(data, np.ndarray) and data.dtype.names:
                x = data['x']
                y = data['y']
                wsi_names = data['wsi_name'] if 'wsi_name' in data.dtype.names else None
                return x, y, wsi_names, expected_uid
            else:
                # Simple array (legacy format)
                if len(data.shape) == 1:
                    x = data[::2]
                    y = data[1::2]
                else:
                    x = data[:, 0]
                    y = data[:, 1]
                return x, y, None, expected_uid
    
    # Otherwise, try to find UID-based file
    uid_pattern = os.path.join(coordinates_dir, f"{class_name}_*.npy")
    uid_files = glob.glob(uid_pattern)
    
    uid = None
    if uid_files:
        # Sort by modification time, use most recent
        uid_files.sort(key=os.path.getmtime, reverse=True)
        coord_path = uid_files[0]
        # Extract UID from filename: {class_name}_{uid}.npy
        filename = os.path.basename(coord_path)
        uid = filename.replace(f"{class_name}_", "").replace(".npy", "")
        print(f"  Found UID-based coordinates: {filename}")
    else:
        # Fall back to legacy format
        coord_path = os.path.join(coordinates_dir, f"{class_name}.npy")
    
    if not os.path.exists(coord_path):
        return None, None, None, None
    
    data = np.load(coord_path, allow_pickle=True)
    
    # Handle structured array (slide2vec format)
    if isinstance(data, np.ndarray) and data.dtype.names:
        x = data['x']
        y = data['y']
        wsi_names = data['wsi_name'] if 'wsi_name' in data.dtype.names else None
        return x, y, wsi_names, uid
    else:
        # Simple array (legacy format)
        if len(data.shape) == 1:
            x = data[::2]
            y = data[1::2]
        else:
            x = data[:, 0]
            y = data[:, 1]
        return x, y, None, uid



def extract_features_from_config(cfg):
    set_seed(42)

    # Read class order and data paths from config
    class_order = list(getattr(cfg.dataset, 'class_order', ['benign', 'tumor']))
    wsi_dir = cfg.dataset.wsi_dir
    coordinates_dir = getattr(cfg.dataset, 'coordinates_dir', None)
    patch_size = getattr(cfg.dataset, 'patch_size', 224)
    use_openslide = getattr(cfg.dataset, 'use_openslide', HAS_OPENSLIDE)
    
    # Prototype coordinates directory (where extract_patches.py saves them)
    proto_coords_dir = os.path.join(cfg.work_dir, 'prototype_coordinates')
    
    # Read UID from latest_uid.txt if run_uid is null
    uid = getattr(cfg, 'run_uid', None)
    if uid is None or uid == 'null':
        uid_file = os.path.join(proto_coords_dir, 'latest_uid.txt')
        if os.path.exists(uid_file):
            with open(uid_file, 'r') as f:
                uid = f.read().strip()
            print(f"Using UID from {uid_file}: {uid}")
    
    save_dir = cfg.features.save_dir
    output_name = cfg.features.medclip_features_pkl

    os.makedirs(save_dir, exist_ok=True)
    
    # Load first coordinate set to verify it exists and get UID if needed
    sample_coords_x, sample_coords_y, sample_wsi_names, coord_uid = load_prototype_coordinates(proto_coords_dir, class_order[0], expected_uid=uid)
    
    # Use coordinate UID if we didn't get one from config
    if uid is None:
        uid = coord_uid
    
    # Resolve UID placeholders in paths if needed
    if uid:
        # Replace ${run_uid} in save_dir if present
        save_dir = save_dir.replace('${run_uid}', uid).replace('None', uid)
        output_name = output_name.replace('${run_uid}', uid).replace('None', uid)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, output_name)
    else:
        save_path = os.path.join(save_dir, output_name)
    
    # Check if features already exist
    if os.path.exists(save_path):
        print(f"\n✓ MedCLIP features already exist at: {save_path}")
        print("  Skipping feature extraction (delete file to re-extract)")
        return save_path  # Return path so caller knows where features are

    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()


    features_dict = {}
    for class_name in class_order:
        features_dict[class_name] = []

        # Load prototype coordinates (using the UID we determined earlier)
        coords_x, coords_y, wsi_names, class_uid = load_prototype_coordinates(proto_coords_dir, class_name, expected_uid=uid)
        
        if coords_x is None:
            print(f"Warning: prototype coordinates not found for {class_name} — skipping class")
            continue

        print(f"\nProcessing {class_name} prototypes ({len(coords_x)} coordinates)...")
        
        # Extract patches at each coordinate
        for i, (x, y) in enumerate(tqdm(zip(coords_x, coords_y), total=len(coords_x), 
                                         desc=f"{class_name}", ncols=100)):
            # Get WSI name from structured array (if available)
            wsi_name = None
            if wsi_names is not None and i < len(wsi_names):
                wsi_name = wsi_names[i]
            
            patch = None
            
            # If we have a WSI name, try that first
            if wsi_name:
                wsi_path = os.path.join(wsi_dir, wsi_name)
                if os.path.exists(wsi_path):
                    try:
                        patch = extract_patch(wsi_path, int(x), int(y), patch_size, use_openslide)
                    except:
                        # If failed, fall through to search all WSIs
                        patch = None
            
            # If we didn't get a patch from specific WSI, search all available WSIs
            if patch is None:
                for wsi_file in os.listdir(wsi_dir):
                    if not wsi_file.lower().endswith(('.tif', '.tiff')):
                        continue
                    
                    wsi_path = os.path.join(wsi_dir, wsi_file)
                    
                    try:
                        # Try to extract patch
                        patch = extract_patch(wsi_path, int(x), int(y), patch_size, use_openslide)
                        break
                    except:
                        # This WSI might not have valid data at this coordinate, try next
                        continue
            
            if patch is None:
                print(f"  Warning: could not extract patch at ({x}, {y}) from any WSI")
                continue
            
            # Ensure RGB
            if len(patch.shape) == 2:
                patch = cv.cvtColor(patch, cv.COLOR_GRAY2RGB)

            transform = get_transform()
            patch_tensor = transform(image=patch)["image"]
            patch_tensor = (patch_tensor - patch_tensor.min()) / (patch_tensor.max() - patch_tensor.min() + 1e-8)

            with torch.no_grad():
                outputs = model.vision_model(patch_tensor.unsqueeze(0).to(device))
                features = outputs.cpu().detach().numpy()

            features_dict[class_name].append({
                'name': f"{class_name}_x{int(x)}_y{int(y)}",
                'features': features
            })

    with open(save_path, 'wb') as f:
        pkl.dump(features_dict, f)

    print(f"\nFeatures saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    extract_features_from_config(cfg)