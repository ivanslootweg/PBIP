"""
Patch-level training dataset for PBIP.

Treats each (WSI, coordinate) pair as a separate training sample, enabling
training on all high-confidence patches selected during prototype extraction.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
from typing import Optional, Tuple, List
import cv2 as cv
from skimage import io
import glob

# Import common utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.common import extract_patch_numpy, extract_patch_openslide

try:
    import openslide as _openslide
    HAS_OPENSLIDE = True
except Exception:
    _openslide = None
    HAS_OPENSLIDE = False


class PatchLevelTrainingDataset(Dataset):
    """
    Patch-level dataset that treats each (WSI, coordinate) as a separate sample.
    
    Instead of loading WSIs and sampling randomly, this loads ALL patches from
    coordinate files (benign_*.npy, tumor_*.npy) created during prototype extraction.
    
    This provides dense training on high-confidence patches selected by pseudo-labels.
    
    Args:
        wsi_dir: Directory containing WSI (.tif) files
        proto_coords_dir: Directory containing prototype coordinate files
        run_uid: Run UID to identify coordinate files (e.g., "10000_50_pseudo-per95_8b5b5ddc")
        class_order: List of class names in order (e.g., ['benign', 'tumor'])
        num_classes: Number of classes
        patch_size: Size of patches to extract (default: 224)
        transform: Albumentations transform pipeline
        use_openslide: Whether to use OpenSlide for extraction
    
    Returns:
        - filename: WSI name
        - patch: Extracted patch image tensor (C, H, W)
        - class_label: One-hot class label (num_classes,)
        - patch_info: Dict with 'wsi_name', 'x', 'y', 'class_name'
    """
    
    def __init__(
        self,
        wsi_dir: str,
        proto_coords_dir: str,
        run_uid: str,
        class_order: List[str] = None,
        num_classes: int = 2,
        patch_size: int = 224,
        transform=None,
        use_openslide: Optional[bool] = None,
    ):
        super(PatchLevelTrainingDataset, self).__init__()
        
        self.wsi_dir = wsi_dir
        self.proto_coords_dir = Path(proto_coords_dir)
        self.run_uid = run_uid
        self.class_order = class_order or ['benign', 'tumor']
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.transform = transform
        self.use_openslide = HAS_OPENSLIDE if use_openslide is None else use_openslide
        
        # Storage for all patches
        self.patch_list = []  # List of (wsi_name, x, y, class_idx)
        self.wsi_cache = {}  # Cache loaded WSIs to avoid repeated file reads
        
        self._load_coordinate_files()
    
    def _load_coordinate_files(self):
        """Load all coordinate files and build patch list."""
        print("\n" + "="*70)
        print("LOADING PATCH-LEVEL TRAINING DATA")
        print("="*70)
        
        total_patches = 0
        
        for class_idx, class_name in enumerate(self.class_order):
            # Find coordinate file for this class
            pattern = f"{class_name}_{self.run_uid}.npy"
            coord_files = list(self.proto_coords_dir.glob(pattern))
            
            if not coord_files:
                # Try without exact UID match (find any file for this class)
                pattern_loose = f"{class_name}_*.npy"
                coord_files = list(self.proto_coords_dir.glob(pattern_loose))
                
                if not coord_files:
                    print(f"WARNING: No coordinate file found for class '{class_name}'")
                    print(f"  Searched for: {self.proto_coords_dir / pattern}")
                    continue
                else:
                    # Use the most recent file
                    coord_files = sorted(coord_files, key=lambda p: p.stat().st_mtime)
                    coord_file = coord_files[-1]
                    print(f"Using coordinate file (no exact UID match): {coord_file.name}")
            else:
                coord_file = coord_files[0]
            
            # Load coordinates
            coords_data = np.load(coord_file, allow_pickle=True)
            
            # Handle structured array format (slide2vec style)
            if coords_data.dtype.names:
                coords_x = coords_data['x']
                coords_y = coords_data['y']
                wsi_names = coords_data['wsi_name'] if 'wsi_name' in coords_data.dtype.names else None
            else:
                # Regular array format
                if len(coords_data.shape) == 2:
                    coords_x = coords_data[:, 0]
                    coords_y = coords_data[:, 1]
                else:
                    coords_x = coords_data[::2]
                    coords_y = coords_data[1::2]
                wsi_names = None
            
            # Build patch list
            n_patches = len(coords_x)
            for i in range(n_patches):
                wsi_name = wsi_names[i] if wsi_names is not None else f"unknown_wsi_{i}"
                if isinstance(wsi_name, bytes):
                    wsi_name = wsi_name.decode('utf-8')
                
                self.patch_list.append({
                    'wsi_name': wsi_name,
                    'x': int(coords_x[i]),
                    'y': int(coords_y[i]),
                    'class_idx': class_idx,
                    'class_name': class_name
                })
            
            total_patches += n_patches
            print(f"Class '{class_name}': {n_patches:,} patches from {coord_file.name}")
        
        print(f"\nTotal training patches: {total_patches:,}")
        print("="*70 + "\n")
    
    def __len__(self) -> int:
        return len(self.patch_list)
    
    def __getitem__(self, index: int) -> Tuple:
        """
        Returns:
            filename: WSI filename (for compatibility)
            patch: Extracted patch tensor (C, H, W)
            cls_label: One-hot class label (num_classes,)
            patch_info: Placeholder (0) for compatibility with existing code
        """
        patch_info = self.patch_list[index]
        wsi_name = patch_info['wsi_name']
        x, y = patch_info['x'], patch_info['y']
        class_idx = patch_info['class_idx']
        
        # Create one-hot class label
        cls_label = np.zeros(self.num_classes, dtype=np.float32)
        cls_label[class_idx] = 1.0
        
        # Build WSI path
        wsi_filename = wsi_name if wsi_name.endswith('.tif') else f"{wsi_name}.tif"
        wsi_path = os.path.join(self.wsi_dir, wsi_filename)
        
        # Extract patch
        if self.use_openslide:
            patch = extract_patch_openslide(wsi_path, x, y, self.patch_size)
        else:
            # Load WSI (with caching)
            if wsi_path not in self.wsi_cache:
                wsi = io.imread(wsi_path)
                if len(wsi.shape) == 2:
                    wsi = cv.cvtColor(wsi, cv.COLOR_GRAY2RGB)
                elif wsi.shape[2] == 4:
                    wsi = cv.cvtColor(wsi, cv.COLOR_RGBA2RGB)
                self.wsi_cache[wsi_path] = wsi
            else:
                wsi = self.wsi_cache[wsi_path]
            
            patch = extract_patch_numpy(wsi, x, y, self.patch_size)
        
        # Ensure RGB
        if len(patch.shape) == 2:
            patch = cv.cvtColor(patch, cv.COLOR_GRAY2RGB)
        
        # Create dummy mask for transform compatibility
        dummy_mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        
        # Apply transform
        if self.transform is not None:
            out = self.transform(image=patch, mask=dummy_mask)
            patch_tensor = out["image"]
        else:
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        
        cls_label_tensor = torch.from_numpy(cls_label).float()
        
        # Return format compatible with existing training code
        # (filename, image, cls_label, gt_label)
        return wsi_name, patch_tensor, cls_label_tensor, 0


class PatchLevelTestDataset(Dataset):
    """
    Patch-level test/validation dataset with ground truth masks.
    
    Similar to PatchLevelTrainingDataset but includes GT masks for evaluation.
    """
    
    def __init__(
        self,
        wsi_dir: str,
        gt_dir: str,
        proto_coords_dir: str,
        run_uid: str,
        class_order: List[str] = None,
        num_classes: int = 2,
        patch_size: int = 224,
        transform=None,
        use_openslide: Optional[bool] = None,
    ):
        super(PatchLevelTestDataset, self).__init__()
        
        self.wsi_dir = wsi_dir
        self.gt_dir = gt_dir
        self.proto_coords_dir = Path(proto_coords_dir)
        self.run_uid = run_uid
        self.class_order = class_order or ['benign', 'tumor']
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.transform = transform
        self.use_openslide = HAS_OPENSLIDE if use_openslide is None else use_openslide
        
        self.patch_list = []
        self._load_coordinate_files()
    
    def _load_coordinate_files(self):
        """Load coordinate files and filter by GT mask availability."""
        print("\n" + "="*70)
        print("LOADING PATCH-LEVEL VALIDATION DATA")
        print("="*70)
        
        total_patches = 0
        skipped_no_gt = 0
        
        for class_idx, class_name in enumerate(self.class_order):
            pattern = f"{class_name}_{self.run_uid}.npy"
            coord_files = list(self.proto_coords_dir.glob(pattern))
            
            if not coord_files:
                pattern_loose = f"{class_name}_*.npy"
                coord_files = list(self.proto_coords_dir.glob(pattern_loose))
                if coord_files:
                    coord_file = sorted(coord_files, key=lambda p: p.stat().st_mtime)[-1]
                else:
                    continue
            else:
                coord_file = coord_files[0]
            
            coords_data = np.load(coord_file, allow_pickle=True)
            
            if coords_data.dtype.names:
                coords_x = coords_data['x']
                coords_y = coords_data['y']
                wsi_names = coords_data['wsi_name'] if 'wsi_name' in coords_data.dtype.names else None
            else:
                if len(coords_data.shape) == 2:
                    coords_x = coords_data[:, 0]
                    coords_y = coords_data[:, 1]
                else:
                    coords_x = coords_data[::2]
                    coords_y = coords_data[1::2]
                wsi_names = None
            
            class_patches = 0
            for i in range(len(coords_x)):
                wsi_name = wsi_names[i] if wsi_names is not None else f"unknown_wsi_{i}"
                if isinstance(wsi_name, bytes):
                    wsi_name = wsi_name.decode('utf-8')
                
                # Check if GT mask exists
                mask_filename = f"{wsi_name}.tif" if not wsi_name.endswith('.tif') else wsi_name
                mask_path = os.path.join(self.gt_dir, mask_filename)
                
                # For benign patches (class 0), accept without GT mask (synthetic all-zero mask)
                # For tumor patches, require GT mask
                has_gt_mask = os.path.exists(mask_path)
                is_benign = (class_idx == 0)
                
                if not has_gt_mask and not is_benign:
                    # Tumor patch without GT mask - skip
                    skipped_no_gt += 1
                    continue
                
                self.patch_list.append({
                    'wsi_name': wsi_name,
                    'x': int(coords_x[i]),
                    'y': int(coords_y[i]),
                    'class_idx': class_idx,
                    'class_name': class_name,
                    'mask_path': mask_path if has_gt_mask else None  # None = synthetic all-zero mask
                })
                class_patches += 1
            
            total_patches += class_patches
            print(f"Class '{class_name}': {class_patches:,} patches")
        
        print(f"\nTotal validation patches: {total_patches:,}")
        if skipped_no_gt > 0:
            print(f"Skipped {skipped_no_gt:,} tumor patches (no GT mask)")
            print(f"Note: Benign patches use synthetic all-zero masks (no tumor annotation needed)")
        print("="*70 + "\n")
    
    def __len__(self) -> int:
        return len(self.patch_list)
    
    def __getitem__(self, index: int) -> Tuple:
        """
        Returns:
            filename: WSI filename
            patch: Extracted patch tensor (C, H, W)
            cls_label: One-hot class label (num_classes,)
            mask: Ground truth mask tensor (H, W)
        """
        patch_info = self.patch_list[index]
        wsi_name = patch_info['wsi_name']
        x, y = patch_info['x'], patch_info['y']
        class_idx = patch_info['class_idx']
        mask_path = patch_info['mask_path']
        
        # Create one-hot class label
        cls_label = np.zeros(self.num_classes, dtype=np.float32)
        cls_label[class_idx] = 1.0
        
        # Build WSI path
        wsi_filename = wsi_name if wsi_name.endswith('.tif') else f"{wsi_name}.tif"
        wsi_path = os.path.join(self.wsi_dir, wsi_filename)
        
        # Extract patch
        if self.use_openslide:
            patch = extract_patch_openslide(wsi_path, x, y, self.patch_size)
        else:
            wsi = io.imread(wsi_path)
            if len(wsi.shape) == 2:
                wsi = cv.cvtColor(wsi, cv.COLOR_GRAY2RGB)
            elif wsi.shape[2] == 4:
                wsi = cv.cvtColor(wsi, cv.COLOR_RGBA2RGB)
            patch = extract_patch_numpy(wsi, x, y, self.patch_size)
        
        if len(patch.shape) == 2:
            patch = cv.cvtColor(patch, cv.COLOR_GRAY2RGB)
        
        # Load mask: real GT mask or synthetic all-zero mask for benign
        if mask_path is not None:
            # Load and extract mask patch from GT
            full_mask = np.array(io.imread(mask_path))
            if len(full_mask.shape) == 3:
                full_mask = full_mask[:, :, 0]  # Take first channel if RGB
            
            # Extract same region from mask
            mask_rgb = np.stack([full_mask]*3, axis=-1)
            mask_patch = extract_patch_numpy(mask_rgb, x, y, self.patch_size)[:, :, 0]
        else:
            # Synthetic all-zero mask for benign patches (no tumor)
            mask_patch = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        
        # Apply transform
        if self.transform is not None:
            out = self.transform(image=patch, mask=mask_patch)
            patch_tensor = out["image"]
            mask_tensor = out["mask"].long()
        else:
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask_patch).long()
        
        cls_label_tensor = torch.from_numpy(cls_label).float()
        
        return wsi_name, patch_tensor, cls_label_tensor, mask_tensor
