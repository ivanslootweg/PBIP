"""
WSI (Whole Slide Image) dataset classes for weak supervision learning with MedCLIP.

Follows slide2vec pattern: coordinates stored with metadata (tile_level, tile_size, spacing).
Patches extracted from WSI at specified coordinates using wholeslidedata or OpenSlide.

Dataset structure:
- WSI files: .tif files (whole slide images)
- Coordinates: .npy files with x, y coords and metadata (tile_level, tile_size_resized, etc.)
- Class labels: image-level weak labels from CSV
- GT masks: pixel-level annotations for validation/test (optional)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import glob
from pathlib import Path
import csv
from typing import Dict, List, Optional, Tuple
from PIL import Image
import cv2 as cv
from skimage import io

# Import common utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.common import load_coordinates, extract_patch_numpy, extract_patch_openslide
from utils.pseudo_labels import PseudoLabelLoader, PatchSelector

try:
    import wholeslidedata as wsd
    HAS_WSD = True
except Exception:
    HAS_WSD = False

try:
    import openslide as _openslide
    HAS_OPENSLIDE = True
except Exception:
    _openslide = None
    HAS_OPENSLIDE = False
import random


class CustomWSIPatchTrainingDataset(Dataset):
    """
    Load patches from WSI files using coordinates, with image-level class labels.
    Patches are extracted from coordinates and returned for MedCLIP embedding.
    
    Args:
        wsi_dir: Directory containing WSI (.tif) files
        coordinates_dir: Directory containing coordinates files (.txt or .npy)
        split_csv: CSV file with columns: 'train', 'val', 'test' containing filenames
        split: Which split to load ('train', 'val', 'test')
        class_labels_dict: Dict mapping filename -> class_labels (array of 0/1)
        num_classes: Number of classes for weak labels
        patch_size: Size of patches to extract (default: 224)
        max_patches: Maximum number of patches to use per WSI (None = use all)
        coordinates_suffix: File suffix for coordinates (.txt or .npy)
    """
    
    def __init__(
        self,
        wsi_dir: str,
        coordinates_dir: str,
        split_csv: str,
        split: str = "train",
        class_labels_dict: Optional[Dict] = None,
        num_classes: int = 4,
        patch_size: int = 224,
        max_patches: Optional[int] = None,
        coordinates_suffix: str = ".npy",
        transform=None,
        use_openslide: Optional[bool] = None,
    ):
        super(CustomWSIPatchTrainingDataset, self).__init__()
        
        self.wsi_dir = wsi_dir
        self.coordinates_dir = coordinates_dir
        self.split = split
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.coordinates_suffix = coordinates_suffix
        self.class_labels_dict = class_labels_dict or {}
        self.transform = transform
        self.use_openslide = HAS_OPENSLIDE if use_openslide is None else use_openslide
        
        self.filenames = []
        self.wsi_paths = []
        self.coordinate_paths = []
        self.class_labels = []
        
        self._load_split_from_csv(split_csv, split)
    
    def _load_split_from_csv(self, split_csv: str, split: str):
        """
        Load filenames from CSV split file with validation.
        
        For TRAINING: requires WSI, coordinates, and label (NO GT mask needed - weak supervision)
        """
        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")
        
        skipped_count = {'no_wsi': 0, 'no_coords': 0, 'no_label': 0}
        skipped_samples = {'no_label': []}
        
        with open(split_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split in row and row[split]:
                    filename = row[split].strip()
                    # Normalize key as basename without extension for lookups
                    base_key = os.path.splitext(os.path.basename(filename))[0]
                    
                    # Construct WSI path (use normalized base and .tif)
                    wsi_name = base_key + ".tif"
                    wsi_path = os.path.join(self.wsi_dir, wsi_name)
                    
                    # Construct coordinates path
                    coord_name = base_key + self.coordinates_suffix
                    coord_path = os.path.join(self.coordinates_dir, coord_name)
                    
                    # Presence checks: require WSI, coordinates, and label entry
                    if not os.path.exists(wsi_path):
                        skipped_count['no_wsi'] += 1
                        continue
                    if not os.path.exists(coord_path):
                        skipped_count['no_coords'] += 1
                        continue
                    if base_key not in self.class_labels_dict:
                        skipped_count['no_label'] += 1
                        if len(skipped_samples['no_label']) < 5:
                            skipped_samples['no_label'].append(base_key)
                        continue

                    self.filenames.append(base_key)
                    self.wsi_paths.append(wsi_path)
                    self.coordinate_paths.append(coord_path)
                    
                    # Get class labels (guaranteed present)
                    cls_label = np.array(self.class_labels_dict[base_key], dtype=np.float32)
                    self.class_labels.append(cls_label)
        
        # Summary
        print(f"\n[{split.upper()} - Training] Loaded {len(self.filenames)} samples")
        if any(skipped_count.values()):
            print(f"  Skipped: {skipped_count['no_wsi']} (no WSI), "
                  f"{skipped_count['no_coords']} (no coords), "
                  f"{skipped_count['no_label']} (no label)")
            if skipped_samples['no_label']:
                print(f"  Example keys without labels: {skipped_samples['no_label'][:5]}")
                print(f"  Available label keys (first 5): {list(self.class_labels_dict.keys())[:5]}")
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, index: int) -> Tuple:
        """
        Returns:
            filename: Original filename
            image: Composite image from extracted patches (224, 224, 3)
            cls_label: Image-level class labels (num_classes,)
            patch_id: Placeholder for compatibility (0)
        """
        filename = self.filenames[index]
        wsi_path = self.wsi_paths[index]
        coord_path = self.coordinate_paths[index]
        cls_label = self.class_labels[index]
        
        # Load WSI or prepare OpenSlide
        if not self.use_openslide:
            wsi = io.imread(wsi_path)
            if len(wsi.shape) == 2:
                wsi = cv.cvtColor(wsi, cv.COLOR_GRAY2RGB)
            elif wsi.shape[2] == 4:
                wsi = cv.cvtColor(wsi, cv.COLOR_RGBA2RGB)

        # Load coordinates
        coords = load_coordinates(coord_path, self.coordinates_suffix, self.max_patches)
        
        # Randomly sample a coordinate per access for training diversity
        if len(coords) > 0:
            idx = random.randrange(len(coords))
            cx, cy = int(coords[idx, 0]), int(coords[idx, 1])
            if self.use_openslide:
                patch = extract_patch_openslide(wsi_path, cx, cy, self.patch_size)
            else:
                patch = extract_patch_numpy(wsi, cx, cy, self.patch_size)
        else:
            # Fallback: return center patch
            if self.use_openslide:
                # Without dimensions, default top-left
                patch = extract_patch_openslide(wsi_path, 0 + self.patch_size//2, 0 + self.patch_size//2, self.patch_size)
            else:
                h, w = wsi.shape[:2]
                patch = extract_patch_numpy(wsi, w // 2, h // 2, self.patch_size)
        
        # Ensure RGB
        if len(patch.shape) == 2:
            patch = cv.cvtColor(patch, cv.COLOR_GRAY2RGB)

        # Apply transform if provided (expects Albumentations with ToTensorV2)
        if self.transform is not None:
            out = self.transform(image=patch)
            patch_tensor = out["image"]
        else:
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        cls_label_tensor = torch.from_numpy(cls_label).float()

        return filename, patch_tensor, cls_label_tensor, 0


class CustomWSIPatchTestDataset(Dataset):
    """
    Load patches from WSI files with ground truth pixel-level masks.
    Used for validation and testing.
    
    Args:
        wsi_dir: Directory containing WSI (.tif) files
        coordinates_dir: Directory containing coordinates files
        split_csv: CSV file with columns: 'train', 'val', 'test'
        gt_dir: Directory containing ground truth masks (.png files)
        split: Which split to load ('val' or 'test')
        class_labels_dict: Dict mapping filename -> class_labels
        num_classes: Number of classes
        patch_size: Size of patches to extract
        max_patches: Maximum number of patches per WSI
        coordinates_suffix: File suffix for coordinates
        mask_suffix: File suffix for GT masks
    """
    
    def __init__(
        self,
        wsi_dir: str,
        coordinates_dir: str,
        split_csv: str,
        gt_dir: str,
        split: str = "valid",
        class_labels_dict: Optional[Dict] = None,
        num_classes: int = 4,
        patch_size: int = 224,
        max_patches: Optional[int] = None,
        coordinates_suffix: str = ".npy",
        mask_suffix: str = ".png",
        transform=None,
        use_openslide: Optional[bool] = None,
    ):
        super(CustomWSIPatchTestDataset, self).__init__()
        
        self.wsi_dir = wsi_dir
        self.coordinates_dir = coordinates_dir
        self.gt_dir = gt_dir
        self.split = split
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.coordinates_suffix = coordinates_suffix
        self.mask_suffix = mask_suffix
        self.class_labels_dict = class_labels_dict or {}
        self.transform = transform
        self.use_openslide = HAS_OPENSLIDE if use_openslide is None else use_openslide
        
        self.filenames = []
        self.wsi_paths = []
        self.coordinate_paths = []
        self.mask_paths = []
        self.class_labels = []
        
        # Normalize split names
        split_key = split if split in ['train', 'val', 'test'] else ('val' if split == 'valid' else split)
        self._load_split_from_csv(split_csv, split_key)
    
    def _load_split_from_csv(self, split_csv: str, split: str):
        """
        Load filenames from CSV split file with validation.
        
        For VAL/TEST: requires WSI, coordinates, GT mask, AND label (for evaluation)
        """
        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")
        
        skipped_count = {'no_wsi': 0, 'no_coords': 0, 'no_gt': 0, 'no_label': 0}
        skipped_samples = {'no_label': [], 'no_gt': []}
        
        with open(split_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split in row and row[split]:
                    filename = row[split].strip()
                    base_key = os.path.splitext(os.path.basename(filename))[0]
                    
                    # Construct WSI path
                    wsi_name = base_key + ".tif"
                    wsi_path = os.path.join(self.wsi_dir, wsi_name)
                    
                    # Construct coordinates path
                    coord_name = base_key + self.coordinates_suffix
                    coord_path = os.path.join(self.coordinates_dir, coord_name)
                    
                    # Construct mask path
                    mask_name = base_key + self.mask_suffix
                    mask_path = os.path.join(self.gt_dir, mask_name)
                    
                    # Presence checks: require WSI, coordinates, GT, and label
                    if not os.path.exists(wsi_path):
                        skipped_count['no_wsi'] += 1
                        continue
                    if not os.path.exists(coord_path):
                        skipped_count['no_coords'] += 1
                        continue
                    if not os.path.exists(mask_path):
                        skipped_count['no_gt'] += 1
                        if len(skipped_samples['no_gt']) < 5:
                            skipped_samples['no_gt'].append(base_key)
                        continue
                    if base_key not in self.class_labels_dict:
                        skipped_count['no_label'] += 1
                        if len(skipped_samples['no_label']) < 5:
                            skipped_samples['no_label'].append(base_key)
                        continue
                    
                    self.filenames.append(base_key)
                    self.wsi_paths.append(wsi_path)
                    self.coordinate_paths.append(coord_path)
                    self.mask_paths.append(mask_path)
                    
                    # Get class labels (guaranteed present)
                    cls_label = np.array(self.class_labels_dict[base_key], dtype=np.float32)
                    self.class_labels.append(cls_label)
        
        # Summary
        print(f"\n[{split.upper()} - Val/Test] Loaded {len(self.filenames)} samples")
        if any(skipped_count.values()):
            print(f"  Skipped: {skipped_count['no_wsi']} (no WSI), "
                  f"{skipped_count['no_coords']} (no coords), "
                  f"{skipped_count['no_gt']} (no GT mask), "
                  f"{skipped_count['no_label']} (no label)")
            if skipped_samples['no_label']:
                print(f"  Example keys without labels: {skipped_samples['no_label'][:5]}")
            if skipped_samples['no_gt']:
                print(f"  Example keys without GT: {skipped_samples['no_gt'][:5]}")
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, index: int) -> Tuple:
        """
        Returns:
            filename: Original filename
            image: Composite image from extracted patches
            cls_label: Image-level class labels
            mask: Ground truth pixel-level mask
        """
        filename = self.filenames[index]
        wsi_path = self.wsi_paths[index]
        coord_path = self.coordinate_paths[index]
        mask_path = self.mask_paths[index]
        cls_label = self.class_labels[index]
        
        # Load WSI or prepare OpenSlide
        if not self.use_openslide:
            wsi = io.imread(wsi_path)
            if len(wsi.shape) == 2:
                wsi = cv.cvtColor(wsi, cv.COLOR_GRAY2RGB)
            elif wsi.shape[2] == 4:
                wsi = cv.cvtColor(wsi, cv.COLOR_RGBA2RGB)
        
        # Load coordinates and extract patch
        coords = load_coordinates(coord_path, self.coordinates_suffix, self.max_patches)
        if len(coords) > 0:
            cx, cy = int(coords[0, 0]), int(coords[0, 1])
            if self.use_openslide:
                patch = extract_patch_openslide(wsi_path, cx, cy, self.patch_size)
            else:
                patch = extract_patch_numpy(wsi, cx, cy, self.patch_size)
        else:
            h, w = wsi.shape[:2]
            cx, cy = w // 2, h // 2
            if self.use_openslide:
                patch = extract_patch_openslide(wsi_path, cx, cy, self.patch_size)
            else:
                patch = extract_patch_numpy(wsi, cx, cy, self.patch_size)
        
        if len(patch.shape) == 2:
            patch = cv.cvtColor(patch, cv.COLOR_GRAY2RGB)
        
        # Load mask and crop the same region as the image patch
        full_mask = np.array(Image.open(mask_path))
        # Build a dummy 3-channel view for consistent padding logic
        mask_rgb = np.stack([full_mask]*3, axis=-1)
        mask_patch = extract_patch_numpy(mask_rgb, cx, cy, self.patch_size)[:, :, 0]

        # Apply transform if provided (expects Albumentations with ToTensorV2 and transpose_mask=True)
        if self.transform is not None:
            out = self.transform(image=patch, mask=mask_patch)
            patch_tensor = out["image"]
            mask_tensor = out["mask"].long()
        else:
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask_patch).long()
        cls_label_tensor = torch.from_numpy(cls_label).float()

        return filename, patch_tensor, cls_label_tensor, mask_tensor


class TileDataset(Dataset):
    """
    Slide2Vec-style tile dataset that loads patches from WSI using coordinates
    stored with metadata (tile_level, tile_size_resized, spacing info).
    
    Coordinates are loaded from .npy files with structure:
    {
        'x': array of x coordinates,
        'y': array of y coordinates,
        'tile_level': array of zoom levels,
        'tile_size_resized': array of resized tile sizes,
        'tile_size': array of target tile sizes,
        'tile_size_lv0': target tile size at level 0,
        'resize_factor': resize factor applied to tiles
    }
    
    Args:
        wsi_path: Path to WSI file
        tile_dir: Directory containing coordinate .npy files
        target_spacing: Target spacing in microns/pixel
        backend: Backend for wholeslidedata ('asap', 'openslide', etc.)
        transforms: Albumentations transforms pipeline
    """
    
    def __init__(
        self,
        wsi_path: str,
        tile_dir: str,
        target_spacing: float = 0.5,
        backend: str = "asap",
        transforms=None,
    ):
        self.path = Path(wsi_path)
        self.target_spacing = target_spacing
        self.backend = backend
        self.name = self.path.stem.replace(" ", "_")
        self.transforms = transforms
        
        # Try to use wholeslidedata, fallback to OpenSlide
        self.use_wsd = HAS_WSD
        
        self.load_coordinates(tile_dir)
    
    def load_coordinates(self, tile_dir: str):
        """Load tile coordinates from .npy file matching WSI name."""
        coord_path = Path(tile_dir, f"{self.name}.npy")
        
        if not coord_path.exists():
            raise FileNotFoundError(f"Coordinates file not found: {coord_path}")
        
        # Load with allow_pickle=True to support structured numpy arrays
        coordinates = np.load(coord_path, allow_pickle=True)
        
        # Handle both structured arrays and regular arrays
        if isinstance(coordinates, np.ndarray) and coordinates.dtype.names:
            # Structured array (slide2vec format)
            self.x = coordinates["x"]
            self.y = coordinates["y"]
            self.tile_level = coordinates.get("tile_level", np.zeros(len(self.x), dtype=int))
            self.tile_size_resized = coordinates.get("tile_size_resized", 
                                                      np.full(len(self.x), 224, dtype=int))
            self.resize_factor = coordinates.get("resize_factor", np.ones(len(self.x)))
            self.tile_size = coordinates.get("tile_size", self.tile_size_resized)
            self.tile_size_lv0 = coordinates.get("tile_size_lv0", [224])[0]
        else:
            # Regular array format (just x, y coordinates)
            if len(coordinates.shape) == 1:
                self.x = coordinates[::2]  # Even indices are x
                self.y = coordinates[1::2]  # Odd indices are y
            else:
                self.x = coordinates[:, 0]
                self.y = coordinates[:, 1]
            
            self.tile_level = np.zeros(len(self.x), dtype=int)
            self.tile_size_resized = np.full(len(self.x), 224, dtype=int)
            self.tile_size = self.tile_size_resized
            self.tile_size_lv0 = 224
            self.resize_factor = np.ones(len(self.x))
        
        self.coordinates = np.column_stack([self.x, self.y]).astype(int)
        self.scaled_coordinates = self.scale_coordinates()
    
    def scale_coordinates(self) -> np.ndarray:
        """Scale coordinates from level 0 to target spacing."""
        if not self.use_wsd:
            # Without WSD, return coordinates as-is (assume already at target spacing)
            return self.coordinates.copy()
        
        try:
            wsi = wsd.WholeSlideImage(str(self.path), backend=self.backend)
            min_spacing = wsi.spacings[0] if wsi.spacings else 1.0
            scale = min_spacing / self.target_spacing
            scaled = (self.coordinates * scale).astype(int)
            return scaled
        except Exception as e:
            print(f"Warning: Could not scale coordinates: {e}, using as-is")
            return self.coordinates.copy()
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns:
            idx: Index
            tile: Patch image tensor (C, H, W)
        """
        if self.use_wsd:
            tile_arr = self._get_patch_wsd(idx)
        else:
            tile_arr = self._get_patch_openslide(idx)
        
        tile = Image.fromarray(tile_arr).convert("RGB")
        
        # Resize if needed
        if self.tile_size[idx] != self.tile_size_resized[idx]:
            tile = tile.resize((self.tile_size[idx], self.tile_size[idx]))
        
        # Apply transforms
        if self.transforms:
            tile = self.transforms(image=np.array(tile))["image"]
        else:
            tile = torch.from_numpy(np.array(tile)).permute(2, 0, 1).float() / 255.0
        
        return idx, tile
    
    def _get_patch_wsd(self, idx: int) -> np.ndarray:
        """Extract patch using wholeslidedata."""
        try:
            wsi = wsd.WholeSlideImage(str(self.path), backend=self.backend)
            tile_level = int(self.tile_level[idx])
            tile_spacing = wsi.spacings[tile_level]
            
            tile_arr = wsi.get_patch(
                int(self.x[idx]),
                int(self.y[idx]),
                int(self.tile_size_resized[idx]),
                int(self.tile_size_resized[idx]),
                spacing=tile_spacing,
                center=False,
            )
            return tile_arr
        except Exception as e:
            print(f"WSD patch extraction failed at ({self.x[idx]}, {self.y[idx]}): {e}")
            # Fallback to OpenSlide
            return self._get_patch_openslide(idx)
    
    def _get_patch_openslide(self, idx: int) -> np.ndarray:
        """Extract patch using OpenSlide."""
        if not HAS_OPENSLIDE:
            raise RuntimeError("OpenSlide not available and WSD patch extraction failed")
        
        half = int(self.tile_size_resized[idx] // 2)
        top_left = (max(0, int(self.x[idx]) - half), max(0, int(self.y[idx]) - half))
        
        try:
            slide = _openslide.OpenSlide(str(self.path))
            region = slide.read_region(top_left, 0, 
                                     (int(self.tile_size_resized[idx]), 
                                      int(self.tile_size_resized[idx])))
            slide.close()
            tile_arr = np.array(region.convert('RGB'))
            return tile_arr
        except Exception as e:
            print(f"OpenSlide patch extraction failed: {e}")
            raise

class CustomWSIPatchTrainingDatasetWithPseudoLabels(CustomWSIPatchTrainingDataset):
    """
    Enhanced training dataset that integrates pseudo-label driven patch selection.
    
    This dataset extends CustomWSIPatchTrainingDataset to support WSI-level classification
    by incorporating attention scores (pseudo-labels) for each patch coordinate. It enables
    the "Selection-then-Prompting" strategy for building high-quality prototype banks.
    
    Only patches with high attention scores (high confidence) are used for prototype
    construction, preventing the prototype bank from being "poisoned" by weak-label artifacts.
    
    Args:
        use_pseudo_labels: Whether to enable pseudo-label filtering
        pseudo_label_dir: Directory containing .pt files with attention scores
        pseudo_label_binary_mode: If True, interpret scores as binary (low/high). 
                                  If False, multi-class (one score per class)
        pseudo_label_selection_strategy: Strategy for selecting patches ('percentile', 
                                        'threshold', 'entropy', 'margin')
        pseudo_label_confidence_threshold: Confidence threshold for selection
        pseudo_label_min_patches: Minimum selected patches per WSI to be usable
        ... (other CustomWSIPatchTrainingDataset arguments)
    """
    
    def __init__(
        self,
        wsi_dir: str,
        coordinates_dir: str,
        split_csv: str,
        split: str = "train",
        class_labels_dict: Optional[Dict] = None,
        num_classes: int = 4,
        patch_size: int = 224,
        max_patches: Optional[int] = None,
        coordinates_suffix: str = ".npy",
        transform=None,
        use_openslide: Optional[bool] = None,
        # Pseudo-label specific arguments
        use_pseudo_labels: bool = True,
        pseudo_label_dir: Optional[str] = None,
        pseudo_label_binary_mode: bool = True,
        pseudo_label_selection_strategy: str = 'percentile',
        pseudo_label_confidence_threshold: float = 0.85,
        pseudo_label_min_patches: int = 5,
        pseudo_label_analyze: bool = True,
    ):
        super().__init__(
            wsi_dir=wsi_dir,
            coordinates_dir=coordinates_dir,
            split_csv=split_csv,
            split=split,
            class_labels_dict=class_labels_dict,
            num_classes=num_classes,
            patch_size=patch_size,
            max_patches=max_patches,
            coordinates_suffix=coordinates_suffix,
            transform=transform,
            use_openslide=use_openslide,
        )
        
        self.use_pseudo_labels = use_pseudo_labels
        self.pseudo_label_dir = pseudo_label_dir
        self.pseudo_label_binary_mode = pseudo_label_binary_mode
        self.pseudo_label_selection_strategy = pseudo_label_selection_strategy
        self.pseudo_label_confidence_threshold = pseudo_label_confidence_threshold
        self.pseudo_label_min_patches = pseudo_label_min_patches
        
        # Initialize pseudo-label components if enabled
        if self.use_pseudo_labels and pseudo_label_dir:
            try:
                self.pseudo_label_loader = PseudoLabelLoader(
                    pseudo_label_dir=pseudo_label_dir,
                    binary_mode=pseudo_label_binary_mode,
                    num_classes=num_classes,
                )
                
                self.patch_selector = PatchSelector(
                    num_classes=num_classes,
                    selection_strategy=pseudo_label_selection_strategy,
                )
                
                # Pre-load and analyze pseudo-labels
                self._initialize_pseudo_labels(pseudo_label_analyze)
                
            except Exception as e:
                print(f"Warning: Failed to initialize pseudo-labels: {e}")
                print("Continuing without pseudo-label filtering")
                self.use_pseudo_labels = False
        else:
            self.pseudo_label_loader = None
            self.patch_selector = None
    
    def _initialize_pseudo_labels(self, analyze: bool = True):
        """Load pseudo-labels for all samples and analyze quality."""
        print("\n" + "="*70)
        print("PSEUDO-LABEL INITIALIZATION")
        print("="*70)
        
        # Load pseudo-labels for all samples
        self.pseudo_label_scores = {}  # wsi_name -> scores tensor
        self.high_conf_patch_indices = {}  # wsi_name -> selected patch indices
        
        skipped_wsis = []
        insufficient_patches = []
        
        for wsi_name in self.filenames:
            try:
                # Load scores for this WSI
                scores = self.pseudo_label_loader.load_wsi_scores(wsi_name)
                self.pseudo_label_scores[wsi_name] = scores
                
                # Select high-confidence patches
                selected_mask = self.patch_selector.select_patches(
                    scores,
                    confidence_threshold=self.pseudo_label_confidence_threshold,
                    return_scores=False
                )
                
                selected_indices = np.where(selected_mask)[0]
                
                # Check minimum threshold
                if len(selected_indices) < self.pseudo_label_min_patches:
                    insufficient_patches.append((wsi_name, len(selected_indices)))
                    self.high_conf_patch_indices[wsi_name] = selected_indices
                else:
                    self.high_conf_patch_indices[wsi_name] = selected_indices
                    
            except Exception as e:
                print(f"Warning: Failed to load pseudo-labels for {wsi_name}: {e}")
                skipped_wsis.append(wsi_name)
        
        # Print statistics
        print(f"\nLoaded pseudo-labels for {len(self.pseudo_label_scores)}/{len(self.filenames)} WSIs")
        
        if skipped_wsis:
            print(f"Skipped {len(skipped_wsis)} WSIs due to errors (first 5):")
            for wsi_name in skipped_wsis[:5]:
                print(f"  - {wsi_name}")
        
        if insufficient_patches:
            print(f"\n{len(insufficient_patches)} WSIs have insufficient high-confidence patches:")
            print(f"  (minimum required: {self.pseudo_label_min_patches})")
            for wsi_name, num_patches in insufficient_patches[:5]:
                print(f"  - {wsi_name}: {num_patches} patches")
        
        # Analyze global statistics if requested
        if analyze and self.pseudo_label_loader:
            try:
                from utils.pseudo_labels import PseudoLabelAnalyzer
                analyzer = PseudoLabelAnalyzer(self.pseudo_label_loader)
                stats = analyzer.analyze_all_wsis()
                
                print(f"\nGlobal Pseudo-Label Statistics:")
                print(f"  Mode: {'Binary' if self.pseudo_label_binary_mode else 'Multi-class'}")
                print(f"  Total WSIs analyzed: {stats['num_wsis']}")
                
                if 'global_stats' in stats:
                    gs = stats['global_stats']
                    if 'mean' in gs:
                        print(f"  Global mean score: {gs['mean']:.4f}")
                        print(f"  Global std: {gs['std']:.4f}")
                        print(f"  Global min: {gs['min']:.4f}")
                        print(f"  Global max: {gs['max']:.4f}")
            except Exception as e:
                print(f"Warning: Could not analyze pseudo-label statistics: {e}")
        
        print("="*70 + "\n")
    
    def get_high_confidence_patches(self, wsi_name: str) -> np.ndarray:
        """
        Get indices of high-confidence patches for a WSI.
        
        Returns:
            numpy array of patch indices selected by the patch selector
        """
        if not self.use_pseudo_labels or wsi_name not in self.high_conf_patch_indices:
            return None
        
        return self.high_conf_patch_indices[wsi_name]
    
    def get_wsi_pseudo_scores(self, wsi_name: str) -> Optional[torch.Tensor]:
        """Get pseudo-label scores for a WSI."""
        if not self.use_pseudo_labels or wsi_name not in self.pseudo_label_scores:
            return None
        
        return self.pseudo_label_scores[wsi_name]
    
    def __getitem__(self, index: int) -> Tuple:
        """
        Returns:
            filename: Original filename
            image: Extracted patch tensor (C, H, W)
            cls_label: Image-level class labels
            patch_id: Placeholder (0)
            pseudo_label: (optional) Pseudo-label score for this patch coordinate
        """
        filename, patch_tensor, cls_label, patch_id = super().__getitem__(index)
        
        # Optionally return pseudo-label information
        if self.use_pseudo_labels:
            wsi_name = self.filenames[index]
            pseudo_scores = self.get_wsi_pseudo_scores(wsi_name)
            
            if pseudo_scores is not None:
                # Return tuple with pseudo-label info
                return filename, patch_tensor, cls_label, patch_id, pseudo_scores
        
        return filename, patch_tensor, cls_label, patch_id