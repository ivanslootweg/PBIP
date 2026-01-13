"""
Common utility functions to reduce code duplication across the codebase.
"""

import numpy as np
import torch
from typing import Tuple, List, Optional

try:
    import openslide as _openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False


def load_coordinates(coord_path: str, coordinates_suffix: str = '.npy', 
                    max_patches: Optional[int] = None) -> np.ndarray:
    """
    Load coordinates from file (.npy or .txt) with optional sampling.
    
    Handles both simple arrays and structured arrays (slide2vec format).
    
    Args:
        coord_path: Path to coordinate file
        coordinates_suffix: File extension (.npy or .txt)
        max_patches: If set, randomly sample this many patches
        
    Returns:
        Array of shape (N, 2) with [x, y] coordinates as int32
    """
    if coordinates_suffix == ".npy":
        coords = np.load(coord_path)
        
        # Check if it's a structured array with named fields (slide2vec format)
        if coords.dtype.names:
            # Extract x, y coordinates from structured array
            x_coords = coords['x']
            y_coords = coords['y']
            coords = np.stack([x_coords, y_coords], axis=1)
            
    elif coordinates_suffix == ".txt":
        coords = np.loadtxt(coord_path)
    else:
        raise ValueError(f"Unsupported coordinates suffix: {coordinates_suffix}")
    
    # Limit to max_patches if specified
    if max_patches and len(coords) > max_patches:
        indices = np.random.choice(len(coords), max_patches, replace=False)
        coords = coords[indices]
    
    return coords.astype(np.int32)


def extract_patch_numpy(wsi: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
    """
    Extract a patch from a numpy array WSI with padding if needed.
    
    Args:
        wsi: WSI as numpy array (H, W, C)
        x: Center x coordinate
        y: Center y coordinate
        patch_size: Size of square patch
        
    Returns:
        Patch of shape (patch_size, patch_size, C)
    """
    import cv2 as cv
    
    h, w = wsi.shape[:2]
    half = patch_size // 2
    
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x1 + patch_size)
    y2 = min(h, y1 + patch_size)
    
    patch = wsi[y1:y2, x1:x2]
    
    # Pad if necessary
    if patch.shape[:2] != (patch_size, patch_size):
        patch = cv.copyMakeBorder(
            patch,
            0, patch_size - patch.shape[0],
            0, patch_size - patch.shape[1],
            cv.BORDER_CONSTANT, value=255
        )
    
    return patch


def extract_patch_openslide(wsi_path: str, x: int, y: int, patch_size: int) -> np.ndarray:
    """
    Extract a patch from WSI using OpenSlide.
    
    Args:
        wsi_path: Path to WSI file
        x: Center x coordinate
        y: Center y coordinate
        patch_size: Size of square patch
        
    Returns:
        Patch as RGB numpy array of shape (patch_size, patch_size, 3)
    """
    if not HAS_OPENSLIDE:
        raise ImportError("OpenSlide not available. Install with: pip install openslide-python")
    
    half = patch_size // 2
    top_left = (max(0, x - half), max(0, y - half))
    
    slide = _openslide.OpenSlide(wsi_path)
    region = slide.read_region(top_left, 0, (patch_size, patch_size))
    slide.close()
    
    region = region.convert('RGB')
    patch = np.array(region)
    
    return patch


def extract_patch(wsi_path: str, x: int, y: int, patch_size: int, 
                 use_openslide: bool = True) -> np.ndarray:
    """
    Extract a patch from WSI file using either OpenSlide or numpy.
    
    Args:
        wsi_path: Path to WSI file
        x: Center x coordinate
        y: Center y coordinate  
        patch_size: Size of square patch
        use_openslide: If True, use OpenSlide; else load as numpy array
        
    Returns:
        Patch as RGB numpy array of shape (patch_size, patch_size, 3)
    """
    import cv2 as cv
    from skimage import io
    
    if use_openslide and HAS_OPENSLIDE:
        return extract_patch_openslide(wsi_path, x, y, patch_size)
    else:
        # Load full WSI as numpy array
        wsi = io.imread(wsi_path)
        
        # Convert to RGB if needed
        if len(wsi.shape) == 2:
            wsi = cv.cvtColor(wsi, cv.COLOR_GRAY2RGB)
        elif wsi.shape[2] == 4:
            wsi = cv.cvtColor(wsi, cv.COLOR_RGBA2RGB)
        
        return extract_patch_numpy(wsi, x, y, patch_size)


def merge_multiscale_predictions(predictions: List[torch.Tensor], 
                                 k_list: List[int], 
                                 method: str = 'mean') -> List[torch.Tensor]:
    """
    Merge all 4 scales of subclass predictions to parent class predictions.
    
    Args:
        predictions: List of [cls1, cls2, cls3, cls4] tensors
        k_list: Number of subclasses per parent class
        method: Merge method ('mean' or 'max')
        
    Returns:
        List of merged predictions [cls1_merge, cls2_merge, cls3_merge, cls4_merge]
    """
    from utils.hierarchical_utils import merge_to_parent_predictions
    
    return [merge_to_parent_predictions(pred, k_list, method=method) 
            for pred in predictions]


def merge_multiscale_cams(cams: List[torch.Tensor], 
                         k_list: List[int], 
                         method: str = 'mean') -> List[torch.Tensor]:
    """
    Merge all 4 scales of subclass CAMs to parent class CAMs.
    
    Args:
        cams: List of [cam1, cam2, cam3, cam4] tensors
        k_list: Number of subclasses per parent class
        method: Merge method ('mean' or 'max')
        
    Returns:
        List of merged CAMs [cam1_merge, cam2_merge, cam3_merge, cam4_merge]
    """
    from utils.hierarchical_utils import merge_subclass_cams_to_parent
    
    return [merge_subclass_cams_to_parent(cam, k_list, method=method) 
            for cam in cams]


def compute_multiscale_loss(predictions: List[torch.Tensor], 
                           labels: torch.Tensor,
                           loss_fn,
                           weights: Tuple[float, float, float, float]) -> torch.Tensor:
    """
    Compute weighted multi-scale classification loss.
    
    Args:
        predictions: List of 4 prediction tensors [cls1, cls2, cls3, cls4]
        labels: Ground truth labels
        loss_fn: Loss function (e.g., BCEWithLogitsLoss)
        weights: Tuple of 4 weights (scale1_weight, scale2_weight, scale3_weight, scale4_weight)
        
    Returns:
        Weighted sum of losses across scales
    """
    losses = [loss_fn(pred, labels) for pred in predictions]
    return sum(w * loss for w, loss in zip(weights, losses))


def get_color_palette(num_classes: int, include_background: bool = True) -> List[List[int]]:
    """
    Generate a color palette for visualization.
    
    Args:
        num_classes: Number of foreground classes
        include_background: If True, add white background color at the end
        
    Returns:
        List of RGB colors as [R, G, B] lists
    """
    base_palette = [
        [255, 0, 0],      # Class 0: Red
        [0, 255, 0],      # Class 1: Green
        [0, 0, 255],      # Class 2: Blue
        [153, 0, 255],    # Class 3: Purple
        [255, 255, 0],    # Class 4: Yellow
        [0, 255, 255],    # Class 5: Cyan
        [255, 128, 0],    # Class 6: Orange
        [128, 0, 255],    # Class 7: Violet
        [255, 0, 128],    # Class 8: Pink
        [128, 255, 0],    # Class 9: Lime
    ]
    
    palette = base_palette[:num_classes]
    
    if include_background:
        palette.append([255, 255, 255])  # White for background
    
    return palette
