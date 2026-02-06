import numpy as np
import os
import h5py
from typing import Optional

def load_coordinates(coord_path: str, coordinates_suffix: Optional[str] = None, max_patches: Optional[int] = None) -> np.ndarray:
    """
    Load coordinates from file (.npy, .txt, or .h5) with optional sampling.
    Handles both simple arrays and structured arrays (slide2vec format).
    """
    coords = []
    
    # Infer suffix if not provided or just use file extension
    if str(coord_path).endswith('.npy'):
        try:
            c = np.load(coord_path, allow_pickle=True) # Allow pickle for structured arrays if needed
            # Handle structured arrays common in pathology (x, y fields)
            if c.dtype.names and ('x' in c.dtype.names) and ('y' in c.dtype.names):
                coords = np.stack([c['x'], c['y']], axis=1)
            else:
                coords = c
        except Exception as e:
            print(f"Error loading .npy {coord_path}: {e}")
            return np.array([])
            
    elif str(coord_path).endswith('.h5'):
        try:
            with h5py.File(coord_path, 'r') as f:
                # Try common keys
                if 'coords' in f:
                    coords = f['coords'][:]
                elif 'coordinates' in f:
                    coords = f['coordinates'][:]
                elif 'patch_coords' in f:
                    coords = f['patch_coords'][:]
                else:
                    keys = list(f.keys())
                    if len(keys) > 0:
                        coords = f[keys[0]][:]
        except Exception as e:
             print(f"Error loading .h5 {coord_path}: {e}")
             return np.array([])
             
    elif str(coord_path).endswith('.txt'):
        try:
            coords = np.loadtxt(coord_path)
        except Exception as e:
             print(f"Error loading .txt {coord_path}: {e}")
             return np.array([])
             
    elif str(coord_path).endswith('.pt'):
         # Added .pt support for this specific project context
        import torch
        try:
            c = torch.load(coord_path)
            coords = c.numpy() if isinstance(c, torch.Tensor) else c
        except Exception as e:
             print(f"Error loading .pt {coord_path}: {e}")
             return np.array([])
             
    else:
        # Fallback or unknown
        pass

    if isinstance(coords, (list, tuple)):
        coords = np.array(coords)
        
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        # Try to fix shape if manageable
        if len(coords.shape) == 1 and len(coords) > 0:
             # Weird case
             pass
        else:
            # print(f"Warning: Unexpected coord shape {coords.shape} for {coord_path}")
            pass

    # Random sampling if requested
    if max_patches is not None and len(coords) > max_patches:
        indices = np.random.choice(len(coords), max_patches, replace=False)
        coords = coords[indices]

    return coords.astype(np.int32)
