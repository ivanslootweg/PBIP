import pandas as pd
import torch
from torch.utils.data import Dataset
import openslide
import numpy as np
import os
from PIL import Image
import h5py
from utils.common import load_coordinates

class PseudoPatchLabelDataset(Dataset):
    def __init__(self, csv_path, config=None, split='train', transform=None, binary_mode=True, scale_attention=True):
        self.df = pd.read_csv(csv_path)
        
        # Determine column names from config or default
        self.col_map = {
            'image_name': 'image_name',
            'wsi_label': 'wsi_label',
            'coords': 'attention_coordinates_path', # Default per user request
            'scores': 'attention_scores_path',
            'split': 'split',
            'wsi_path': 'wsi_path'
        }
        
        self.wsi_dir = None
        self.wsi_ext = '.tif'
        
        if config:
            if 'columns' in config:
                self.col_map.update(config['columns'])
            self.wsi_dir = config.get('wsi_dir')
            self.wsi_ext = config.get('wsi_extension', '.tif')
            
        col_split = self.col_map['split']
        if col_split in self.df.columns:
            self.df = self.df[self.df[col_split] == split].reset_index(drop=True)
            
        self.transform = transform
        self.binary_mode = binary_mode
        self.scale_attention = scale_attention
        
        self.samples = [] 
        self.wsi_data = [] 
        
        print(f"Initializing PseudoPatchLabelDataset with {len(self.df)} slides...")
        self._prepare_index()

    def _prepare_index(self):
        col_name = self.col_map['image_name']
        col_coords = self.col_map['coords']
        col_scores = self.col_map['scores']
        col_label = self.col_map['wsi_label']
        col_wsi_path = self.col_map.get('wsi_path', 'wsi_path')
        
        for idx, row in self.df.iterrows():
            # WSI Path construction
            # Priority: Absolute path in CSV > wsi_dir + image_name
            wsi_path = None
            if col_wsi_path in row and pd.notna(row[col_wsi_path]):
                wsi_path = str(row[col_wsi_path])
            
            if not wsi_path or not os.path.exists(wsi_path):
                # Fallback to constructing from dir
                fname = str(row[col_name])
                if self.wsi_dir:
                    name_part = fname
                    if not name_part.lower().endswith(tuple(['.tif', '.svs', '.ndpi', '.mrxs', '.tiff'])):
                        name_part += self.wsi_ext
                    wsi_path = os.path.join(self.wsi_dir, name_part)
                else:
                    if not wsi_path: wsi_path = fname # Keep what we had if no directory

            if not os.path.exists(wsi_path):
                # Only warn if we really can't find it, might be just missing from this split or valid skip
                # print(f"Warning: WSI not found for {row[col_name]} at {wsi_path}")
                continue

            # Coords
            if col_coords not in row or pd.isna(row[col_coords]):
                 continue
                 
            coords_path = row[col_coords]
            coords = load_coordinates(coords_path)
                
            if len(coords) == 0:
                continue

            # Scores
            if col_scores not in row or pd.isna(row[col_scores]):
                continue

            attn_path = row[col_scores]
            try:
                # Use safe globals or weights_only=False if trusted. 
                # User's error suggests they have complex types (numpy globals).
                attn_data = torch.load(attn_path, map_location='cpu', weights_only=False)
                
                # Handle Dict vs Tensor
                attn_scores = None
                if isinstance(attn_data, dict):
                    if 'patch_logits' in attn_data:
                        attn_scores = attn_data['patch_logits']
                    elif 'logits' in attn_data: # Fallback, though likely slide logits
                        attn_scores = attn_data['logits'] 
                    else:
                        # Try to find any tensor that matches coords length
                        for k, v in attn_data.items():
                            if isinstance(v, torch.Tensor) and len(v) == len(coords):
                                attn_scores = v
                                break
                elif isinstance(attn_data, torch.Tensor):
                    attn_scores = attn_data
                elif isinstance(attn_data, np.ndarray):
                    attn_scores = attn_data
                
                if attn_scores is None:
                    print(f"Skipping {row[col_name]}: Could not find scores/patch_logits in {attn_path}")
                    continue

                if isinstance(attn_scores, torch.Tensor):
                    attn_scores = attn_scores.numpy()
            except Exception as e:
                print(f"Error loading scores from {attn_path}: {e}")
                continue
            
            # Check length mismatch
            if len(coords) != len(attn_scores):
                l = min(len(coords), len(attn_scores))
                coords = coords[:l]
                attn_scores = attn_scores[:l]

            if self.scale_attention:
                if len(attn_scores.shape) == 1:
                     min_v = attn_scores.min()
                     max_v = attn_scores.max()
                     if max_v - min_v > 0:
                         attn_scores = (attn_scores - min_v) / (max_v - min_v)
                else:
                    for c in range(attn_scores.shape[1]):
                        min_v = attn_scores[:, c].min()
                        max_v = attn_scores[:, c].max()
                        if max_v - min_v > 0:
                            attn_scores[:, c] = (attn_scores[:, c] - min_v) / (max_v - min_v)

            wsi_label = row[col_label]
            
            self.wsi_data.append({
                'path': wsi_path,
                'coords': coords,
                'scores': attn_scores,
                'wsi_label': wsi_label,
                'slide_ob': None,
                'annotation_path': row.get('annotation_path', None)
            })
            
            num_patches = len(coords)
            for p_idx in range(num_patches):
                self.samples.append((idx, p_idx))

    def __len__(self):
        return len(self.samples)
        
    def _get_slide(self, wsi_idx):
        data = self.wsi_data[wsi_idx]
        if data['slide_ob'] is None:
            try:
                data['slide_ob'] = openslide.OpenSlide(data['path'])
            except Exception as e:
                # print(f"Error opening slide {data['path']}: {e}")
                return None
        return data['slide_ob']

    def __getitem__(self, idx):
        wsi_idx, p_idx = self.samples[idx]
        data = self.wsi_data[wsi_idx]
        
        coord = data['coords'][p_idx]
        score = data['scores'][p_idx]
        wsi_label = data['wsi_label']
        
        img = np.zeros((224, 224, 3), dtype=np.uint8) 

        slide = self._get_slide(wsi_idx)
        if slide:
            try:
                patch_size = 224
                # Coordinate files (like from CLAM) are usually (x, y) top-left at level 0
                x, y = int(coord[0]), int(coord[1])
                img = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
                img = np.array(img)
            except Exception as e:
                pass
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
            
        target = score
        
        if self.binary_mode and target is not None:
            if float(wsi_label) == 0.0: # Negative WSI
                target = 0.0 # Force 0
            else:
                if isinstance(target, (list, np.ndarray)) and len(target.shape) > 0 and target.shape[0] > 1:
                     target = target[0]
        
        # Handle Multiclass target: If target is [C0, C1, ...], and we want `target` to be tensor of shape [C]
        if not self.binary_mode:
            # PBIP trainer expects targets to be used in loss_fn(pred, targets)
            # If targets is an array [ProbC1, ProbC2...], we just return it.
            pass

        mask = np.zeros((224, 224), dtype=np.uint8)
        # TODO: Implement mask loading if annotation_path is a readable mask (e.g. tif)
        # XML Support would require rasterization lib.
        
        return img, target, mask
