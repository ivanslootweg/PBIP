"""

NEW APPROACH:
The pipeline uses precomputed patch features (from encoders like Virchow2, DinoV3)
loaded via FeatureWSIDataset (datasets/feature_dataset.py). This eliminates the need for:
- On-the-fly patch extraction from WSI files
- Pixel-level segmentation predictions
- Multi-scale CAM generation

CLASSES REMOVED:
- PatchLevelTrainingDataset
- PatchLevelTestDataset

MIGRATION GUIDE:
1. Ensure patch_features_dir is set in your config (e.g., /path/to/virchow2_features/)
2. Use datasets/feature_dataset.py FeatureWSIDataset instead
3. Update config: use_feature_based_training: true (or set patch_features_dir)

Dataset for precomputed patch features.

Returns tuples: (wsi_name, patch_features, cls_label, global_feature)
- patch_features: torch.Tensor (n_patches, feature_dim)
- cls_label: torch.Tensor (num_classes,)
- global_feature: torch.Tensor (feature_dim,) loaded from global_feature_dir or mean of patches

Supported patch feature file formats: .pt (torch.save), .npy, .npz, .pkl
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import pickle as pkl
from PIL import Image
from xml.etree import ElementTree as ET
import time

# local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.trainutils import load_class_labels_from_csv
from utils.encoders import EncoderFactory


class FeatureWSIDataset(Dataset):
    def __init__(
        self,
        patch_features_dir: str,
        split_csv: str,
        labels_csv: str,
        split: str = "train",
        num_classes: int = 2,
        file_suffixes: List[str] = None,
        global_feature_dir: Optional[str] = None,
        wsi_dir: Optional[str] = None,
        auto_generate_patch_features: bool = True,
        auto_generate_global_features: bool = True,
        patch_encoder: str = "virchow2",
        device: Optional[torch.device] = None,
        gt_dir: Optional[str] = None,  # Directory with ground truth masks for segmentation metrics
        binary_mode: bool = False,  # True: masks are binary (0/1), False: masks are multi-class (0,1,2,...)
        coordinates_dir: Optional[str] = None,  # Directory with patch coordinate files
        coordinates_suffix: str = "_patches.h5",  # Coordinate file suffix
        patch_size: int = 224,  # Patch size used during feature extraction
        fallback_patch_features_dir: Optional[str] = None,  # Fallback feature directory
        fallback_coordinates_dir: Optional[str] = None,  # Fallback coordinate directory
        fallback_coordinates_suffix: Optional[str] = None,  # Fallback coordinate suffix
        pseudo_label_dir: Optional[str] = None,  # Directory with pseudo-labels (attention scores)
        verbose: bool = False,  # Enable detailed timing logs
    ):
        super().__init__()
        self.verbose = verbose
        self.patch_features_dir = Path(patch_features_dir)
        self.global_feature_dir = Path(global_feature_dir) if global_feature_dir else None
        self.split_csv = split_csv
        self.labels_csv = labels_csv
        self.split = split
        self.num_classes = num_classes
        self.wsi_dir = Path(wsi_dir) if wsi_dir else None
        self.auto_generate_patch_features = auto_generate_patch_features
        self.auto_generate_global_features = auto_generate_global_features
        self.patch_encoder = patch_encoder
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Optional: Ground truth masks for segmentation evaluation
        self.gt_dir = Path(gt_dir) if gt_dir is not None else None
        self.binary_mode = binary_mode
        self.coordinates_dir = Path(coordinates_dir) if coordinates_dir is not None else None
        self.coordinates_suffix = coordinates_suffix
        self.patch_size = patch_size
        
        # Simple coordinate caching (per-WSI) to avoid reloading same files repeatedly
        self._coord_cache = {}  # {wsi_basename: (coords_x, coords_y, coords_array)}
        
        # Fallback directories
        self.fallback_patch_features_dir = Path(fallback_patch_features_dir) if fallback_patch_features_dir else None
        self.fallback_coordinates_dir = Path(fallback_coordinates_dir) if fallback_coordinates_dir else None
        self.fallback_coordinates_suffix = fallback_coordinates_suffix if fallback_coordinates_suffix else coordinates_suffix
        
        # Pseudo-label directory
        self.pseudo_label_dir = pseudo_label_dir
        self.benign_pseudo_label_floor = None  # Will be computed on first access

        if file_suffixes is None:
            self.suffixes = ['.pt', '.npy', '.npz', '.pkl', '.h5',"_patches.h5"]
        else:
            self.suffixes = file_suffixes

        # Create patch features directory if it doesn't exist (will generate on demand)
        self.patch_features_dir.mkdir(parents=True, exist_ok=True)
        
        # Create global features directory if it doesn't exist
        if self.global_feature_dir:
            self.global_feature_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature extractors for on-the-fly generation
        self.patch_encoder_model = None
        self.patch_feature_dim = None
        self.global_encoder_model = None
        self.global_feature_dim = None

        # Load split and labels
        self.filenames = []  # basenames without extensions
        self.class_labels = load_class_labels_from_csv(labels_csv, num_classes)

        # Read split CSV and collect filenames in the requested split column
        import csv
        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")

        with open(split_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.split in row and row[self.split]:
                    basename = os.path.splitext(os.path.basename(row[self.split].strip()))[0]
                    if basename not in self.class_labels:
                        # skip if no slide-level label
                        continue
                    self.filenames.append(basename)

        if not self.filenames:
            raise ValueError(f"No items found for split '{self.split}' in split CSV with available labels")

        # Compute 0.1 percentile floor for benign pseudo-labels from training set positive cases
        if self.split == "train" and self.pseudo_label_dir:
            self.benign_pseudo_label_floor = self._compute_pseudo_label_percentile()
            if self.benign_pseudo_label_floor is not None:
                print(f"[PSEUDO-LABELS] Using 0.1 percentile floor for benign class: {self.benign_pseudo_label_floor:.4f}")

    def _compute_pseudo_label_percentile(self) -> Optional[float]:
        """
        Compute 0.1 percentile of positive class pseudo-labels from training set.
        This value will be used as the pseudo-label floor for benign (negative) cases.
        
        Returns:
            0.1 percentile value, or None if not enough positive pseudo-labels found
        """
        if not self.pseudo_label_dir or not Path(self.pseudo_label_dir).exists():
            return None
        
        pseudo_label_extensions = ['.pt', '.pth', '.npy', '.npz', '.pkl']
        all_scores = []
        
        # Iterate through training samples and collect pseudo-labels from positive class
        for basename in self.filenames:
            # Get slide label
            slide_label = self.class_labels.get(basename)
            if slide_label is None:
                continue
            
            # Check if positive class (class 1 for binary, or not 0 for multiclass)
            is_positive = False
            if isinstance(slide_label, (int, np.integer)):
                is_positive = (slide_label != 0)
            elif isinstance(slide_label, (list, tuple)):
                # One-hot encoded: class 1 is [0, 1, ...]
                if len(slide_label) > 1:
                    is_positive = (slide_label[1] == 1)
                else:
                    is_positive = (slide_label[0] != 0)
            elif isinstance(slide_label, np.ndarray):
                # One-hot encoded: class 1 is [0, 1, ...]
                if len(slide_label) > 1:
                    is_positive = (slide_label[1] == 1 or slide_label.argmax() == 1)
                else:
                    is_positive = (slide_label[0] != 0)
            
            if not is_positive:
                continue
            
            # Try to load pseudo-label file
            pseudo_label_path = None
            for ext in pseudo_label_extensions:
                candidate = Path(self.pseudo_label_dir) / f"{basename}{ext}"
                if candidate.exists():
                    pseudo_label_path = candidate
                    break
            
            if pseudo_label_path is None:
                continue  # Skip if file not found
            
            try:
                # Load and extract scores
                ext = pseudo_label_path.suffix.lower()
                scores = None
                
                if ext in ['.pt', '.pth']:
                    data = torch.load(pseudo_label_path, map_location='cpu', weights_only=False)
                    if isinstance(data, dict):
                        for key in ['patch_logits', 'attention_scores', 'scores', 'attention', 'predictions']:
                            if key in data:
                                scores = data[key]
                                break
                    elif isinstance(data, torch.Tensor):
                        scores = data
                elif ext == '.npy':
                    scores = np.load(pseudo_label_path)
                elif ext == '.npz':
                    data = np.load(pseudo_label_path)
                    scores = data[data.files[0]]
                elif ext == '.pkl':
                    import pickle
                    with open(pseudo_label_path, 'rb') as f:
                        data = pickle.load(f)
                    if isinstance(data, dict):
                        for key in ['patch_logits', 'attention_scores', 'scores', 'attention', 'predictions']:
                            if key in data:
                                scores = data[key]
                                break
                    else:
                        scores = data
                
                if scores is not None:
                    # Convert to numpy if needed
                    if isinstance(scores, torch.Tensor):
                        scores = scores.cpu().numpy()
                    # Flatten and collect
                    scores = np.asarray(scores).flatten()
                    all_scores.extend(scores)
            
            except Exception as e:
                print(f"[PSEUDO-LABELS] Warning: Failed to load pseudo-labels from {pseudo_label_path}: {e}")
                continue
        
        if len(all_scores) == 0:
            print(f"[PSEUDO-LABELS] Warning: No positive class pseudo-labels found to compute percentile")
            return None
        
        # Compute 0.1 percentile
        percentile_value = float(np.percentile(all_scores, 0.1))
        print(f"[PSEUDO-LABELS] Computed from {len(all_scores)} scores: 0.1 percentile = {percentile_value:.6f}")
        
        return percentile_value

    def _find_feature_file(self, basename: str) -> Optional[Path]:
        # Try primary directory first
        for s in self.suffixes:
            p = self.patch_features_dir / (basename + s)
            if p.exists():
                return p
        
        # Try fallback directory
        if self.fallback_patch_features_dir:
            for s in self.suffixes:
                p = self.fallback_patch_features_dir / (basename + s)
                if p.exists():
                    return p
        
        return None

    def _load_feature_file(self, path: Path) -> torch.Tensor:
        ext = path.suffix.lower()
        if ext == '.pt':
            data = torch.load(path, map_location='cpu')
            # Expect tensor or dict with 'features' key
            if isinstance(data, dict):
                if 'features' in data:
                    f = data['features']
                else:
                    # try to find first tensor
                    tensor_keys = [k for k, v in data.items() if isinstance(v, (torch.Tensor, np.ndarray))]
                    if tensor_keys:
                        f = data[tensor_keys[0]]
                    else:
                        raise ValueError(f"No tensor-like object found in {path}")
            elif isinstance(data, torch.Tensor):
                f = data
            elif isinstance(data, np.ndarray):
                f = torch.from_numpy(data)
            else:
                raise ValueError(f"Unsupported data type in {path}: {type(data)}")
        elif ext == '.npy':
            arr = np.load(path, allow_pickle=True)
            f = torch.from_numpy(arr)
        elif ext == '.npz':
            arr = np.load(path)
            # pick first array
            key = list(arr.files)[0]
            f = torch.from_numpy(arr[key])
        elif ext == '.pkl':
            with open(path, 'rb') as fobj:
                data = pkl.load(fobj)
            if isinstance(data, dict) and 'features' in data:
                f = torch.from_numpy(np.array(data['features']))
            else:
                f = torch.from_numpy(np.array(data))
        elif ext == '.h5':
            import h5py
            with h5py.File(path, 'r') as h5f:
                # Try common key names for features
                if 'features' in h5f:
                    f = torch.from_numpy(h5f['features'][:])
                elif 'data' in h5f:
                    f = torch.from_numpy(h5f['data'][:])
                else:
                    # Use first dataset found
                    key = list(h5f.keys())[0]
                    f = torch.from_numpy(h5f[key][:])
        else:
            raise ValueError(f"Unsupported feature file extension: {ext}")

        # Ensure float tensor
        if isinstance(f, np.ndarray):
            f = torch.from_numpy(f)
        if isinstance(f, torch.Tensor):
            return f.float()
        else:
            raise ValueError(f"Failed to convert features to tensor for {path}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        t_start = time.time()
        basename = self.filenames[idx]
        
        # Time: Find feature file
        t_find_feat = time.time()
        feat_path = self._find_feature_file(basename)
        if feat_path is None:
            if self.verbose:
                print(f"[TIMING] {basename}: feature_find={time.time() - t_find_feat:.3f}s - FILE NOT FOUND")
            raise FileNotFoundError(f"Feature file not found for {basename}")
        
        # Time: Load features
        t_load_feat = time.time()
        feat = self._load_feature_file(feat_path)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        t_feat_load = time.time() - t_load_feat

        # Load class label
        cls_label = self.class_labels.get(basename)
        if cls_label is None:
            raise ValueError(f"No class label found for {basename} in labels_csv")
        cls_label_tensor = torch.from_numpy(np.array(cls_label)).float()
        
        # Time: Global features
        t_global = time.time()
        if self.global_feature_dir is not None:
            global_feat = self._load_global_feature(basename)
        elif self.auto_generate_global_features:
            print(f"Auto-generating global features for {basename}...")
            global_feat = self._extract_global_features_titan(None, feat, basename)
        else:
            # Fallback to patch mean
            global_feat = feat.mean(dim=0)
        t_global_load = time.time() - t_global

        # Time: GT labels
        t_gt = time.time()
        patch_labels = None
        if self.gt_dir is not None:
            patch_labels = self._load_patch_labels(basename, feat.shape[0], cls_label_tensor)
        t_gt_load = time.time() - t_gt
        
        # Time: Pseudo-labels
        t_pseudo = time.time()
        pseudo_labels = self._load_pseudo_labels(basename, feat.shape[0], cls_label_tensor)
        t_pseudo_load = time.time() - t_pseudo
        
        t_total = time.time() - t_start
        # Log timing for slow items (> 0.5s)
        if t_total > 0.5:
            if self.verbose:
                print(f"[TIMING] {basename}: feat_load={t_feat_load:.3f}s global={t_global_load:.3f}s gt={t_gt_load:.3f}s pseudo={t_pseudo_load:.3f}s total={t_total:.3f}s")
        
        return basename, feat, cls_label_tensor, global_feat, patch_labels, pseudo_labels
    
    def _load_patch_labels(
        self,
        basename: str,
        num_patches: int,
        slide_label: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Load ground truth patch labels for segmentation metrics.
        
        Note: Masks follow the same binary logic as binary_mode in config:
        - binary_mode=True: masks are binary (0=benign/class0, 1=tumor/class1)
        - binary_mode=False: masks are multi-class (0,1,2,...,num_classes-1)
        
        Args:
            basename: WSI basename
            num_patches: Number of patches to match
            
        Returns:
            Tensor of shape (num_patches,) with class labels, or None if not found
        """
        t_gt_start = time.time()
        # Try different mask file formats
        mask_extensions = ['.npy', '.pt', '.npz', '.png', '.tif', '.xml']

        for ext in mask_extensions:
            mask_path = self.gt_dir / f"{basename}_patch_labels{ext}"
            if not mask_path.exists():
                mask_path = self.gt_dir / f"{basename}{ext}"
            
            if not mask_path.exists():
                continue

            t_ext_start = time.time()
            try:
                # === CASE 1: Direct per-patch labels stored ===
                if ext in ['.npy', '.pt', '.npz']:
                    if ext == '.npy':
                        labels = np.load(mask_path)
                    elif ext == '.pt':
                        labels = torch.load(mask_path, weights_only=True)
                        if isinstance(labels, torch.Tensor):
                            labels = labels.numpy()
                    elif ext == '.npz':
                        data = np.load(mask_path)
                        labels = data[data.files[0]]

                    labels_tensor = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels.long()

                    # Handle shape mismatch
                    if labels_tensor.numel() != num_patches:
                        print(f"Warning: Patch labels for {basename} have {labels_tensor.numel()} elements, expected {num_patches}. Truncating/padding.")
                        if labels_tensor.numel() > num_patches:
                            labels_tensor = labels_tensor[:num_patches]
                        else:
                            padding = torch.zeros(num_patches - labels_tensor.numel(), dtype=torch.long)
                            labels_tensor = torch.cat([labels_tensor, padding])
                    t_ext_elapsed = time.time() - t_ext_start
                    if t_ext_elapsed > 0.5:
                        if self.verbose:
                            print(f"[TIMING_GT] {basename}: loaded {ext} in {t_ext_elapsed:.3f}s")
                    return labels_tensor

                # === CASE 2: Image mask (e.g., .tif) that must be mapped via coordinates ===
                elif ext in ['.png', '.tif']:
                    from PIL import Image
                    Image.MAX_IMAGE_PIXELS = None  # Allow large whole-slide masks
                    mask = np.array(Image.open(mask_path))

                    # Require coordinates to map patches to pixels (use cache to avoid reloading)
                    coords = self._get_cached_coordinates(basename)
                    if coords is None or len(coords) == 0:
                        print(f"Warning: No coordinates found for {basename}; cannot derive patch labels from mask {mask_path}")
                        return None

                    # Ensure coords length matches available patches
                    n_coords = len(coords)
                    if n_coords != num_patches:
                        # If mismatch is large, skip segmentation to avoid incorrect labels
                        mismatch_ratio = abs(n_coords - num_patches) / max(1, num_patches)
                        if mismatch_ratio > 0.2:
                            print(f"Warning: Coordinates ({n_coords}) and patch features ({num_patches}) differ significantly for {basename}. Skipping segmentation labels for this slide.")
                            return None
                        min_len = min(n_coords, num_patches)
                        print(f"Warning: Coordinates ({n_coords}) and patch features ({num_patches}) differ for {basename}. Using first {min_len} entries.")
                        coords = coords[:min_len]
                        num_patches = min_len

                    patch_labels = np.zeros(num_patches, dtype=np.int64)
                    for idx, (x, y) in enumerate(coords):
                        x = int(x)
                        y = int(y)
                        patch_region = mask[y:y + self.patch_size, x:x + self.patch_size]
                        if patch_region.size == 0:
                            continue
                        if self.binary_mode:
                            # Calculate percentage of annotated pixels (non-zero)
                            annotated_pixels = (patch_region > 0).sum()
                            total_pixels = patch_region.size
                            coverage_percentage = (annotated_pixels / total_pixels) * 100
                            # Patch is positive if >= 1% of pixels are annotated
                            patch_labels[idx] = 1 if coverage_percentage >= 1.0 else 0
                        else:
                            # Majority label within the patch
                            vals, counts = np.unique(patch_region, return_counts=True)
                            patch_labels[idx] = int(vals[counts.argmax()])

                    patch_labels_tensor = torch.from_numpy(patch_labels).long()

                    # Enforce benign slide constraint: if slide label is class 0, force all patch labels to 0
                    if slide_label is not None and slide_label.numel() > 0:
                        slide_class = int(slide_label.argmax().item()) if slide_label.dim() > 0 else int(slide_label.item())
                        if slide_class == 0:
                            return torch.zeros(num_patches, dtype=torch.long)

                    return patch_labels_tensor

                # === CASE 3: XML annotations (polygon overlays, e.g., ASAP/QuPath) ===
                elif ext in ['.xml']:
                    t_xml_coords = time.time()
                    coords = self._load_coordinates(basename)
                    t_xml_coords = time.time() - t_xml_coords
                    if coords is None or len(coords) == 0:
                        print(f"Warning: No coordinates found for {basename}; cannot derive patch labels from XML {mask_path}")
                        return None

                    n_coords = len(coords)
                    if n_coords != num_patches:
                        mismatch_ratio = abs(n_coords - num_patches) / max(1, num_patches)
                        if mismatch_ratio > 0.2:
                            print(f"Warning: Coordinates ({n_coords}) and patch features ({num_patches}) differ significantly for {basename}. Skipping segmentation labels for this slide.")
                            return None
                        min_len = min(n_coords, num_patches)
                        print(f"Warning: Coordinates ({n_coords}) and patch features ({num_patches}) differ for {basename}. Using first {min_len} entries.")
                        coords = coords[:min_len]
                        num_patches = min_len

                    t_xml_parse = time.time()
                    polygons, poly_labels = self._parse_xml_annotations(mask_path)
                    t_xml_parse = time.time() - t_xml_parse
                    if len(polygons) == 0:
                        print(f"Warning: No polygons parsed from XML {mask_path}")
                        return None

                    t_xml_match = time.time()
                    patch_labels = np.zeros(num_patches, dtype=np.int64)
                    half_patch = self.patch_size / 2.0
                    for idx, (x, y) in enumerate(coords):
                        # Calculate pixel coverage within this patch across all polygons
                        x_start, y_start = int(x), int(y)
                        x_end = min(x_start + self.patch_size, int(x) + self.patch_size)
                        y_end = min(y_start + self.patch_size, int(y) + self.patch_size)
                        
                        # Count annotated pixels (pixels inside any polygon)
                        annotated_pixel_count = 0
                        total_pixel_count = (x_end - x_start) * (y_end - y_start)
                        
                        if total_pixel_count == 0:
                            patch_labels[idx] = 0
                            continue
                        
                        # Check each pixel in the patch
                        for px in range(x_start, x_end):
                            for py in range(y_start, y_end):
                                # Check if this pixel is inside any polygon
                                for poly_idx, poly in enumerate(polygons):
                                    if self._point_in_polygon(float(px), float(py), poly):
                                        annotated_pixel_count += 1
                                        break
                        
                        # Calculate coverage percentage
                        coverage_percentage = (annotated_pixel_count / total_pixel_count) * 100
                        
                        # Assign label based on coverage (1% threshold)
                        if self.binary_mode:
                            patch_labels[idx] = 1 if coverage_percentage >= 1.0 else 0
                        else:
                            # For multiclass: if coverage >= 1%, assign the class of the first matching polygon
                            if coverage_percentage >= 1.0:
                                for poly_idx, poly in enumerate(polygons):
                                    # Find which polygon covers the most pixels in this patch
                                    pass
                                # Simple approach: check center point for class
                                cx = float(x) + half_patch
                                cy = float(y) + half_patch
                                label = 0
                                for poly_idx, poly in enumerate(polygons):
                                    if self._point_in_polygon(cx, cy, poly):
                                        label = poly_labels[poly_idx]
                                        break
                                patch_labels[idx] = label
                            else:
                                patch_labels[idx] = 0
                    
                    t_xml_match = time.time() - t_xml_match
                    if self.verbose:
                        print(f"[TIMING_XML] {basename}: coords={t_xml_coords:.3f}s parse={t_xml_parse:.3f}s match={t_xml_match:.3f}s")

                    patch_labels_tensor = torch.from_numpy(patch_labels).long()

                    if slide_label is not None and slide_label.numel() > 0:
                        slide_class = int(slide_label.argmax().item()) if slide_label.dim() > 0 else int(slide_label.item())
                        if slide_class == 0:
                            return torch.zeros(num_patches, dtype=torch.long)

                    return patch_labels_tensor

            except Exception as e:
                print(f"Warning: Failed to load patch labels for {basename}: {e}")
                return None

        # If no labels were loaded but slide is benign, still return zeros to enforce constraint
        if slide_label is not None and slide_label.numel() > 0:
            slide_class = int(slide_label.argmax().item()) if slide_label.dim() > 0 else int(slide_label.item())
            if slide_class == 0:
                return torch.zeros(num_patches, dtype=torch.long)

        return None

    def _parse_xml_annotations(self, xml_path: Path):
        """Parse polygon annotations from an XML file (ASAP/QuPath style)."""
        t_parse_start = time.time()
        polygons = []
        labels = []

        try:
            t_xml_read = time.time()
            tree = ET.parse(xml_path)
            t_xml_read = time.time() - t_xml_read
            root = tree.getroot()
            
            t_xml_iter = time.time()
            for ann in root.findall('.//Annotation'):
                # Each <Annotation> can contain multiple <Coordinates> blocks (outer/holes)
                for coords_block in ann.findall('.//Coordinates'):
                    points = []
                    for coord in coords_block.findall('.//Coordinate'):
                        x = float(coord.attrib.get('X', 0))
                        y = float(coord.attrib.get('Y', 0))
                        points.append((x, y))
                    if len(points) < 3:
                        continue

                    # Infer class id
                    raw_label = ann.attrib.get('PartOfGroup') or ann.attrib.get('Name') or ''
                    label = self._normalize_annotation_label(raw_label)
                    labels.append(label)
                    polygons.append(points)
            t_xml_iter = time.time() - t_xml_iter
        except Exception as e:
            print(f"Warning: Failed to parse XML annotations from {xml_path}: {e}")
            return [], []

        t_total = time.time() - t_parse_start
        if t_total > 0.5:
            if self.verbose:
                print(f"[TIMING_XML_PARSE] {xml_path.name}: xml_read={t_xml_read:.3f}s iter={t_xml_iter:.3f}s total={t_total:.3f}s n_polys={len(polygons)}")

        return polygons, labels

    def _normalize_annotation_label(self, raw_label: str) -> int:
        """Convert XML annotation label text to class index."""
        if raw_label is None:
            raw_label = ''
        raw_label = str(raw_label).strip()

        # Numeric labels
        if raw_label.isdigit():
            return min(int(raw_label), max(0, self.num_classes - 1))

        lower = raw_label.lower()
        tumor_keys = {
            'tumor', 'cancer', 'positive', 'malignant', 'bcc', 'scc', 'basal',
            'carcinoma', 'lesion', 'met', 'metastasis'
        }
        background_keys = {'background', 'benign', 'normal', 'negative', 'stroma', 'fat'}

        if lower in tumor_keys:
            return min(1, max(0, self.num_classes - 1))
        if lower in background_keys:
            return 0

        # Default foreground
        return min(1, max(0, self.num_classes - 1))

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """Ray casting algorithm to test if point is inside polygon."""
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]

            intersects = ((y1 > y) != (y2 > y)) and (
                x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
            )
            if intersects:
                inside = not inside
        return inside
    
    def _load_pseudo_labels(
        self,
        basename: str,
        num_patches: int,
        slide_label: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Load pseudo-labels (attention scores) with zero fallback for benign class.
        
        Args:
            basename: WSI identifier
            num_patches: Number of patches (for shape validation)
            slide_label: Slide-level label tensor (for class identification)
            
        Returns:
            Tensor of shape (num_patches,) with attention scores, or None if not using pseudo-labels
        """
        # Check if pseudo-labels are configured
        pseudo_label_dir = getattr(self, 'pseudo_label_dir', None)
        if pseudo_label_dir is None:
            return None
        
        # Try to load pseudo-label file
        pseudo_label_extensions = ['.pt', '.pth', '.npy', '.npz', '.pkl']
        pseudo_label_path = None
        
        for ext in pseudo_label_extensions:
            candidate = Path(pseudo_label_dir) / f"{basename}{ext}"
            if candidate.exists():
                pseudo_label_path = candidate
                break
        
        # If not found and this is benign class (class 0), return percentile floor value
        if pseudo_label_path is None:
            # Check if this is benign class
            is_benign = False
            if slide_label is not None:
                if slide_label.dim() == 1 and len(slide_label) > 0:
                    # One-hot encoded: class 0 is [1, 0, ...]
                    is_benign = (slide_label[0] == 1)
                elif slide_label.dim() == 0:
                    # Single class index
                    is_benign = (slide_label.item() == 0)
            
            if is_benign:
                # Return 0.1 percentile floor (or 0 as fallback if not computed)
                floor_value = self.benign_pseudo_label_floor if self.benign_pseudo_label_floor is not None else 0.0
                return torch.full((num_patches,), floor_value, dtype=torch.float32)
            else:
                # For positive class, missing pseudo-labels is an error
                raise FileNotFoundError(
                    f"Pseudo-label file not found for {basename} in {pseudo_label_dir}.\n"
                    f"This is required for positive class (BCC). Tried extensions: {pseudo_label_extensions}"
                )
        
        # Load the pseudo-label file
        ext = pseudo_label_path.suffix.lower()
        if ext in ['.pt', '.pth']:
            data = torch.load(pseudo_label_path, map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                # Try common keys
                for key in ['patch_logits', 'attention_scores', 'scores', 'attention', 'predictions']:
                    if key in data:
                        scores = data[key]
                        break
                else:
                    raise ValueError(f"Could not find attention scores in {pseudo_label_path}. Keys: {list(data.keys())}")
            else:
                scores = data
            
            if isinstance(scores, np.ndarray):
                scores = torch.from_numpy(scores)
            
            # Handle binary mode: if 2D with shape (N, 2), take class 1 (tumor) scores
            if scores.dim() == 2 and scores.shape[1] == 2:
                scores = scores[:, 1]  # Tumor class
            
            return scores.float()
        
        elif ext in ['.npy', '.npz']:
            if ext == '.npz':
                data = np.load(pseudo_label_path)
                scores = data[data.files[0]]
            else:
                scores = np.load(pseudo_label_path)
            
            scores = torch.from_numpy(scores).float()
            if scores.dim() == 2 and scores.shape[1] == 2:
                scores = scores[:, 1]
            return scores
        
        else:
            raise ValueError(f"Unsupported pseudo-label file format: {ext}")

    def _load_coordinates(self, basename: str) -> Optional[np.ndarray]:
        """Load patch coordinates for a WSI if available."""
        t_coord_start = time.time()
        if self.coordinates_dir is None:
            return None
        
        # Try primary directory first
        candidates = [
            self.coordinates_dir / f"{basename}{self.coordinates_suffix}",
            self.coordinates_dir / f"{basename}_patches{self.coordinates_suffix}",
            self.coordinates_dir / f"{basename}.npy",
            self.coordinates_dir / f"{basename}.npz",
        ]
        coord_path = None
        for p in candidates:
            if p.exists():
                coord_path = p
                break
        
        # Try fallback directory
        if coord_path is None and self.fallback_coordinates_dir:
            candidates_fallback = [
                self.fallback_coordinates_dir / f"{basename}{self.fallback_coordinates_suffix}",
                self.fallback_coordinates_dir / f"{basename}_patches{self.fallback_coordinates_suffix}",
                self.fallback_coordinates_dir / f"{basename}.npy",
                self.fallback_coordinates_dir / f"{basename}.npz",
            ]
            for p in candidates_fallback:
                if p.exists():
                    coord_path = p
                    break
        
        if coord_path is None:
            return None

        try:
            if coord_path.suffix.lower() in ['.npy']:
                coords = np.load(coord_path)
            elif coord_path.suffix.lower() in ['.npz']:
                data = np.load(coord_path)
                coords = data[data.files[0]]
            elif coord_path.suffix.lower() in ['.h5', '.hdf5']:
                with h5py.File(coord_path, 'r') as f:
                    if 'coords' in f:
                        coords = f['coords'][:]
                    else:
                        # Take the first dataset
                        first_key = list(f.keys())[0]
                        coords = f[first_key][:]
            else:
                # Fallback: try text
                coords = np.loadtxt(coord_path)
        except Exception as e:
            print(f"Warning: Failed to load coordinates from {coord_path}: {e}")
            return None

        # Ensure shape (N, 2)
        coords = np.array(coords)
        if coords.ndim == 1:
            # Flat array: reshape to (N, 2) pairs
            if coords.size % 2 != 0:
                print(f"Warning {basename} : Coordinate array has odd size {coords.size} from shape {coords.shape}, cannot form pairs")
                return None
            coords = coords.reshape(-1, 2)
        elif coords.ndim == 2:
            # Already 2D: ensure second dimension is at least 2
            if coords.shape[1] < 2:
                print(f"Warning {basename} : Coordinate array has shape {coords.shape}, expected (N, 2)")
                return None
            if coords.shape[1] > 2:
                coords = coords[:, :2]
        else:
            print(f"Warning {basename} : Unexpected coordinate array shape {coords.shape}")
            return None
        
        t_coord_elapsed = time.time() - t_coord_start
        if t_coord_elapsed > 0.1:
            if self.verbose:
                print(f"[TIMING_COORD_LOAD] {basename}: loaded {coord_path.name} with {len(coords)} coords in {t_coord_elapsed:.3f}s")
        return coords
    
    def _get_cached_coordinates(self, basename: str) -> Optional[np.ndarray]:
        """Load coordinates once per WSI and cache them to avoid repeated disk access."""
        # Check cache first (safe: cache is just a dict, no side effects)
        if basename in self._coord_cache:
            return self._coord_cache[basename]
        
        # Load from disk
        t_coord_load = time.time()
        coords = self._load_coordinates(basename)
        t_coord_load = time.time() - t_coord_load
        
        # Cache the result (even if None, to avoid repeated failed attempts)
        self._coord_cache[basename] = coords
        
        if t_coord_load > 0.1:
            if self.verbose:
                print(f"[TIMING_COORD] {basename}: loaded coordinates in {t_coord_load:.3f}s (cache miss)")
        
        return coords
    
    def _load_patch_encoder(self):
        """Lazily initialize and cache patch encoder (Virchow2)."""
        if self.patch_encoder_model is None:
            print(f"Loading {self.patch_encoder} patch encoder on-the-fly...")
            self.patch_encoder_model, self.patch_feature_dim = EncoderFactory.create_encoder(
                self.patch_encoder, device=self.device
            )
        return self.patch_encoder_model, self.patch_feature_dim
    
    def _load_global_encoder(self):
        """Lazily initialize and cache global encoder (TITAN for slide-level features)."""
        if self.global_encoder_model is None:
            print("Loading TITAN global encoder on-the-fly...")
            try:
                from transformers import AutoModel
                self.global_encoder_model = AutoModel.from_pretrained(
                    "MahmoodLab/TITAN", trust_remote_code=True
                )
                self.global_encoder_model = self.global_encoder_model.to(self.device)
                self.global_encoder_model.eval()
                # TITAN outputs 768-dim features
                self.global_feature_dim = 768
            except Exception as e:
                print(f"Warning: Failed to load TITAN encoder: {e}. Will use patch mean as fallback.")
                self.global_feature_dim = self.patch_feature_dim if self.patch_feature_dim else 1280
        return self.global_encoder_model, self.global_feature_dim
    
    def _extract_patch_features_virchow2(self, wsi_path: str, basename: str) -> torch.Tensor:
        """Extract patch features using Virchow2 encoder on-the-fly."""
        if not Path(wsi_path).exists():
            raise FileNotFoundError(f"WSI file not found: {wsi_path}")
        
        encoder, feature_dim = self._load_patch_encoder()
        
        print(f"Extracting Virchow2 patch features for {basename}...")
        try:
            from openslide import OpenSlide
            slide = OpenSlide(wsi_path)
            
            # Extract patches at 224x224 resolution (Virchow2 standard)
            patch_size = 224
            level = 0
            patches = []
            
            # Simple grid-based extraction
            width, height = slide.dimensions
            for y in range(0, height - patch_size, patch_size):
                for x in range(0, width - patch_size, patch_size):
                    try:
                        patch = slide.read_region((x, y), level, (patch_size, patch_size))
                        patch = patch.convert('RGB')
                        patch_tensor = torch.from_numpy(np.array(patch)).permute(2, 0, 1).float() / 255.0
                        
                        with torch.no_grad():
                            feat = encoder(patch_tensor.unsqueeze(0).to(self.device))
                        patches.append(feat.cpu())
                    except Exception as e:
                        continue
            
            if not patches:
                raise RuntimeError(f"Failed to extract any patches from {basename}")
            
            features = torch.cat(patches, dim=0)  # (n_patches, feature_dim)
            
            # Save to disk
            output_path = self.patch_features_dir / (basename + '.pt')
            torch.save(features, output_path)
            print(f"✓ Saved {basename} patch features to {output_path}")
            
            return features
        except Exception as e:
            raise RuntimeError(f"Patch feature extraction failed for {basename}: {e}")
    
    def _extract_global_features_titan(self, wsi_path: str, patch_features: torch.Tensor, basename: str) -> torch.Tensor:
        """Extract global (slide-level) features using TITAN encoder on-the-fly."""
        encoder, feature_dim = self._load_global_encoder()
        
        if encoder is None:
            # Fallback to mean of patch features
            print(f"TITAN not available, using patch mean as global feature for {basename}")
            return patch_features.mean(dim=0)
        
        print(f"Extracting TITAN global features for {basename}...")
        try:
            # TITAN expects patch features + coordinates
            # For simplicity, aggregate patch features
            with torch.no_grad():
                patch_feats = patch_features.unsqueeze(0).to(self.device)  # (1, n_patches, patch_dim)
                # Use simple aggregation through TITAN
                global_feat = encoder.encode_slide_from_patch_features(
                    patch_feats, None, None
                )
            
            if isinstance(global_feat, torch.Tensor):
                global_feat = global_feat.squeeze().cpu()
            else:
                # Fallback
                global_feat = patch_features.mean(dim=0).cpu()
            
            # Save to disk if global_feature_dir exists
            if self.global_feature_dir:
                output_path = self.global_feature_dir / (basename + '.pt')
                torch.save(global_feat, output_path)
                print(f"✓ Saved {basename} global features to {output_path}")
            
            return global_feat
        except Exception as e:
            print(f"Warning: TITAN global extraction failed for {basename}: {e}. Using patch mean.")
            return patch_features.mean(dim=0).cpu()
    
    def _load_global_feature(self, basename: str) -> torch.Tensor:
        """Load global (slide-level) feature from file or compute from patches."""
        # Try to find global feature file
        if self.global_feature_dir:
            for s in self.suffixes:
                p = self.global_feature_dir / (basename + s)
                if p.exists():
                    try:
                        return self._load_feature_file(p).squeeze()
                    except Exception as e:
                        print(f"Warning: Failed to load global feature from {p}: {e}")
                        break
        
        # Fallback: compute from patch features
        print(f"Warning: No global feature found for {basename}, using patch mean")
        feat_path = self._find_feature_file(basename)
        if feat_path is None:
            raise FileNotFoundError(f"Cannot compute global feature: no patch feature file for {basename}")
        feat = self._load_feature_file(feat_path)
        return feat.mean(dim=0) if feat.dim() > 1 else feat


# End of file
