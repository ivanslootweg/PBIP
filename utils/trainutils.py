import os
import os.path
import csv
from typing import Dict, Optional

import torch.distributed as dist

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets.wsi_dataset import CustomWSIPatchTrainingDataset, CustomWSIPatchTestDataset
# NOTE: PatchLevelTrainingDataset and PatchLevelTestDataset have been removed
# The new feature-based pipeline uses FeatureWSIDataset instead
# Legacy patch-level datasets are no longer supported



def load_class_labels_from_csv(labels_csv: str, num_classes: int = None) -> Dict:
    """
    Load image-level class labels from CSV file.
    
    Expected CSV format:
        image_name,label1,label2,label3,label4
        image1.npy,0,1,1,0
        image2.npy,1,0,0,1
    
    Or single label per row:
        image_name,label
        image1.npy,"0,1,1,0"
        
    Or single class index (auto-converted to one-hot):
        image_name,label
        image1.npy,0  -> converted to [1, 0, 0, ...] (class 0)
        image2.npy,2  -> converted to [0, 0, 1, ...] (class 2)
    
    Args:
        labels_csv: Path to CSV file with class labels
        num_classes: Number of classes for one-hot encoding (auto-inferred if None)
        
    Returns:
        Dict mapping image_name -> numpy array of class labels
    """
    class_labels_dict = {}
    single_label_indices = []  # Track single-label indices for auto-inference
    
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    
    # First pass: collect data and detect format
    rows_data = []
    with open(labels_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_data.append(row)
            
            # Get label columns (all except image_name)
            label_cols = [col for col in row.keys() if col != 'image_name']
            
            if len(label_cols) == 1:
                label_str = row[label_cols[0]].strip()
                # Check if it's a single integer (not comma/space separated)
                if ',' not in label_str and len(label_str.split()) == 1:
                    try:
                        single_label_indices.append(int(label_str))
                    except ValueError:
                        pass
    
    # Infer num_classes if needed and we have single-label format
    if num_classes is None and single_label_indices:
        num_classes = max(single_label_indices) + 1
        print(f"Auto-detected {num_classes} classes from label indices: {sorted(set(single_label_indices))}")
    
    # Second pass: convert labels
    for row in rows_data:
        image_name_raw = row['image_name'].strip()
        # Normalize to basename without extension to avoid mismatch
        image_key = os.path.splitext(os.path.basename(image_name_raw))[0]
        
        # Get label columns (all except image_name)
        label_cols = [col for col in row.keys() if col != 'image_name']
        
        if len(label_cols) == 1:
            # Single label column - parse as comma-separated or space-separated
            label_str = row[label_cols[0]].strip()
            # Try parsing as comma or space-separated values
            if ',' in label_str:
                labels = [int(x.strip()) for x in label_str.split(',')]
            else:
                # Could be a single integer or space-separated
                parts = label_str.split()
                if len(parts) == 1:
                    # Single class index - convert to one-hot encoding
                    class_idx = int(parts[0])
                    if num_classes is None:
                        raise ValueError(f"Cannot convert single label {class_idx} to one-hot without num_classes specified")
                    if class_idx >= num_classes or class_idx < 0:
                        raise ValueError(f"Label {class_idx} for {image_key} is out of range [0, {num_classes-1}]")
                    
                    # Create one-hot encoded label
                    labels = [0] * num_classes
                    labels[class_idx] = 1
                else:
                    labels = [int(x.strip()) for x in parts]
        else:
            # Multiple label columns
            labels = [int(row[col].strip()) for col in label_cols]
        
        class_labels_dict[image_key] = labels
    
    return class_labels_dict


def get_custom_dataset(cfg, split="valid", verbose=False):
    """
    Load custom WSI patch-based dataset with weak image-level labels.
    
    Two modes:
    1. Patch-level: Uses prototype coordinate files directly (recommended for weak supervision)
    2. WSI-level: Loads from split CSV with random sampling (original behavior)
    
    Config requirements:
        dataset:
            name: "custom_wsi"
            wsi_dir: "/path/to/wsi/files"
            use_patch_level_dataset: true  # NEW: enables patch-level training
            
            # For patch-level mode:
            class_order: [benign, tumor]
            
            # For WSI-level mode:
            coordinates_dir: "/path/to/coordinates"
            split_csv: "/path/to/split.csv"
            labels_csv: "/path/to/labels.csv"
            
            # Common:
            gt_dir: "/path/to/ground_truth"
            num_classes: 2
            patch_size: 224
    
    Args:
        cfg: Configuration object
        split: "valid" or "test" for val_dataset (will be normalized to "val" to match CSV columns)
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Build transforms for WSI dataset
    MEAN, STD = get_wsi_normalization()
    train_transforms = [
        A.Normalize(MEAN, STD),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(),
        ToTensorV2(transpose_mask=True),
    ]
    val_transforms = [
        A.Normalize(MEAN, STD),
        ToTensorV2(transpose_mask=True),
    ]
    transform = {
        "train": A.Compose(train_transforms),
        "val": A.Compose(val_transforms),
    }
    
    # Normalize split names: 'valid' -> 'val' to match CSV column names
    val_split = "val" if split == "valid" else split
    
    # Check if patch-level dataset should be used
    use_patch_level = getattr(cfg.dataset, 'use_patch_level_dataset', True)
    
    if use_patch_level:
        # If precomputed patch features are provided, use feature-based dataset
        patch_features_dir = getattr(cfg.dataset, 'patch_features_dir', None)
        if patch_features_dir:
            print("Using FEATURE-BASED training dataset (precomputed patch features)")
            from datasets.feature_dataset import FeatureWSIDataset
            
            # Get global_feature_dir from config (optional)
            global_feature_dir = getattr(cfg.dataset, 'global_feature_dir', None)
            
            # Get wsi_dir and auto-generation settings from config
            wsi_dir = getattr(cfg.dataset, 'wsi_dir', None)
            auto_generate_patch_features = getattr(cfg.dataset, 'auto_generate_patch_features', True)
            auto_generate_global_features = getattr(cfg.dataset, 'auto_generate_global_features', False)
            patch_encoder = getattr(cfg.model, 'patch_encoder', 'virchow2')
            
            # Optional: gt_dir for ground truth segmentation masks (validation only)
            gt_dir = getattr(cfg.dataset, 'gt_dir', None)
            binary_mode = getattr(cfg.dataset, 'binary_mode', False)
            coordinates_dir = getattr(cfg.dataset, 'coordinates_dir', None)
            coordinates_suffix = getattr(cfg.dataset, 'coordinates_suffix', '_patches.h5')
            patch_size = getattr(cfg.dataset, 'patch_size', 224)
            
            # Fallback directories
            fallback_patch_features_dir = getattr(cfg.dataset, 'fallback_patch_features_dir', None)
            fallback_coordinates_dir = getattr(cfg.dataset, 'fallback_coordinates_dir', None)
            fallback_coordinates_suffix = getattr(cfg.dataset, 'fallback_coordinates_suffix', None)
            
            # Pseudo-label directory
            pseudo_label_dir = getattr(cfg.dataset, 'pseudo_label_dir', None)

            train_dataset = FeatureWSIDataset(
                patch_features_dir=patch_features_dir,
                split_csv=cfg.dataset.split_csv,
                labels_csv=cfg.dataset.labels_csv,
                split="train",
                num_classes=cfg.dataset.num_classes,
                global_feature_dir=global_feature_dir,
                wsi_dir=wsi_dir,
                auto_generate_patch_features=auto_generate_patch_features,
                verbose=verbose,
                auto_generate_global_features=auto_generate_global_features,
                patch_encoder=patch_encoder,
                gt_dir=None,  # No GT masks needed for training
                binary_mode=binary_mode,
                coordinates_dir=coordinates_dir,
                coordinates_suffix=coordinates_suffix,
                patch_size=patch_size,
                fallback_patch_features_dir=fallback_patch_features_dir,
                fallback_coordinates_dir=fallback_coordinates_dir,
                fallback_coordinates_suffix=fallback_coordinates_suffix,
                pseudo_label_dir=pseudo_label_dir,
            )

            val_dataset = FeatureWSIDataset(
                patch_features_dir=patch_features_dir,
                split_csv=cfg.dataset.split_csv,
                labels_csv=cfg.dataset.labels_csv,
                split=val_split,
                num_classes=cfg.dataset.num_classes,
                global_feature_dir=global_feature_dir,
                wsi_dir=wsi_dir,
                auto_generate_patch_features=auto_generate_patch_features,
                auto_generate_global_features=auto_generate_global_features,
                patch_encoder=patch_encoder,
                gt_dir=gt_dir,  # Enable GT masks for validation if available
                binary_mode=binary_mode,
                coordinates_dir=coordinates_dir,
                coordinates_suffix=coordinates_suffix,
                patch_size=patch_size,
                fallback_patch_features_dir=fallback_patch_features_dir,
                fallback_coordinates_dir=fallback_coordinates_dir,
                fallback_coordinates_suffix=fallback_coordinates_suffix,
                pseudo_label_dir=pseudo_label_dir,
                verbose=verbose,
            )

        else:
            # Legacy patch-level dataset is no longer supported
            raise ValueError(
                "patch_features_dir must be specified in config. "
                "The legacy PatchLevelTrainingDataset has been removed. "
                "Please run 'python features/extract_patch_features.py' first "
                "to generate patch features, then specify 'patch_features_dir' in your config."
            )
        
    else:
        print("Using WSI-LEVEL training dataset (random sampling from split CSV)")
        
        # Load class labels (num_classes will be auto-inferred from data)
        class_labels_dict = load_class_labels_from_csv(
            cfg.dataset.labels_csv, 
            num_classes=cfg.dataset.get('num_classes', None)
        )
        
        # Create training dataset (weak labels only)
        train_dataset = CustomWSIPatchTrainingDataset(
            wsi_dir=cfg.dataset.wsi_dir,
            coordinates_dir=cfg.dataset.coordinates_dir,
            split_csv=cfg.dataset.split_csv,
            split="train",
            class_labels_dict=class_labels_dict,
            num_classes=cfg.dataset.num_classes,
            patch_size=getattr(cfg.dataset, 'patch_size', 224),
            max_patches=getattr(cfg.dataset, 'max_patches', None),
            coordinates_suffix=getattr(cfg.dataset, 'coordinates_suffix', '.npy'),
            transform=transform["train"],
            use_openslide=getattr(cfg.dataset, 'use_openslide', None),
        )
        
        # Create validation dataset (with GT masks if available)
        val_dataset = CustomWSIPatchTestDataset(
            wsi_dir=cfg.dataset.wsi_dir,
            coordinates_dir=cfg.dataset.coordinates_dir,
            split_csv=cfg.dataset.split_csv,
            gt_dir=cfg.dataset.gt_dir,
            split=val_split,
            class_labels_dict=class_labels_dict,
            num_classes=cfg.dataset.num_classes,
            patch_size=getattr(cfg.dataset, 'patch_size', 224),
            max_patches=getattr(cfg.dataset, 'max_patches', None),
            coordinates_suffix=getattr(cfg.dataset, 'coordinates_suffix', '.npy'),
            mask_suffix=getattr(cfg.dataset, 'mask_suffix', '.png'),
            transform=transform["val"],
            use_openslide=getattr(cfg.dataset, 'use_openslide', None),
        )
    
    return train_dataset, val_dataset


def get_wsi_normalization():
    """
    Get standard normalization statistics for WSI patch data.
    
    These are the mean and std from histopathology image datasets.
    
    Returns:
        Tuple of (mean, std) for normalization
    """
    mean = [0.66791496, 0.47791372, 0.70623304]
    std = [0.1736589, 0.22564577, 0.19820057]
    return mean, std


def all_reduced(x, n_gpus):
    dist.all_reduce(x)
    x /= n_gpus

