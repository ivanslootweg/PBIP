import os
import os.path
import csv
from typing import Dict, Optional

import torch.distributed as dist

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets.bcss import BCSSTestDataset, BCSSTrainingDataset, BCSSWSSSDataset
from datasets.wsi_dataset import CustomWSIPatchTrainingDataset, CustomWSIPatchTestDataset


def get_wsss_dataset(cfg):
    MEAN, STD = get_mean_std(cfg.dataset.name)

    transform = {
        "train": A.Compose([
            A.Normalize(MEAN, STD),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(),
            ToTensorV2(transpose_mask=True),
        ]),
        "val": A.Compose([
            A.Normalize(MEAN, STD),
            ToTensorV2(transpose_mask=True),
        ])
    }
    train_dataset = BCSSWSSSDataset(cfg.dataset.train_root, mask_name=cfg.dataset.mask_root,transform=transform["train"])
    val_dataset = BCSSTestDataset(cfg.dataset.val_root, split="valid", transform=transform["val"])

    return train_dataset, val_dataset


def get_cls_dataset(cfg, split="valid", p=0.5, enable_rotation=True):
    MEAN, STD = get_mean_std(cfg.dataset.name)
    
    # 构建训练时的变换列表
    train_transforms = [
        A.Normalize(MEAN, STD),
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
    ]
    
    # 根据参数决定是否添加旋转
    if enable_rotation:
        train_transforms.append(A.RandomRotate90())
    
    train_transforms.append(ToTensorV2(transpose_mask=True))
    
    transform = {
        "train": A.Compose(train_transforms),
        "val": A.Compose([
            A.Normalize(MEAN, STD),
            ToTensorV2(transpose_mask=True),
        ]),
    }


    train_dataset = BCSSTrainingDataset(cfg.dataset.train_root, transform=transform["train"])
    val_dataset = BCSSTestDataset(cfg.dataset.val_root, split, transform=transform["val"])

    return train_dataset, val_dataset


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


def get_custom_dataset(cfg, split="valid"):
    """
    Load custom WSI patch-based dataset with weak image-level labels.
    Patches are extracted from WSI files using coordinates.
    
    Config requirements:
        dataset:
            name: "custom_wsi"
            wsi_dir: "/path/to/wsi/files"
            coordinates_dir: "/path/to/coordinates"
            split_csv: "/path/to/split.csv"  # columns: train, val, test
            labels_csv: "/path/to/labels.csv"  # columns: image_name, label1, label2, ...
            gt_dir: "/path/to/ground_truth"  # for val/test
            num_classes: 4
            patch_size: 224  # optional
            coordinates_suffix: ".npy"  # or ".txt"
    
    Args:
        cfg: Configuration object
        split: "valid" or "test" for val_dataset
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Load class labels (num_classes will be auto-inferred from data)
    class_labels_dict = load_class_labels_from_csv(
        cfg.dataset.labels_csv, 
        num_classes=cfg.dataset.get('num_classes', None)
    )
    
    # Build transforms consistent with BCSS pipeline
    MEAN, STD = get_mean_std(cfg.dataset.name)
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
    val_split = "val" if split == "valid" else split
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


def get_mean_std(dataset):
    norm = [[0.66791496, 0.47791372, 0.70623304], [0.1736589,  0.22564577, 0.19820057]]
    return norm[0], norm[1]


def all_reduced(x, n_gpus):
    dist.all_reduce(x)
    x /= n_gpus

