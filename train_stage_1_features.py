'''
Training script for pre-extracted MedCLIP features on WSI patches.

This script trains on pre-computed MedCLIP embeddings instead of raw images.
The features are already extracted from patches of WSI files.

Usage:
python train_stage_1_features.py --config ./work_dirs/custom/config.yaml --gpu 0

Configuration file should include:
  dataset:
    name: wsi
    feature_dir: /path/to/medclip/features
    split_csv: /path/to/split.csv
    labels_csv: /path/to/labels.csv
    gt_dir: /path/to/ground_truth
    num_classes: 4
'''

import argparse
import datetime
import os
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.trainutils import get_custom_dataset, load_class_labels_from_csv
from utils.optimizer import PolyWarmupAdamW
from utils.pyutils import set_seed, AverageMeter
from utils.evaluate import ConfusionMatrixAllClass

start_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--gpu", type=int, default=0, help="gpu id")
args = parser.parse_args()


def cal_eta(time0, cur_iter, total_iter):
    """Calculate elapsed time and estimated time to completion"""
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def aggregate_patch_features(patch_features, aggregation_method="mean"):
    """
    Aggregate patch-level features to image-level features.
    
    Args:
        patch_features: (n_patches, feature_dim) tensor
        aggregation_method: 'mean', 'max', or 'attention'
        
    Returns:
        Image-level feature: (feature_dim,) tensor
    """
    if aggregation_method == "mean":
        return patch_features.mean(dim=0)
    elif aggregation_method == "max":
        return patch_features.max(dim=0)[0]
    elif aggregation_method == "attention":
        # Simple attention-based aggregation
        # Compute attention weights as softmax of feature norms
        norms = torch.norm(patch_features, dim=1, keepdim=True)
        weights = F.softmax(norms, dim=0)
        return (patch_features * weights).sum(dim=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")


def train(cfg):
    
    print("\nInitializing feature-based training...")
    torch.backends.cudnn.benchmark = True
    
    num_workers = min(10, os.cpu_count())
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    # ============ prepare data ============
    print("\nPreparing datasets...")
    train_dataset, val_dataset = get_custom_dataset(cfg, split="valid")
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Efficient data loading configuration
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.samples_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    # ============ prepare model ============
    # For feature-based training, we create a simple classification head
    # that operates on aggregated Virchow2 features
    
    feature_dim = 256  # Virchow2 feature dimension - UPDATE based on your features
    num_classes = cfg.dataset.num_classes
    
    # Simple MLP classification head
    classification_head = nn.Sequential(
        nn.Linear(feature_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    classification_head.to(device)
    classification_head.train()
    
    # Mixed precision training setup
    iters_per_epoch = len(train_loader)
    cfg.train.max_iters = cfg.train.epoch * iters_per_epoch
    cfg.train.eval_iters = iters_per_epoch
    cfg.scheduler.warmup_iter = cfg.scheduler.warmup_iter * iters_per_epoch
    scaler = torch.cuda.amp.GradScaler()
    
    # Optimizer configuration
    optimizer = PolyWarmupAdamW(
        params=classification_head.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )

    # Loss function
    loss_function = nn.BCEWithLogitsLoss().to(device)
    
    aggregation_method = getattr(cfg.train, 'feature_aggregation', 'mean')
    print(f"Using {aggregation_method} aggregation for patch features")

    print("\nStarting training...")
    train_loader_iter = iter(train_loader)
    
    for n_iter in range(cfg.train.max_iters):
        try:
            img_name, patch_features, cls_labels, gt_label = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            img_name, patch_features, cls_labels, gt_label = next(train_loader_iter)

        # patch_features shape: (batch_size, n_patches, feature_dim)
        # cls_labels shape: (batch_size, num_classes)
        batch_size = patch_features.shape[0]
        
        # Aggregate patch features to image-level features
        image_features = []
        for i in range(batch_size):
            agg_feature = aggregate_patch_features(
                patch_features[i].to(device),
                aggregation_method=aggregation_method
            )
            image_features.append(agg_feature)
        
        image_features = torch.stack(image_features)  # (batch_size, feature_dim)
        cls_labels = cls_labels.to(device).float()
        
        with torch.cuda.amp.autocast():
            # Forward pass through classification head
            logits = classification_head(image_features)  # (batch_size, num_classes)
            
            # Compute classification loss
            loss = loss_function(logits, cls_labels)

        # Gradient scaling for mixed precision
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (n_iter + 1) % 100 == 0:
            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']
            
            torch.cuda.synchronize()
            
            pred = (torch.sigmoid(logits) > 0.5).float()
            all_acc = (pred == cls_labels).all(dim=1).float().mean() * 100
            avg_acc = ((pred == cls_labels).float().mean(dim=0)).mean() * 100
            
            print(
                f"Iter: {n_iter + 1}/{cfg.train.max_iters}; "
                f"Elapsed: {delta}; ETA: {eta}; "
                f"LR: {cur_lr:.3e}; Loss: {loss.item():.4f}; "
                f"Acc: {all_acc:.2f}/{avg_acc:.2f}"
            )
        
        # Regular validation
        if (n_iter + 1) % cfg.train.eval_iters == 0 or (n_iter + 1) == cfg.train.max_iters:
            print(f"\n[Validation at iter {n_iter + 1}]")
            validate_feature_based(
                model=classification_head,
                val_loader=val_loader,
                device=device,
                aggregation_method=aggregation_method,
                num_classes=num_classes
            )

    print("\nTraining completed!")
    print(f"Total elapsed time: {datetime.datetime.now() - start_time}")


def validate_feature_based(model, val_loader, device, aggregation_method="mean", num_classes=4):
    """
    Validation function for feature-based model.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for img_name, patch_features, cls_labels, mask in val_loader:
            batch_size = patch_features.shape[0]
            
            # Aggregate patch features
            image_features = []
            for i in range(batch_size):
                agg_feature = aggregate_patch_features(
                    patch_features[i].to(device),
                    aggregation_method=aggregation_method
                )
                image_features.append(agg_feature)
            
            image_features = torch.stack(image_features)
            cls_labels = cls_labels.to(device).float()
            
            logits = model(image_features)
            pred = (torch.sigmoid(logits) > 0.5).float()
            
            all_preds.append(pred.cpu().numpy())
            all_labels.append(cls_labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute accuracy
    all_acc = (all_preds == all_labels).all(axis=1).mean() * 100
    avg_acc = ((all_preds == all_labels).mean(axis=0)).mean() * 100
    
    print(f"  Val Acc: {all_acc:.2f} / {avg_acc:.2f}")
    
    model.train()


if __name__ == "__main__":
    if args.config is None:
        raise ValueError("Please provide a config file using --config")
    
    cfg = OmegaConf.load(args.config)
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    
    train(cfg)
