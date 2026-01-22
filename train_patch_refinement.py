#!/usr/bin/env python3
"""
Simplified training script for feature-based patch-level refinement.

This is the NEW recommended training approach for the feature-based pipeline:
- Loads precomputed patch features (from virchow2 or dinov3)
- OR auto-generates Virchow2 patch features from WSI if patch_features_dir is missing
- OR auto-generates TITAN global features if global_feature_dir is missing
- Refines patch-level predictions using prototype matching

Usage:
    python train_patch_refinement.py --config work_dirs/custom_wsi_template.yaml --gpu 0

Key differences from train_stage_1.py (deprecated):
- No pixel-level dense prediction
- No multi-scale CAM generation
- No on-the-fly patch extraction
- Simpler model architecture (no ClsNetwork)
- Direct patch feature refinement with prototype matching
- On-the-fly feature generation if features missing (auto_generate_patch_features, auto_generate_global_features)
"""

import argparse
import datetime
import os
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import yaml
import wandb
import numpy as np
import hashlib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.trainutils import get_custom_dataset
from utils.optimizer import PolyWarmupAdamW
from utils.pyutils import set_seed, AverageMeter, build_uid_from_config
from utils.encoders import EncoderFactory, get_encoder_config
from utils.prototype_guided_attention import PrototypeGuidedAttention
from utils.refinement_losses import CombinedRefinementLoss
from utils.validation_metrics import ValidationMetrics

start_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None, required=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--verbose", action="store_true", help="Enable detailed timing logs")
args = parser.parse_args()


class PatchRefinementModel(nn.Module):
    """
    Simple model for patch-level prototype-based refinement.
    
    Unlike the old ClsNetwork (which outputs 4 multi-scale CAMs for dense prediction),
    this model:
    1. Takes patch features + global features as input
    2. Concatenates global feature with each patch feature
    3. Computes similarity to prototype bank
    4. Refines predictions by combining prototype match scores with raw attention
    
    No spatial/CNN processing needed - features are already extracted.
    """
    
    def __init__(self, feature_dim, num_classes, global_feature_dim=None, 
                 prototype_features=None, k_list=None, nk=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.global_feature_dim = global_feature_dim if global_feature_dim else feature_dim
        
        # Combined feature dimension (patch + global)
        self.combined_dim = feature_dim + self.global_feature_dim
        
        # Create prototype bank and prototype matching component
        from utils.prototype_guided_attention import PrototypeBank
        
        # Initialize prototype bank
        self.prototype_bank = PrototypeBank(
            num_classes=num_classes,
            feature_dim=feature_dim,
            max_prototypes_per_class=100  # Can make configurable if needed
        )
        
        # Load prototypes into the bank if provided
        if prototype_features is not None and k_list is not None:
            # TODO: Load prototypes from the provided features into the bank
            # For now, the bank will be empty and prototype matching will be skipped
            pass
        
        # Prototype matching component
        self.prototype_guided_attention = PrototypeGuidedAttention(
            prototype_bank=self.prototype_bank,
            num_classes=num_classes,
            attention_type='cosine_sim',
            temperature=1.0
        )
        
        # Optional: projection layer to normalize concatenated features
        self.feature_projection = nn.Linear(self.combined_dim, feature_dim)
        
    def forward(self, patch_features, global_features, raw_attention=None):
        """
        Args:
            patch_features: (batch_size, num_patches, feature_dim) or (num_patches, feature_dim)
            global_features: (batch_size, global_feature_dim) or (global_feature_dim,)
            raw_attention: (batch_size, num_patches, num_classes) or (num_patches, num_classes)
        
        Returns:
            refined_attention: (batch_size, num_patches, num_classes) refined predictions
        """
        # Handle batched input
        is_batched = patch_features.dim() == 3
        
        if is_batched:
            batch_size, num_patches, _ = patch_features.shape
            # Expand global features to match patches: (batch_size, 1, global_dim) -> (batch_size, num_patches, global_dim)
            if global_features.dim() == 2:
                global_features_expanded = global_features.unsqueeze(1).expand(batch_size, num_patches, -1)
            else:
                # Already expanded
                global_features_expanded = global_features
        else:
            # Single sample
            num_patches = patch_features.shape[0]
            batch_size = 1
            if global_features.dim() == 1:
                global_features_expanded = global_features.unsqueeze(0).expand(num_patches, -1)
            else:
                global_features_expanded = global_features
        
        # Concatenate patch features with global features
        combined_features = torch.cat([patch_features, global_features_expanded], dim=-1)
        
        # Project back to original feature dimension
        projected_features = self.feature_projection(combined_features)
        projected_features_norm = F.normalize(projected_features, p=2, dim=-1)
        
        # Flatten for prototype matching if batched
        if is_batched:
            projected_flat = projected_features_norm.view(-1, self.feature_dim)  # (batch_size * num_patches, D)
            if raw_attention is not None:
                raw_attention_flat = raw_attention.view(-1, raw_attention.shape[-1])  # (batch_size * num_patches, num_classes)
            else:
                raw_attention_flat = None
        else:
            projected_flat = projected_features_norm
            raw_attention_flat = raw_attention
        
        # Compute prototype matching scores (flattened)
        match_scores_flat = self.prototype_guided_attention.compute_match_scores(projected_flat)
        
        # Combine with raw attention if provided
        if raw_attention_flat is not None:
            refined_attention_flat = self.prototype_guided_attention.combine_with_raw_attention(
                    match_scores_flat, raw_attention_flat
            )
        else:
            # Use prototype scores directly if no raw attention
                refined_attention_flat = F.softmax(match_scores_flat, dim=-1)
        
        # Reshape back to batch format if needed
        if is_batched:
            refined_attention = refined_attention_flat.view(batch_size, num_patches, -1)
            match_scores = match_scores_flat.view(batch_size, num_patches, self.num_classes)
        else:
            refined_attention = refined_attention_flat
            match_scores = match_scores_flat
        
        return refined_attention, match_scores


def load_config(config_path):
    """Load and validate configuration."""
    with open(config_path) as f:
        cfg = OmegaConf.load(f)
    
    # Verify required settings for feature-based pipeline
    if not getattr(cfg.dataset, 'patch_features_dir', None):
        raise ValueError(
            "ERROR: patch_features_dir not set in config.\n"
            "This is required for feature-based training pipeline.\n"
            "Set dataset.patch_features_dir to path containing precomputed patch features."
        )
    
    # No pseudo-labels required in the feature-based pipeline
    
    return cfg


def main():
    print("\n" + "="*80)
    print("PATCH-LEVEL PROTOTYPE-BASED REFINEMENT TRAINING")
    print("="*80)
    print(f"Start time: {start_time}")
    print(f"Config: {args.config}")
    
    # Load configuration
    cfg = load_config(args.config)
    print(f"✓ Config loaded: use_feature_based_training={getattr(cfg.dataset, 'use_feature_based_training', False)}")
    
    # Resolve run_uid from config (only once per run, never concatenate)
    uid = build_uid_from_config(cfg)
    cfg.run_uid = uid  # Only set once, never concatenate
    print(f"✓ Generated run_uid from config: {uid}")
    
    # Resolve ${run_uid} placeholders via OmegaConf (no manual string replace)
    def _resolve(value):
        return OmegaConf.to_container(OmegaConf.create({'v': value}), resolve=True)['v']

    cfg.model.label_feature_path = _resolve(cfg.model.label_feature_path)
    if hasattr(cfg, 'output_dirs'):
        if hasattr(cfg.output_dirs, 'ckpt_dir'):
            cfg.output_dirs.ckpt_dir = _resolve(cfg.output_dirs.ckpt_dir)
        if hasattr(cfg.output_dirs, 'pred_dir'):
            cfg.output_dirs.pred_dir = _resolve(cfg.output_dirs.pred_dir)
        if hasattr(cfg.output_dirs, 'train_log_dir'):
            cfg.output_dirs.train_log_dir = _resolve(cfg.output_dirs.train_log_dir)
    
    # Set random seed
    set_seed(0)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    
    # Create output directories
    os.makedirs(cfg.output_dirs.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.output_dirs.pred_dir, exist_ok=True)
    os.makedirs(cfg.output_dirs.train_log_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("LOADING DATASETS")
    print(f"{'='*80}")
    
    # Load datasets (feature-based pipeline)
    train_dataset, val_dataset = get_custom_dataset(cfg, verbose=args.verbose)
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Custom collate function for variable-length patch sequences
    def collate_variable_patches(batch):
        """
        Collate function for batches with variable number of patches per WSI.
        Returns lists instead of stacked tensors for patch_features.
        Handles optional patch_labels and pseudo_labels (None if not available).
        """
        wsi_names = [item[0] for item in batch]
        patch_features = [item[1] for item in batch]  # List of tensors with different shapes
        cls_labels = torch.stack([item[2] for item in batch])
        global_features = torch.stack([item[3] for item in batch])
        
        # Optional: ground truth patch labels (for segmentation metrics)
        patch_labels = [item[4] for item in batch] if len(batch[0]) > 4 else None
        
        # Optional: pseudo-labels (attention scores for benign vs positive discrimination)
        pseudo_labels = [item[5] for item in batch] if len(batch[0]) > 5 else None
        
        return wsi_names, patch_features, cls_labels, global_features, patch_labels, pseudo_labels
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.samples_per_gpu,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_variable_patches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.samples_per_gpu,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_variable_patches
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Get encoder configuration to determine feature dimension
    encoder_config = get_encoder_config(cfg.model.patch_encoder)
    feature_dim = encoder_config['feature_dim']
    print(f"✓ Patch encoder: {cfg.model.patch_encoder} (feature_dim={feature_dim})")
    
    # Load prototype features if available
    print(f"\n{'='*80}")
    print("INITIALIZING MODEL")
    print(f"{'='*80}")
    
    prototype_features = None
    k_list = None
    nk = None
    
    try:
        # Try to load label features for prototype matching
        import pickle
        label_fea_path = cfg.model.label_feature_path
        if os.path.exists(label_fea_path):
            with open(label_fea_path, 'rb') as f:
                label_fea_info = pickle.load(f)
                prototype_features = label_fea_info['features']
                k_list = label_fea_info['k_list']
                nk = label_fea_info.get('nk', 5)
                
                # Convert to torch tensor if needed
                if isinstance(prototype_features, np.ndarray):
                    prototype_features = torch.from_numpy(prototype_features).float()
                elif not isinstance(prototype_features, torch.Tensor):
                    prototype_features = torch.tensor(prototype_features).float()
                
                print(f"✓ Loaded prototype features: {prototype_features.shape}")
                print(f"  - K (subclasses): {k_list}")
                print(f"  - Nk (representatives): {nk}")
        else:
            print(f"⚠ Prototype features not found at {label_fea_path}")
            print("  Training will refine with attention scores only (no prototype matching)")
    except Exception as e:
        print(f"⚠ Could not load prototype features: {e}")
    
    # Create model
    model = PatchRefinementModel(
        feature_dim=feature_dim,
        num_classes=cfg.dataset.num_classes,
        global_feature_dim=feature_dim,  # Same as patch features unless specified
        prototype_features=prototype_features,
        k_list=k_list,
        nk=nk
    )
    model = model.to(device)
    
    print(f"✓ Model initialized on device {device}")
    print(f"  - Patch feature dimension: {feature_dim}")
    print(f"  - Global feature dimension: {feature_dim}")
    print(f"  - Combined dimension: {feature_dim + feature_dim}")
    print(f"  - Number of classes: {cfg.dataset.num_classes}")
    
    # Create classifier head for slide-level predictions
    classifier_head = nn.Linear(feature_dim, cfg.dataset.num_classes)
    classifier_head = classifier_head.to(device)
    print(f"✓ Classifier head created: Linear({feature_dim}, {cfg.dataset.num_classes})")
    
    # Optimizer (include classifier head parameters)
    optimizer = PolyWarmupAdamW(
        params=list(model.parameters()) + list(classifier_head.parameters()),
        lr=cfg.optimizer.learning_rate,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.weight_decay,
        warmup_iter=cfg.scheduler.warmup_iter,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        max_iter=cfg.train.epoch * len(train_loader),
        power=cfg.scheduler.power
    )
    
    # Loss function: Slide-level CE + Prototype contrastive
    from utils.refinement_losses import CombinedRefinementLoss
    
    # Check if we have a prototype bank to use for contrastive loss
    if prototype_features is not None:
        # Import PrototypeBank to wrap prototype features
        from utils.prototype_guided_attention import PrototypeBank
        
        # Create prototype bank from loaded features
        prototype_bank = PrototypeBank(
            num_classes=cfg.dataset.num_classes,
            feature_dim=feature_dim,
            max_prototypes_per_class=100,
        )
        
        # Initialize prototype bank with loaded features
        # prototype_features shape: (K_total, D) where K_total = sum(k_list)
        # Need to split by class according to k_list
        start_idx = 0
        for class_id, k in enumerate(k_list):
            end_idx = start_idx + k * nk  # k subclasses * nk representatives each
            class_prototypes = prototype_features[start_idx:end_idx]
            prototype_bank.add_prototypes(class_id, class_prototypes)
            start_idx = end_idx
            print(f"  Added {len(class_prototypes)} prototypes for class {class_id}")
        
        # Create combined loss: slide-level CE + contrastive
        combined_dim = feature_dim + feature_dim  # patch features + global features
        criterion = CombinedRefinementLoss(
            prototype_bank=prototype_bank,
            classifier_head=classifier_head,
            num_classes=cfg.dataset.num_classes,
            contrastive_weight=getattr(cfg.train, 'contrastive_loss_weight', 0.5),
            temperature=getattr(cfg.train, 'infonce_temperature', 0.07),
            combined_feature_dim=combined_dim,
            feature_dim=feature_dim,
        ).to(device)
        print(f"✓ Using CombinedRefinementLoss: Slide-level CE + Prototype contrastive")
        print(f"  - Combined feature dim: {combined_dim} -> projected to {feature_dim}")
    else:
        # Fallback to simple cross-entropy if no prototypes
        criterion = nn.CrossEntropyLoss()
        print(f"⚠ No prototypes available, using basic CrossEntropyLoss")
    
    # W&B initialization (if enabled)
    use_wandb = getattr(cfg.wandb, 'enabled', True)
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg),
            tags=[cfg.model.patch_encoder, 'feature-based', 'patch-refinement'],
            notes=f"Patch-level refinement with {cfg.model.patch_encoder} features"
        )
        print(f"✓ W&B initialized: {cfg.wandb.project}")
    
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}")
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(cfg.train.epoch):
        print(f"\n--- Epoch {epoch+1}/{cfg.train.epoch} ---")
        
        # Training
        model.train()
        train_loss_meter = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Train", total=len(train_loader))
        for batch_idx, batch in enumerate(pbar):
            t_batch_start = time.time()
            
            # Get batch data
            # FeatureWSIDataset now returns: (wsi_name, patch_features, cls_label, global_feature, patch_labels, pseudo_labels)
            # patch_features is a list of tensors (variable length)
            wsi_names, patch_features_list, cls_labels, global_features, patch_labels, pseudo_labels = batch
            
            optimizer.zero_grad()
            
            # Process each WSI in the batch and accumulate gradients
            batch_losses = []
            for i in range(len(wsi_names)):
                # GPU transfer
                t_gpu_start = time.time()
                patch_feats = patch_features_list[i].unsqueeze(0).to(device)  # (1, n_patches, D)
                global_feat = global_features[i].unsqueeze(0).to(device)  # (1, D)
                cls_label = cls_labels[i].unsqueeze(0).to(device)  # (1, num_classes)
                t_gpu = time.time() - t_gpu_start
                
                # CONSERVATIVE OPTIMIZATION: Patch Sampling
                # If N is too large (e.g. 160k), gradients become the bottleneck in loss/backward
                # Randomly sample a representative subset for training
                max_p = cfg.model.get('max_patches_training', 4096)
                num_patches_raw = patch_feats.shape[1]
                if num_patches_raw > max_p:
                    indices = torch.randperm(num_patches_raw, device=device)[:max_p]
                    patch_feats = patch_feats[:, indices, :]
                
                # Forward pass
                t_forward_start = time.time()
                refined_attention, match_scores = model(patch_feats, global_feat)
                t_forward = time.time() - t_forward_start
                
                # Prepare combined features for loss
                batch_size, num_patches, feature_dim = patch_feats.shape
                global_feature_dim = global_feat.shape[-1]
                
                # Expand global features to match patches
                global_features_expanded = global_feat.unsqueeze(1).expand(batch_size, num_patches, -1)
                combined_features = torch.cat([patch_feats, global_features_expanded], dim=-1)
                
                # Loss computation
                t_loss_start = time.time()
                if isinstance(criterion, CombinedRefinementLoss):
                    loss, loss_info = criterion(combined_features, cls_label)
                    
                    # Log detailed loss components
                    if batch_idx % 10 == 0 and use_wandb and i == 0:
                        train_metrics = {
                            'train/patch_ce_loss': loss_info.get('patch_ce_loss', 0.0),
                            'train/slide_ce_loss': loss_info.get('slide_ce_loss', 0.0),
                            'train/contrastive_loss': loss_info.get('contrastive_loss', 0.0),
                            'train/total_loss': loss_info.get('total_loss', 0.0),
                        }
                        wandb.log(train_metrics)
                else:
                    # Fallback: basic cross-entropy
                    loss = criterion(refined_attention.view(-1, cfg.dataset.num_classes), 
                                   cls_label.argmax(dim=1).unsqueeze(1).expand(-1, refined_attention.shape[1]).reshape(-1))
                t_loss = time.time() - t_loss_start
                
                # Normalize loss by batch size for gradient accumulation
                loss = loss / len(wsi_names)
                
                # Backward pass (gradients accumulate across batch)
                t_backward_start = time.time()
                loss.backward()
                t_backward = time.time() - t_backward_start
                
                batch_losses.append(loss.item() * len(wsi_names))  # Unnormalize for logging
                
                # Log timing for first sample in batch
                if i == 0 and batch_idx % 5 == 0:
                    if args.verbose:
                        print(f"[BATCH_TIMING] idx={batch_idx} gpu={t_gpu:.3f}s forward={t_forward:.3f}s loss={t_loss:.3f}s backward={t_backward:.3f}s")
            
            # Single optimizer step after processing all samples in batch
            optimizer.step()
            
            # Update metrics with average loss across batch
            train_loss_meter.update(np.mean(batch_losses))
            pbar.set_postfix({'loss': train_loss_meter.avg})
            
            t_batch_total = time.time() - t_batch_start
            if batch_idx % 5 == 0:
                print(f"[BATCH_TOTAL] idx={batch_idx} total_batch_time={t_batch_total:.3f}s n_items={len(wsi_names)}")
        
        print(f"Train Loss: {train_loss_meter.avg:.4f}")
        
        # Validation
        if (epoch + 1) % 1 == 0:  # Validate every epoch
            model.eval()
            val_loss_meter = AverageMeter()
            
            # Initialize validation metrics tracker
            val_metrics = ValidationMetrics(
                num_classes=cfg.dataset.num_classes,
                class_names=getattr(cfg.dataset, 'class_names', None)
            )
            
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"Val", total=len(val_loader))
                for batch_idx, batch in enumerate(pbar):
                    t_val_batch_start = time.time()
                    
                    wsi_names, patch_features_list, cls_labels, global_features, patch_labels_list, pseudo_labels_list = batch
                    
                    # Process each WSI separately (variable length)
                    batch_losses = []
                    for i in range(len(wsi_names)):
                        patch_feats = patch_features_list[i].unsqueeze(0).to(device)  # (1, n_patches, D)
                        global_feat = global_features[i].unsqueeze(0).to(device)  # (1, D)
                        cls_label = cls_labels[i].unsqueeze(0).to(device)  # (1, num_classes)
                        
                        # CONSERVATIVE OPTIMIZATION: Patch Sampling
                        # For extremely large slides (e.g. 160k patches), gradients and loss computation
                        # become the bottleneck. Even in validation, large tensors can be slow.
                        max_p_val = cfg.model.get('max_patches_training', 4096) * 4  # Allow 4x more for validation
                        num_patches_raw = patch_feats.shape[1]
                        if num_patches_raw > max_p_val:
                            # Use fixed seed for deterministic validation sampling
                            gen = torch.Generator(device=device)
                            gen.manual_seed(42)
                            indices = torch.randperm(num_patches_raw, device=device, generator=gen)[:max_p_val]
                            patch_feats = patch_feats[:, indices, :]
                            if patch_labels_list is not None and patch_labels_list[i] is not None:
                                # Sample GT labels too
                                # patch_labels_list[i] is (n_patches,)
                                patch_labels_list[i] = patch_labels_list[i][indices.cpu()]
                        
                        # Get ground truth patch labels if available (sampled above if needed)
                        patch_gt = None
                        if patch_labels_list is not None and patch_labels_list[i] is not None:
                            patch_gt = patch_labels_list[i].unsqueeze(0).to(device)  # (1, n_patches)
                        
                        # Forward pass
                        refined_attention, match_scores = model(patch_feats, global_feat)
                        
                        # Combine features for loss (patch + global context)
                        batch_size, num_patches, feature_dim = patch_feats.shape
                        global_feature_dim = global_feat.shape[-1]
                        global_features_expanded = global_feat.unsqueeze(1).expand(batch_size, num_patches, -1)
                        combined_features = torch.cat([patch_feats, global_features_expanded], dim=-1)
                        
                        # Loss computation: Patch-level CE + Slide-level CE + Contrastive
                        if isinstance(criterion, CombinedRefinementLoss):
                            loss, loss_info = criterion(combined_features, cls_label)
                            
                            # === EXTRACT METRICS BY MODULE ===
                            
                            # 1. ClassifierNet metrics (slide-level classification)
                            # Get slide-level logits from aggregated patch predictions
                            patch_logits = criterion.classifier(
                                criterion.feature_projection(combined_features.view(-1, combined_features.shape[-1]))
                                if criterion.feature_projection is not None
                                else combined_features.view(-1, combined_features.shape[-1])
                            ).view(batch_size, num_patches, cfg.dataset.num_classes)
                            slide_logits = patch_logits.mean(dim=1)  # (1, num_classes)
                            
                            # Track loss components separately for detailed logging
                            if not hasattr(val_metrics, 'patch_ce_losses'):
                                val_metrics.patch_ce_losses = []
                                val_metrics.slide_ce_losses = []
                            val_metrics.patch_ce_losses.append(loss_info.get('patch_ce_loss', 0.0))
                            val_metrics.slide_ce_losses.append(loss_info.get('slide_ce_loss', 0.0))
                            
                            val_metrics.update_classification(
                                slide_logits=slide_logits,
                                slide_labels=cls_label,
                                classifier_loss=0.0  # Already tracked above
                            )
                            
                            # Track prototype contrastive loss
                            val_metrics.prototype_losses.append(loss_info.get('contrastive_loss', 0.0))
                            
                            # Segmentation metrics (patch predictions)
                            val_metrics.update_segmentation(
                                patch_predictions=patch_logits,  # (1, n_patches, num_classes)
                                patch_labels=patch_gt,  # (1, n_patches) or None
                                slide_labels=cls_label  # (1, num_classes) or (1,)
                            )
                            
                        else:
                            loss = criterion(refined_attention.view(-1, cfg.dataset.num_classes),
                                           cls_label.argmax(dim=1).unsqueeze(1).expand(-1, refined_attention.shape[1]).reshape(-1))
                        
                        batch_losses.append(loss.item())
                        val_metrics.total_losses.append(loss.item())
                    
                    val_loss_meter.update(np.mean(batch_losses))
                    pbar.set_postfix({'loss': val_loss_meter.avg})
                    
                    t_val_batch_total = time.time() - t_val_batch_start
                    if args.verbose and batch_idx % 5 == 0:
                        print(f"[VAL_BATCH_TOTAL] idx={batch_idx} total_time={t_val_batch_total:.3f}s n_items={len(wsi_names)}")
            
            # Compute all validation metrics
            all_val_metrics = val_metrics.compute_all_metrics()
            
            print(f"\n{'='*80}")
            print(f"VALIDATION RESULTS - Epoch {epoch + 1}")
            print(f"{'='*80}")
            print(f"Val Loss: {val_loss_meter.avg:.4f}")
            
            # Print key metrics by module
            if 'classifier/f1_macro' in all_val_metrics:
                print(f"\n[Classification - Slide Level]")
                print(f"  F1 (macro): {all_val_metrics.get('classifier/f1_macro', 0):.4f}")
                print(f"  AUROC (macro): {all_val_metrics.get('classifier/auroc_macro', 0):.4f}")
                print(f"  AUPRC (macro): {all_val_metrics.get('classifier/auprc_macro', 0):.4f}")
            
            if 'segmentation/mean_iou' in all_val_metrics:
                print(f"\n[Segmentation - Patch Level]")
                print(f"  Mean IoU: {all_val_metrics['segmentation/mean_iou']:.4f}")
            
            print(f"{'='*80}\n")
            
            # Save best model
            if val_loss_meter.avg < best_loss:
                best_loss = val_loss_meter.avg
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }
                ckpt_path = os.path.join(cfg.output_dirs.ckpt_dir, 'best_model.pth')
                torch.save(checkpoint, ckpt_path)
                print(f"✓ Saved best model: {ckpt_path}")
            
            # Log to W&B with streamlined metrics
            if use_wandb:
                wandb_log = {
                    'epoch': epoch + 1,
                }
                
                # Add validation loss components
                wandb_log['val/patch_ce_loss'] = np.mean(val_metrics.patch_ce_losses) if hasattr(val_metrics, 'patch_ce_losses') else 0.0
                wandb_log['val/slide_ce_loss'] = np.mean(val_metrics.slide_ce_losses) if hasattr(val_metrics, 'slide_ce_losses') else 0.0
                wandb_log['val/contrastive_loss'] = np.mean(val_metrics.prototype_losses) if len(val_metrics.prototype_losses) > 0 else 0.0
                wandb_log['val/total_loss'] = val_loss_meter.avg
                
                # Add essential validation metrics (classifier and segmentation only)
                for metric_name, metric_value in all_val_metrics.items():
                    # Only log classifier/ and segmentation/ metrics
                    if metric_name.startswith('classifier/') or metric_name.startswith('segmentation/'):
                        wandb_log[f'val/{metric_name}'] = metric_value
                
                wandb.log(wandb_log)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {cfg.output_dirs.ckpt_dir}")
    print(f"Total time: {datetime.datetime.now() - start_time}")
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
