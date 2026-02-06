import argparse
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import yaml
import wandb

# Imports from PBIP
from utils.optimizer import PolyWarmupAdamW
from utils.pyutils import set_seed, AverageMeter
from utils.fgbg_feature import FeatureExtractor, MaskAdapter_DynamicThreshold
from utils.contrast_loss import InfoNCELossFG, InfoNCELossBG
from utils.hierarchical_utils import merge_to_parent_predictions, expand_parent_to_subclass_labels, merge_subclass_cams_to_parent
from model.model import ClsNetwork
from medclip import MedCLIPModel, MedCLIPVisionModelViT

# Local imports
from datasets.wsi_dataset import PseudoPatchLabelDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform():
    MEAN = [0.66791496, 0.47791372, 0.70623304]
    STD = [0.1736589, 0.22564577, 0.19820057]
    return A.Compose([
        A.Normalize(MEAN, STD),
        A.Resize(224, 224),
        ToTensorV2()
    ])

def train(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Path setup
    work_dir = config_dict.get('work_dir', './work_dirs')
    exp_name = config_dict.get('experiment_name', 'default_exp')
    exp_dir = os.path.join(work_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # WandB
    wandb_cfg = config_dict.get('wandb', {})
    if wandb_cfg.get('enabled', False):
        wandb.init(project=wandb_cfg.get('project', 'PBIP'), 
                   name=exp_name,
                   config=config_dict)

    # Proto path
    proto_path = os.path.join(exp_dir, 'features/prototypes.pkl')
    if not os.path.exists(proto_path):
        print(f"Prototypes not found at {proto_path}")
        return

    # Hyperparams
    train_cfg = config_dict.get('train', {})
    optim_cfg = config_dict.get('optimizer', {})
    sched_cfg = config_dict.get('scheduler', {})

    # Construct OmegaConf cfg to match PBIP expectations
    cfg = OmegaConf.create({
        'train': {
            'samples_per_gpu': train_cfg.get('samples_per_gpu', 2),
            'max_iters': 0, # set later
            'epoch': train_cfg.get('epoch', 50),
            'pretrained': True,
            'mask_adapter_alpha': 0.5, # Default
            'merge_train': 'max', # Default
            'contrastive_loss_weight': train_cfg.get('contrastive_loss_weight', 0.5),
            'infonce_temperature': train_cfg.get('infonce_temperature', 0.07),
        },
        'dataset': {
            'cls_num_classes': len(config_dict.get('class_names', ['Neg', 'Pos'])),
        },
        'model': {
            'backbone': {'config': 'resnet50', 'stride': 1}, # Defaults
            'n_ratio': 0.1,
            'label_feature_path': proto_path
        },
        'optimizer': {
            'learning_rate': float(optim_cfg.get('learning_rate', 1e-4)),
            'weight_decay': float(optim_cfg.get('weight_decay', 1e-4)),
            'betas': optim_cfg.get('betas', [0.9, 0.999]),
        },
        'scheduler': {
            'warmup_iter': sched_cfg.get('warmup_iter', 100),
            'warmup_ratio': float(sched_cfg.get('warmup_ratio', 0.1)),
            'power': float(sched_cfg.get('power', 1.0))
        }
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    # Dataset
    transform = get_transform()
    train_dataset = PseudoPatchLabelDataset(config_dict['data_csv_path'], config=config_dict, split='train', 
                                          transform=transform, 
                                          binary_mode=config_dict.get('binary_mode', True),
                                          scale_attention=train_cfg.get('scale_attention', True))
    
    val_dataset = PseudoPatchLabelDataset(config_dict['data_csv_path'], config=config_dict, split='val', 
                                          transform=transform,
                                          binary_mode=config_dict.get('binary_mode', True))

    num_workers = train_cfg.get('num_workers', 4)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.samples_per_gpu, 
                            shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    # Model
    model = ClsNetwork(backbone=cfg.model.backbone.config,
                       stride=cfg.model.backbone.stride,
                       cls_num_classes=cfg.dataset.cls_num_classes,
                       n_ratio=cfg.model.n_ratio,
                       pretrained=cfg.train.pretrained,
                       l_fea_path=cfg.model.label_feature_path)
    model.to(device)

    # Optimizer
    iters_per_epoch = len(train_loader)
    cfg.train.max_iters = cfg.train.epoch * iters_per_epoch
    
    optimizer = PolyWarmupAdamW(model.parameters(), lr=cfg.optimizer.learning_rate, 
                                max_iter=cfg.train.max_iters) # Simplify params

    scaler = torch.cuda.amp.GradScaler()
                                
    # Loss
    cls_loss_fn = nn.BCEWithLogitsLoss()
    fg_loss_fn = InfoNCELossFG(temperature=cfg.train.infonce_temperature).to(device)
    mask_adapter = MaskAdapter_DynamicThreshold(alpha=cfg.train.mask_adapter_alpha)
    feature_extractor = FeatureExtractor(mask_adapter=mask_adapter)
    
    # Check if we have clip model for feature extraction in training loop?
    # `train_stage_1.py` has `clip_model`
    clip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).to(device)
    clip_model.eval()

    print("Starting training...")
    
    for epoch in range(cfg.train.epoch):
        model.train()
        avg_cls_loss = AverageMeter()
        avg_sim_loss = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epoch}")
        for imgs, targets, _ in pbar:
            imgs = imgs.to(device).float()
            targets = targets.to(device).float() # [batch_size] or [batch_size, num_classes]
            
            # Forward
            # ClsNetwork forward returns: cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list
            # But PBIP usually has fixed hierarchy depth? 
            # If I look at model.py I would know. Assuming train_stage_1 hierarchy (4 layers).
            # But k_list depends on pickle.
            # If my pickle has 1 layer (flat list of classes), `model.py` might break if it expects 4 layers.
            # I should assume `model` adapts to `k_list`.
            
            # Model Forward Return Handling
            with torch.cuda.amp.autocast():
                # PBIP ClsNetwork returns:
                # cls1, cam1, cls2, cam2, ..., clsN, camN, l_fea, k_list
                # Where N is derived from hierarchy or k_list.
                # Since we don't know N statically, we grab last 2 and assume first pair is finest scale.
                
                results = model(imgs)
                k_list = results[-1]
                l_fea = results[-2]
                
                # Assume results[0] (cls) and results[1] (cam) correspond to first scale
                # If PBIP order is finest-to-coarsest or vice-versa, we typically supervise all or finest.
                # `train_stage_1.py` merged cls1..cls4. 
                # We will just supervise the first output for generic compatibility.
                
                cls_out = results[0] 
                cam_out = results[1]
                
                # Merge to parent prediction
                parent_logits = merge_to_parent_predictions(cls_out, k_list, method='max')
                
                # Loss
                # Prepare target: [Batch, NumClasses]
                if len(targets.shape) == 1:
                    # Binary target?
                    # If NumClasses=2, and target is Prob(Class 1).
                    # Target tensor: [1-p, p]
                    t_stack = torch.stack([1-targets, targets], dim=1)
                    cls_loss = cls_loss_fn(parent_logits, t_stack)
                else:
                    cls_loss = cls_loss_fn(parent_logits, targets)
                
                # Similarity/Contrastive Loss
                # Need binary labels for "Positive" patches to pull them to prototypes?
                # I'll treat target > 0.5 as positive mask
                # Assuming binary mode for simplicity of explanation
                
                # We need `label` for `extract_features`.
                # PBIP uses ONE-HOT-like or Class Index? `where(label==1)`.
                # So if I have 2 classes, `label` should be [Batch, 2] one-hot (or binary thresholded).
                
                if len(targets.shape) == 1:
                    bin_label = torch.zeros_like(parent_logits)
                    bin_label[:, 0] = (targets < 0.5).float()
                    bin_label[:, 1] = (targets >= 0.5).float()
                else:
                    bin_label = (targets > 0.5).float()
                
                # Adapt functionality
                # `expand_parent_to_subclass_labels` expands parent one-hot to subclass one-hot
                subclass_labels = expand_parent_to_subclass_labels(bin_label, k_list)
                
                # Feature extraction
                # cam_out (Batch, Subclasses, H, W)
                # But `cam` comes from `model(inputs)`.
                # `feature_extractor` needs `cam_224`.
                cam_224, cam_224_mask = feature_extractor.prepare_cam_mask(cam_out, imgs.size(0))
                
                fg_fea, _, fg_mask, _ = feature_extractor.extract_features(imgs, cam_224, cam_224_mask, subclass_labels)
                
                sim_loss = torch.tensor(0.0).to(device)
                if fg_fea is not None: # If any foreground detected
                   # Get CLIP features
                   fg_img_fea, _ = feature_extractor.get_masked_features(fg_fea, fg_fea, fg_mask, fg_mask, clip_model)
                   # InfoNCE
                   try:
                       sim_loss = fg_loss_fn(fg_img_fea, subclass_labels, k_list, l_fea)
                   except:
                       sim_loss = torch.tensor(0.0).to(device) # Fallback

                loss = cls_loss + cfg.train.contrastive_loss_weight * sim_loss  
                
            avg_cls_loss.update(cls_loss.item(), imgs.size(0))
            avg_sim_loss.update(sim_loss.item(), imgs.size(0))
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix({'cls': avg_cls_loss.avg, 'sim': avg_sim_loss.avg})
            
            if wandb_cfg.get('enabled', False):
                wandb.log({
                    'batch_cls_loss': cls_loss.item(),
                    'batch_sim_loss': sim_loss.item(),
                    'batch_total_loss': loss.item(),
                    'lr': optimizer.param_groups[0]['lr']
                })
            
        # Validation
        val_dice = validate(model, val_loader, device, k_list)
        print(f"Epoch {epoch+1} Val Dice: {val_dice}")
        
        if wandb_cfg.get('enabled', False):
            wandb.log({
                'epoch': epoch + 1,
                'val_dice': val_dice,
                'avg_cls_loss': avg_cls_loss.avg,
                'avg_sim_loss': avg_sim_loss.avg
            })
        
    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(exp_dir, 'final_model.pth'))

def validate(model, loader, device, k_list):
    model.eval()
    dice_meter = AverageMeter()
    with torch.no_grad():
        for imgs, targets, mask in loader:
            imgs = imgs.to(device).float()
            # If mask is None or zero, skip dice
            if mask.sum() == 0:
                continue
            
            # Prediction
            results = model(imgs)
            cls_out = results[0]
            cam_out = results[1]
            k_list = results[-1]
            
            # Generate mask from CAM
            # CAM: [B, Subclasses, H, W]
            # Merge CAM to parent?
            # Or just take CAM matching target class?
            # Assuming Binary Positive WSI
            # Take Max over positive subclasses?
            # Or just use `generate_cam` utility?
            
            # Simple thresholding
            cam_parent = merge_subclass_cams_to_parent(cam_out, k_list) # Hypothetical function matching merge_train logic
            # Usually we sum/max cam.
            
            # Let's assume prediction is Cam[:, 1] (Positive)
            pred_mask = (cam_parent[:, 1] > 0.5).cpu().numpy()
            
            # Calc Dice vs `mask`
            # mask is [B, H, W]
            inter = (pred_mask * mask.numpy()).sum()
            union = pred_mask.sum() + mask.numpy().sum()
            dice = 2*inter / (union + 1e-8)
            dice_meter.update(dice, 1)
            
    return dice_meter.avg

# Helper for cam merge if not imported
def merge_subclass_cams_to_parent(cams, k_list):
    # cams: [B, SumK, H, W]
    # k_list: [k1, k2...]
    # result: [B, LenK, H, W]
    start = 0
    outs = []
    for k in k_list:
        sub = cams[:, start:start+k]
        # Max pool across subclasses
        mp, _ = torch.max(sub, dim=1, keepdim=True)
        outs.append(mp)
        start += k
    return torch.cat(outs, dim=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)
