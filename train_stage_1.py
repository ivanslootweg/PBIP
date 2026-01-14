'''
python train_cls.py --config ./work_dirs/bcss/classification/config.yaml

For custom WSI dataset with coordinates:
python train_stage_1.py --config ./work_dirs/custom/config.yaml --gpu 0
'''
import argparse
import datetime
import os
import numpy as np
import cv2 as cv
from omegaconf import OmegaConf
from tqdm import tqdm
import ttach as tta
from skimage import morphology
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.trainutils import get_cls_dataset, get_custom_dataset
from utils.optimizer import PolyWarmupAdamW
from utils.pyutils import set_seed,AverageMeter
from utils.evaluate import ConfusionMatrixAllClass
from utils.fgbg_feature import FeatureExtractor, MaskAdapter_DynamicThreshold
from utils.contrast_loss import InfoNCELossFG, InfoNCELossBG
from utils.hierarchical_utils import pair_features, merge_to_parent_predictions, merge_subclass_cams_to_parent, expand_parent_to_subclass_labels
from utils.validate import validate, generate_cam
from utils.common import merge_multiscale_predictions, compute_multiscale_loss
from utils.encoders import EncoderFactory, get_encoder_config
from model.model import ClsNetwork

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


def extract_and_cache_features(dataset, encoder, cfg, encoder_name, device):
    """
    Extract patch features using encoder and cache them to disk.
    Skip extraction if features already exist.
    Also generates thumbnails for GT masks and image data of 10x10 patch areas.
    """
    from utils.common import load_coordinates, extract_patch_numpy, extract_patch_openslide
    from datasets.wsi_dataset import HAS_OPENSLIDE
    
    # Create feature cache directory
    features_dir = Path(cfg.work_dir) / "cached_features" / encoder_name
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Create thumbnail directories
    thumbnails_dir = Path(cfg.work_dir) / "thumbnails"
    mask_thumb_dir = thumbnails_dir / "gt_masks"
    image_thumb_dir = thumbnails_dir / "image_data"
    mask_thumb_dir.mkdir(parents=True, exist_ok=True)
    image_thumb_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Feature cache directory: {features_dir}")
    print(f"Thumbnail directories: {mask_thumb_dir}, {image_thumb_dir}")
    
    use_openslide = HAS_OPENSLIDE
    if hasattr(dataset, 'use_openslide'):
        use_openslide = dataset.use_openslide
    
    encoder.eval()
    skipped = 0
    extracted = 0
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Extracting features"):
            # Get sample info
            if hasattr(dataset, 'filenames'):
                filename = dataset.filenames[idx]
                wsi_path = dataset.wsi_paths[idx]
                coord_path = dataset.coordinate_paths[idx]
            else:
                continue
            
            feature_path = features_dir / f"{filename}.pt"
            
            # Check if we need to generate thumbnails (even if features exist)
            should_generate_thumbnails = False
            if hasattr(dataset, 'gt_dir'):
                gt_dir = dataset.gt_dir
                mask_suffix = getattr(dataset, 'mask_suffix', '.png')
                mask_path = os.path.join(gt_dir, filename + mask_suffix)
                
                if os.path.exists(mask_path):
                    # Check if any thumbnail exists for this file
                    existing_thumbs = list(mask_thumb_dir.glob(f"{filename}_grid*.png"))
                    if len(existing_thumbs) == 0:
                        should_generate_thumbnails = True
            
            # Check if feature already exists
            if feature_path.exists():
                skipped += 1
                
                # Generate thumbnails if needed even though features exist
                if should_generate_thumbnails:
                    # Load coordinates for thumbnail generation
                    patch_size = getattr(dataset, 'patch_size', 224)
                    max_patches = getattr(dataset, 'max_patches', None)
                    coords_suffix = getattr(dataset, 'coordinates_suffix', '.npy')
                    coords = load_coordinates(coord_path, coords_suffix, max_patches)
                    
                    if len(coords) > 0:
                        patch_coords = [(int(c[0]), int(c[1])) for c in coords]
                        generate_thumbnails(wsi_path, mask_path, patch_coords, patch_size,
                                          filename, mask_thumb_dir, image_thumb_dir,
                                          use_openslide, dataset)
                continue
            
            # Load coordinates
            patch_size = getattr(dataset, 'patch_size', 224)
            max_patches = getattr(dataset, 'max_patches', None)
            coords_suffix = getattr(dataset, 'coordinates_suffix', '.npy')
            
            coords = load_coordinates(coord_path, coords_suffix, max_patches)
            
            if len(coords) == 0:
                continue
            
            # Extract patches and features
            patches = []
            patch_coords = []
            
            # Open WSI once for all patches (major speedup)
            if use_openslide:
                try:
                    from utils.common import _openslide
                    slide = _openslide.OpenSlide(wsi_path)
                    half = patch_size // 2
                except Exception as e:
                    print(f"Failed to open WSI {filename}: {e}")
                    continue
            else:
                # Load entire WSI once for numpy-based extraction
                try:
                    wsi = cv.imread(wsi_path)
                    if wsi is None:
                        from skimage import io
                        wsi = io.imread(wsi_path)
                    if len(wsi.shape) == 2:
                        wsi = cv.cvtColor(wsi, cv.COLOR_GRAY2RGB)
                    elif wsi.shape[2] == 4:
                        wsi = cv.cvtColor(wsi, cv.COLOR_RGBA2RGB)
                except Exception as e:
                    print(f"Failed to load WSI {filename}: {e}")
                    continue
            
            for coord in coords:
                cx, cy = int(coord[0]), int(coord[1])
                
                try:
                    if use_openslide:
                        # Extract patch using already-opened slide
                        top_left = (max(0, cx - half), max(0, cy - half))
                        region = slide.read_region(top_left, 0, (patch_size, patch_size))
                        region = region.convert('RGB')
                        patch = np.array(region)
                    else:
                        # Extract patch from already-loaded WSI
                        patch = extract_patch_numpy(wsi, cx, cy, patch_size)
                    
                    if len(patch.shape) == 2:
                        patch = cv.cvtColor(patch, cv.COLOR_GRAY2RGB)
                    
                    patches.append(patch)
                    patch_coords.append((cx, cy))
                except:
                    continue
            
            # Close slide if using OpenSlide
            if use_openslide:
                slide.close()
            
            if len(patches) == 0:
                continue
            
            # Convert to tensor and extract features
            patch_tensors = []
            for patch in patches:
                patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                patch_tensors.append(patch_tensor)
            
            # Batch process with smaller chunks to avoid OOM
            feature_batch_size = getattr(cfg.dataset, 'feature_batch_size', 32)
            all_features = []
            
            for i in range(0, len(patch_tensors), feature_batch_size):
                batch_end = min(i + feature_batch_size, len(patch_tensors))
                batch = torch.stack(patch_tensors[i:batch_end]).to(device)
                
                with torch.amp.autocast('cuda'):
                    if hasattr(encoder, 'vision_model'):
                        # MedCLIP
                        batch_features = encoder.vision_model(batch)
                    else:
                        # Virchow2 or DinoV3
                        batch_features = encoder(batch)
                
                all_features.append(batch_features.cpu())
                
                # Clean up GPU memory
                del batch, batch_features
                torch.cuda.empty_cache()
            
            # Concatenate all features
            features = torch.cat(all_features, dim=0)
            
            # Save features and coordinates
            torch.save({
                'features': features,
                'coordinates': patch_coords,
                'filename': filename
            }, feature_path)
            
            extracted += 1
            
            # Generate thumbnails if GT mask exists
            if hasattr(dataset, 'gt_dir'):
                gt_dir = dataset.gt_dir
                mask_suffix = getattr(dataset, 'mask_suffix', '.png')
                mask_path = os.path.join(gt_dir, filename + mask_suffix)
                
                if os.path.exists(mask_path):
                    generate_thumbnails(wsi_path, mask_path, patch_coords, patch_size,
                                      filename, mask_thumb_dir, image_thumb_dir,
                                      use_openslide, dataset)
    
    print(f"\nFeature extraction complete:")
    print(f"  Extracted: {extracted}")
    print(f"  Skipped (already cached): {skipped}")
    print(f"  Total: {len(dataset)}")


def generate_thumbnails(wsi_path, mask_path, patch_coords, patch_size,
                       filename, mask_thumb_dir, image_thumb_dir,
                       use_openslide, dataset):
    """
    Generate thumbnails for 10x10 patch areas where at least 1 patch has tumor.
    Creates both GT mask thumbnails and image data thumbnails.
    """
    # Load GT mask
    gt_mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    if gt_mask is None:
        return
    
    # Load WSI
    if use_openslide:
        import openslide
        wsi = openslide.OpenSlide(wsi_path)
        wsi_width, wsi_height = wsi.dimensions
    else:
        wsi_img = cv.imread(wsi_path)
        if wsi_img is None:
            from skimage import io
            wsi_img = io.imread(wsi_path)
        if len(wsi_img.shape) == 2:
            wsi_img = cv.cvtColor(wsi_img, cv.COLOR_GRAY2RGB)
        elif wsi_img.shape[2] == 4:
            wsi_img = cv.cvtColor(wsi_img, cv.COLOR_RGBA2RGB)
        wsi_height, wsi_width = wsi_img.shape[:2]
    
    # Find 10x10 patch area with tumor
    # Group coordinates into 10x10 grids
    grid_size = 10
    grid_pixel_size = grid_size * patch_size
    
    # Find all possible 10x10 grids
    grids = {}
    for cx, cy in patch_coords:
        # Determine grid origin (top-left corner)
        grid_x = (cx // grid_pixel_size) * grid_pixel_size
        grid_y = (cy // grid_pixel_size) * grid_pixel_size
        grid_key = (grid_x, grid_y)
        
        if grid_key not in grids:
            grids[grid_key] = []
        grids[grid_key].append((cx, cy))
    
    # Check each grid for tumor presence
    for grid_idx, (grid_origin, grid_patches) in enumerate(grids.items()):
        grid_x, grid_y = grid_origin
        
        # Check if this grid region has tumor in GT mask
        # Sample the mask at grid region
        mask_x1 = max(0, grid_x)
        mask_y1 = max(0, grid_y)
        mask_x2 = min(gt_mask.shape[1], grid_x + grid_pixel_size)
        mask_y2 = min(gt_mask.shape[0], grid_y + grid_pixel_size)
        
        if mask_x2 <= mask_x1 or mask_y2 <= mask_y1:
            continue
        
        grid_mask_region = gt_mask[mask_y1:mask_y2, mask_x1:mask_x2]
        
        # Check for tumor (non-zero pixels)
        if not np.any(grid_mask_region > 0):
            continue
        
        # This grid has tumor! Generate thumbnails
        
        # 1. GT Mask Thumbnail (low resolution, e.g., 100x100 px for 10x10 patches)
        thumb_size = 100
        mask_thumbnail = cv.resize(grid_mask_region, (thumb_size, thumb_size),
                                  interpolation=cv.INTER_NEAREST)
        mask_thumb_path = mask_thumb_dir / f"{filename}_grid{grid_idx}.png"
        cv.imwrite(str(mask_thumb_path), mask_thumbnail)
        
        # 2. Image Data Thumbnail
        img_x1 = max(0, grid_x)
        img_y1 = max(0, grid_y)
        img_x2 = min(wsi_width, grid_x + grid_pixel_size)
        img_y2 = min(wsi_height, grid_y + grid_pixel_size)
        
        if img_x2 <= img_x1 or img_y2 <= img_y1:
            continue
        
        if use_openslide:
            # Extract region from WSI
            region_width = img_x2 - img_x1
            region_height = img_y2 - img_y1
            grid_img_region = wsi.read_region((img_x1, img_y1), 0,
                                             (region_width, region_height))
            grid_img_region = np.array(grid_img_region.convert('RGB'))
        else:
            grid_img_region = wsi_img[img_y1:img_y2, img_x1:img_x2]
        
        img_thumbnail = cv.resize(grid_img_region, (thumb_size, thumb_size),
                                 interpolation=cv.INTER_LINEAR)
        img_thumb_path = image_thumb_dir / f"{filename}_grid{grid_idx}.png"
        cv.imwrite(str(img_thumb_path), cv.cvtColor(img_thumbnail, cv.COLOR_RGB2BGR))

def train(cfg):
    
    print("\nInitializing training...")
    torch.backends.cudnn.benchmark = True  # Enable cudnn auto optimization
    
    num_workers = min(10, os.cpu_count())  # Optimize worker count based on CPU cores
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)
    
    # Load patch encoder for prototype feature extraction
    encoder_name = getattr(cfg.model, 'patch_encoder', 'medclip')
    print(f"\nInitializing patch encoder: {encoder_name}")
    encoder_config = get_encoder_config(encoder_name)
    print(f"  Description: {encoder_config['description']}")
    print(f"  Output dimension: {encoder_config['feature_dim']}")
    
    clip_model, feature_dim = EncoderFactory.create_encoder(
        encoder_name=encoder_name, 
        device=device
    )
    print(f"Patch encoder loaded successfully")
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    # ============ prepare data ============
    print("\nPreparing datasets...")
    
    # Check dataset type from config
    dataset_name = getattr(cfg.dataset, 'name', 'bcss')
    
    if dataset_name == 'custom_wsi':
        # Custom WSI dataset with coordinates
        print(f"Loading custom WSI dataset: {dataset_name}")
        train_dataset, val_dataset = get_custom_dataset(cfg, split="valid")
    else:
        # Default BCSS dataset
        print(f"Loading BCSS dataset: {dataset_name}")
        train_dataset, val_dataset = get_cls_dataset(cfg, split="valid")
    
    # Efficient data loading configuration
    train_loader = DataLoader(train_dataset,
                            batch_size=cfg.train.samples_per_gpu,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=True,
                            prefetch_factor=2,
                            persistent_workers=True)
    
    
    val_loader = DataLoader(val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True,
                          persistent_workers=True)

    # ============ prepare model ============
    num_classes = getattr(cfg.dataset, 'num_classes', 4)
    model = ClsNetwork(backbone=cfg.model.backbone.config,
                    stride=cfg.model.backbone.stride,
                    num_classes=num_classes,
                    n_ratio=cfg.model.n_ratio,
                    pretrained=cfg.train.pretrained,
                    l_fea_path=cfg.model.label_feature_path)
    
    # Mixed precision training setup
    iters_per_epoch = len(train_loader)
    cfg.train.max_iters = cfg.train.epoch * iters_per_epoch
    cfg.train.eval_iters = iters_per_epoch
    cfg.scheduler.warmup_iter = cfg.scheduler.warmup_iter * iters_per_epoch
    scaler = torch.cuda.amp.GradScaler()
    
    model.to(device)
    model.train()

    # Optimizer configuration (from paper Section 4.1)
    # AdamW with lr=5e-5, weight_decay=1e-3, polynomial decay
    optimizer = PolyWarmupAdamW(
        params=model.parameters(),
        lr=cfg.optimizer.learning_rate,  # 5e-5 from paper
        weight_decay=cfg.optimizer.weight_decay,  # 1e-3 from paper
        betas=cfg.optimizer.betas,  # (0.9, 0.999) default Adam betas
        warmup_iter=cfg.scheduler.warmup_iter,  # Warmup iterations
        max_iter=cfg.train.max_iters,  # Total training iterations
        warmup_ratio=cfg.scheduler.warmup_ratio,  # Initial LR ratio during warmup
        power=cfg.scheduler.power  # 1.0 = linear decay (polynomial scheduler)
    )

    # Loss functions and feature extractor setup
    loss_function = nn.BCEWithLogitsLoss().to(device)
    mask_adapter = MaskAdapter_DynamicThreshold(alpha=cfg.train.mask_adapter_alpha,)
    feature_extractor = FeatureExtractor(mask_adapter=mask_adapter)
    
    # InfoNCE temperature from config (default 1.0 from paper Section 4.1)
    temperature = getattr(cfg.train, 'temperature', 1.0)
    fg_loss_fn = InfoNCELossFG(temperature=temperature).to(device)
    bg_loss_fn = InfoNCELossBG(temperature=temperature).to(device)
    
    # CAM regularization weight from config (default 0.01 from paper Section 3.2)
    lambda_cam = getattr(cfg.train, 'lambda_cam', 0.01)
    
    best_fuse234_dice = 0.0

    print("\nStarting training...")
    train_loader_iter = iter(train_loader)
    
    for n_iter in range(cfg.train.max_iters):
        try:
            img_name, inputs, cls_labels, gt_label = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, gt_label = next(train_loader_iter)

        inputs = inputs.to(device).float()
        cls_labels = cls_labels.to(device).float()
        
        with torch.cuda.amp.autocast():
            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list = model(inputs)

            # Merge subclass predictions to parent class predictions
            cls1_merge, cls2_merge, cls3_merge, cls4_merge = merge_multiscale_predictions(
                [cls1, cls2, cls3, cls4], k_list, method=cfg.train.merge_train)

            # Generate binary masks for feature extraction
            subclass_labels = expand_parent_to_subclass_labels(cls_labels, k_list)
            cls4_expand=expand_parent_to_subclass_labels(cls4_merge, k_list)
            cls4_bir=(cls4>cls4_expand).float()*subclass_labels

            # Extract foreground and background features
            batch_info = feature_extractor.process_batch(inputs, cam4, cls4_bir, clip_model)
            fg_features, bg_features = batch_info['fg_features'], batch_info['bg_features']

            # Pair features with text embeddings
            set_info = pair_features(fg_features, bg_features, l_fea, cls4_bir)
            fg_features, bg_features, fg_pro, bg_pro = set_info['fg_features'], set_info['bg_features'], set_info['fg_text'], set_info['bg_text']
                
            # Compute contrastive losses
            fg_loss = fg_loss_fn(fg_features, fg_pro, bg_pro)
            bg_loss = bg_loss_fn(bg_features, fg_pro, bg_pro)
            
            # Multi-scale classification losses (from paper Section 3.2, Equation 3)
            # Different weights for each feature pyramid level
            weights = (cfg.train.scale1_weight, cfg.train.scale2_weight,
                      cfg.train.scale3_weight, cfg.train.scale4_weight)
            cls_loss = compute_multiscale_loss(
                [cls1_merge, cls2_merge, cls3_merge, cls4_merge], 
                cls_labels, loss_function, weights)
            
            # CAM regularization with lambda_cam weight (from paper Section 3.2)
            cam_reg = lambda_cam * torch.mean(cam4)
            
            # Total loss: classification + contrastive (λ_c) + CAM regularization
            loss = cls_loss + (fg_loss + bg_loss + cam_reg) * cfg.train.contrastive_weight

        # Gradient scaling for mixed precision
        optimizer.zero_grad(set_to_none=True)  # More efficient gradient clearing 5358
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (n_iter + 1) % 100 == 0:
            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']
            
            # Synchronize for accurate timing measurement
            torch.cuda.synchronize()
            
            cls_pred4 = (torch.sigmoid(cls4_merge) > 0.5).float()
            all_cls_acc4 = (cls_pred4 == cls_labels).all(dim=1).float().mean() * 100
            avg_cls_acc4 = ((cls_pred4 == cls_labels).float().mean(dim=0)).mean() * 100
            
            print(
                f"Iter: {n_iter + 1}/{cfg.train.max_iters}; "
                f"Elapsed: {delta}; ETA: {eta}; "
                f"LR: {cur_lr:.3e}; Loss: {loss.item():.4f}; "
                f"Acc4: {all_cls_acc4:.2f}/{avg_cls_acc4:.2f}"
            )
        # Regular validation and model saving
        if (n_iter + 1) % cfg.train.eval_iters == 0 or (n_iter + 1) == cfg.train.max_iters:
            val_all_acc4, val_avg_acc4, fuse234_score, val_cls_loss = validate(
                model=model,
                data_loader=val_loader,
                cfg=cfg,
                cls_loss_func=loss_function
            )   
            print("Validation results:")
            print(f"Val all acc4: {val_all_acc4:.6f}")
            print(f"Val avg acc4: {val_avg_acc4:.6f}")
            print(f"Fuse234 score: {fuse234_score}, mIOU: {fuse234_score[:-1].mean():.4f}")
            
            if fuse234_score[:-1].mean() > best_fuse234_dice:
                best_fuse234_dice = fuse234_score[:-1].mean()
                save_path = os.path.join(cfg.output_dirs.ckpt_dir, "best_cam.pth")
                
                # Save best model checkpoint
                torch.save(
                    {
                        "cfg": cfg,
                        "iter": n_iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    },
                    save_path,
                    _use_new_zipfile_serialization=True
                )
                print(f"\nSaved best model with mIOU: {best_fuse234_dice:.4f}")

    torch.cuda.empty_cache()
    end_time = datetime.datetime.now()
    total_training_time = end_time - start_time
    print(f'Total training time: {total_training_time}')

    
    print("\n" + "="*80)
    print("POST-TRAINING EVALUATION AND CAM GENERATION")
    print("="*80)
 
    print("\nPreparing test dataset...")

    # Load test dataset based on dataset type
    if dataset_name == 'custom_wsi':
        train_dataset, test_dataset = get_custom_dataset(cfg, split="test")
    else:
        train_dataset, test_dataset = get_cls_dataset(cfg, split="test", enable_rotation=False, p=0.0)
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    

    test_loader = DataLoader(test_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True)

    print("\n1. Testing on test dataset...")
    print("-" * 50)
    
    test_all_acc4, test_avg_acc4, fuse234_score, test_cls_loss = validate(
                model=model,
                data_loader=test_loader,
                cfg=cfg,
                cls_loss_func=loss_function)   

    
    print("Testing results:")
    print(f"Test all acc4: {test_all_acc4:.6f}")
    print(f"Test avg acc4: {test_avg_acc4:.6f}")
    print(f"Fuse234 score: {fuse234_score}, mIOU: {fuse234_score[:-1].mean():.4f}")
    
    print("\nPer-class IoU scores:")
    for i, score in enumerate(fuse234_score[:-1]):
        print(f"  Class {i}: {score:.6f}")
    if len(fuse234_score) > cfg.dataset.num_classes:
        print(f"  Background: {fuse234_score[-1]:.6f}")

    print("\n2. Extracting patch features with caching...")
    print("-" * 50)
    extract_and_cache_features(train_dataset, clip_model, cfg, encoder_name, device)
    
    print("\n3. Generating CAMs for complete training dataset...")
    print("-" * 50)
    
    train_cam_loader = DataLoader(train_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            persistent_workers=True)

    print(f"Generating CAMs for all {len(train_dataset)} training samples...")
    print(f"Output directory: {cfg.output_dirs.pred_dir}")

    best_model_path = os.path.join(cfg.output_dirs.ckpt_dir, "best_cam.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        # weights_only=False needed for PyTorch 2.6+ to load DictConfig
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        best_iter = checkpoint.get("iter", "unknown")
        print(f"✓ Best model loaded successfully! (Saved at iteration: {best_iter})")
    else:
        print("⚠ Warning: Best model checkpoint not found, using current model state")
        print(f"Expected path: {best_model_path}")
    
    generate_cam(model=model, data_loader=train_cam_loader, cfg=cfg)
    
    print("\nFiles generated:")
    print(f"  • Training CAM visualizations: {cfg.output_dirs.pred_dir}/*.png")
    print(f"  • Model checkpoint: {cfg.output_dirs.ckpt_dir}/best_cam.pth")
    print("="*80)
    
    
if __name__ == "__main__":
    cfg = OmegaConf.load(args.config)
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    # Create output directories with timestamp for checkpoints
    cfg.output_dirs.ckpt_dir = os.path.join(cfg.output_dirs.ckpt_dir, timestamp)
    
    os.makedirs(cfg.work_dir, exist_ok=True)
    os.makedirs(cfg.output_dirs.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.output_dirs.pred_dir, exist_ok=True)

    print('\nArgs: %s' % args)
    print('\nConfigs: %s' % cfg)

    set_seed(0)
    train(cfg=cfg)
