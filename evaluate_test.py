import argparse
import torch
import yaml
import os
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from model.model import ClsNetwork
from datasets.wsi_dataset import PseudoPatchLabelDataset
from train_patch_refinement import get_transform, merge_subclass_cams_to_parent
from utils.pyutils import AverageMeter

def evaluate(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Path setup
    work_dir = config_dict.get('work_dir', './work_dirs')
    exp_name = config_dict.get('experiment_name', 'default_exp')
    exp_dir = os.path.join(work_dir, exp_name)
    
    proto_path = os.path.join(exp_dir, 'features/prototypes.pkl')
    model_path = os.path.join(exp_dir, 'final_model.pth')

    if not os.path.exists(proto_path):
        print(f"Prototypes not loading from {proto_path}")
        return

    # Dataset
    transform = get_transform()
    test_dataset = PseudoPatchLabelDataset(config_dict['data_csv_path'], config=config_dict, split='test', 
                                          transform=transform,
                                          binary_mode=config_dict.get('binary_mode', True))
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Needs to match train set num classes
    cls_num_classes = len(config_dict.get('class_names', ['Neg', 'Pos']))

    # Load Model structure (reuse train logic or default)
    # Ideally save config/args with model. 
    # We will assume config matches training.
    
    cfg = OmegaConf.create({
        'train': {'pretrained': False},
        'dataset': {'cls_num_classes': cls_num_classes},
        'model': {
            'backbone': {'config': 'resnet50', 'stride': 1},
            'n_ratio': 0.1,
            'label_feature_path': proto_path
        }
    })
    
    model = ClsNetwork(backbone=cfg.model.backbone.config,
                       stride=cfg.model.backbone.stride,
                       cls_num_classes=cfg.dataset.cls_num_classes,
                       n_ratio=cfg.model.n_ratio,
                       pretrained=cfg.train.pretrained,
                       l_fea_path=cfg.model.label_feature_path)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}")
        return

    model.to(device)
    model.eval()
    
    dice_meter = AverageMeter()
    
    print("Starting evaluation...")
    with torch.no_grad():
        for imgs, targets, mask in tqdm(test_loader):
            # mask expected: [B, H, W] or [B, 1, H, W]
            if mask is None or mask.sum() == 0:
                continue
                
            imgs = imgs.to(device).float()
            
            results = model(imgs)
            cam_out = results[1] # [B, Subclasses, H, W]
            k_list = results[-1]
            
            # Merge
            cam_parent = merge_subclass_cams_to_parent(cam_out, k_list)
            
            # Assuming Binary Positive (Class 1) for Dice
            # If Multiclass: Need to know WHICH class to Dice against?
            # User says "DCI for all those for which we have segmentations".
            # Usually mask is binary for "Tumor" (Class 1).
            
            pred_mask = (cam_parent[:, 1] > 0.5).cpu().numpy()
            
            mask_np = mask.numpy()
            if len(mask_np.shape) == 4: mask_np = mask_np.squeeze(1)
            
            inter = (pred_mask * mask_np).sum()
            union = pred_mask.sum() + mask_np.sum()
            dice = 2*inter / (union + 1e-8)
            
            dice_meter.update(dice, imgs.size(0))
            
    print(f"Test Set DICE: {dice_meter.avg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.config)
