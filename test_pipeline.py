#!/usr/bin/env python3
"""
Diagnostic script to verify the full prototype extraction → MedCLIP → clustering pipeline.
"""

import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
import argparse
import numpy as np
from PIL import Image

def count_files(directory):
    """Count files in a directory recursively."""
    if not os.path.exists(directory):
        return 0
    return len(list(Path(directory).rglob("*"))) - 1  # -1 for the directory itself

def check_prototypes(cfg):
    """Check what prototypes were extracted."""
    print("\n=== PROTOTYPE EXTRACTION CHECK ===")
    proto_dir = Path(cfg.features.proto_image_dir)
    proto_coords_dir = proto_dir.parent / 'prototype_coordinates'
    
    class_order = list(getattr(cfg.dataset, 'class_order', ['benign', 'tumor']))
    
    prototypes_ready = True
    for class_name in class_order:
        class_dir = proto_dir / class_name
        coords_file = proto_coords_dir / f"{class_name}.npy"
        
        # Check coordinates file first
        coords_exist = coords_file.exists()
        
        # Count images
        if class_dir.exists():
            images = list(class_dir.glob("*.png"))
            print(f"\n{class_name.upper()}:")
            print(f"  ✓ Images extracted: {len(images)}")
            
            # Check image sizes
            if images:
                img = Image.open(images[0])
                print(f"  ✓ Image size: {img.size}")
                print(f"  ✓ Sample images: {[img.name for img in images[:3]]}")
        else:
            print(f"\n{class_name.upper()}:")
            if coords_exist:
                print(f"  ✓ Coordinates available but images not extracted yet")
            else:
                print(f"  ✗ No prototype directory or coordinates found")
            prototypes_ready = False
        
        # Check coordinates file
        if coords_exist:
            coords_data = np.load(coords_file, allow_pickle=True)
            if hasattr(coords_data, 'dtype') and coords_data.dtype.names:
                print(f"  ✓ Coordinates: {len(coords_data)} samples")
            else:
                print(f"  ✓ Coordinates shape: {coords_data.shape}")
    
    return prototypes_ready

def check_medclip_features(cfg):
    """Check if MedCLIP features were extracted."""
    print("\n=== MEDCLIP FEATURES CHECK ===")
    features_dir = Path(cfg.features.save_dir)
    medclip_pkl = features_dir / cfg.features.medclip_features_pkl
    label_feature_pkl = features_dir / cfg.features.label_feature_pkl
    
    medclip_ready = False
    clustering_ready = False
    
    if medclip_pkl.exists():
        print(f"✓ MedCLIP features file exists: {medclip_pkl.name}")
        # Try to load and check
        import pickle
        try:
            with open(medclip_pkl, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                print(f"  ✓ Classes: {list(data.keys())}")
                for key in list(data.keys())[:2]:
                    if isinstance(data[key], list):
                        print(f"    {key}: {len(data[key])} features")
                medclip_ready = True
        except Exception as e:
            print(f"  ⚠ Could not load: {e}")
    else:
        print(f"✗ MedCLIP features file NOT FOUND: {medclip_pkl}")
    
    if label_feature_pkl.exists():
        print(f"✓ Label features (clustered) file exists: {label_feature_pkl.name}")
        clustering_ready = True
    else:
        print(f"✗ Label features file NOT FOUND: {label_feature_pkl}")
    
    return medclip_ready, clustering_ready

def check_training_setup(cfg):
    """Check if training would work with current config."""
    print("\n=== TRAINING SETUP CHECK ===")
    
    # Check model config
    print(f"Model backbone: {cfg.model.backbone.config}")
    print(f"Number of classes: {cfg.dataset.num_classes}")
    
    # Check label feature path
    label_feature_path = cfg.model.label_feature_path
    print(f"Label feature path (from config): {label_feature_path}")
    
    # Try to resolve the path
    expanded_path = label_feature_path
    for key in ['features.save_dir', 'features.label_feature_pkl']:
        if '${' + key + '}' in expanded_path:
            parts = key.split('.')
            val = cfg
            for part in parts:
                val = getattr(val, part)
            expanded_path = expanded_path.replace('${' + key + '}', str(val))
    
    print(f"Resolved path: {expanded_path}")
    
    if os.path.exists(expanded_path):
        print(f"✓ Label features file exists")
    else:
        print(f"✗ Label features file NOT FOUND")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    
    print("=" * 80)
    print("PIPELINE DIAGNOSTIC CHECK")
    print("=" * 80)
    
    prototypes_ready = check_prototypes(cfg)
    medclip_ready, clustering_ready = check_medclip_features(cfg)
    check_training_setup(cfg)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    
    # Determine what steps need to be done
    steps_todo = []
    
    if not medclip_ready:
        steps_todo.append({
            'num': len(steps_todo) + 1,
            'name': 'Extract MedCLIP features',
            'cmd': 'python3 features/excract_medclip_proces.py --config work_dirs/custom_wsi_template.yaml'
        })
    
    if not clustering_ready:
        steps_todo.append({
            'num': len(steps_todo) + 1,
            'name': 'Run k-means clustering',
            'cmd': 'python3 features/k_mean_cos_per_class.py --config work_dirs/custom_wsi_template.yaml'
        })
    
    if clustering_ready:
        steps_todo.append({
            'num': len(steps_todo) + 1,
            'name': 'Train segmentation model',
            'cmd': 'python3 train_stage_1.py --config work_dirs/custom_wsi_template.yaml --gpu 0'
        })
    
    if steps_todo:
        for step in steps_todo:
            print(f"{step['num']}. {step['name']}:")
            print(f"   {step['cmd']}\n")
    else:
        print("⚠ Cannot proceed: Missing required files")
        print("Please extract prototypes first.")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
