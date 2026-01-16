"""
Example script: WSI-Level Classification with Pseudo-Label Guided Prototype Selection

This script demonstrates the complete workflow for WSI classification using:
1. Attention scores (pseudo-labels) from a pretrained MIL model
2. High-confidence patch selection for prototype bank initialization
3. Prototype-guided attention for WSI-level classification
4. Multi-stage refinement (optional)

Prerequisites:
- Attention scores saved as .pt files (one per WSI)
- WSI images and patch coordinates
- Split CSV file

Usage:
    python examples/wsi_training_with_prototypes.py --config ./work_dirs/custom/config.yaml
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# Project imports
from datasets.wsi_dataset import CustomWSIPatchTrainingDatasetWithPseudoLabels
from utils.pseudo_labels import PseudoLabelLoader, PatchSelector
from utils.prototype_guided_attention import create_wsi_classifier_with_prototypes
from utils.prototype_refinement import MultiStageRefinementPipeline


class WSITrainingPipeline:
    """
    Full pipeline for WSI classification with pseudo-label guided prototypes.
    """
    
    def __init__(self, cfg, device='cuda'):
        self.cfg = cfg
        self.device = device
        self.best_accuracy = 0.0
        
        print("="*70)
        print("WSI CLASSIFICATION WITH PSEUDO-LABEL GUIDED PROTOTYPES")
        print("="*70)
    
    def load_data(self):
        """Load training dataset with pseudo-labels."""
        print("\nLoading training dataset...")
        
        self.train_dataset = CustomWSIPatchTrainingDatasetWithPseudoLabels(
            wsi_dir=self.cfg.dataset.wsi_dir,
            coordinates_dir=self.cfg.dataset.coordinates_dir,
            split_csv=self.cfg.dataset.split_csv,
            split='train',
            class_labels_dict=self._load_class_labels(),
            num_classes=self.cfg.dataset.num_classes,
            patch_size=self.cfg.dataset.patch_size,
            max_patches=self.cfg.dataset.max_patches,
            use_pseudo_labels=self.cfg.dataset.use_pseudo_labels,
            pseudo_label_dir=self.cfg.dataset.pseudo_label_dir,
            pseudo_label_binary_mode=self.cfg.dataset.pseudo_label_binary_mode,
            pseudo_label_selection_strategy=self.cfg.dataset.pseudo_label_selection_strategy,
            pseudo_label_confidence_threshold=self.cfg.dataset.pseudo_label_confidence_threshold,
            pseudo_label_min_patches=self.cfg.dataset.pseudo_label_min_patches,
            pseudo_label_analyze=self.cfg.dataset.pseudo_label_analyze,
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.samples_per_gpu,
            shuffle=True,
            num_workers=4,
        )
        
        print(f"✓ Loaded {len(self.train_dataset)} training samples")
    
    def initialize_prototypes(self):
        """Initialize prototype bank from high-confidence patches."""
        print("\nInitializing prototype bank...")
        
        # Collect high-confidence patches for each class
        class_patches = {class_id: [] for class_id in range(self.cfg.dataset.num_classes)}
        
        # Assume class labels are stored in dataset
        wsi_labels = {}
        for idx in range(len(self.train_dataset)):
            filename = self.train_dataset.filenames[idx]
            cls_label = self.train_dataset.class_labels[idx]
            # Get primary class (argmax for multi-hot or single value for binary)
            wsi_labels[filename] = int(np.argmax(cls_label))
        
        # Extract features from high-confidence patches
        # This is a placeholder - in practice, you'd extract actual features from encoder
        print("Collecting high-confidence patches by class...")
        
        for idx in range(len(self.train_dataset)):
            wsi_name = self.train_dataset.filenames[idx]
            high_conf_indices = self.train_dataset.get_high_confidence_patches(wsi_name)
            wsi_class = wsi_labels[wsi_name]
            
            if high_conf_indices is not None and len(high_conf_indices) > 0:
                # In practice, extract features from actual patches
                # For demo, use random features
                num_patches = len(high_conf_indices)
                dummy_features = torch.randn(num_patches, 512)
                class_patches[wsi_class].append(dummy_features)
        
        # Convert to tensors
        for class_id in range(self.cfg.dataset.num_classes):
            if len(class_patches[class_id]) > 0:
                class_patches[class_id] = torch.cat(class_patches[class_id], dim=0)
                print(f"Class {class_id}: {class_patches[class_id].shape[0]} high-confidence patches")
        
        # Initialize multi-stage pipeline
        self.classifier, self.prototype_bank = create_wsi_classifier_with_prototypes(
            feature_dim=512,
            num_classes=self.cfg.dataset.num_classes,
            max_prototypes_per_class=50,
            attention_type='cosine_sim',
        )
        
        self.pipeline = MultiStageRefinementPipeline(
            prototype_bank=self.prototype_bank,
            classifier=self.classifier,
            num_classes=self.cfg.dataset.num_classes,
            max_refinement_iterations=self.cfg.train.get('refinement_iterations', 3),
        )
        
        # Initialize prototypes
        self.pipeline.initialize_prototypes_from_patches(
            class_patches,
            n_clusters_per_class=self.cfg.train.get('n_clusters_per_class', 5)
        )
        
        self.classifier = self.classifier.to(self.device)
        print("✓ Prototype bank initialized")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.train.epoch}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Handle variable batch format (with/without pseudo-labels)
            if len(batch_data) == 5:  # With pseudo-labels
                filenames, images, labels, _, pseudo_scores = batch_data
            else:
                filenames, images, labels, _ = batch_data
                pseudo_scores = None
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits, info = self.classifier(images.unsqueeze(0) if images.dim() == 3 else images)
            
            # Compute loss (basic cross-entropy for demo)
            loss = F.cross_entropy(logits, labels.argmax(dim=1) if labels.dim() > 1 else labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                true_labels = labels.argmax(dim=1) if labels.dim() > 1 else labels
                correct += (preds == true_labels).sum().item()
                total += true_labels.size(0)
            
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        print(f"Epoch {epoch+1}/{self.cfg.train.epoch} - "
             f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy
    
    def train(self):
        """Train the WSI classifier."""
        print("\nStarting training...")
        
        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.cfg.optimizer.learning_rate,
        )
        
        for epoch in range(self.cfg.train.epoch):
            loss, accuracy = self.train_epoch(epoch)
            
            # Save checkpoint if best
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self._save_checkpoint(epoch)
    
    def _save_checkpoint(self, epoch):
        """Save model checkpoint."""
        ckpt_dir = Path(self.cfg.work_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        path = ckpt_dir / f"best_model_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.classifier.state_dict(),
            'prototype_bank_state_dict': self.prototype_bank.state_dict(),
            'accuracy': self.best_accuracy,
        }, path)
        
        print(f"✓ Saved checkpoint: {path}")
    
    def _load_class_labels(self):
        """Load class labels from CSV."""
        from utils.trainutils import load_class_labels_from_csv
        
        labels_csv = self.cfg.dataset.labels_csv
        num_classes = self.cfg.dataset.num_classes
        
        return load_class_labels_from_csv(labels_csv, num_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./work_dirs/custom_wsi_template.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    # Load configuration
    cfg = OmegaConf.load(args.config)
    
    # Create pipeline
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    pipeline = WSITrainingPipeline(cfg, device=device)
    
    # Run training
    try:
        pipeline.load_data()
        pipeline.initialize_prototypes()
        pipeline.train()
        print("\n✓ Training completed successfully!")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
