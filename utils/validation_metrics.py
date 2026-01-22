"""
Comprehensive validation metrics for WSI classification and segmentation.

Organized by module:
- ClassifierNet: Slide-level classification metrics
- PrototypeBank: Prototype matching and attention metrics  
- Segmentation: Patch-level segmentation metrics (when GT available)
"""

import numpy as np
import torch
import torch.nn.functional as F
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    hamming_loss
)
from typing import Dict, List, Tuple, Optional
from .evaluate import ConfusionMatrixAllClass


class ValidationMetrics:
    """Compute comprehensive validation metrics organized by module."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all accumulators."""
        # Classification predictions
        self.slide_predictions = []  # (N,) predicted class indices
        self.slide_probabilities = []  # (N, num_classes) predicted probabilities
        self.slide_labels = []  # (N,) ground truth class indices
        
        # Prototype matching scores
        self.prototype_match_scores = []  # List of (n_patches, K) match scores per WSI
        self.prototype_attention_weights = []  # List of (n_patches,) attention weights
        
        # Segmentation (when GT available)
        self.patch_predictions = []  # List of patch predictions per WSI
        self.patch_labels = []  # List of patch GT labels per WSI (if available)
        self.has_segmentation_gt = False
        
        # Loss components
        self.classifier_losses = []
        self.prototype_losses = []
        self.total_losses = []
    
    def update_classification(
        self,
        slide_logits: torch.Tensor,
        slide_labels: torch.Tensor,
        classifier_loss: float
    ):
        """
        Update classification metrics from ClassifierNet module.
        
        Args:
            slide_logits: (batch_size, num_classes) logits from classifier
            slide_labels: (batch_size,) or (batch_size, num_classes) ground truth
            classifier_loss: Scalar classification loss value
        """
        # Convert to numpy
        probs = F.softmax(slide_logits, dim=1).cpu().numpy()  # (batch, num_classes)
        preds = slide_logits.argmax(dim=1).cpu().numpy()  # (batch,)
        
        # Handle one-hot encoded labels
        if slide_labels.dim() == 2:
            labels = slide_labels.argmax(dim=1).cpu().numpy()
        else:
            labels = slide_labels.cpu().numpy()
        
        self.slide_predictions.extend(preds)
        self.slide_probabilities.extend(probs)
        self.slide_labels.extend(labels)
        self.classifier_losses.append(classifier_loss)
    
    def update_prototype_matching(
        self,
        match_scores: torch.Tensor,
        attention_weights: torch.Tensor,
        prototype_loss: float
    ):
        """
        Update prototype matching metrics from PrototypeBank module.
        
        Args:
            match_scores: (batch, n_patches, K) cosine similarity scores
            attention_weights: (batch, n_patches) refined attention weights
            prototype_loss: Scalar contrastive loss value
        """
        # Store per-WSI scores (handle batch)
        for i in range(match_scores.shape[0]):
            self.prototype_match_scores.append(match_scores[i].cpu().numpy())
            self.prototype_attention_weights.append(attention_weights[i].cpu().numpy())
        
        self.prototype_losses.append(prototype_loss)
    
    def update_segmentation(
        self,
        patch_predictions: torch.Tensor,
        patch_labels: Optional[torch.Tensor] = None,
        slide_labels: Optional[torch.Tensor] = None,
    ):
        """
        Update patch-level segmentation metrics (when GT available).
        
        Args:
            patch_predictions: (batch, n_patches, num_classes) patch logits
            patch_labels: (batch, n_patches) ground truth patch labels (optional)
            slide_labels: (batch, num_classes) or (batch,) slide-level labels (optional)
        """
        # Store per-WSI predictions only when GT available to keep lengths consistent
        for i in range(patch_predictions.shape[0]):
            if patch_labels is None:
                continue
            if patch_labels[i] is None:
                continue
            # If slide label is benign (class 0), force patch GT to zeros to enforce constraint
            if slide_labels is not None:
                slide_label_i = slide_labels[i]
                slide_class = int(slide_label_i.argmax().item()) if slide_label_i.dim() > 0 else int(slide_label_i.item())
                if slide_class == 0:
                    patch_gt = torch.zeros_like(patch_labels[i])
                else:
                    patch_gt = patch_labels[i]
            else:
                patch_gt = patch_labels[i]

            patch_preds = patch_predictions[i].argmax(dim=-1).cpu().numpy()
            patch_gt = patch_gt.cpu().numpy()
            
            # Handle length mismatch: truncate to min length
            min_len = min(len(patch_preds), len(patch_gt))
            if len(patch_preds) != len(patch_gt):
                patch_preds = patch_preds[:min_len]
                patch_gt = patch_gt[:min_len]
            
            self.patch_predictions.append(patch_preds)
            self.patch_labels.append(patch_gt)
            self.has_segmentation_gt = True
    
    def compute_classification_metrics(self) -> Dict[str, float]:
        """
        Compute slide-level classification metrics from ClassifierNet.
        
        Returns:
            Dictionary with classification metrics organized by module.
        """
        metrics = {}
        
        if len(self.slide_predictions) == 0:
            return metrics
        
        # Convert to numpy arrays
        y_true = np.array(self.slide_labels)
        y_pred = np.array(self.slide_predictions)
        y_prob = np.array(self.slide_probabilities)
        
        # === OVERALL CLASSIFICATION METRICS ===
        metrics['classifier/loss'] = np.mean(self.classifier_losses)
        
        # Macro-averaged F1 (equal weight per class)
        metrics['classifier/f1_macro'] = f1_score(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # === PER-CLASS METRICS ===
        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            
            # Binary mask for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            # F1 score
            metrics[f'classifier/class_{class_idx}_{class_name}/f1'] = f1_score(
                y_true_binary, y_pred_binary, zero_division=0
            )
            
            # AUROC and AUPRC (if we have positive samples)
            if len(np.unique(y_true_binary)) > 1:
                metrics[f'classifier/class_{class_idx}_{class_name}/auroc'] = roc_auc_score(
                    y_true_binary, y_prob[:, class_idx]
                )
                metrics[f'classifier/class_{class_idx}_{class_name}/auprc'] = average_precision_score(
                    y_true_binary, y_prob[:, class_idx]
                )
        
        # === AUROC METRICS ===
        # Try One-vs-Rest AUROC if we have multiple classes with samples
        try:
            if self.num_classes == 2:
                # Binary classification
                metrics['classifier/auroc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multi-class OvR
                metrics['classifier/auroc_macro'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='macro'
                )
        except ValueError as e:
            # Not enough classes represented in validation set
            pass
        
        # === AUPRC METRICS ===
        # Macro-averaged AUPRC across all classes
        try:
            if self.num_classes == 2:
                # Binary classification
                metrics['classifier/auprc'] = average_precision_score(y_true, y_prob[:, 1])
            else:
                # Multi-class: compute per-class AUPRC and average
                auprc_per_class = []
                for class_idx in range(self.num_classes):
                    y_true_binary = (y_true == class_idx).astype(int)
                    if len(np.unique(y_true_binary)) > 1:
                        auprc_per_class.append(average_precision_score(y_true_binary, y_prob[:, class_idx]))
                
                if len(auprc_per_class) > 0:
                    metrics['classifier/auprc_macro'] = np.mean(auprc_per_class)
        except ValueError as e:
            # Not enough classes represented
            pass
        

        return metrics
    
    def compute_prototype_metrics(self) -> Dict[str, float]:
        """
        Compute prototype matching metrics from PrototypeBank module.
        
        Returns:
            Dictionary with prototype metrics.
        """
        metrics = {}
        
        if len(self.prototype_match_scores) == 0:
            return metrics
        
        # Aggregate statistics across all WSIs
        all_match_scores = np.concatenate(self.prototype_match_scores, axis=0)  # (total_patches, K)
        all_attention = np.concatenate(self.prototype_attention_weights, axis=0)  # (total_patches,)
        
        # === PROTOTYPE MATCHING STATISTICS ===
        metrics['prototype/loss'] = np.mean(self.prototype_losses)
        
        # Match score statistics
        metrics['prototype/match_score_mean'] = all_match_scores.mean()
        metrics['prototype/match_score_std'] = all_match_scores.std()
        metrics['prototype/match_score_max'] = all_match_scores.max()
        metrics['prototype/match_score_min'] = all_match_scores.min()
        
        # Best match per patch
        best_matches = all_match_scores.max(axis=1)  # (total_patches,)
        metrics['prototype/best_match_mean'] = best_matches.mean()
        metrics['prototype/best_match_std'] = best_matches.std()
        
        # Attention weight statistics
        metrics['prototype/attention_mean'] = all_attention.mean()
        metrics['prototype/attention_std'] = all_attention.std()
        metrics['prototype/attention_max'] = all_attention.max()
        metrics['prototype/attention_min'] = all_attention.min()
        
        # Attention sparsity (how concentrated are the weights?)
        attention_entropy = -np.sum(all_attention * np.log(all_attention + 1e-10))
        metrics['prototype/attention_entropy'] = attention_entropy
        
        # Prototype utilization (how many prototypes are being used?)
        top_prototype_indices = all_match_scores.argmax(axis=1)  # (total_patches,)
        unique_prototypes = len(np.unique(top_prototype_indices))
        total_prototypes = all_match_scores.shape[1]
        metrics['prototype/utilization_ratio'] = unique_prototypes / total_prototypes
        metrics['prototype/num_active_prototypes'] = unique_prototypes
        
        return metrics
    
    def compute_segmentation_metrics(self) -> Dict[str, float]:
        """
        Compute patch-level segmentation metrics (when GT available).
        
        Returns:
            Dictionary with segmentation metrics organized by module.
        """
        metrics = {}
        
        if not self.has_segmentation_gt or len(self.patch_labels) == 0:
            return metrics
        
        # Flatten all patch predictions and labels
        y_pred = np.concatenate(self.patch_predictions, axis=0)  # (total_patches,)
        y_true = np.concatenate(self.patch_labels, axis=0)  # (total_patches,)
        
        # === PER-CLASS SEGMENTATION METRICS ===
        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            
            # Binary mask for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            # IoU (Jaccard Index)
            intersection = np.logical_and(y_true_binary, y_pred_binary).sum()
            union = np.logical_or(y_true_binary, y_pred_binary).sum()
            iou = intersection / union if union > 0 else 0
            
            metrics[f'segmentation/class_{class_idx}_{class_name}/iou'] = iou
        
        # === MEAN IoU (excluding background if class 0) ===
        # Compute per-class IoU
        ious = []
        for class_idx in range(self.num_classes):
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            intersection = np.logical_and(y_true_binary, y_pred_binary).sum()
            union = np.logical_or(y_true_binary, y_pred_binary).sum()
            
            if union > 0:
                ious.append(intersection / union)
        
        # Mean IoU (excluding background if it's class 0)
        if len(ious) > 1:
            metrics['segmentation/mean_iou'] = np.mean(ious[1:])  # Exclude background
        elif len(ious) > 0:
            metrics['segmentation/mean_iou'] = np.mean(ious)
        
        return metrics
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics across all modules.
        
        Returns:
            Dictionary with all metrics organized by module.
        """
        all_metrics = {}
        
        # Classification metrics (from ClassifierNet)
        all_metrics.update(self.compute_classification_metrics())
        
        # Prototype metrics (from PrototypeBank)
        all_metrics.update(self.compute_prototype_metrics())
        
        # Segmentation metrics (from patch predictions, if GT available)
        all_metrics.update(self.compute_segmentation_metrics())
        
        # Overall loss
        if len(self.total_losses) > 0:
            all_metrics['total_loss'] = np.mean(self.total_losses)
        
        return all_metrics
