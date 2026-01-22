"""
Loss functions for patch refinement with prototype matching.

Simplified approach for sparse tumor scenarios:
1. Slide-level classification loss (directly supervises with GT slide label)
2. Prototype contrastive loss (aligns features with prototype bank)

No bag-level aggregation - just direct supervision at slide level.
Uses mean pooling of patch features to create slide representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class SlideLevelClassificationLoss(nn.Module):
    """
    Direct slide-level supervision using ground-truth slide labels.
    
    For sparse tumor scenarios, directly supervise the slide-level prediction
    (via mean pooling of patch features) instead of trying to aggregate
    noisy individual patch predictions.
    
    Flow:
    1. Mean pool patch features → slide_feature (batch_size, feature_dim)
    2. Pass through classifier → slide_logits (batch_size, num_classes)
    3. Compare with GT slide label using CrossEntropyLoss
    
    Args:
        num_classes: Number of classes
    """
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        slide_logits: torch.Tensor,
        slide_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute slide-level classification loss.
        
        Args:
            slide_logits: Slide-level logits from classifier, shape (batch_size, num_classes)
            slide_labels: Ground-truth slide labels, shape (batch_size, num_classes) one-hot
                         or (batch_size,) class indices
        
        Returns:
            loss: Scalar loss
            info: Dictionary with loss info
        """
        info = {}
        
        # Convert labels to indices if needed
        if slide_labels.dim() == 2:
            slide_label_idx = slide_labels.argmax(dim=1)
        else:
            slide_label_idx = slide_labels.long()
        
        loss = self.criterion(slide_logits, slide_label_idx)
        info['slide_ce_loss'] = loss.item()
        
        return loss, info


class PrototypeContrastiveLoss(nn.Module):
    """
    Contrastive loss to align patches with prototypes using InfoNCE.
    
    For each patch:
    - Pull toward prototypes from same predicted class (positive keys)
    - Push away from prototypes from different classes (negative keys)
    
    Uses InfoNCE (NT-Xent) loss: -log(exp(sim(patch, pos_proto)) / sum(exp(sim(patch, all_protos))))
    
    Args:
        prototype_bank: PrototypeBank instance
        num_classes: Number of classes
        temperature: Temperature for softmax in InfoNCE
    """
    
    def __init__(
        self,
        prototype_bank,
        num_classes: int = 2,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.prototype_bank = prototype_bank
        self.num_classes = num_classes
        self.temperature = temperature
    
    def forward(
        self,
        patch_features: torch.Tensor,
        patch_pred_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute InfoNCE-style contrastive loss.
        
        Args:
            patch_features: Patch features, shape (num_patches, feature_dim)
            patch_pred_labels: Predicted patch class labels, shape (num_patches,)
        
        Returns:
            loss: Scalar loss
            info: Dictionary with loss info
        """
        info = {}
        device = patch_features.device
        
        # Normalize features for cosine similarity
        patch_features_norm = F.normalize(patch_features, p=2, dim=1)
        
        # Get all prototypes from bank
        all_prototypes = self.prototype_bank.get_all_prototypes()
        
        if not all_prototypes or len(all_prototypes) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), info
        
        # Collect all prototypes and their class labels
        all_proto_list = []
        all_proto_labels_list = []
        for class_id in range(self.num_classes):
            protos = all_prototypes.get(class_id, None)
            if protos is not None and len(protos) > 0:
                all_proto_list.append(protos)
                all_proto_labels_list.extend([class_id] * len(protos))
        
        if len(all_proto_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), info
        
        all_proto_features = torch.cat(all_proto_list, dim=0)  # (num_protos, D)
        all_proto_features_norm = F.normalize(all_proto_features, p=2, dim=1)
        all_proto_labels = torch.tensor(all_proto_labels_list, device=device, dtype=torch.long)
        
        # Compute logits: cosine similarities scaled by temperature
        logits = torch.mm(patch_features_norm, all_proto_features_norm.t()) / self.temperature
        # (num_patches, num_protos)
        
        # For each patch, positive keys are prototypes from same predicted class
        losses = []
        num_patches = patch_features.shape[0]
        
        for i in range(num_patches):
            patch_class = patch_pred_labels[i].item() if isinstance(patch_pred_labels[i], torch.Tensor) else int(patch_pred_labels[i])
            patch_logits = logits[i]  # (num_protos,)
            
            # Positive mask: prototypes from same class as this patch's prediction
            pos_mask = (all_proto_labels == patch_class)
            
            if pos_mask.sum() == 0:
                # No prototypes for this class, skip
                continue
            
            # InfoNCE loss: -log(mean(exp(pos_logits)) / sum(exp(all_logits)))
            # Numerically stable: -mean(pos) + logsumexp(all)
            pos_logits = patch_logits[pos_mask]
            
            # Use logsumexp for numerical stability
            loss_i = -pos_logits.mean() + torch.logsumexp(patch_logits, dim=0)
            losses.append(loss_i)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), info
        
        contrastive_loss = torch.stack(losses).mean()
        info['contrastive_loss'] = contrastive_loss.item()
        
        return contrastive_loss, info


class CombinedRefinementLoss(nn.Module):
    """
    Combined loss for patch classification with prototype matching.
    
    Matches PBIP logic but adapted for patch-level (not pixel-level) classification:
    
    PBIP: For each PIXEL, match against class prototypes using multi-scale context
    User: For each PATCH, match against class prototypes using patch + global context
    
    Supervision: weak (slide-level labels applied to all patches, like pixels in PBIP)
    
    Args:
        prototype_bank: PrototypeBank instance
        classifier_head: nn.Linear(feature_dim, num_classes) for slide-level predictions
        num_classes: Number of classes
        contrastive_weight: Weight for contrastive loss
        temperature: Temperature for InfoNCE contrastive loss
    """
    
    def __init__(
        self,
        prototype_bank,
        classifier_head,
        num_classes: int = 2,
        contrastive_weight: float = 0.5,
        temperature: float = 0.07,
        combined_feature_dim: int = None,  # NEW: dimension of combined features (patch + global)
        feature_dim: int = None,  # NEW: dimension expected by classifier
    ):
        super().__init__()
        self.num_classes = num_classes
        self.contrastive_weight = contrastive_weight
        self.classifier = classifier_head
        
        # Feature dimension for classifier input
        self.feature_dim = feature_dim if feature_dim else combined_feature_dim
        
        # If combined features are larger than classifier expects, add projection
        if combined_feature_dim and feature_dim and combined_feature_dim != feature_dim:
            self.feature_projection = nn.Linear(combined_feature_dim, feature_dim)
            print(f"  Added feature projection: {combined_feature_dim} -> {feature_dim}")
        else:
            self.feature_projection = None
        
        # Slide-level classification loss (supervises aggregated patch predictions)
        self.slide_ce = nn.CrossEntropyLoss()
        
        # Patch-level classification loss (weak supervision: all patches get slide label)
        self.patch_ce = nn.CrossEntropyLoss()
        
        # Prototype contrastive loss (InfoNCE: patches pulled toward matching prototypes)
        self.contrastive_loss = PrototypeContrastiveLoss(
            prototype_bank=prototype_bank,
            num_classes=num_classes,
            temperature=temperature,
        )
    
    def forward(
        self,
        patch_features_combined: torch.Tensor,
        slide_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss following PBIP logic (adapted for patches).
        
        Args:
            patch_features_combined: Patch features (possibly concatenated with global),
                                    shape (batch_size, num_patches, feature_dim)
                                    This is the equivalent of PBIP's hierarchical features
            slide_labels: Ground-truth slide labels, shape (batch_size, num_classes) one-hot
                         or (batch_size,) class indices
        
        Returns:
            total_loss: Combined loss
            info: Dictionary with loss components for logging
            
        Loss components (matching PBIP):
        1. Patch-level CE: Supervises individual patch predictions with weak slide label
           (like PBIP supervises pixels with image label)
        2. Slide-level CE: Aggregated patch predictions must match GT slide label
           (like PBIP fuses pixel predictions to slide level)
        3. Prototype contrastive: InfoNCE loss pulling patches to prototypes
           (like PBIP's ℒ_SIM for foreground/background separation)
        """
        info = {}
        
        # Handle single slide
        if patch_features_combined.dim() == 2:
            patch_features_combined = patch_features_combined.unsqueeze(0)
            if slide_labels.dim() == 1:
                slide_labels = slide_labels.unsqueeze(0)
        
        batch_size, num_patches, combined_dim = patch_features_combined.shape
        
        # Convert labels to indices if needed
        if slide_labels.dim() == 2:
            slide_label_idx = slide_labels.argmax(dim=1)  # (batch_size,)
        else:
            slide_label_idx = slide_labels.long()  # (batch_size,)
        
        # ===== 1. PATCH-LEVEL CROSS-ENTROPY (weak supervision) =====
        # Like PBIP: each pixel/patch inherits slide label as supervision
        patch_features_flat = patch_features_combined.view(-1, combined_dim)  # (batch*num_patches, combined_dim)
        
        # Project to classifier input dimension if needed
        if self.feature_projection is not None:
            patch_features_projected = self.feature_projection(patch_features_flat)  # (batch*num_patches, feature_dim)
        else:
            patch_features_projected = patch_features_flat
        
        patch_logits = self.classifier(patch_features_projected)  # (batch*num_patches, num_classes)
        
        # Expand slide labels to all patches (weak supervision)
        patch_labels_expanded = slide_label_idx.unsqueeze(1).expand(-1, num_patches)  # (batch, num_patches)
        patch_labels_flat = patch_labels_expanded.reshape(-1)  # (batch*num_patches,)
        
        patch_ce_loss = self.patch_ce(patch_logits, patch_labels_flat)
        info['patch_ce_loss'] = patch_ce_loss.item()
        
        # ===== 2. SLIDE-LEVEL AGGREGATION CHECK =====
        # Aggregate patch logits to slide level via mean pooling
        patch_logits_reshaped = patch_logits.view(batch_size, num_patches, self.num_classes)  # (batch, num_patches, num_classes)
        slide_logits = patch_logits_reshaped.mean(dim=1)  # (batch, num_classes) - mean aggregate
        
        slide_ce_loss = self.slide_ce(slide_logits, slide_label_idx)
        info['slide_ce_loss'] = slide_ce_loss.item()
        
        # ===== 3. PROTOTYPE CONTRASTIVE LOSS (InfoNCE) =====
        # Use predicted patch labels from classifier
        patch_pred_labels = patch_logits.argmax(dim=1)  # (batch*num_patches,)
        
        # Use projected features for contrastive loss (same dimension as prototypes)
        contrastive_loss, contrastive_info = self.contrastive_loss(
            patch_features_projected, patch_pred_labels
        )
        info.update(contrastive_info)
        
        # ===== COMBINE LOSSES =====
        # Total: patch_ce + slide_ce + contrastive_weight * contrastive_loss
        # (like PBIP: ℒ_CLS at multiple scales + ℒ_SIM with prototypes)
        total_loss = patch_ce_loss + slide_ce_loss + self.contrastive_weight * contrastive_loss
        info['total_loss'] = total_loss.item()
        
        return total_loss, info
