"""
Prototype-Guided Attention Mechanism for WSI Classification.

This module implements the MIL (Multiple Instance Learning) aggregation with
prototype-guided attention, replacing standard global pooling with similarity-based
attention weighting. This bridges PBIP's prototype matching with WSI-level classification.

Key concepts:
- Patches are weighted by their similarity to class-specific prototypes
- A slide-level representation is formed by aggregating patch features weighted by
  their prototype similarity scores
- This replaces standard MIL attention with explainable, prototype-grounded decisions

Architecture flow:
1. Extract patch features from WSI (e.g., using MedCLIP)
2. Load class prototypes from the Prototype Bank
3. Compute similarity between each patch and class prototypes
4. Use similarities as attention weights for patch aggregation
5. Classify slide based on aggregated features and similarity scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
from .wsi_common import (
    normalize_features,
    validate_class_id,
    aggregate_features_weighted,
    cosine_similarity_matrix,
)


class PrototypeBank(nn.Module):
    """
    Stores and manages class-specific prototypes learned from high-confidence patches.
    
    Structure:
    - prototypes[class_id]: List of prototype features for that class
    - prototype_info[class_id]: Metadata (counts, confidence scores, etc.)
    
    Args:
        num_classes: Number of classes
        feature_dim: Dimension of feature vectors
        max_prototypes_per_class: Maximum prototypes to store per class
    """
    
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        max_prototypes_per_class: int = 100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.max_prototypes_per_class = max_prototypes_per_class
        
        # Register buffers for each class's prototypes
        for class_id in range(num_classes):
            self.register_buffer(
                f'class_{class_id}_prototypes',
                torch.zeros(max_prototypes_per_class, feature_dim)
            )
            self.register_buffer(
                f'class_{class_id}_counts',
                torch.zeros(max_prototypes_per_class, dtype=torch.long)
            )
        
        self.is_initialized = False
    
    def add_prototypes(
        self,
        class_id: int,
        features: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Add prototypes for a class (typically from clustering high-confidence patches).
        
        Args:
            class_id: Target class ID
            features: Feature vectors, shape (num_features, feature_dim)
            confidences: Optional confidence scores for each feature
            
        Returns:
            Number of prototypes added
        """
        validate_class_id(class_id, self.num_classes)
        
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.feature_dim}, "
                           f"got {features.shape[1]}")
        
        # Normalize features using shared utility
        features_norm = normalize_features(features)
        
        # Store prototypes (up to max)
        num_to_add = min(len(features), self.max_prototypes_per_class)
        
        prototypes_buffer = getattr(self, f'class_{class_id}_prototypes')
        counts_buffer = getattr(self, f'class_{class_id}_counts')
        
        prototypes_buffer[:num_to_add] = features_norm[:num_to_add]
        counts_buffer[:num_to_add] = 1  # Count of patches in each prototype
        
        self.is_initialized = True
        return num_to_add
    
    def get_class_prototypes(self, class_id: int) -> torch.Tensor:
        """Get non-zero prototypes for a class."""
        prototypes_buffer = getattr(self, f'class_{class_id}_prototypes')
        counts_buffer = getattr(self, f'class_{class_id}_counts')
        
        # Return only prototypes with non-zero counts
        valid_mask = counts_buffer > 0
        return prototypes_buffer[valid_mask]
    
    def get_all_prototypes(self) -> Dict[int, torch.Tensor]:
        """Get all prototypes organized by class."""
        result = {}
        for class_id in range(self.num_classes):
            result[class_id] = self.get_class_prototypes(class_id)
        return result
    
    def save(self, path: str):
        """Save prototype bank to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state_dict = {
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'max_prototypes_per_class': self.max_prototypes_per_class,
            'state_dict': self.state_dict(),
        }
        torch.save(state_dict, path)
        print(f"Prototype bank saved to {path}")
    
    def load(self, path: str):
        """Load prototype bank from disk."""
        state_dict = torch.load(path)
        
        # Verify compatibility
        if state_dict['feature_dim'] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: {state_dict['feature_dim']} vs "
                           f"{self.feature_dim}")
        
        self.load_state_dict(state_dict['state_dict'])
        self.is_initialized = True
        print(f"Prototype bank loaded from {path}")


class PrototypeGuidedAttention(nn.Module):
    """
    Computes attention weights based on patch similarity to class prototypes.
    
    For each patch, computes cosine similarity to all class prototypes, then
    uses these similarities as the basis for attention weight computation.
    
    Args:
        prototype_bank: PrototypeBank instance
        num_classes: Number of classes
        attention_type: Type of attention aggregation
                       - 'cosine_sim': Use max similarity as attention
                       - 'softmax': Softmax over class similarities
                       - 'mean': Mean similarity to class prototypes
    """
    
    def __init__(
        self,
        prototype_bank: PrototypeBank,
        num_classes: int,
        attention_type: str = 'cosine_sim',
        temperature: float = 1.0,
    ):
        super().__init__()
        self.prototype_bank = prototype_bank
        self.num_classes = num_classes
        self.attention_type = attention_type
        self.temperature = temperature
        
        if attention_type not in ['cosine_sim', 'softmax', 'mean']:
            raise ValueError(f"Unknown attention type: {attention_type}")
    
    def forward(
        self,
        patch_features: torch.Tensor,
        class_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prototype-guided attention weights for patches.
        
        Args:
            patch_features: Patch features, shape (num_patches, feature_dim)
            class_id: If specified, compute attention only for this class;
                     otherwise compute for all classes
            
        Returns:
            attention_weights: Attention weights, shape (num_patches,) or (num_patches, num_classes)
            similarity_matrix: Similarity matrix, shape (num_patches, num_prototypes) or 
                              (num_patches, num_classes_prototypes)
        """
        # Normalize patch features using shared utility
        patch_features_norm = normalize_features(patch_features)
        
        if class_id is not None:
            # Single class: compute similarity to class prototypes
            class_prototypes = self.prototype_bank.get_class_prototypes(class_id)
            
            if len(class_prototypes) == 0:
                # No prototypes for this class, return uniform weights
                num_patches = patch_features.shape[0]
                return torch.ones(num_patches) / num_patches, None
            
            # Compute cosine similarities: (num_patches, num_prototypes)
            similarities = torch.mm(patch_features_norm, class_prototypes.t())
            
            # Compute attention weights
            attention_weights = self._compute_attention_weights(similarities)
            
            return attention_weights, similarities
        
        else:
            # Multi-class: compute attention for each class separately
            all_prototypes = self.prototype_bank.get_all_prototypes()
            
            attention_weights_list = []
            similarity_matrices = []
            
            for class_id in range(self.num_classes):
                class_prototypes = all_prototypes[class_id]
                
                if len(class_prototypes) == 0:
                    # No prototypes, use uniform weights
                    attention_weights_list.append(
                        torch.ones(patch_features.shape[0]) / patch_features.shape[0]
                    )
                    similarity_matrices.append(None)
                else:
                    similarities = torch.mm(patch_features_norm, class_prototypes.t())
                    weights = self._compute_attention_weights(similarities)
                    
                    attention_weights_list.append(weights)
                    similarity_matrices.append(similarities)
            
            # Stack attention weights: (num_patches, num_classes)
            attention_weights = torch.stack(attention_weights_list, dim=1)
            
            return attention_weights, similarity_matrices
    
    def _compute_attention_weights(self, similarities: torch.Tensor) -> torch.Tensor:
        """
        Convert similarity scores to attention weights.
        
        Args:
            similarities: Similarity matrix, shape (num_patches, num_prototypes)
            
        Returns:
            Attention weights, shape (num_patches,)
        """
        if self.attention_type == 'cosine_sim':
            # Use max similarity as confidence (patch similarity to best matching prototype)
            weights, _ = torch.max(similarities, dim=1)
            # Scale to [0, 1] range
            weights = (weights + 1) / 2  # Maps [-1, 1] to [0, 1]
        
        elif self.attention_type == 'softmax':
            # Softmax over prototypes for each patch
            weights = F.softmax(similarities / self.temperature, dim=1)
            # Aggregate across prototypes (mean)
            weights = weights.mean(dim=1)
        
        elif self.attention_type == 'mean':
            # Mean similarity to all prototypes
            weights = similarities.mean(dim=1)
            # Scale to [0, 1] range
            weights = (weights + 1) / 2
        
        # Normalize to sum to 1 for aggregation
        weights = weights / (weights.sum() + 1e-8)
        
        return weights
    
    def aggregate_patches(
        self,
        patch_features: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate patch features using attention weights.
        
        Args:
            patch_features: Patch features, shape (num_patches, feature_dim)
            attention_weights: Attention weights, shape (num_patches,) or (num_patches, num_classes)
            
        Returns:
            Aggregated features, shape (feature_dim,) or (num_classes, feature_dim)
        """
        if attention_weights.dim() == 1:
            # Single-class aggregation
            # Expand weights: (num_patches,) -> (num_patches, 1)
            weights_expanded = attention_weights.unsqueeze(1)
            aggregated = (patch_features * weights_expanded).sum(dim=0)
        else:
            # Multi-class aggregation
            # weights shape: (num_patches, num_classes)
            # Expand for broadcasting: (num_patches, num_classes, 1)
            weights_expanded = attention_weights.unsqueeze(2)
            # Expand features: (num_patches, feature_dim) -> (num_patches, 1, feature_dim)
            patch_features_expanded = patch_features.unsqueeze(1)
            # Weighted sum: (num_patches, num_classes, feature_dim) -> (num_classes, feature_dim)
            aggregated = (patch_features_expanded * weights_expanded).sum(dim=0)
        
        return aggregated


class WSIClassifierWithPrototypes(nn.Module):
    """
    WSI-level classifier that uses prototype-guided attention for aggregation.
    
    Pipeline:
    1. Takes patch features as input (extracted from WSI patches)
    2. Computes attention weights using similarity to class prototypes
    3. Aggregates patch features using attention weights
    4. Classifies the aggregated slide representation
    
    Args:
        feature_dim: Dimension of patch features (e.g., 512 for MedCLIP)
        num_classes: Number of classes
        prototype_bank: PrototypeBank instance
        attention_type: Type of prototype-guided attention
        classifier_hidden_dim: Hidden dimension for classification head
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        prototype_bank: PrototypeBank,
        attention_type: str = 'cosine_sim',
        classifier_hidden_dim: int = 256,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.prototype_bank = prototype_bank
        
        # Prototype-guided attention module
        self.attention = PrototypeGuidedAttention(
            prototype_bank=prototype_bank,
            num_classes=num_classes,
            attention_type=attention_type,
        )
        
        # Classification head (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(classifier_hidden_dim, num_classes),
        )
    
    def forward(
        self,
        patch_features: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Classify a WSI based on its patches and prototype similarity.
        
        Args:
            patch_features: Patch features, shape (num_patches, feature_dim) or
                           (batch_size, num_patches, feature_dim)
            return_attention: If True, return attention maps for interpretability
            
        Returns:
            logits: Classification logits, shape (num_classes,) or (batch_size, num_classes)
            info: Dictionary with auxiliary information (attention weights, similarities, etc.)
        """
        info = {}
        
        # Handle batched input
        is_batched = patch_features.dim() == 3
        if is_batched:
            batch_size = patch_features.shape[0]
            num_patches = patch_features.shape[1]
            patch_features = patch_features.reshape(-1, self.feature_dim)
        else:
            batch_size = 1
            num_patches = patch_features.shape[0]
        
        # Compute prototype-guided attention
        attention_weights, similarities = self.attention(patch_features)
        info['attention_weights'] = attention_weights
        info['similarities'] = similarities
        
        # Aggregate patches using attention
        aggregated_features = self.attention.aggregate_patches(
            patch_features,
            attention_weights
        )
        
        if is_batched:
            # Reshape back to batch format
            aggregated_features = aggregated_features.view(batch_size, self.feature_dim)
        
        info['aggregated_features'] = aggregated_features
        
        # Classify
        logits = self.classifier(aggregated_features)
        
        return logits, info
    
    def get_attention_for_class(
        self,
        patch_features: torch.Tensor,
        class_id: int,
    ) -> torch.Tensor:
        """Get attention weights for patches relative to a specific class."""
        attention_weights, _ = self.attention(patch_features, class_id=class_id)
        return attention_weights


def create_wsi_classifier_with_prototypes(
    feature_dim: int,
    num_classes: int,
    max_prototypes_per_class: int = 100,
    attention_type: str = 'cosine_sim',
    classifier_hidden_dim: int = 256,
) -> Tuple[WSIClassifierWithPrototypes, PrototypeBank]:
    """
    Convenience function to create WSI classifier with prototype bank.
    
    Returns:
        (classifier, prototype_bank) tuple
    """
    prototype_bank = PrototypeBank(
        num_classes=num_classes,
        feature_dim=feature_dim,
        max_prototypes_per_class=max_prototypes_per_class,
    )
    
    classifier = WSIClassifierWithPrototypes(
        feature_dim=feature_dim,
        num_classes=num_classes,
        prototype_bank=prototype_bank,
        attention_type=attention_type,
        classifier_hidden_dim=classifier_hidden_dim,
    )
    
    return classifier, prototype_bank
