"""
Common utilities for WSI-level classification with prototypes.

This module contains shared functionality across pseudo_labels, prototype_guided_attention,
and prototype_refinement modules to eliminate code duplication.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.cluster import KMeans


def normalize_features(features: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    L2-normalize feature vectors.
    
    Args:
        features: Feature tensor
        dim: Dimension along which to normalize
        
    Returns:
        Normalized features
    """
    return F.normalize(features, p=2, dim=dim)


def validate_class_id(class_id: int, num_classes: int) -> None:
    """
    Validate that class_id is within valid range.
    
    Args:
        class_id: Class ID to validate
        num_classes: Total number of classes
        
    Raises:
        ValueError: If class_id is out of range
    """
    if class_id < 0 or class_id >= num_classes:
        raise ValueError(
            f"Invalid class_id {class_id}. Must be in range [0, {num_classes-1}]"
        )


def compute_entropy(probs: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Compute entropy of probability distributions.
    
    Args:
        probs: Probability tensor, shape (..., num_classes)
        dim: Dimension along which to compute entropy
        
    Returns:
        Entropy values
    """
    return -(probs * torch.log(probs + 1e-8)).sum(dim=dim)


def compute_margin(scores: torch.Tensor) -> torch.Tensor:
    """
    Compute margin between top-2 class scores.
    
    Args:
        scores: Score tensor, shape (num_samples, num_classes)
        
    Returns:
        Margin values, shape (num_samples,)
    """
    if scores.dim() == 1:
        # Binary case
        return torch.abs(scores - 0.5) * 2
    
    # Multi-class: margin between top-2 scores
    top2_scores = torch.topk(scores, k=2, dim=1)[0]
    return top2_scores[:, 0] - top2_scores[:, 1]


def cluster_features_kmeans(
    features: torch.Tensor,
    n_clusters: int,
    normalize: bool = True,
    random_state: int = 42,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Cluster features using k-means and return cluster centers.
    
    Args:
        features: Feature tensor, shape (num_samples, feature_dim)
        n_clusters: Number of clusters
        normalize: Whether to L2-normalize features before clustering
        random_state: Random seed for reproducibility
        
    Returns:
        (cluster_centers, labels) tuple where:
            - cluster_centers: Tensor of shape (n_clusters, feature_dim)
            - labels: numpy array of cluster assignments
    """
    # Convert to numpy
    if normalize:
        features_np = normalize_features(features).cpu().numpy()
    else:
        features_np = features.cpu().numpy()
    
    # Adjust n_clusters if fewer samples
    n_clusters = min(n_clusters, features.shape[0])
    
    if n_clusters == 0:
        raise ValueError("Cannot cluster 0 samples")
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(features_np)
    
    # Convert centers back to tensor
    centers = torch.from_numpy(kmeans.cluster_centers_).float()
    
    return centers, labels


def binary_to_multiclass_scores(
    scores: torch.Tensor,
    num_classes: int = 2
) -> torch.Tensor:
    """
    Convert binary scores to multi-class format.
    
    Args:
        scores: Binary scores, shape (num_samples,)
        num_classes: Number of classes (default: 2)
        
    Returns:
        Multi-class scores, shape (num_samples, num_classes)
    """
    if scores.dim() != 1:
        raise ValueError("Input must be 1D tensor for binary scores")
    
    if num_classes == 2:
        # Binary: stack [1-score, score]
        prob_class0 = 1 - scores
        prob_class1 = scores
        return torch.stack([prob_class0, prob_class1], dim=1)
    else:
        raise ValueError(f"Cannot convert binary to {num_classes} classes")


def compute_confidence_from_scores(
    scores: torch.Tensor,
    method: str = 'max'
) -> torch.Tensor:
    """
    Compute confidence values from raw scores.
    
    Args:
        scores: Score tensor, shape (num_samples,) or (num_samples, num_classes)
        method: Confidence computation method
            - 'max': Maximum score
            - 'entropy': 1 - normalized_entropy
            - 'margin': Margin between top-2 classes
            
    Returns:
        Confidence values, shape (num_samples,)
    """
    if scores.dim() == 1:
        # Binary scores - return as-is
        return scores
    
    if method == 'max':
        return scores.max(dim=1)[0]
    
    elif method == 'entropy':
        probs = torch.softmax(scores, dim=1)
        entropy = compute_entropy(probs, dim=1)
        num_classes = scores.shape[1]
        # Normalize to [0, 1] where 1 = most confident
        return 1 - (entropy / np.log(num_classes))
    
    elif method == 'margin':
        margin = compute_margin(scores)
        # Normalize to [0, 1]
        return torch.clamp(margin, 0, 1)
    
    else:
        raise ValueError(f"Unknown confidence method: {method}")


def get_statistics_dict(tensor: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics for a tensor.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Dictionary with mean, std, min, max, median
    """
    return {
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'median': tensor.median().item(),
    }


def aggregate_features_weighted(
    features: torch.Tensor,
    weights: torch.Tensor,
    normalize_weights: bool = True
) -> torch.Tensor:
    """
    Aggregate features using weights.
    
    Args:
        features: Feature tensor, shape (num_samples, feature_dim)
        weights: Weight tensor, shape (num_samples,) or (num_samples, num_classes)
        normalize_weights: Whether to normalize weights to sum to 1
        
    Returns:
        Aggregated features, shape (feature_dim,) or (num_classes, feature_dim)
    """
    if normalize_weights:
        if weights.dim() == 1:
            weights = weights / (weights.sum() + 1e-8)
        else:
            weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)
    
    if weights.dim() == 1:
        # Single aggregation
        weights_expanded = weights.unsqueeze(1)
        return (features * weights_expanded).sum(dim=0)
    else:
        # Multi-class aggregation
        weights_expanded = weights.unsqueeze(2)
        features_expanded = features.unsqueeze(1)
        return (features_expanded * weights_expanded).sum(dim=0)


def cosine_similarity_matrix(
    features1: torch.Tensor,
    features2: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute cosine similarity matrix between two feature sets.
    
    Args:
        features1: First feature set, shape (N, D)
        features2: Second feature set, shape (M, D)
        normalize: Whether to L2-normalize features
        
    Returns:
        Similarity matrix, shape (N, M)
    """
    if normalize:
        features1 = normalize_features(features1, dim=1)
        features2 = normalize_features(features2, dim=1)
    
    return torch.mm(features1, features2.t())
