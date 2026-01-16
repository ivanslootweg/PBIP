"""
Prototype Bank Refinement for WSI Classification.

This module implements the iterative "Expectation-Maximization" style loop for
refining prototypes based on correctly classified WSIs. Over iterations, the
prototype bank learns more accurate visual definitions of each class.

Algorithm:
Stage 1 (Expectation): Use current prototypes to classify all WSIs
Stage 2 (Maximization): Collect high-confidence patches from correctly classified WSIs
                        and update prototypes via clustering
Repeat: This refines the visual definition until convergence

This bridges the two-stage PBIP approach with WSI-level MIL paradigm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .wsi_common import (
    normalize_features,
    cluster_features_kmeans,
    compute_confidence_from_scores,
)


class PrototypeBankRefinement:
    """
    Manages iterative refinement of prototypes based on WSI classification results.
    
    Args:
        prototype_bank: PrototypeBank instance to refine
        num_classes: Number of classes
        feature_dim: Dimension of feature vectors
        refinement_strategy: Strategy for selecting patches for prototype update
                           - 'high_confidence': Select patches with high prediction confidence
                           - 'margin': Select patches with large margin between top-2 classes
                           - 'entropy': Select patches with low entropy
        n_clusters_per_class: Number of clusters (prototypes) per class
    """
    
    def __init__(
        self,
        prototype_bank,
        num_classes: int,
        feature_dim: int,
        refinement_strategy: str = 'high_confidence',
        n_clusters_per_class: int = 5,
    ):
        self.prototype_bank = prototype_bank
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.refinement_strategy = refinement_strategy
        self.n_clusters_per_class = n_clusters_per_class
        
        # History tracking
        self.refinement_history = []
    
    def update_prototypes_from_wsi_features(
        self,
        wsi_features_dict: Dict[str, Tuple[torch.Tensor, int, float]],
        high_confidence_threshold: float = 0.8,
        min_patches_per_class: int = 10,
    ) -> Dict:
        """
        Update prototypes from a collection of WSI features and their predictions.
        
        Args:
            wsi_features_dict: Dictionary mapping wsi_name -> (features, pred_class, confidence)
                              where features shape is (num_patches, feature_dim)
            high_confidence_threshold: Confidence threshold for patch selection
            min_patches_per_class: Minimum patches needed to update class prototypes
            
        Returns:
            Statistics dictionary with refinement information
        """
        stats = {
            'total_wsis': len(wsi_features_dict),
            'correctly_classified': 0,
            'patches_selected_per_class': {},
            'prototypes_updated_per_class': {},
        }
        
        # Collect high-confidence patches for each class
        class_patches = {class_id: [] for class_id in range(self.num_classes)}
        
        for wsi_name, (features, pred_class, confidence) in wsi_features_dict.items():
            # Use only patches from high-confidence predictions
            if confidence >= high_confidence_threshold:
                stats['correctly_classified'] += 1
                
                # Select patches for prototype update
                selected_patch_indices = self._select_patches_for_refinement(
                    features,
                    confidence,
                )
                
                if len(selected_patch_indices) > 0:
                    selected_features = features[selected_patch_indices]
                    class_patches[pred_class].append(selected_features)
        
        # Update prototypes using selected patches
        for class_id in range(self.num_classes):
            if len(class_patches[class_id]) > 0:
                all_patches = torch.cat(class_patches[class_id], dim=0)
                
                stats['patches_selected_per_class'][class_id] = len(all_patches)
                
                if len(all_patches) >= min_patches_per_class:
                    num_updated = self._update_class_prototypes(class_id, all_patches)
                    stats['prototypes_updated_per_class'][class_id] = num_updated
        
        # Record in history
        self.refinement_history.append(stats)
        
        return stats
    
    def _select_patches_for_refinement(
        self,
        features: torch.Tensor,
        confidence: float,
    ) -> np.ndarray:
        """
        Select patches to use for prototype refinement.
        
        Uses different strategies based on configuration.
        
        Args:
            features: Patch features, shape (num_patches, feature_dim)
            confidence: Prediction confidence for this WSI
            
        Returns:
            Boolean mask or indices of selected patches
        """
        num_patches = features.shape[0]
        
        if self.refinement_strategy == 'high_confidence':
            # Select patches based on confidence-weighted sampling
            # Higher confidence WSIs contribute more patches
            num_to_select = max(1, int(num_patches * confidence))
            
            # Randomly select patches (weighted by feature norms as proxy for confidence)
            feature_norms = torch.norm(features, dim=1)
            probs = feature_norms / (feature_norms.sum() + 1e-8)
            
            selected_indices = np.random.choice(
                num_patches,
                size=num_to_select,
                replace=False,
                p=probs.cpu().numpy()
            )
            return selected_indices
        
        elif self.refinement_strategy == 'margin':
            # Select patches with largest margin between predictions
            # (This requires access to classifier which we don't have here)
            # For now, use all patches
            return np.arange(num_patches)
        
        elif self.refinement_strategy == 'entropy':
            # Select patches with low entropy (most confident predictions)
            # Again, requires classifier access
            # Use all patches as fallback
            return np.arange(num_patches)
        
        else:
            # Default: use all patches
            return np.arange(num_patches)
    
    def _update_class_prototypes(
        self,
        class_id: int,
        features: torch.Tensor,
    ) -> int:
        """
        Update prototypes for a class using k-means clustering.
        
        Args:
            class_id: Target class ID
            features: Patch features for this class, shape (num_patches, feature_dim)
            
        Returns:
            Number of prototypes added
        """
        # Determine number of clusters
        n_clusters = min(self.n_clusters_per_class, len(features))
        
        if n_clusters < 1:
            return 0
        
        # Perform k-means clustering using shared utility
        try:
            new_prototypes, _ = cluster_features_kmeans(
                features,
                n_clusters=n_clusters,
                normalize=True,
                random_state=42
            )
            
            # Add to prototype bank
            self.prototype_bank.add_prototypes(class_id, new_prototypes)
            
            return n_clusters
        
        except Exception as e:
            print(f"Warning: Failed to cluster prototypes for class {class_id}: {e}")
            return 0
    
    def get_refinement_statistics(self) -> Dict:
        """Get statistics from all refinement iterations."""
        if not self.refinement_history:
            return {}
        
        stats = {
            'num_iterations': len(self.refinement_history),
            'total_correctly_classified': sum(
                h['correctly_classified'] for h in self.refinement_history
            ),
            'avg_correctly_classified': np.mean([
                h['correctly_classified'] for h in self.refinement_history
            ]),
        }
        
        return stats


class MultiStageRefinementPipeline:
    """
    Implements the multi-stage refinement loop for WSI-level PBIP.
    
    Pipeline:
    Stage 0 (Initialization): Bootstrap prototypes from initial high-confidence patches
    Stage 1 (EM-Loop): 
        - E-step: Classify all WSIs using current prototypes
        - M-step: Refine prototypes from correctly classified WSIs
        - Repeat until convergence or max iterations
    
    Args:
        prototype_bank: PrototypeBank instance
        classifier: WSI classifier using prototypes
        num_classes: Number of classes
        max_refinement_iterations: Maximum EM iterations
        convergence_threshold: Threshold for determining convergence
    """
    
    def __init__(
        self,
        prototype_bank,
        classifier,
        num_classes: int,
        max_refinement_iterations: int = 5,
        convergence_threshold: float = 0.01,
    ):
        self.prototype_bank = prototype_bank
        self.classifier = classifier
        self.num_classes = num_classes
        self.max_refinement_iterations = max_refinement_iterations
        self.convergence_threshold = convergence_threshold
        
        self.refinement_manager = PrototypeBankRefinement(
            prototype_bank=prototype_bank,
            num_classes=num_classes,
            feature_dim=classifier.feature_dim,
        )
        
        self.stage_results = []
    
    def initialize_prototypes_from_patches(
        self,
        initial_patches_by_class: Dict[int, torch.Tensor],
        n_clusters_per_class: int = 5,
    ) -> Dict:
        """
        Initialize prototype bank from initial high-confidence patches.
        
        Args:
            initial_patches_by_class: Dict mapping class_id -> features tensor
                                     shape (num_patches, feature_dim)
            n_clusters_per_class: Number of clusters per class
            
        Returns:
            Initialization statistics
        """
        stats = {
            'stage': 'initialization',
            'prototypes_per_class': {},
        }
        
        print("\n" + "="*70)
        print("STAGE 0: PROTOTYPE BANK INITIALIZATION")
        print("="*70)
        
        for class_id, features in initial_patches_by_class.items():
            if features.shape[0] < n_clusters_per_class:
                print(f"Warning: Class {class_id} has {features.shape[0]} patches, "
                     f"fewer than n_clusters={n_clusters_per_class}")
                n_clusters = features.shape[0]
            else:
                n_clusters = n_clusters_per_class
            
            # Cluster patches to get prototypes using shared utility
            try:
                prototypes, _ = cluster_features_kmeans(
                    features,
                    n_clusters=n_clusters,
                    normalize=True,
                    random_state=42
                )
                
                num_added = self.prototype_bank.add_prototypes(class_id, prototypes)
                
                stats['prototypes_per_class'][class_id] = num_added
                print(f"Class {class_id}: Initialized {num_added} prototypes from "
                     f"{features.shape[0]} patches")
            
            except Exception as e:
                print(f"Error initializing prototypes for class {class_id}: {e}")
        
        print("="*70 + "\n")
        self.stage_results.append(stats)
        return stats
    
    def run_em_refinement_loop(
        self,
        wsi_dataloader,
        wsi_labels,
        device: str = 'cuda',
    ) -> List[Dict]:
        """
        Run Expectation-Maximization refinement loop.
        
        Args:
            wsi_dataloader: DataLoader yielding (wsi_name, features, wsi_labels)
            wsi_labels: Dictionary mapping wsi_name -> ground_truth_class
            device: Device for computation
            
        Returns:
            List of statistics for each EM iteration
        """
        print("\n" + "="*70)
        print("MULTI-STAGE REFINEMENT (EXPECTATION-MAXIMIZATION)")
        print("="*70)
        
        em_results = []
        prev_accuracy = 0.0
        
        for iteration in range(self.max_refinement_iterations):
            print(f"\n--- EM Iteration {iteration + 1}/{self.max_refinement_iterations} ---")
            
            # E-STEP: Classify all WSIs
            print("E-step: Classifying WSIs with current prototypes...")
            wsi_predictions = self._classify_all_wsis(wsi_dataloader, device)
            
            # Evaluate classification accuracy
            correct = sum(
                1 for wsi_name, pred_class in wsi_predictions.items()
                if pred_class == wsi_labels[wsi_name]
            )
            accuracy = correct / len(wsi_predictions)
            
            print(f"Classification accuracy: {accuracy:.4f} ({correct}/{len(wsi_predictions)})")
            
            # Check convergence
            if abs(accuracy - prev_accuracy) < self.convergence_threshold:
                print(f"Converged! Accuracy improvement < {self.convergence_threshold}")
                break
            
            # M-STEP: Refine prototypes from correctly classified WSIs
            print("M-step: Refining prototypes from correctly classified WSIs...")
            
            # Collect features from correctly classified WSIs
            wsi_features_for_refinement = {}
            for wsi_name, (pred_class, confidence) in wsi_predictions.items():
                if pred_class == wsi_labels[wsi_name]:
                    # Get features for this WSI (need to retrieve from dataloader)
                    wsi_features_for_refinement[wsi_name] = (
                        None,  # Would need to cache features from E-step
                        pred_class,
                        confidence
                    )
            
            # Update prototypes (requires cached features, implementation pending)
            # refinement_stats = self.refinement_manager.update_prototypes_from_wsi_features(
            #     wsi_features_for_refinement
            # )
            
            iteration_result = {
                'iteration': iteration + 1,
                'accuracy': accuracy,
                'num_correct': correct,
                'num_total': len(wsi_predictions),
                # 'refinement_stats': refinement_stats,
            }
            em_results.append(iteration_result)
            prev_accuracy = accuracy
        
        print("\n" + "="*70)
        print(f"EM Refinement completed: {len(em_results)} iterations")
        print("="*70 + "\n")
        
        self.stage_results.extend(em_results)
        return em_results
    
    def _classify_all_wsis(
        self,
        wsi_dataloader,
        device: str = 'cuda',
    ) -> Dict[str, Tuple[int, float]]:
        """
        Classify all WSIs using current prototype bank.
        
        Args:
            wsi_dataloader: DataLoader yielding (wsi_name, features)
            device: Device for computation
            
        Returns:
            Dictionary mapping wsi_name -> (pred_class, confidence)
        """
        self.classifier.eval()
        predictions = {}
        
        with torch.no_grad():
            for batch in tqdm(wsi_dataloader, desc="Classifying WSIs"):
                wsi_names = batch[0]  # WSI names
                patch_features = batch[1].to(device)  # (batch_size, num_patches, feature_dim)
                
                # Classify
                logits, _ = self.classifier(patch_features)
                
                # Get predictions
                pred_classes = torch.argmax(logits, dim=1)
                confidences = torch.softmax(logits, dim=1).max(dim=1)[0]
                
                for wsi_name, pred_class, confidence in zip(
                    wsi_names, pred_classes.cpu(), confidences.cpu()
                ):
                    predictions[wsi_name] = (
                        int(pred_class),
                        float(confidence)
                    )
        
        return predictions
    
    def get_final_statistics(self) -> Dict:
        """Get overall statistics from multi-stage refinement."""
        return {
            'num_stages': len(self.stage_results),
            'stage_results': self.stage_results,
        }
