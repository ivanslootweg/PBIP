import argparse
import os
import pickle as pkl
import numpy as np
import torch
import yaml
from sklearn.metrics.pairwise import cosine_similarity
import re

class CosineSimilarityKMeans:
    def __init__(self, n_clusters, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(random_state)
        
    def fit_predict(self, X):
        n_samples = X.shape[0]
        if n_samples < self.n_clusters:
            # If not enough samples, just use what we have as centers
            self.cluster_centers_ = X
            # Pad if strictly needed or return smaller
            return np.arange(n_samples), np.eye(n_samples), torch.from_numpy(self.cluster_centers_)

        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx]
        
        for _ in range(self.max_iter):
            similarities = cosine_similarity(X, self.cluster_centers_)
            new_labels = np.argmax(similarities, axis=1)
            
            old_centers = self.cluster_centers_.copy()
            for i in range(self.n_clusters):
                cluster_samples = X[new_labels == i]
                if len(cluster_samples) > 0:
                    self.cluster_centers_[i] = cluster_samples.mean(axis=0)
                    self.cluster_centers_[i] /= (np.linalg.norm(self.cluster_centers_[i]) + 1e-8)
            
            if np.allclose(old_centers, self.cluster_centers_):
                break
                
        return new_labels, similarities, torch.from_numpy(self.cluster_centers_)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Read from experiment-specific features dir
    work_dir = config.get('work_dir', './work_dirs')
    experiment_name = config.get('experiment_name', 'default_experiment')
    features_dir = os.path.join(work_dir, experiment_name, 'features')
    features_path = os.path.join(features_dir, 'prototype_features.pkl')
    
    if not os.path.exists(features_path):
        print(f"Features file not found at {features_path}. Please run extract_patches.py first.")
        return

    with open(features_path, 'rb') as f:
        features_dict = pkl.load(f)
        
    n_clusters = config.get('n_clusters', 4)
    n_proto_per_cluster = config.get('n_prototypes_per_cluster', 1)
    
    # K * Nk
    target_k = n_clusters * n_proto_per_cluster
    
    class_names = config.get('class_names', ["Negative", "Positive"])
    k_list = [target_k] * len(class_names)
    
    all_centers = []
    prototype_metadata = {} 

    for class_idx, class_name in enumerate(class_names):
        print(f"Clustering {class_name} with K={target_k}...")
        
        if class_name not in features_dict:
            print(f"Warning: {class_name} not in features dict.")
            dummy_protos = torch.randn(target_k, 512) 
            all_centers.append(dummy_protos)
            continue
            
        items = features_dict[class_name]
        if len(items) == 0:
             print(f"Warning: No items for {class_name}.")
             dummy_protos = torch.randn(target_k, 512) 
             all_centers.append(dummy_protos)
             continue
             
        feats = np.array([x['features'].squeeze() for x in items])
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        
        kmeans = CosineSimilarityKMeans(n_clusters=target_k, random_state=42)
        labels, sims, centers = kmeans.fit_predict(feats)
        
        all_centers.append(centers)
        
        class_protos = []
        # Support case where we have fewer centers than target_k
        actual_k = centers.shape[0]
        
        for c_id in range(actual_k):
            # Best match for center c_id
            col = sims[:, c_id]
            best_idx = np.argmax(col)
            item = items[best_idx]
            name = item['name']
            class_protos.append({
                'name': name,
                'cluster_idx': c_id,
                'similarity': float(col[best_idx])
            })
        prototype_metadata[class_name] = class_protos

    all_centers_tensor = torch.cat(all_centers, dim=0)
    
    # Save to experiment specific dir
    work_dir = config.get('work_dir', './work_dirs')
    exp_name = config.get('experiment_name', 'default_exp')
    exp_dir = os.path.join(work_dir, exp_name)
    
    save_dir = os.path.join(exp_dir, 'features')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "prototypes.pkl")
    viz_path = os.path.join(save_dir, "prototypes_viz.pkl")
    
    if os.path.exists(save_path) and os.path.exists(viz_path):
        print(f"Prototypes already exist at {save_path}. Skipping clustering.")
        return
    
    save_info = {
        'features': all_centers_tensor,
        'k_list': k_list,
        'class_order': class_names, # Save class order for training mapping
        'cumsum_k': np.cumsum([0] + k_list)
    }
    
    with open(save_path, 'wb') as f:
        pkl.dump(save_info, f)
        
    with open(viz_path, 'wb') as f:
        pkl.dump(prototype_metadata, f)
        
    print(f"Saved prototypes to {save_path}")
    print(f"Saved viz metadata to {viz_path}")

if __name__ == "__main__":
    main()
