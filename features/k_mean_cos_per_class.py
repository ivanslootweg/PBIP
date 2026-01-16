import os
import pickle as pkl
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from omegaconf import OmegaConf
import argparse
import glob

class CosineSimilarityKMeans:
    def __init__(self, n_clusters, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(random_state)
        
    def fit_predict(self, X):
        n_samples = X.shape[0]
        
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
                    self.cluster_centers_[i] /= np.linalg.norm(self.cluster_centers_[i])
            
            if np.allclose(old_centers, self.cluster_centers_):
                break
                
        return new_labels, similarities, torch.from_numpy(self.cluster_centers_)

def cluster_features_per_class(cfg):
    # Read UID from config or latest_uid.txt
    uid = getattr(cfg, 'run_uid', None)
    if uid is None or uid == 'null':
        proto_coords_dir = os.path.join(cfg.work_dir, 'prototype_coordinates')
        uid_file = os.path.join(proto_coords_dir, 'latest_uid.txt')
        if os.path.exists(uid_file):
            with open(uid_file, 'r') as f:
                uid = f.read().strip()
            print(f"Using UID from {uid_file}: {uid}")
    
    # Resolve UID in save_dir
    save_dir = cfg.features.save_dir
    if uid:
        save_dir = save_dir.replace('${run_uid}', uid).replace('None', uid)
    
    base_medclip_name = cfg.features.medclip_features_pkl.replace('.pkl', '')
    if uid:
        base_medclip_name = base_medclip_name.replace('${run_uid}', uid).replace('None', uid)
    
    # Construct the input path
    input_pkl = os.path.join(save_dir, base_medclip_name + '.pkl')
    
    if not os.path.exists(input_pkl):
        raise FileNotFoundError(f"MedCLIP features not found at: {input_pkl}\\nPlease run MedCLIP extraction first.")
    
    print(f"Loading MedCLIP features from: {input_pkl}")
    
    with open(input_pkl, 'rb') as f:
        features_dict = pkl.load(f)

    k_list = list(cfg.features.k_list)
    nk = getattr(cfg.features, 'nk', 5)  # Paper uses Nk=5 representative images per subclass
    class_order = list(getattr(cfg.dataset, 'class_order', ['benign', 'tumor']))
    
    if len(k_list) != len(class_order):
        raise ValueError(f"features.k_list must have {len(class_order)} values (one per parent class in class_order), got {len(k_list)}")

    print(f"\nUsing Nk={nk} representative images per subclass (from paper)")
    print(f"Total features to store: K × Nk = {sum(k_list)} × {nk} = {sum(k_list) * nk}")

    all_representative_features = []
    representative_indices_per_class = {}
    
    for class_name, k in zip(class_order, k_list):
        print(f"\n{'='*70}")
        print(f"Class: {class_name} (K={k} subclasses, Nk={nk} representatives per subclass)")
        print(f"{'='*70}")
        
        class_features = []
        class_names = []
        
        for item in features_dict[class_name]:
            class_features.append(item['features'].squeeze())
            class_names.append(item['name'])
        
        features_array = np.array(class_features)
        features_norm = features_array / np.linalg.norm(features_array, axis=1, keepdims=True)
        
        kmeans = CosineSimilarityKMeans(n_clusters=k, random_state=42)
        cluster_labels, similarities, cluster_centers = kmeans.fit_predict(features_norm)
        
        class_representative_features = []
        representative_indices_per_class[class_name] = {}
        
        for cluster_idx in range(k):
            cluster_mask = cluster_labels == cluster_idx
            cluster_similarities = similarities[cluster_mask][:, cluster_idx]
            
            # Select top Nk closest images to cluster center
            top_nk_indices = np.argsort(cluster_similarities)[-nk:][::-1]
            cluster_sample_indices = np.where(cluster_mask)[0][top_nk_indices]
            
            print(f"\nSubclass {cluster_idx + 1} - Top {nk} closest images to center:")
            subclass_features = []
            for idx, sample_idx in enumerate(cluster_sample_indices, 1):
                similarity = similarities[sample_idx, cluster_idx]
                print(f"{idx}. {class_names[sample_idx]} (similarity: {similarity:.4f})")
                # Get the normalized feature for this representative image
                subclass_features.append(features_norm[sample_idx])
            
            cluster_size = np.sum(cluster_labels == cluster_idx)
            print(f"Total samples in cluster: {cluster_size}")
            
            # Stack features for this subclass: (Nk, feature_dim)
            subclass_features_array = np.array(subclass_features)
            class_representative_features.append(torch.from_numpy(subclass_features_array))
            representative_indices_per_class[class_name][cluster_idx] = cluster_sample_indices.tolist()
        
        # Stack all subclasses for this class: (K, Nk, feature_dim)
        class_features_tensor = torch.stack(class_representative_features)
        # Flatten to (K*Nk, feature_dim) for compatibility with model
        class_features_tensor = class_features_tensor.reshape(-1, class_features_tensor.shape[-1])
        all_representative_features.append(class_features_tensor)
    
    # Concatenate all classes: (total_features, feature_dim)
    all_features_tensor = torch.cat(all_representative_features, dim=0)
    
    save_info = {
        'features': all_features_tensor,
        'k_list': k_list,
        'nk': nk,
        'class_order': class_order,
        'cumsum_k': np.cumsum([0] + k_list),
        'representative_indices': representative_indices_per_class
    }
    
    # Use same save_dir already resolved with UID
    os.makedirs(save_dir, exist_ok=True)
    
    # Use UID in label features filename if available
    base_label_name = cfg.features.label_feature_pkl.replace('.pkl', '')
    if uid:
        base_label_name = base_label_name.replace('${run_uid}', uid).replace('None', uid)
    
    save_path = os.path.join(save_dir, base_label_name + '.pkl')
    
    with open(save_path, 'wb') as f:
        pkl.dump(save_info, f)
    
    print(f"\n{'='*70}")
    print(f"Information saved to {save_path}")
    print(f"Feature tensor shape: {all_features_tensor.shape}")
    print(f"  Total features: {all_features_tensor.shape[0]} = {sum(k_list)} classes × {nk} representatives")
    print(f"  Feature dimension: {all_features_tensor.shape[1]}")
    print(f"K list: {k_list}")
    print(f"Nk (representatives per subclass): {nk}")
    print(f"Cumulative sum of k: {save_info['cumsum_k']}")
    print("\nClass feature index ranges:")
    for i, class_name in enumerate(class_order):
        start_idx = save_info['cumsum_k'][i] * nk
        end_idx = save_info['cumsum_k'][i+1] * nk
        print(f"{class_name}: {start_idx} to {end_idx} ({(end_idx - start_idx)} features)")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cluster_features_per_class(cfg)