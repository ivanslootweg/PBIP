"""
TSNE Visualization of Prototype Feature Space

Creates two side-by-side TSNE plots:
1. Left: Colored by class (e.g., benign vs BCC)
2. Right: Colored by subcluster (showing all subclusters with different markers)
"""

import argparse
import os
import pickle as pkl
import numpy as np
import yaml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Paths
    work_dir = config.get('work_dir', './work_dirs')
    experiment_name = config.get('experiment_name', 'default_experiment')
    features_dir = os.path.join(work_dir, experiment_name, 'features')
    prototypes_path = os.path.join(features_dir, 'prototypes.pkl')
    
    # Output guard
    output_path = os.path.join(features_dir, 'prototype_tsne.png')
    if os.path.exists(output_path):
        print(f"TSNE visualization already exists at {output_path}. Skipping.")
        return
    
    if not os.path.exists(prototypes_path):
        print(f"Prototypes file not found at {prototypes_path}. Please run clustering first.")
        return
    
    print("Loading prototypes...")
    with open(prototypes_path, 'rb') as f:
        prototype_metadata = pkl.load(f)
    
    # Extract features and metadata
    all_features = []
    all_classes = []
    all_subclusters = []
    class_names = list(prototype_metadata.keys())
    
    for class_name, protos in prototype_metadata.items():
        for proto in protos:
            all_features.append(proto['features'].flatten())
            all_classes.append(class_name)
            all_subclusters.append(f"{class_name} Sub{proto['cluster_idx']}")
    
    all_features = np.array(all_features)
    
    print(f"Running TSNE on {len(all_features)} prototypes...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)-1))
    tsne_results = tsne.fit_transform(all_features)
    
    # Create two-panel plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define colors and markers for classes
    class_colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    class_color_map = {name: class_colors[i] for i, name in enumerate(class_names)}
    
    # Define markers for subclusters (circles and triangles alternating)
    markers = ['o', '^', 's', 'D', 'v', 'P', '*', 'X']
    
    # LEFT PLOT: Colored by Class
    ax1.set_title('TSNE Projection: Colored by Class', fontsize=14, fontweight='bold')
    ax1.set_xlabel('TSNE Dimension 1', fontsize=12)
    ax1.set_ylabel('TSNE Dimension 2', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for class_name in class_names:
        mask = np.array([c == class_name for c in all_classes])
        color = 'coral' if 'benign' in class_name.lower() else 'gray'
        ax1.scatter(
            tsne_results[mask, 0], 
            tsne_results[mask, 1],
            c=color,
            s=100,
            alpha=0.7,
            label=class_name,
            edgecolors='black',
            linewidths=0.5
        )
    
    ax1.legend(loc='upper right', fontsize=10)
    
    # RIGHT PLOT: Colored by Subcluster
    ax2.set_title('TSNE Projection: Colored by Subcluster', fontsize=14, fontweight='bold')
    ax2.set_xlabel('TSNE Dimension 1', fontsize=12)
    ax2.set_ylabel('TSNE Dimension 2', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Get unique subclusters
    unique_subclusters = sorted(set(all_subclusters))
    
    # Create color map for subclusters
    subcluster_info = {}
    for sc in unique_subclusters:
        class_name = sc.split(' Sub')[0]
        sub_idx = int(sc.split('Sub')[1])
        
        # Base color from class
        if 'benign' in class_name.lower():
            # Shades of blue for benign
            base_colors = ['#D4E6F1', '#A9CCE3', '#7FB3D5', '#5499C7', '#2E86C1', '#21618C']
            color = base_colors[sub_idx % len(base_colors)]
            marker = 'o'
        else:
            # Shades of red/orange for BCC
            base_colors = ['#F5CBA7', '#F0B27A', '#EB984E', '#E67E22', '#CA6F1E', '#AF601A']
            color = base_colors[sub_idx % len(base_colors)]
            marker = '^'
        
        subcluster_info[sc] = {'color': color, 'marker': marker}
    
    # Plot each subcluster
    for subcluster in unique_subclusters:
        mask = np.array([s == subcluster for s in all_subclusters])
        info = subcluster_info[subcluster]
        ax2.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            c=info['color'],
            marker=info['marker'],
            s=100,
            alpha=0.7,
            label=subcluster,
            edgecolors='black',
            linewidths=0.5
        )
    
    # Create legend with two columns
    ax2.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"TSNE visualization saved to {output_path}")

if __name__ == "__main__":
    main()
