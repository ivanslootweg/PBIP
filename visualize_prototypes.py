"""
Visualize Prototype Bank and Subclusters

Generates comprehensive static visualizations of the prototype bank created by k_mean_cos_per_class.py:
- Grid view of all prototypes organized by class and subcluster
- Feature space projections (t-SNE/UMAP)
- Cluster composition statistics
- Representative sample thumbnails

Usage:
    python visualize_prototypes.py --config work_dirs/custom_wsi_template.yaml
    
    # Or specify UID explicitly
    python visualize_prototypes.py --config work_dirs/custom_wsi_template.yaml --uid 500_th0-9980_top_attention_abc123

Output:
    Saves figures to: work_dir/runs/{uid}/visualizations/
"""

import os
import sys
import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import hashlib

# Import common utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils.common import extract_patch_numpy, extract_patch_openslide
from utils.pyutils import build_uid_from_config

try:
    import openslide
    HAS_OPENSLIDE = True
except:
    from skimage import io
    HAS_OPENSLIDE = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except:
    HAS_TSNE = False
    print("Warning: scikit-learn not found. t-SNE visualization will be skipped.")

try:
    import umap
    HAS_UMAP = True
except:
    HAS_UMAP = False
    print("Note: UMAP not installed. Using t-SNE only.")


def load_prototype_data(cfg, uid):
    """Load prototype bank and exemplar features."""
    # Since we set cfg.run_uid, we can trust OmegaConf resolution primarily
    save_dir = cfg.features.save_dir
    
    # Fallback/Safety: Check if 'None' still persists due to raw string access
    if uid and 'None' in save_dir and uid not in save_dir:
         save_dir = save_dir.replace('None', uid)

    # Load label features (prototype bank)
    base_label_name = cfg.features.label_feature_pkl.replace('.pkl', '')
    
    label_pkl_path = os.path.join(save_dir, base_label_name + '.pkl')
    
    if not os.path.exists(label_pkl_path):
        import glob
        # Try to find robustly if file name is slightly different
        pattern = os.path.join(save_dir, "label_*.pkl") # Simplified pattern
        candidates = glob.glob(pattern)
        if candidates:
            print(f"Warning: Exact match not found, but found candidate: {candidates[0]}")
            label_pkl_path = candidates[0]
        else:
             # Fallback to check for old complex naming
            pattern_complex = os.path.join(save_dir, f"label_fea_pro_{uid}.pkl")
            if os.path.exists(pattern_complex):
                 print(f"Warning: Exact match not found, but found legacy format: {pattern_complex}")
                 label_pkl_path = pattern_complex
            else:
                 raise FileNotFoundError(f"Prototype bank not found: {label_pkl_path}")
    
    print(f"Loading prototype bank from: {label_pkl_path}")
    with open(label_pkl_path, 'rb') as f:
        label_data = pkl.load(f)
    
    # Load exemplar features (original patch info)
    patch_encoder = cfg.features.features_for_prototype_clusters.replace('.pkl', '')
    encoder_name = getattr(cfg.model, 'patch_encoder', 'medclip')
    if encoder_name.lower().strip() != 'medclip':
        patch_encoder = patch_encoder.replace('medclip', encoder_name)
    
    # Removed UID interpolation for filename as we simplified config
    
    exemplar_pkl_path = os.path.join(save_dir, patch_encoder + '.pkl')
    
    exemplar_data = None
    if os.path.exists(exemplar_pkl_path):
        print(f"Loading exemplar features from: {exemplar_pkl_path}")
        with open(exemplar_pkl_path, 'rb') as f:
            exemplar_data = pkl.load(f)
    else:
        print(f"Warning: Exemplar features not found at {exemplar_pkl_path}")
        print("  Visualizations will be limited to feature space plots only")
    
    return label_data, exemplar_data


def extract_patch_image(wsi_path, x, y, patch_size, use_openslide=True):
    """Extract a patch image from WSI."""
    if use_openslide and HAS_OPENSLIDE:
        return extract_patch_openslide(wsi_path, x, y, patch_size)
    else:
        return extract_patch_numpy(wsi_path, x, y, patch_size)


def load_patch_images(exemplar_data, class_name, representative_indices, wsi_dir, patch_size, use_openslide):
    """Load actual patch images for representative samples."""
    if exemplar_data is None:
        return None
    
    # Get feature list for this class
    if class_name not in exemplar_data:
        print(f"Warning: Class '{class_name}' not found in exemplar data")
        return None
    
    feature_list = exemplar_data[class_name]
    
    # Load images for each subcluster
    patch_images = {}
    for cluster_idx, sample_indices in representative_indices.items():
        cluster_images = []
        for idx in sample_indices:
            if idx >= len(feature_list):
                print(f"Warning: Index {idx} out of range for {class_name}")
                continue
            
            item = feature_list[idx]
            wsi_name = item['name']
            x, y = item['coords']
            
            # Construct WSI path
            wsi_path = os.path.join(wsi_dir, wsi_name)

            # Fallback: exemplar entries may omit extension; try common slide suffixes
            if not os.path.exists(wsi_path) and Path(wsi_name).suffix == '':
                for ext in ('.tif', '.tiff', '.svs'):
                    candidate = wsi_path + ext
                    if os.path.exists(candidate):
                        wsi_path = candidate
                        break
            
            try:
                patch_img = extract_patch_image(wsi_path, x, y, patch_size, use_openslide)
                if patch_img is not None:
                    cluster_images.append({
                        'image': patch_img,
                        'wsi_name': wsi_name,
                        'coords': (x, y)
                    })
            except Exception as e:
                print(f"Warning: Failed to extract patch from {wsi_name} at ({x}, {y}): {e}")
                continue
        
        patch_images[cluster_idx] = cluster_images
    
    return patch_images


def plot_prototype_grid(label_data, exemplar_data, patch_images_by_class, save_path):
    """Generate grid view of all prototypes organized by class and subcluster."""
    k_list = label_data['k_list']
    nk = label_data['nk']
    class_order = label_data['class_order']
    representative_indices = label_data['representative_indices']
    
    # Calculate grid dimensions
    max_k = max(k_list)
    n_classes = len(class_order)
    
    # Create figure with subplots for each class
    fig = plt.figure(figsize=(max_k * 2.5, n_classes * nk * 0.8 + 2))
    gs = GridSpec(n_classes, 1, figure=fig, hspace=0.4)
    
    for class_idx, class_name in enumerate(class_order):
        k = k_list[class_idx]
        
        # Create subplot grid for this class
        ax_class = fig.add_subplot(gs[class_idx])
        ax_class.axis('off')
        ax_class.set_title(f'{class_name.upper()} (K={k} subclusters, Nk={nk} representatives each)',
                          fontsize=14, fontweight='bold', pad=10)
        
        # Get patch images for this class
        patch_images = patch_images_by_class.get(class_name, {}) if patch_images_by_class else {}
        
        # Create nested grid for subclusters
        inner_gs = gs[class_idx].subgridspec(nk, k, wspace=0.1, hspace=0.1)
        
        for sub_idx in range(k):
            cluster_images = patch_images.get(sub_idx, [])
            
            for nk_idx in range(nk):
                ax = fig.add_subplot(inner_gs[nk_idx, sub_idx])
                
                if nk_idx < len(cluster_images):
                    # Show actual patch image
                    img_data = cluster_images[nk_idx]
                    ax.imshow(img_data['image'])
                    ax.axis('off')
                    
                    # Add border and label
                    if nk_idx == 0:
                        ax.set_title(f'Sub {sub_idx}', fontsize=9, fontweight='bold')
                    
                    # Add WSI name as text (small)
                    wsi_basename = Path(img_data['wsi_name']).stem[:12]
                    ax.text(0.5, -0.05, wsi_basename, transform=ax.transAxes,
                           fontsize=6, ha='center', va='top')
                else:
                    # Empty placeholder
                    ax.imshow(np.ones((224, 224, 3)) * 0.9)
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                           fontsize=10, ha='center', va='center', color='gray')
    
    plt.suptitle('Prototype Bank: Representative Samples per Subcluster', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved prototype grid to: {save_path}")


def plot_feature_space(label_data, exemplar_data, save_path, method='tsne'):
    """Plot feature space projection (t-SNE or UMAP)."""
    if method == 'tsne' and not HAS_TSNE:
        print("Skipping t-SNE: scikit-learn not available")
        return
    if method == 'umap' and not HAS_UMAP:
        print("Skipping UMAP: umap-learn not available")
        return
    
    features = label_data['features'].cpu().numpy()
    k_list = label_data['k_list']
    nk = label_data['nk']
    class_order = label_data['class_order']
    cumsum_k = label_data['cumsum_k']
    
    # Generate labels for each prototype
    labels = []
    class_ids = []
    subcluster_ids = []
    
    for class_idx, class_name in enumerate(class_order):
        k = k_list[class_idx]
        for sub_idx in range(k):
            for _ in range(nk):
                labels.append(f'{class_name}_sub{sub_idx}')
                class_ids.append(class_idx)
                subcluster_ids.append(sub_idx)
    
    labels = np.array(labels)
    class_ids = np.array(class_ids)
    subcluster_ids = np.array(subcluster_ids)
    
    # Dimensionality reduction
    print(f"Computing {method.upper()} projection...")
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    else:  # umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(features) - 1))
    
    embedding = reducer.fit_transform(features)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Color by class
    ax = axes[0]
    colors = plt.cm.Set1(np.linspace(0, 1, len(class_order)))
    
    for class_idx, class_name in enumerate(class_order):
        mask = class_ids == class_idx
        ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                  c=[colors[class_idx]], label=class_name, 
                  alpha=0.7, s=80, edgecolors='black', linewidths=0.5)
    
    ax.set_title(f'{method.upper()} Projection: Colored by Class', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    
    # Plot 2: Color by subcluster (within class)
    ax = axes[1]
    
    for class_idx, class_name in enumerate(class_order):
        k = k_list[class_idx]
        class_mask = class_ids == class_idx
        class_embedding = embedding[class_mask]
        class_subclusters = subcluster_ids[class_mask]
        
        # Use different color map per class
        if class_idx == 0:
            cmap = plt.cm.Blues
        else:
            cmap = plt.cm.Reds
        
        subcluster_colors = cmap(np.linspace(0.3, 0.9, k))
        
        for sub_idx in range(k):
            sub_mask = class_subclusters == sub_idx
            if np.any(sub_mask):
                ax.scatter(class_embedding[sub_mask, 0], class_embedding[sub_mask, 1],
                          c=[subcluster_colors[sub_idx]], 
                          label=f'{class_name} Sub{sub_idx}',
                          alpha=0.7, s=80, edgecolors='black', linewidths=0.5,
                          marker='o' if class_idx == 0 else '^')
    
    ax.set_title(f'{method.upper()} Projection: Colored by Subcluster', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'Prototype Feature Space ({method.upper()})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {method.upper()} projection to: {save_path}")


def plot_cluster_statistics(label_data, exemplar_data, save_path):
    """Generate cluster composition statistics and charts."""
    k_list = label_data['k_list']
    nk = label_data['nk']
    class_order = label_data['class_order']
    representative_indices = label_data['representative_indices']
    features = label_data['features'].cpu().numpy()
    cumsum_k = label_data['cumsum_k']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Prototypes per class (bar chart)
    ax = axes[0, 0]
    class_counts = [k * nk for k in k_list]
    bars = ax.bar(class_order, class_counts, color=['skyblue', 'salmon'][:len(class_order)],
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Prototypes', fontsize=12, fontweight='bold')
    ax.set_title('Prototypes per Class', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Subclusters per class
    ax = axes[0, 1]
    x = np.arange(len(class_order))
    width = 0.35
    
    bars1 = ax.bar(x, k_list, width, label='Subclusters (K)', 
                   color='steelblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width, [nk] * len(class_order), width, 
                   label='Representatives per subcluster (Nk)',
                   color='coral', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Clustering Parameters', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(class_order)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Feature space coverage (variance per class)
    ax = axes[1, 0]
    
    class_variances = []
    for class_idx, class_name in enumerate(class_order):
        start_idx = cumsum_k[class_idx] * nk
        end_idx = cumsum_k[class_idx + 1] * nk
        class_features = features[start_idx:end_idx]
        
        # Compute variance across feature dimensions
        variance = np.var(class_features, axis=0).mean()
        class_variances.append(variance)
    
    bars = ax.bar(class_order, class_variances, 
                  color=['lightblue', 'lightcoral'][:len(class_order)],
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Mean Feature Variance', fontsize=12, fontweight='bold')
    ax.set_title('Feature Diversity per Class', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, var in zip(bars, class_variances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{var:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
PROTOTYPE BANK SUMMARY
{'=' * 40}

Total Prototypes: {features.shape[0]}
Feature Dimension: {features.shape[1]}

Configuration:
  • Nk (representatives per subcluster): {nk}
  • Total subclusters: {sum(k_list)}

Per-Class Breakdown:
"""
    
    for class_idx, class_name in enumerate(class_order):
        k = k_list[class_idx]
        n_proto = k * nk
        start_idx = cumsum_k[class_idx] * nk
        end_idx = cumsum_k[class_idx + 1] * nk
        
        summary_text += f"\n{class_name.upper()}:\n"
        summary_text += f"  • Subclusters (K): {k}\n"
        summary_text += f"  • Total prototypes: {n_proto}\n"
        summary_text += f"  • Feature indices: {start_idx}–{end_idx}\n"
        summary_text += f"  • Feature variance: {class_variances[class_idx]:.6f}\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Prototype Bank Statistics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved cluster statistics to: {save_path}")


def plot_similarity_heatmap(label_data, save_path):
    """Plot similarity heatmap between all prototypes."""
    features = label_data['features'].cpu().numpy()
    k_list = label_data['k_list']
    nk = label_data['nk']
    class_order = label_data['class_order']
    cumsum_k = label_data['cumsum_k']
    
    # Normalize features
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(features_norm, features_norm.T)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', fontsize=12, fontweight='bold')
    
    # Add class boundaries
    for class_idx in range(len(class_order)):
        boundary = cumsum_k[class_idx + 1] * nk
        if boundary < features.shape[0]:
            ax.axhline(y=boundary - 0.5, color='black', linewidth=2)
            ax.axvline(x=boundary - 0.5, color='black', linewidth=2)
    
    # Add class labels
    tick_positions = []
    tick_labels = []
    for class_idx, class_name in enumerate(class_order):
        start_idx = cumsum_k[class_idx] * nk
        end_idx = cumsum_k[class_idx + 1] * nk
        mid_idx = (start_idx + end_idx) / 2
        tick_positions.append(mid_idx)
        tick_labels.append(class_name.upper())
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=12, fontweight='bold')
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=12, fontweight='bold')
    
    ax.set_title('Prototype Similarity Matrix (Cosine)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Prototype Index', fontsize=12)
    ax.set_ylabel('Prototype Index', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved similarity heatmap to: {save_path}")


def visualize_prototypes(cfg, uid=None, skip_images=False):
    """Main visualization function."""
    
    # Determine UID
    if uid is None:
        uid = build_uid_from_config(cfg)
        print(f"Generated UID from config: {uid}")
    
    # IMPORTANT: Inject UID into config so interpolations (${run_uid}) work natively
    cfg.run_uid = uid
    
    # Load data
    label_data, exemplar_data = load_prototype_data(cfg, uid)
    
    # Create output directory
    work_dir = Path(cfg.work_dir)
    if uid:
        vis_dir = work_dir / 'runs' / uid / 'visualizations'
    else:
        vis_dir = work_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Generating visualizations...")
    print(f"Output directory: {vis_dir}")
    print(f"{'='*70}\n")
    
    # Load patch images if available
    patch_images_by_class = {}
    if not skip_images and exemplar_data is not None:
        wsi_dir = cfg.dataset.wsi_dir
        patch_size = getattr(cfg.dataset, 'patch_size', 224)
        use_openslide = getattr(cfg.dataset, 'use_openslide', HAS_OPENSLIDE)
        class_order = label_data['class_order']
        representative_indices = label_data['representative_indices']
        
        print("Loading patch images for representative samples...")
        for class_name in class_order:
            if class_name in representative_indices:
                print(f"  Loading {class_name} patches...")
                patch_images = load_patch_images(
                    exemplar_data, class_name, representative_indices[class_name],
                    wsi_dir, patch_size, use_openslide
                )
                if patch_images:
                    patch_images_by_class[class_name] = patch_images
        print()
    
    # Generate visualizations
    print("Generating figures...\n")
    
    # 1. Prototype grid (only if we have images)
    if patch_images_by_class:
        grid_path = vis_dir / 'prototype_grid.png'
        plot_prototype_grid(label_data, exemplar_data, patch_images_by_class, grid_path)
    else:
        print("⊘ Skipping prototype grid (no patch images available)")
    
    # 2. Feature space projections
    if HAS_TSNE:
        tsne_path = vis_dir / 'feature_space_tsne.png'
        plot_feature_space(label_data, exemplar_data, tsne_path, method='tsne')
    
    if HAS_UMAP:
        umap_path = vis_dir / 'feature_space_umap.png'
        plot_feature_space(label_data, exemplar_data, umap_path, method='umap')
    
    # 3. Cluster statistics
    stats_path = vis_dir / 'cluster_statistics.png'
    plot_cluster_statistics(label_data, exemplar_data, stats_path)
    
    # 4. Similarity heatmap
    heatmap_path = vis_dir / 'similarity_heatmap.png'
    plot_similarity_heatmap(label_data, heatmap_path)
    
    print(f"\n{'='*70}")
    print(f"✓ All visualizations saved to: {vis_dir}")
    print(f"{'='*70}\n")
    
    # Print summary
    print("Generated files:")
    for f in sorted(vis_dir.glob('*.png')):
        print(f"  • {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize prototype bank and subclusters")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--uid', type=str, default=None, 
                       help='Run UID (auto-detected from config if not provided)')
    parser.add_argument('--skip_images', action='store_true',
                       help='Skip loading patch images (faster, only shows feature space plots)')
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    visualize_prototypes(cfg, uid=args.uid, skip_images=args.skip_images)
