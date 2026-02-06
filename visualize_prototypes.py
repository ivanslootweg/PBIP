import argparse
import os
import pickle as pkl
import yaml
import pandas as pd
import openslide
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    work_dir = config.get('work_dir', './work_dirs')
    exp_name = config.get('experiment_name', 'default_exp')
    exp_dir = os.path.join(work_dir, exp_name)
    
    viz_path = os.path.join(exp_dir, 'features/prototypes_viz.pkl')
    
    if not os.path.exists(viz_path):
        print(f"Viz file not found: {viz_path}")
        return
        
    with open(viz_path, 'rb') as f:
        prototype_metadata = pkl.load(f)

    # Build filename -> path map
    data_csv = config.get('data_csv_path')
    df = pd.read_csv(data_csv)
    
    # Config/Columns
    col_map = {
        'image_name': 'image_name',
    }
    if 'columns' in config:
        col_map.update(config['columns'])
        
    wsi_dir = config.get('wsi_dir')
    wsi_ext = config.get('wsi_extension', '.tif')
    
    path_map = {}
    
    col_wsi_path = col_map.get('wsi_path', 'wsi_path')

    for _, row in df.iterrows():
        fname = str(row[col_map['image_name']])
        
        # Construct full path
        full_path = None
        if col_wsi_path in row and pd.notna(row[col_wsi_path]):
             full_path = str(row[col_wsi_path])
        
        if not full_path or not os.path.exists(full_path):
            if wsi_dir:
                name_part = fname
                if not name_part.lower().endswith(tuple(['.tif', '.svs', '.ndpi', '.mrxs', '.tiff'])):
                    name_part += wsi_ext
                full_path = os.path.join(wsi_dir, name_part)
            else:
                 if not full_path: full_path = fname
            
        base = os.path.basename(fname)
        path_map[base] = full_path
    
    save_dir = os.path.join(exp_dir, 'figures/prototypes')
    os.makedirs(save_dir, exist_ok=True)
    
    # Skip if visualizations already exist
    overview_exists = os.path.exists(os.path.join(save_dir, 'prototype_bank_overview.png'))
    if overview_exists:
        print(f"Prototype visualizations already exist in {save_dir}. Skipping.")
        return
    
    # Get config parameters
    n_clusters = config.get('n_clusters', 5)
    n_proto_per_cluster = config.get('n_prototypes_per_cluster', 6)
    
    all_class_overviews = []
    
    for class_name, protos in prototype_metadata.items():
        print(f"Visualizing {class_name} ({len(protos)} prototypes)...")
        
        # Group by cluster
        clusters = {}
        for proto in protos:
            cluster_id = proto.get('cluster_idx', 0)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(proto)
        
        # Extract patches for each cluster
        cluster_images = {}
        for cluster_id in sorted(clusters.keys()):
            cluster_patches = []
            for proto in clusters[cluster_id][:n_proto_per_cluster]:  # Limit to Nk
                try:
                    name = proto['name']
                    wsi_basename, x_str, y_str = name.rsplit('_', 2)
                    x = int(x_str)
                    y = int(y_str)
                    
                    full_path = path_map.get(wsi_basename)
                    if not full_path or not os.path.exists(full_path):
                        continue
                    
                    slide = openslide.OpenSlide(full_path)
                    patch = slide.read_region((x, y), 0, (224, 224)).convert("RGB")
                    cluster_patches.append(patch)
                except Exception as e:
                    pass
            
            if cluster_patches:
                cluster_images[cluster_id] = cluster_patches
        
        # Create grid visualization for this class
        if cluster_images:
            class_overview = create_prototype_overview(cluster_images, class_name, 
                                                      n_clusters, n_proto_per_cluster, save_dir)
            all_class_overviews.append((class_name, class_overview))
    
    # Create combined overview
    if len(all_class_overviews) > 1:
        create_combined_overview(all_class_overviews, save_dir)
    
    print(f"Visualizations saved to {save_dir}")

def create_prototype_overview(cluster_images, class_name, n_clusters, n_proto_per_cluster, save_dir):
    """Create a grid overview of prototypes organized by subcluster."""
    patch_size = 224
    margin = 10
    title_height = 60
    label_width = 80
    
    # Calculate dimensions
    n_cols = len(cluster_images)
    n_rows = max(len(patches) for patches in cluster_images.values())
    
    grid_width = n_cols * patch_size + (n_cols + 1) * margin + label_width
    grid_height = n_rows * patch_size + (n_rows + 1) * margin + title_height
    
    # Create canvas
    canvas = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load font, fallback to default
    try:
        font_title = ImageFont.truetype("arial.ttf", 24)
        font_label = ImageFont.truetype("arial.ttf", 16)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
    
    # Draw title
    title = f"{class_name.upper()} (K={n_clusters} subclusters, Nk={n_proto_per_cluster} representatives each)"
    draw.text((margin, margin), title, fill='black', font=font_title)
    
    # Draw grid
    for col_idx, (cluster_id, patches) in enumerate(sorted(cluster_images.items())):
        # Draw subcluster label
        label_x = label_width + col_idx * (patch_size + margin) + margin
        label_y = title_height - 20
        draw.text((label_x, label_y), f"Sub {cluster_id}", fill='black', font=font_label)
        
        # Draw patches
        for row_idx, patch in enumerate(patches):
            x = label_width + col_idx * (patch_size + margin) + margin
            y = title_height + row_idx * (patch_size + margin) + margin
            canvas.paste(patch, (x, y))
    
    # Save individual class overview
    class_path = os.path.join(save_dir, f"{class_name}_overview.png")
    canvas.save(class_path)
    print(f"  Saved {class_name} overview to {class_path}")
    
    return canvas

def create_combined_overview(class_overviews, save_dir):
    """Combine multiple class overviews into a single image."""
    margin = 20
    
    # Calculate total dimensions
    max_width = max(img.width for _, img in class_overviews)
    total_height = sum(img.height for _, img in class_overviews) + margin * (len(class_overviews) + 1)
    
    # Create combined canvas
    combined = Image.new('RGB', (max_width, total_height), 'white')
    
    # Paste each class overview
    y_offset = margin
    for class_name, img in class_overviews:
        combined.paste(img, (0, y_offset))
        y_offset += img.height + margin
    
    # Save combined overview
    combined_path = os.path.join(save_dir, "prototype_bank_overview.png")
    combined.save(combined_path)
    print(f"  Saved combined overview to {combined_path}")

if __name__ == "__main__":
    main()
