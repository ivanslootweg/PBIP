"""
Interactive Prototype Bank Visualization (HTML Dashboard)

Generates an interactive HTML dashboard for exploring the prototype bank:
- Hover over patches to see WSI source, coordinates, and similarity scores
- Interactive feature space plots with zoom/pan
- Click-to-highlight prototypes across visualizations
- All-in-one standalone HTML file (no server needed)

Usage:
    python visualize_prototypes_interactive.py --config work_dirs/custom_wsi_template.yaml
    
    # Or specify UID explicitly
    python visualize_prototypes_interactive.py --config work_dirs/custom_wsi_template.yaml --uid 500_th0-9980_top_attention_abc123

Output:
    Opens browser with interactive dashboard at: work_dir/runs/{uid}/visualizations/prototype_dashboard.html
"""

import os
import sys
import argparse
import pickle as pkl
import numpy as np
import base64
from io import BytesIO
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import hashlib
from PIL import Image

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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except:
    HAS_PLOTLY = False
    print("ERROR: plotly is required for interactive visualizations")
    print("Install with: pip install plotly")
    sys.exit(1)

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
    # Resolve ${run_uid} via OmegaConf now that cfg.run_uid is set upstream
    save_dir = OmegaConf.to_container(OmegaConf.create({'save_dir': cfg.features.save_dir}), resolve=True)['save_dir']
    
    # Load label features (prototype bank)
    base_label_name = cfg.features.label_feature_pkl.replace('.pkl', '')
    
    label_pkl_path = os.path.join(save_dir, base_label_name + '.pkl')
    
    if not os.path.exists(label_pkl_path):
        import glob
        pattern = os.path.join(save_dir, "label_*.pkl")
        candidates = glob.glob(pattern)
        if candidates:
            label_pkl_path = candidates[0]
        else:
            # Fallback for old format
            complex_name = f"label_fea_pro_{uid}.pkl"
            if os.path.exists(os.path.join(save_dir, complex_name)):
                label_pkl_path = os.path.join(save_dir, complex_name)
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
    
    exemplar_pkl_path = os.path.join(save_dir, patch_encoder + '.pkl')
    
    exemplar_data = None
    if os.path.exists(exemplar_pkl_path):
        print(f"Loading exemplar features from: {exemplar_pkl_path}")
        with open(exemplar_pkl_path, 'rb') as f:
            exemplar_data = pkl.load(f)
    else:
        print(f"Warning: Exemplar features not found at {exemplar_pkl_path}")
        print("  Interactive grid will be limited to feature plots only")
    
    return label_data, exemplar_data


def extract_patch_image(wsi_path, x, y, patch_size, use_openslide=True):
    """Extract a patch image from WSI."""
    if use_openslide and HAS_OPENSLIDE:
        return extract_patch_openslide(wsi_path, x, y, patch_size)
    else:
        return extract_patch_numpy(wsi_path, x, y, patch_size)


def image_to_base64(img_array, thumbnail_size=(112, 112)):
    """Convert numpy image array to base64 string for HTML embedding."""
    # Convert to PIL Image
    img = Image.fromarray(img_array.astype('uint8'))
    
    # Resize for web display
    img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
    
    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"


def load_patch_images_with_metadata(exemplar_data, class_name, representative_indices, 
                                     wsi_dir, patch_size, use_openslide):
    """Load patch images and metadata for interactive display."""
    if exemplar_data is None:
        return None
    
    if class_name not in exemplar_data:
        print(f"Warning: Class '{class_name}' not found in exemplar data")
        return None
    
    feature_list = exemplar_data[class_name]
    
    # Load images and metadata for each subcluster
    patch_data = {}
    for cluster_idx, sample_indices in representative_indices.items():
        cluster_data = []
        for rank, idx in enumerate(sample_indices, 1):
            if idx >= len(feature_list):
                continue
            
            item = feature_list[idx]
            wsi_name = item['name']
            x, y = item['coords']
            
            wsi_path = os.path.join(wsi_dir, wsi_name)
            
            try:
                patch_img = extract_patch_image(wsi_path, x, y, patch_size, use_openslide)
                if patch_img is not None:
                    # Convert to base64 for HTML embedding
                    img_base64 = image_to_base64(patch_img)
                    
                    cluster_data.append({
                        'image_base64': img_base64,
                        'wsi_name': Path(wsi_name).stem,
                        'coords': (int(x), int(y)),
                        'rank': rank,
                        'index': idx
                    })
            except Exception as e:
                print(f"Warning: Failed to extract patch from {wsi_name} at ({x}, {y}): {e}")
                continue
        
        patch_data[cluster_idx] = cluster_data
    
    return patch_data


def create_interactive_grid(label_data, patch_data_by_class):
    """Create interactive grid of prototypes using plotly."""
    k_list = label_data['k_list']
    nk = label_data['nk']
    class_order = label_data['class_order']
    
    # Create HTML for grid layout
    html_sections = []
    
    for class_idx, class_name in enumerate(class_order):
        k = k_list[class_idx]
        patch_data = patch_data_by_class.get(class_name, {})
        
        # Create grid for this class
        grid_html = f"""
        <div class="class-section">
            <h2 class="class-title">{class_name.upper()} (K={k} subclusters, Nk={nk} representatives each)</h2>
            <div class="prototype-grid" style="display: grid; grid-template-columns: repeat({k}, 1fr); gap: 20px;">
        """
        
        for sub_idx in range(k):
            cluster_data = patch_data.get(sub_idx, [])
            
            grid_html += f"""
            <div class="subcluster-column">
                <h3 class="subcluster-title">Subcluster {sub_idx}</h3>
                <div class="patches-column">
            """
            
            for rank in range(nk):
                if rank < len(cluster_data):
                    data = cluster_data[rank]
                    grid_html += f"""
                    <div class="patch-container">
                        <img src="{data['image_base64']}" class="patch-image" 
                             title="WSI: {data['wsi_name']}, Coords: {data['coords']}, Rank: {data['rank']}">
                        <div class="patch-label">{data['wsi_name'][:12]}</div>
                        <div class="patch-rank">Rank {data['rank']}</div>
                    </div>
                    """
                else:
                    grid_html += """
                    <div class="patch-container">
                        <div class="patch-placeholder">N/A</div>
                    </div>
                    """
            
            grid_html += """
                </div>
            </div>
            """
        
        grid_html += """
            </div>
        </div>
        """
        
        html_sections.append(grid_html)
    
    return "\n".join(html_sections)


def create_feature_space_plot(label_data, method='tsne'):
    """Create interactive feature space scatter plot."""
    if method == 'tsne' and not HAS_TSNE:
        return None
    if method == 'umap' and not HAS_UMAP:
        return None
    
    features = label_data['features'].cpu().numpy()
    k_list = label_data['k_list']
    nk = label_data['nk']
    class_order = label_data['class_order']
    cumsum_k = label_data['cumsum_k']
    
    # Generate labels
    class_labels = []
    subcluster_labels = []
    hover_texts = []
    
    for class_idx, class_name in enumerate(class_order):
        k = k_list[class_idx]
        for sub_idx in range(k):
            for rank in range(nk):
                class_labels.append(class_name)
                subcluster_labels.append(f'{class_name}_Sub{sub_idx}')
                hover_texts.append(f"Class: {class_name}<br>Subcluster: {sub_idx}<br>Rank: {rank + 1}")
    
    # Dimensionality reduction
    print(f"Computing {method.upper()} projection...")
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    else:  # umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(features) - 1))
    
    embedding = reducer.fit_transform(features)
    
    # Create interactive scatter plot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'{method.upper()} Projection: By Class',
            f'{method.upper()} Projection: By Subcluster'
        ),
        horizontal_spacing=0.1
    )
    
    # Plot 1: Color by class
    for class_name in class_order:
        mask = [cl == class_name for cl in class_labels]
        x_vals = embedding[mask, 0]
        y_vals = embedding[mask, 1]
        hovers = [hover_texts[i] for i, m in enumerate(mask) if m]
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                name=class_name,
                text=hovers,
                hovertemplate='<b>%{text}</b><br>X: %{x:.6f}<br>Y: %{y:.6f}<extra></extra>',
                marker=dict(size=10, line=dict(width=1, color='white'))
            ),
            row=1, col=1
        )
    
    # Plot 2: Color by subcluster
    unique_subclusters = sorted(set(subcluster_labels))
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    
    for i, subcluster in enumerate(unique_subclusters):
        mask = [sc == subcluster for sc in subcluster_labels]
        x_vals = embedding[mask, 0]
        y_vals = embedding[mask, 1]
        hovers = [hover_texts[j] for j, m in enumerate(mask) if m]
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                name=subcluster,
                text=hovers,
                hovertemplate='<b>%{text}</b><br>X: %{x:.6f}<br>Y: %{y:.6f}<extra></extra>',
                marker=dict(size=10, color=colors[i % len(colors)], line=dict(width=1, color='white'))
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text=f"{method.upper()} Dimension 1", row=1, col=1)
    fig.update_yaxes(title_text=f"{method.upper()} Dimension 2", row=1, col=1)
    fig.update_xaxes(title_text=f"{method.upper()} Dimension 1", row=1, col=2)
    fig.update_yaxes(title_text=f"{method.upper()} Dimension 2", row=1, col=2)
    
    fig.update_layout(
        title_text=f'Prototype Feature Space ({method.upper()})',
        title_font_size=20,
        height=600,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig


def create_similarity_heatmap(label_data):
    """Create interactive similarity heatmap."""
    features = label_data['features'].cpu().numpy()
    k_list = label_data['k_list']
    nk = label_data['nk']
    class_order = label_data['class_order']
    cumsum_k = label_data['cumsum_k']
    
    # Normalize features
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity
    similarity_matrix = np.dot(features_norm, features_norm.T)
    
    # Create hover text
    hover_text = []
    for i in range(len(features)):
        row = []
        for j in range(len(features)):
            row.append(f"Prototype {i} ↔ Prototype {j}<br>Similarity: {similarity_matrix[i, j]:.3f}")
        hover_text.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorscale='RdYlBu_r',
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Cosine<br>Similarity")
    ))
    
    # Add class boundary annotations
    shapes = []
    for class_idx in range(len(class_order)):
        boundary = cumsum_k[class_idx + 1] * nk - 0.5
        if boundary < len(features):
            shapes.append(dict(type='line', x0=-0.5, x1=len(features)-0.5, 
                             y0=boundary, y1=boundary, line=dict(color='black', width=2)))
            shapes.append(dict(type='line', y0=-0.5, y1=len(features)-0.5,
                             x0=boundary, x1=boundary, line=dict(color='black', width=2)))
    
    fig.update_layout(
        title='Prototype Similarity Matrix (Cosine)',
        title_font_size=20,
        xaxis_title='Prototype Index',
        yaxis_title='Prototype Index',
        height=700,
        shapes=shapes
    )
    
    return fig


def create_statistics_plot(label_data):
    """Create interactive statistics dashboard."""
    k_list = label_data['k_list']
    nk = label_data['nk']
    class_order = label_data['class_order']
    features = label_data['features'].cpu().numpy()
    cumsum_k = label_data['cumsum_k']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Prototypes per Class',
            'Clustering Parameters',
            'Feature Diversity per Class',
            'Prototype Bank Summary'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'table'}]]
    )
    
    # Plot 1: Prototypes per class
    class_counts = [k * nk for k in k_list]
    fig.add_trace(
        go.Bar(
            x=class_order,
            y=class_counts,
            name='Prototypes',
            marker_color=['skyblue', 'salmon'][:len(class_order)],
            text=class_counts,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Prototypes: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Plot 2: Clustering parameters
    fig.add_trace(
        go.Bar(
            x=class_order,
            y=k_list,
            name='K (subclusters)',
            marker_color='steelblue',
            text=k_list,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Subclusters: %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Plot 3: Feature diversity
    class_variances = []
    for class_idx in range(len(class_order)):
        start_idx = cumsum_k[class_idx] * nk
        end_idx = cumsum_k[class_idx + 1] * nk
        class_features = features[start_idx:end_idx]
        variance = np.var(class_features, axis=0).mean()
        class_variances.append(variance)
    
    fig.add_trace(
        go.Bar(
            x=class_order,
            y=class_variances,
            name='Mean Variance',
            marker_color=['lightblue', 'lightcoral'][:len(class_order)],
            text=[f'{v:.6f}' for v in class_variances],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Variance: %{y:.6f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Plot 4: Summary table
    summary_headers = ['Metric', 'Value']
    summary_values = [
        ['Total Prototypes', features.shape[0]],
        ['Feature Dimension', features.shape[1]],
        ['Nk (representatives)', nk],
        ['Total Subclusters', sum(k_list)],
    ]
    
    for class_idx, class_name in enumerate(class_order):
        summary_values.append([f'{class_name} - K', k_list[class_idx]])
        summary_values.append([f'{class_name} - Prototypes', k_list[class_idx] * nk])
    
    fig.add_trace(
        go.Table(
            header=dict(values=summary_headers, fill_color='paleturquoise', font=dict(size=14, color='black')),
            cells=dict(values=list(zip(*summary_values)), fill_color='lavender', font=dict(size=12))
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Class", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Class", row=1, col=2)
    fig.update_yaxes(title_text="K", row=1, col=2)
    fig.update_xaxes(title_text="Class", row=2, col=1)
    fig.update_yaxes(title_text="Variance", row=2, col=1)
    
    fig.update_layout(
        title_text='Prototype Bank Statistics',
        title_font_size=20,
        height=800,
        showlegend=False
    )
    
    return fig


def generate_html_dashboard(label_data, patch_data_by_class, feature_plot, heatmap_plot, stats_plot, save_path):
    """Generate complete HTML dashboard."""
    
    # Generate grid HTML
    grid_html = create_interactive_grid(label_data, patch_data_by_class) if patch_data_by_class else "<p>No patch images available</p>"
    
    # Convert plotly figures to HTML
    feature_plot_html = feature_plot.to_html(include_plotlyjs=False, div_id="feature-plot") if feature_plot else ""
    heatmap_plot_html = heatmap_plot.to_html(include_plotlyjs=False, div_id="heatmap-plot") if heatmap_plot else ""
    stats_plot_html = stats_plot.to_html(include_plotlyjs=False, div_id="stats-plot") if stats_plot else ""
    
    # Create HTML template
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prototype Bank Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .section {{
            margin: 40px 0;
        }}
        .section-title {{
            font-size: 24px;
            font-weight: bold;
            color: #444;
            margin-bottom: 20px;
            border-left: 5px solid #4CAF50;
            padding-left: 15px;
        }}
        .class-section {{
            margin: 30px 0;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 8px;
        }}
        .class-title {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        .prototype-grid {{
            display: grid;
            gap: 20px;
        }}
        .subcluster-column {{
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .subcluster-title {{
            font-size: 16px;
            font-weight: bold;
            color: #34495e;
            margin-bottom: 10px;
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }}
        .patches-column {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .patch-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .patch-image {{
            width: 112px;
            height: 112px;
            border: 2px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .patch-image:hover {{
            transform: scale(1.1);
            border-color: #4CAF50;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .patch-label {{
            font-size: 10px;
            color: #666;
            margin-top: 5px;
            text-align: center;
        }}
        .patch-rank {{
            font-size: 9px;
            color: #999;
            font-style: italic;
        }}
        .patch-placeholder {{
            width: 112px;
            height: 112px;
            background-color: #e0e0e0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
            border-radius: 4px;
        }}
        .plot-container {{
            margin: 20px 0;
        }}
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #ddd;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            background-color: #f0f0f0;
            border: none;
            border-radius: 5px 5px 0 0;
            font-size: 16px;
            transition: background-color 0.3s;
        }}
        .tab:hover {{
            background-color: #e0e0e0;
        }}
        .tab.active {{
            background-color: #4CAF50;
            color: white;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Prototype Bank Interactive Dashboard</h1>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('grid')">Prototype Grid</button>
            <button class="tab" onclick="showTab('features')">Feature Space</button>
            <button class="tab" onclick="showTab('heatmap')">Similarity Matrix</button>
            <button class="tab" onclick="showTab('stats')">Statistics</button>
        </div>
        
        <div id="grid" class="tab-content active">
            <div class="section-title">Representative Samples per Subcluster</div>
            {grid_html}
        </div>
        
        <div id="features" class="tab-content">
            <div class="section-title">Feature Space Projections</div>
            <div class="plot-container">
                {feature_plot_html}
            </div>
        </div>
        
        <div id="heatmap" class="tab-content">
            <div class="section-title">Prototype Similarity Matrix</div>
            <div class="plot-container">
                {heatmap_plot_html}
            </div>
        </div>
        
        <div id="stats" class="tab-content">
            <div class="section-title">Cluster Statistics</div>
            <div class="plot-container">
                {stats_plot_html}
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
    """
    
    # Write to file
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"✓ Saved interactive dashboard to: {save_path}")


def visualize_prototypes_interactive(cfg, uid=None, skip_images=False, open_browser=True):
    """Main visualization function for interactive HTML."""
    
    # Determine UID
    if uid is None:
        uid = build_uid_from_config(cfg)
        print(f"Generated UID from config: {uid}")
    
    # Inject UID into config
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
    print(f"Generating interactive visualizations...")
    print(f"Output directory: {vis_dir}")
    print(f"{'='*70}\n")
    
    # Load patch images if available
    patch_data_by_class = {}
    if not skip_images and exemplar_data is not None:
        wsi_dir = cfg.dataset.wsi_dir
        patch_size = getattr(cfg.dataset, 'patch_size', 224)
        use_openslide = getattr(cfg.dataset, 'use_openslide', HAS_OPENSLIDE)
        class_order = label_data['class_order']
        representative_indices = label_data['representative_indices']
        
        print("Loading patch images for interactive display...")
        for class_name in class_order:
            if class_name in representative_indices:
                print(f"  Loading {class_name} patches...")
                patch_data = load_patch_images_with_metadata(
                    exemplar_data, class_name, representative_indices[class_name],
                    wsi_dir, patch_size, use_openslide
                )
                if patch_data:
                    patch_data_by_class[class_name] = patch_data
        print()
    
    # Generate interactive plots
    print("Generating interactive plots...\n")
    
    # Feature space plot
    feature_plot = None
    if HAS_TSNE:
        feature_plot = create_feature_space_plot(label_data, method='tsne')
        print("✓ Created t-SNE feature space plot")
    elif HAS_UMAP:
        feature_plot = create_feature_space_plot(label_data, method='umap')
        print("✓ Created UMAP feature space plot")
    
    # Similarity heatmap
    heatmap_plot = create_similarity_heatmap(label_data)
    print("✓ Created similarity heatmap")
    
    # Statistics plot
    stats_plot = create_statistics_plot(label_data)
    print("✓ Created statistics dashboard")
    
    # Generate HTML dashboard
    dashboard_path = vis_dir / 'prototype_dashboard.html'
    print("\nGenerating HTML dashboard...")
    generate_html_dashboard(label_data, patch_data_by_class, feature_plot, 
                           heatmap_plot, stats_plot, dashboard_path)
    
    print(f"\n{'='*70}")
    print(f"✓ Interactive dashboard saved to: {dashboard_path}")
    print(f"{'='*70}\n")
    
    # Open in browser
    if open_browser:
        import webbrowser
        print(f"Opening dashboard in browser...")
        webbrowser.open(f'file://{dashboard_path.absolute()}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interactive HTML prototype bank visualization")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--uid', type=str, default=None, 
                       help='Run UID (auto-detected from config if not provided)')
    parser.add_argument('--skip_images', action='store_true',
                       help='Skip loading patch images (faster, feature plots only)')
    parser.add_argument('--no_browser', action='store_true',
                       help='Do not automatically open browser')
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    visualize_prototypes_interactive(cfg, uid=args.uid, skip_images=args.skip_images, 
                                    open_browser=not args.no_browser)
