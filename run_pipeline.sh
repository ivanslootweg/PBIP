#!/bin/bash

# Stop execution if any command fails
set -e

# Configuration file (default to config.yaml if not provided)
CONFIG=${1:-config.yaml}

if [ ! -f "$CONFIG" ]; then
    echo "Error: Configuration file '$CONFIG' not found!"
    exit 1
fi

echo "============================================================"
echo "Starting PBIP Pipeline with config: $CONFIG"
echo "============================================================"

# Step 1: Feature Extraction
echo ""
echo "[Step 1/5] Extracting Prototypes (Features)..."
python3 extract_patches.py --config "$CONFIG"

# Step 2: Clustering
echo ""
echo "[Step 2/6] Clustering Prototypes..."
python3 features/k_mean_cos_per_class.py --config "$CONFIG"

# Step 3: TSNE Visualization
echo ""
echo "[Step 3/6] Generating TSNE Visualization..."
python3 visualize_tsne.py --config "$CONFIG"

# Step 4: Prototype Patch Visualization (Optional but recommended)
echo ""
echo "[Step 4/6] Visualizing Prototype Patches..."
python3 visualize_prototypes.py --config "$CONFIG"

# Step 5: Training
echo ""
echo "[Step 5/6] Training Patch Refinement..."
python3 train_patch_refinement.py --config "$CONFIG"

# Step 6: Evaluation
echo ""
echo "[Step 6/6] Evaluating Test Performance..."
python3 evaluate_test.py --config "$CONFIG"

echo ""
echo "============================================================"
echo "Pipeline Completed Successfully!"
echo "============================================================"
