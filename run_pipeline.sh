#!/bin/bash
# PBIP End-to-End Training Pipeline
# This script runs the complete pipeline from prototype extraction to training
export PYTHONPATH="${PYTHONPATH}:/data/pathology/projects/ivan/cloned_tools/PBIP"
set -e  # Exit on error

echo "=========================================="
echo "PBIP Training Pipeline"
echo "=========================================="

# Configuration
CONFIG_FILE="./work_dirs/custom_wsi_template.yaml"
GPU_ID=0
K_LIST="3 3"  # k-means clusters per class [benign, tumor]

# Step 1: Extract prototype coordinates from ground truth masks
echo ""
echo "=========================================="
echo "Step 1/4: Extracting Patch Coordinates"
echo "=========================================="
# python3 features/extract_patches.py \
#     --config work_dirs/custom_wsi_template.yaml \
#     --num_per_wsi 1000 \
#     --num_wsis_per_class 100 \
#     --generate_thumbnails \
#     --num_thumbnail_samples 5

python3 features/extract_patches.py \
    --config work_dirs/custom_wsi_template.yaml \
    --num_per_wsi 100 \
    --num_wsis_per_class 50 \
    --generate_thumbnails \
    --num_thumbnail_samples 5

echo "✓ Patch coordinates extracted and thumbnails generated"

# Step 2: Extract MedCLIP features from prototype images
echo ""
echo "=========================================="
echo "Step 2/4: Extracting MedCLIP Features"
echo "=========================================="
python3 features/excract_medclip_proces.py --config work_dirs/custom_wsi_template.yaml
#     --proto_dir /data/pathology/projects/ivan/WSS/PBIP/prototypes \
#     --output_dir /data/pathology/projects/ivan/WSS/PBIP/image_features

echo "✓ MedCLIP features extracted"

# Step 3: Run k-means clustering to generate label features
echo ""
echo "=========================================="
echo "Step 3/4: K-means Clustering (k_list=$K_LIST)"
echo "=========================================="
python3 features/k_mean_cos_per_class.py  --config work_dirs/custom_wsi_template.yaml
    # --k_list $K_LIST \
    # --input_pkl /data/pathology/projects/ivan/WSS/PBIP/image_features/medclip_exemplars.pkl \
    # --output_dir /data/pathology/projects/ivan/WSS/PBIP/image_features

echo "✓ K-means clustering completed"

# Step 4: Train the model
echo ""
echo "=========================================="
echo "Step 4/4: Training PBIP Model"
echo "=========================================="
python3 train_stage_1.py \
    --config $CONFIG_FILE \
    --gpu $GPU_ID

echo ""
echo "=========================================="
echo "Pipeline Completed Successfully!"
echo "=========================================="
echo "Results saved to:"
echo "  - Checkpoints: /data/pathology/projects/ivan/WSS/PBIP/checkpoints/"
echo "  - Predictions: /data/pathology/projects/ivan/WSS/PBIP/predictions/"
echo "  - Logs: /data/pathology/projects/ivan/WSS/PBIP/training_logs/"
