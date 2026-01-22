#!/bin/bash
###############################################################################
# Full Feature-Based Training Pipeline for PBIP
# 
# This script runs the complete pipeline:
# 0. Extract patch coordinates from prototypes
# 1. Extract patch features using configurable encoder (Virchow2 or DinoV3)
# 2. Cluster features into prototype bank (k-means)
# 3. Train patch-level prototype refinement model
# 4. Visualize prototype bank (static plots)
# 5. Generate interactive prototype dashboard (HTML)
# 6. Run pipeline diagnostics
# 7. Evaluate on test set
#
# Usage:
#   bash pipeline.sh --config work_dirs/custom_wsi_template.yaml --gpu 0
#
# Requirements:
#   - Config file with proper settings (patch_encoder: virchow2 or dinov3)
#   - WSI files and split CSV with labels
#   - Optional: precomputed global slide-level features (auto-generated if missing)
###############################################################################
export PYTHONPATH="${PYTHONPATH}:/data/pathology/projects/ivan/cloned-tools/PBIP"
export GPU=0
export CONFIG="work_dirs/custom_wsi_template.yaml"
set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
CONFIG=""
GPU=0
VERBOSE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE="--verbose"
      shift
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

if [ -z "$CONFIG" ]; then
  echo -e "${RED}Error: --config is required${NC}"
  echo "Usage: bash pipeline.sh --config <config.yaml> [--gpu <gpu_id>] [--verbose]"
  exit 1
fi

if [ ! -f "$CONFIG" ]; then
  echo -e "${RED}Error: Config file not found: $CONFIG${NC}"
  exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PBIP Feature-Based Training Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Config: ${GREEN}$CONFIG${NC}"
echo -e "GPU: ${GREEN}$GPU${NC}"
echo ""

# Step 0: Extract prototype patch coordinates
echo -e "${YELLOW}[STEP 0/6] Extracting prototype patch coordinates...${NC}"
export CUDA_VISIBLE_DEVICES="$GPU"
python3 -u features/extract_patches.py --config "$CONFIG" 
if [ $? -ne 0 ]; then
  echo -e "${RED}Patch extraction failed${NC}"
  exit 1
fi
echo -e "${GREEN}✓ Patch extraction complete${NC}"
echo ""

# Step 1: Extract patch features using configured encoder (Virchow2 or DinoV3)
echo -e "${YELLOW}[STEP 1/6] Extracting prototype exemplar features...${NC}"
export CUDA_VISIBLE_DEVICES="$GPU"
python3 -u features/extract_patch_features.py --config "$CONFIG"
if [ $? -ne 0 ]; then
  echo -e "${RED}Feature extraction failed. Check config patch_encoder setting.${NC}"
  exit 1
fi
echo -e "${GREEN}✓ Feature extraction complete${NC}"
echo ""

# Step 2: Cluster features into prototype bank
echo -e "${YELLOW}[STEP 2/6] Running k-means clustering...${NC}"
python3 -u features/k_mean_cos_per_class.py --config "$CONFIG"
if [ $? -ne 0 ]; then
  echo -e "${RED}K-means clustering failed${NC}"
  exit 1
fi
echo -e "${GREEN}✓ Clustering complete${NC}"
echo ""

# Step 4: Visualize prototype bank (static plots)
echo -e "${YELLOW}[STEP 3/6] Generating prototype visualizations...${NC}"
python3 -u visualize_prototypes.py --config "$CONFIG"
if [ $? -ne 0 ]; then
  echo -e "${RED}Warning: Prototype visualization failed (non-critical)${NC}"
else
  echo -e "${GREEN}✓ Prototype visualizations saved${NC}"
fi
echo ""




# Step 5: Generate interactive prototype dashboard
echo -e "${YELLOW}[STEP 4/6] Creating interactive prototype dashboard...${NC}"
python3 -u visualize_prototypes_interactive.py --config "$CONFIG"
if [ $? -ne 0 ]; then
  echo -e "${RED}Warning: Interactive dashboard failed (non-critical)${NC}"
else
  echo -e "${GREEN}✓ Interactive dashboard created${NC}"
fi
echo ""

# Step 3: Train patch refinement model (using Virchow2 or DinoV3 features)
echo -e "${YELLOW}[STEP 5/6] Training patch-level refinement model...${NC}"
python3 -u train_patch_refinement.py --config "$CONFIG" --gpu "$GPU" $VERBOSE
if [ $? -ne 0 ]; then
  echo -e "${RED}Training failed${NC}"
  exit 1
fi
echo -e "${GREEN}✓ Training complete${NC}"
echo ""



# Step 6: Evaluate on test set (if test_pipeline.py exists)
echo -e "${YELLOW}[STEP 6/6] Evaluating on test set...${NC}"
if [ -f "evaluate_test.py" ]; then
  python3 -u evaluate_test.py --config "$CONFIG" --gpu "$GPU"
  if [ $? -ne 0 ]; then
    echo -e "${RED}Warning: Test evaluation failed (non-critical)${NC}"
  else
    echo -e "${GREEN}✓ Test evaluation complete${NC}"
  fi
else
  echo -e "${YELLOW}Note: evaluate_test.py not found, skipping test evaluation${NC}"
  echo -e "${YELLOW}Use the trained checkpoint for manual evaluation${NC}"
fi
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to:"
echo "  - Checkpoints: work_dir/runs/\${run_uid}/checkpoints/"
echo "  - Predictions: work_dir/runs/\${run_uid}/predictions/"
echo "  - Visualizations: work_dir/runs/\${run_uid}/visualizations/"
echo "  - Training logs: work_dir/runs/\${run_uid}/training_logs/"
echo ""
echo "Next steps:"
echo "  - Check wandb dashboard for training metrics and visualizations"
echo "  - Review prototype dashboard in visualizations/ directory"
echo "  - Analyze test set predictions in predictions/ directory"
echo ""
