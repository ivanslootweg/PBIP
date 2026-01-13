#!/bin/bash

# Script to filter split_csv to only include images present in labels_csv
# Only keeps train/val/test columns and filters to available images
# Usage: bash create_split_dummy.sh

set -e

# File paths
LABELS_CSV="/data/pathology/projects/ivan/WSS/labels_dummy.csv"
SPLIT_CSV="/data/pathology/projects/ivan/DeepDerma/documents/classifier_splits/mohs_based_on_2024_test/splits_0.csv"
OUTPUT_CSV="/data/pathology/projects/ivan/WSS/split_dummy.csv"

echo "Creating filtered split CSV..."
echo "  Labels CSV: $LABELS_CSV"
echo "  Split CSV: $SPLIT_CSV"
echo "  Output CSV: $OUTPUT_CSV"
echo ""

# Extract image names from labels CSV (skip header, get first column)
echo "Reading available images from labels CSV..."
TEMP_IMAGES=$(mktemp)
tail -n +2 "$LABELS_CSV" | cut -d',' -f1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' > "$TEMP_IMAGES"
NUM_AVAILABLE=$(wc -l < "$TEMP_IMAGES")
echo "Found $NUM_AVAILABLE available images"

# Filter using Python for better CSV handling
python3 << 'PYTHON_SCRIPT'
import pandas as pd
import sys

labels_csv = "/data/pathology/projects/ivan/WSS/labels_dummy.csv"
split_csv = "/data/pathology/projects/ivan/DeepDerma/documents/classifier_splits/mohs_based_on_2024_test/splits_0.csv"
output_csv = "/data/pathology/projects/ivan/WSS/split_dummy.csv"

# Read available images
labels_df = pd.read_csv(labels_csv)
available = set(labels_df['image_name'].str.strip().unique())
print(f"Available images: {len(available)}")

# Read split CSV
split_df = pd.read_csv(split_csv)
print(f"Split CSV columns: {list(split_df.columns)}")

# Only keep train/val/test columns
cols_to_keep = [col for col in ['train', 'val', 'test'] if col in split_df.columns]
split_df = split_df[cols_to_keep].copy()

print(f"Using columns: {cols_to_keep}")

# Filter each column to only include available images
for col in cols_to_keep:
    def filter_image(x):
        if pd.isna(x):
            return None
        img_name = str(x).strip()
        if img_name in available:
            return img_name
        return None
    
    split_df[col] = split_df[col].apply(filter_image)

# Remove rows that are completely empty
split_df = split_df.dropna(how='all')

# Save result
split_df.to_csv(output_csv, index=False)

# Report
print(f"\nFiltered results:")
for col in cols_to_keep:
    non_null = split_df[col].notna().sum()
    print(f"  {col}: {non_null} samples")

total_unique = split_df.stack().nunique()
print(f"  Total unique images: {total_unique}")
print(f"\n✓ Output saved to: {output_csv}")
print(f"\nFirst 5 rows:")
print(split_df.head())

PYTHON_SCRIPT

# Cleanup
rm -f "$TEMP_IMAGES"

