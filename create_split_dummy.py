#!/usr/bin/env python3
"""
Memory-efficient script to filter split_csv to only include images present in labels_csv.
Only keeps train/val/test columns and filters to available images.

Usage:
    python3 create_split_dummy.py [--labels LABELS_CSV] [--split SPLIT_CSV] [--output OUTPUT_CSV]
"""

import csv
import sys
from pathlib import Path
from argparse import ArgumentParser


def load_available_images(labels_csv):
    """
    Load available image names from labels CSV into a set.
    Memory efficient: reads file once, stores only image names.
    
    Args:
        labels_csv: Path to labels CSV file
        
    Returns:
        Set of available image names
    """
    available = set()
    with open(labels_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'image_name' in row:
                img_name = row['image_name'].strip()
                if img_name:
                    available.add(img_name)
    return available


def process_split_csv(split_csv, labels_csv, output_csv):
    """
    Filter split CSV to only include images in labels CSV.
    Removes empty cells and packs values to the top of each column.
    Memory efficient: processes line by line, only keeps train/val/test columns.
    
    Args:
        split_csv: Path to original split CSV
        labels_csv: Path to labels CSV
        output_csv: Path to output split CSV
    """
    # Load available images
    print(f"Loading available images from {labels_csv}...", file=sys.stderr)
    available = load_available_images(labels_csv)
    num_available = len(available)
    print(f"  Found {num_available} available images", file=sys.stderr)
    
    # Process split CSV
    print(f"\nFiltering split CSV...", file=sys.stderr)
    
    # First pass: collect all values per column
    columns_data = {'train': [], 'val': [], 'test': []}
    
    with open(split_csv, 'r') as infile:
        reader = csv.DictReader(infile)
        
        # Determine which columns to keep
        fieldnames = [col for col in ['train', 'val', 'test'] if col in reader.fieldnames]
        
        if not fieldnames:
            raise ValueError(f"No train/val/test columns found in {split_csv}")
        
        print(f"  Using columns: {fieldnames}", file=sys.stderr)
        
        # Collect all valid values from each column
        for row in reader:
            for col in fieldnames:
                value = row.get(col, '').strip() if col in row else ''
                if value and value in available:
                    columns_data[col].append(value)
    
    # Second pass: write output with packed columns
    max_rows = max(len(columns_data[col]) for col in fieldnames)
    
    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write rows with all non-empty values packed to top
        for i in range(max_rows):
            row = {}
            for col in fieldnames:
                if i < len(columns_data[col]):
                    row[col] = columns_data[col][i]
                else:
                    row[col] = ''
            writer.writerow(row)
    
    # Report results
    print(f"\nFiltered and compacted results:", file=sys.stderr)
    for col in fieldnames:
        count = len(columns_data[col])
        print(f"  {col}: {count} samples", file=sys.stderr)
    
    total_unique = sum(len(columns_data[col]) for col in fieldnames)
    print(f"  Total entries: {total_unique}", file=sys.stderr)
    print(f"  Output rows: {max_rows}", file=sys.stderr)
    print(f"\n✓ Output saved to: {output_csv}", file=sys.stderr)


def main():
    parser = ArgumentParser(description="Filter split CSV to only include labeled images")
    parser.add_argument(
        '--labels',
        default='/data/pathology/projects/ivan/WSS/labels_dummy.csv',
        help='Path to labels CSV file'
    )
    parser.add_argument(
        '--split',
        default='/data/pathology/projects/ivan/DeepDerma/documents/classifier_splits/mohs_based_on_2024_test/splits_0.csv',
        help='Path to original split CSV file'
    )
    parser.add_argument(
        '--output',
        default='/data/pathology/projects/ivan/WSS/split_dummy.csv',
        help='Path to output split CSV file'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    labels_csv = Path(args.labels)
    split_csv = Path(args.split)
    output_csv = Path(args.output)
    
    if not labels_csv.exists():
        print(f"Error: Labels CSV not found: {labels_csv}", file=sys.stderr)
        sys.exit(1)
    
    if not split_csv.exists():
        print(f"Error: Split CSV not found: {split_csv}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if needed
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Process
    try:
        process_split_csv(str(split_csv), str(labels_csv), str(output_csv))
        print(f"\n✓ Successfully created {output_csv}", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
