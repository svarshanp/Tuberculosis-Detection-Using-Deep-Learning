"""
Data Preprocessing for Tuberculosis Detection
- Scans dataset, removes corrupt images
- Splits into train/val/test (stratified)
- Computes class weights
- Saves metadata CSV
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "archive (8)", "Dataset of Tuberculosis Chest X-rays Images")
NORMAL_DIR = os.path.join(DATASET_DIR, "Normal Chest X-rays")
TB_DIR = os.path.join(DATASET_DIR, "TB Chest X-rays")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

IMG_SIZE = (224, 224)
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


def scan_and_validate_images(directory, label):
    """Scan directory for valid images, skip corrupt ones."""
    valid_images = []
    corrupt_count = 0

    if not os.path.exists(directory):
        print(f"WARNING: Directory not found: {directory}")
        return valid_images, corrupt_count

    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        ext = os.path.splitext(fname)[1].lower()

        if ext not in VALID_EXTENSIONS:
            continue

        try:
            with Image.open(fpath) as img:
                img.verify()
            valid_images.append({'filepath': fpath, 'filename': fname, 'label': label, 'label_name': 'TB' if label == 1 else 'Normal'})
        except Exception:
            corrupt_count += 1
            print(f"  Corrupt image skipped: {fname}")

    return valid_images, corrupt_count


def create_splits(df, test_size=0.1, val_size=0.1, random_state=42):
    """Create stratified train/val/test splits."""
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )

    # Second split: train vs val (val_size relative to train+val)
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, random_state=random_state, stratify=train_val_df['label']
    )

    return train_df, val_df, test_df


def compute_class_weights(df):
    """Compute class weights to handle imbalance."""
    counts = df['label'].value_counts().to_dict()
    total = len(df)
    n_classes = len(counts)
    weights = {}
    for cls, count in counts.items():
        weights[cls] = total / (n_classes * count)
    return weights


def copy_split_images(df, split_name):
    """Copy images into organized split directories."""
    split_dir = os.path.join(DATA_DIR, split_name)
    for label_name in ['Normal', 'TB']:
        label_dir = os.path.join(split_dir, label_name)
        os.makedirs(label_dir, exist_ok=True)

    for _, row in df.iterrows():
        src = row['filepath']
        dst_dir = os.path.join(split_dir, row['label_name'])
        dst = os.path.join(dst_dir, row['filename'])
        if not os.path.exists(dst):
            shutil.copy2(src, dst)


def get_image_stats(df, sample_size=200):
    """Get image dimension and channel statistics from a sample."""
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    widths, heights, channels = [], [], []

    for _, row in sample.iterrows():
        try:
            with Image.open(row['filepath']) as img:
                w, h = img.size
                c = len(img.getbands())
                widths.append(w)
                heights.append(h)
                channels.append(c)
        except Exception:
            continue

    return {
        'width_mean': np.mean(widths), 'width_std': np.std(widths),
        'height_mean': np.mean(heights), 'height_std': np.std(heights),
        'width_min': int(np.min(widths)), 'width_max': int(np.max(widths)),
        'height_min': int(np.min(heights)), 'height_max': int(np.max(heights)),
        'channels': dict(pd.Series(channels).value_counts()),
    }


def main():
    print("=" * 60)
    print("TUBERCULOSIS DETECTION - DATA PREPROCESSING")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1: Scan and validate images
    print("\n[1/5] Scanning and validating images...")
    normal_images, normal_corrupt = scan_and_validate_images(NORMAL_DIR, label=0)
    tb_images, tb_corrupt = scan_and_validate_images(TB_DIR, label=1)

    print(f"  Normal: {len(normal_images)} valid, {normal_corrupt} corrupt")
    print(f"  TB:     {len(tb_images)} valid, {tb_corrupt} corrupt")
    print(f"  Total:  {len(normal_images) + len(tb_images)} valid images")

    # Step 2: Create DataFrame
    print("\n[2/5] Creating dataset DataFrame...")
    all_images = normal_images + tb_images
    df = pd.DataFrame(all_images)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Class distribution:\n{df['label_name'].value_counts().to_string()}")

    # Step 3: Image statistics
    print("\n[3/5] Computing image statistics...")
    stats = get_image_stats(df)
    print(f"  Avg dimensions: {stats['width_mean']:.0f} x {stats['height_mean']:.0f}")
    print(f"  Range: ({stats['width_min']}-{stats['width_max']}) x ({stats['height_min']}-{stats['height_max']})")

    # Step 4: Create splits
    print("\n[4/5] Creating stratified train/val/test splits (80/10/10)...")
    train_df, val_df, test_df = create_splits(df)

    print(f"  Train: {len(train_df)} images")
    print(f"    Normal: {(train_df['label'] == 0).sum()}, TB: {(train_df['label'] == 1).sum()}")
    print(f"  Val:   {len(val_df)} images")
    print(f"    Normal: {(val_df['label'] == 0).sum()}, TB: {(val_df['label'] == 1).sum()}")
    print(f"  Test:  {len(test_df)} images")
    print(f"    Normal: {(test_df['label'] == 0).sum()}, TB: {(test_df['label'] == 1).sum()}")

    # Compute class weights
    class_weights = compute_class_weights(train_df)
    print(f"\n  Class weights: {class_weights}")

    # Step 5: Copy images to split directories
    print("\n[5/5] Organizing images into split directories...")
    copy_split_images(train_df, 'train')
    copy_split_images(val_df, 'val')
    copy_split_images(test_df, 'test')
    print("  Done!")

    # Save metadata
    train_df.to_csv(os.path.join(DATA_DIR, 'train_metadata.csv'), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, 'val_metadata.csv'), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, 'test_metadata.csv'), index=False)
    df.to_csv(os.path.join(DATA_DIR, 'full_metadata.csv'), index=False)

    # Save preprocessing info
    info = {
        'total_images': len(df),
        'normal_count': len(normal_images),
        'tb_count': len(tb_images),
        'corrupt_normal': normal_corrupt,
        'corrupt_tb': tb_corrupt,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'class_weights': {str(k): v for k, v in class_weights.items()},
        'image_stats': {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in stats.items()},
        'img_size': list(IMG_SIZE),
    }
    with open(os.path.join(DATA_DIR, 'preprocessing_info.json'), 'w') as f:
        json.dump(info, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("DATA PREPROCESSING COMPLETE!")
    print(f"Metadata saved to: {DATA_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
