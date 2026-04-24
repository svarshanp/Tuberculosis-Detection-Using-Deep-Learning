"""
Exploratory Data Analysis for Tuberculosis Detection
- Class distribution, sample images, pixel intensity, mean images
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
IMG_SIZE = (224, 224)

sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams.update({'figure.facecolor': '#0e1117', 'axes.facecolor': '#1a1a2e',
                      'text.color': 'white', 'axes.labelcolor': 'white',
                      'xtick.color': 'white', 'ytick.color': 'white',
                      'figure.dpi': 120, 'savefig.bbox': 'tight'})


def plot_class_distribution(df):
    """Plot class distribution bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    counts = df['label_name'].value_counts()
    colors = ['#00d4aa', '#ff6b6b']
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_title('Class Distribution', fontsize=16, fontweight='bold', color='white')
    axes[0].set_ylabel('Number of Images', fontsize=12)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                     str(val), ha='center', va='bottom', fontsize=14, fontweight='bold', color='white')

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors,
                textprops={'color': 'white', 'fontsize': 13}, startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    axes[1].set_title('Class Proportions', fontsize=16, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'class_distribution.png'))
    plt.close()
    print("  Saved: class_distribution.png")


def plot_sample_images(df, n_samples=5):
    """Plot sample images from each class."""
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 8))

    for idx, label_name in enumerate(['Normal', 'TB']):
        subset = df[df['label_name'] == label_name].sample(n=n_samples, random_state=42)
        for j, (_, row) in enumerate(subset.iterrows()):
            img = cv2.imread(row['filepath'])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                axes[idx, j].imshow(img)
            axes[idx, j].axis('off')
            if j == 0:
                axes[idx, j].set_ylabel(label_name, fontsize=16, fontweight='bold', color='white', rotation=0, labelpad=60)

    fig.suptitle('Sample Chest X-Ray Images', fontsize=20, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sample_images.png'))
    plt.close()
    print("  Saved: sample_images.png")


def plot_image_dimensions(df, sample_size=500):
    """Plot distribution of image dimensions."""
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    widths, heights, labels = [], [], []

    for _, row in sample.iterrows():
        try:
            with Image.open(row['filepath']) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                labels.append(row['label_name'])
        except Exception:
            continue

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Width distribution
    axes[0].hist([w for w, l in zip(widths, labels) if l == 'Normal'],
                 bins=30, alpha=0.7, color='#00d4aa', label='Normal')
    axes[0].hist([w for w, l in zip(widths, labels) if l == 'TB'],
                 bins=30, alpha=0.7, color='#ff6b6b', label='TB')
    axes[0].set_title('Width Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Width (px)')
    axes[0].legend()

    # Height distribution
    axes[1].hist([h for h, l in zip(heights, labels) if l == 'Normal'],
                 bins=30, alpha=0.7, color='#00d4aa', label='Normal')
    axes[1].hist([h for h, l in zip(heights, labels) if l == 'TB'],
                 bins=30, alpha=0.7, color='#ff6b6b', label='TB')
    axes[1].set_title('Height Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Height (px)')
    axes[1].legend()

    # Scatter
    for lbl, color in [('Normal', '#00d4aa'), ('TB', '#ff6b6b')]:
        mask = [l == lbl for l in labels]
        w_f = [w for w, m in zip(widths, mask) if m]
        h_f = [h for h, m in zip(heights, mask) if m]
        axes[2].scatter(w_f, h_f, alpha=0.5, color=color, label=lbl, s=15)
    axes[2].set_title('Width vs Height', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Width (px)')
    axes[2].set_ylabel('Height (px)')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'image_dimensions.png'))
    plt.close()
    print("  Saved: image_dimensions.png")


def plot_pixel_intensity(df, sample_size=100):
    """Plot pixel intensity histograms per class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (label_name, color) in enumerate([('Normal', '#00d4aa'), ('TB', '#ff6b6b')]):
        subset = df[df['label_name'] == label_name].sample(n=min(sample_size, (df['label_name'] == label_name).sum()), random_state=42)
        all_pixels = []
        for _, row in subset.iterrows():
            try:
                img = cv2.imread(row['filepath'], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (64, 64))  # smaller for speed
                    all_pixels.extend(img.flatten().tolist())
            except Exception:
                continue

        axes[idx].hist(all_pixels, bins=50, color=color, alpha=0.8, density=True, edgecolor='white', linewidth=0.3)
        axes[idx].set_title(f'{label_name} - Pixel Intensity', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Pixel Value')
        axes[idx].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pixel_intensity.png'))
    plt.close()
    print("  Saved: pixel_intensity.png")


def plot_mean_images(df, sample_size=200):
    """Compute and plot mean image per class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, label_name in enumerate(['Normal', 'TB']):
        subset = df[df['label_name'] == label_name].sample(n=min(sample_size, (df['label_name'] == label_name).sum()), random_state=42)
        images = []
        for _, row in subset.iterrows():
            try:
                img = cv2.imread(row['filepath'], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    images.append(img.astype(np.float32))
            except Exception:
                continue

        if images:
            mean_img = np.mean(images, axis=0)
            axes[idx].imshow(mean_img, cmap='bone')
            axes[idx].set_title(f'Mean {label_name} X-Ray', fontsize=14, fontweight='bold')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'mean_images.png'))
    plt.close()
    print("  Saved: mean_images.png")


def plot_split_distribution(data_dir):
    """Plot the train/val/test split distribution."""
    splits = {}
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(data_dir, f'{split}_metadata.csv')
        if os.path.exists(csv_path):
            split_df = pd.read_csv(csv_path)
            splits[split] = split_df['label_name'].value_counts().to_dict()

    if not splits:
        print("  No split metadata found, skipping split distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(splits))
    width = 0.35

    normal_counts = [splits[s].get('Normal', 0) for s in splits]
    tb_counts = [splits[s].get('TB', 0) for s in splits]

    bars1 = ax.bar(x - width/2, normal_counts, width, label='Normal', color='#00d4aa', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, tb_counts, width, label='TB', color='#ff6b6b', edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Split', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Dataset Split Distribution', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits.keys()])
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 5, str(int(h)),
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'split_distribution.png'))
    plt.close()
    print("  Saved: split_distribution.png")


def main():
    print("=" * 60)
    print("TUBERCULOSIS DETECTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load metadata
    metadata_path = os.path.join(DATA_DIR, 'full_metadata.csv')
    if not os.path.exists(metadata_path):
        print("ERROR: Run data_preprocessing.py first!")
        return

    df = pd.read_csv(metadata_path)
    print(f"\nDataset: {len(df)} images")
    print(f"Classes: {df['label_name'].value_counts().to_dict()}")

    print("\n[1/6] Plotting class distribution...")
    plot_class_distribution(df)

    print("[2/6] Plotting sample images...")
    plot_sample_images(df)

    print("[3/6] Plotting image dimensions...")
    plot_image_dimensions(df)

    print("[4/6] Plotting pixel intensity...")
    plot_pixel_intensity(df)

    print("[5/6] Computing mean images...")
    plot_mean_images(df)

    print("[6/6] Plotting split distribution...")
    plot_split_distribution(DATA_DIR)

    print("\n" + "=" * 60)
    print("EDA COMPLETE! All plots saved to:", RESULTS_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
