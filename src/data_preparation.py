import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import random

# Set random seed untuk reproducibility
random.seed(42)
np.random.seed(42)

# Konfigurasi path
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs" / "figures"

# Pastikan output directory ada
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    """
    Memuat dataset dari train.csv dan menampilkan informasi dasar.

    Returns:
        pd.DataFrame: DataFrame berisi informasi dataset
    """
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    csv_path = DATASET_DIR / "train.csv"
    df = pd.read_csv(csv_path)

    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    return df


def explore_class_distribution(df):
    """
    Mengeksplorasi dan memvisualisasikan distribusi kelas.

    Args:
        df (pd.DataFrame): DataFrame dataset
    """
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Hitung distribusi kelas
    class_counts = df["label"].value_counts().sort_index()

    print("\nClass distribution:")
    for label, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  Class {label}: {count} samples ({percentage:.2f}%)")

    # Visualisasi distribusi kelas
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    axes[0].bar(
        class_counts.index, class_counts.values, color="skyblue", edgecolor="black"
    )
    axes[0].set_xlabel("Class Label", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    axes[0].set_title("Class Distribution - Bar Chart", fontsize=14, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (label, count) in enumerate(class_counts.items()):
        axes[0].text(
            label, count + 10, str(count), ha="center", va="bottom", fontweight="bold"
        )

    # Pie chart
    colors = plt.cm.Set3(range(len(class_counts)))
    axes[1].pie(
        class_counts.values,
        labels=[f"Class {i}" for i in class_counts.index],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
    )
    axes[1].set_title("Class Distribution - Pie Chart", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300, bbox_inches="tight")
    print(
        f"\n✓ Class distribution plot saved to: {OUTPUT_DIR / 'class_distribution.png'}"
    )
    plt.close()


def visualize_sample_images(df, samples_per_class=3):
    """
    Memvisualisasikan sample gambar dari setiap kelas.

    Args:
        df (pd.DataFrame): DataFrame dataset
        samples_per_class (int): Jumlah sample per kelas yang ditampilkan
    """
    print("\n" + "=" * 60)
    print("VISUALIZING SAMPLE IMAGES")
    print("=" * 60)

    classes = sorted(df["label"].unique())
    n_classes = len(classes)

    fig, axes = plt.subplots(
        n_classes, samples_per_class, figsize=(samples_per_class * 3, n_classes * 3)
    )

    for i, class_label in enumerate(classes):
        # Ambil sample dari kelas ini
        class_samples = df[df["label"] == class_label].sample(
            n=samples_per_class, random_state=42
        )

        for j, (_, row) in enumerate(class_samples.iterrows()):
            img_path = DATASET_DIR / "train" / row["filename"]
            img = Image.open(img_path)

            ax = axes[i, j] if n_classes > 1 else axes[j]
            ax.imshow(img)
            ax.axis("off")

            if j == 0:
                ax.set_title(
                    f"Class {class_label}", fontsize=12, fontweight="bold", loc="left"
                )

    plt.suptitle(
        "Sample Images from Each Class", fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_images.png", dpi=300, bbox_inches="tight")
    print(f"\n✓ Sample images saved to: {OUTPUT_DIR / 'sample_images.png'}")
    plt.close()


def analyze_image_properties(df, n_samples=100):
    """
    Menganalisis properti gambar seperti ukuran dan aspect ratio.

    Args:
        df (pd.DataFrame): DataFrame dataset
        n_samples (int): Jumlah sample untuk analisis
    """
    print("\n" + "=" * 60)
    print("ANALYZING IMAGE PROPERTIES")
    print("=" * 60)

    # Sample random images
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)

    widths = []
    heights = []
    aspect_ratios = []

    for _, row in sample_df.iterrows():
        img_path = DATASET_DIR / "train" / row["filename"]
        try:
            img = Image.open(img_path)
            width, height = img.size
            widths.append(width)
            heights.append(height)
            aspect_ratios.append(width / height)
        except Exception as e:
            print(f"  Warning: Could not process {row['filename']}: {e}")

    # Statistik
    print(f"\nImage dimensions (based on {len(widths)} samples):")
    print(
        f"  Width  - Mean: {np.mean(widths):.1f}, Min: {np.min(widths)}, Max: {np.max(widths)}"
    )
    print(
        f"  Height - Mean: {np.mean(heights):.1f}, Min: {np.min(heights)}, Max: {np.max(heights)}"
    )
    print(
        f"  Aspect Ratio - Mean: {np.mean(aspect_ratios):.2f}, Min: {np.min(aspect_ratios):.2f}, Max: {np.max(aspect_ratios):.2f}"
    )

    # Visualisasi
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(widths, bins=20, color="lightblue", edgecolor="black")
    axes[0].set_xlabel("Width (pixels)", fontweight="bold")
    axes[0].set_ylabel("Frequency", fontweight="bold")
    axes[0].set_title("Image Width Distribution", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist(heights, bins=20, color="lightgreen", edgecolor="black")
    axes[1].set_xlabel("Height (pixels)", fontweight="bold")
    axes[1].set_ylabel("Frequency", fontweight="bold")
    axes[1].set_title("Image Height Distribution", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].hist(aspect_ratios, bins=20, color="lightcoral", edgecolor="black")
    axes[2].set_xlabel("Aspect Ratio (W/H)", fontweight="bold")
    axes[2].set_ylabel("Frequency", fontweight="bold")
    axes[2].set_title("Aspect Ratio Distribution", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "image_properties.png", dpi=300, bbox_inches="tight")
    print(f"\n✓ Image properties plot saved to: {OUTPUT_DIR / 'image_properties.png'}")
    plt.close()


def create_train_val_split(df, val_split=0.2):
    """
    Membuat split train-validation dengan stratified sampling.

    Args:
        df (pd.DataFrame): DataFrame dataset
        val_split (float): Proporsi data untuk validation

    Returns:
        tuple: (train_df, val_df)
    """
    print("\n" + "=" * 60)
    print("CREATING TRAIN-VALIDATION SPLIT")
    print("=" * 60)

    from sklearn.model_selection import train_test_split
    import json

    # Create label encoder (string to integer mapping)
    unique_labels = sorted(df["label"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    print("\n✓ Label encoding created:")
    for label, idx in label_to_idx.items():
        print(f"  {idx}: {label}")

    # Save label mapping
    label_mapping = {
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "class_names": unique_labels,
    }

    mapping_path = DATASET_DIR / "label_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(label_mapping, f, indent=4)
    print(f"\n✓ Label mapping saved to: {mapping_path}")

    # Stratified split untuk menjaga proporsi kelas
    train_df, val_df = train_test_split(
        df, test_size=val_split, stratify=df["label"], random_state=42
    )

    print(f"\n✓ Split created successfully!")
    print(f"  Training set: {len(train_df)} samples ({(1-val_split)*100:.0f}%)")
    print(f"  Validation set: {len(val_df)} samples ({val_split*100:.0f}%)")

    # Verifikasi distribusi kelas
    print("\nClass distribution in splits:")
    print("\nTraining set:")
    for label, count in train_df["label"].value_counts().sort_index().items():
        print(f"  Class {label}: {count} samples")

    print("\nValidation set:")
    for label, count in val_df["label"].value_counts().sort_index().items():
        print(f"  Class {label}: {count} samples")

    # Simpan split ke CSV
    train_df.to_csv(DATASET_DIR / "train_split.csv", index=False)
    val_df.to_csv(DATASET_DIR / "val_split.csv", index=False)

    print(f"\n✓ Split files saved:")
    print(f"  {DATASET_DIR / 'train_split.csv'}")
    print(f"  {DATASET_DIR / 'val_split.csv'}")

    return train_df, val_df


def main():
    """
    Fungsi utama untuk menjalankan semua analisis data.
    """
    print("\n" + "=" * 60)
    print("DATA PREPARATION AND EXPLORATION")
    print("Indonesian Food Classification Dataset")
    print("=" * 60)

    # Load dataset
    df = load_dataset()

    # Eksplorasi distribusi kelas
    explore_class_distribution(df)

    # Visualisasi sample images
    visualize_sample_images(df, samples_per_class=3)

    # Analisis properti gambar
    analyze_image_properties(df, n_samples=100)

    # Buat train-validation split
    train_df, val_df = create_train_val_split(df, val_split=0.2)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED!")

if __name__ == "__main__":
    main()
