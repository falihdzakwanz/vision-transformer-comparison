"""
Model Evaluation Script
=======================
Script untuk mengevaluasi model Swin dan DeiT dengan berbagai metrik:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Inference Time
- Parameter Count

Author: 122140132_FalihDzakwanZuhdi
Course: Deep Learning - Semester Ganjil 2025/2026
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import timm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Konfigurasi
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class IndonesianFoodDataset(Dataset):
    """Custom Dataset untuk Indonesian Food Classification."""

    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Load label mapping
        import json
        label_mapping_path = DATASET_DIR / "label_mapping.json"
        if label_mapping_path.exists():
            with open(label_mapping_path, 'r') as f:
                label_mapping = json.load(f)
            self.label_to_idx = label_mapping['label_to_idx']
            self.class_names = label_mapping['class_names']
        else:
            # Create on-the-fly if not exists
            unique_labels = sorted(self.df['label'].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.class_names = unique_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.root_dir / self.df.iloc[idx]["filename"]
        image = Image.open(img_name).convert("RGB")
        
        # Get label (handle both string and integer labels)
        label_value = self.df.iloc[idx]["label"]
        if isinstance(label_value, str):
            label = self.label_to_idx[label_value]
        else:
            label = int(label_value)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_val_transform():
    """Transform untuk validation/test."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def load_model(model_name, num_classes=5, weights_path=None, device="cuda"):
    """
    Load model dengan weights yang sudah di-train.

    Args:
        model_name (str): 'swin' atau 'deit'
        num_classes (int): Jumlah kelas
        weights_path (str): Path ke model weights
        device (str): Device untuk inference

    Returns:
        model: Loaded model
    """
    print(f"\nLoading {model_name.upper()} model...")

    if model_name == "swin":
        model = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes
        )
    elif model_name == "deit":
        model = timm.create_model(
            "deit_tiny_patch16_224", pretrained=False, num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load weights
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"✓ Weights loaded from: {weights_path}")
    else:
        print(f"⚠ No weights found at: {weights_path}")

    model = model.to(device)
    model.eval()

    return model


def count_parameters(model):
    """
    Menghitung jumlah parameter model.

    Returns:
        dict: Dictionary dengan informasi parameter
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    model_size_mb = total_params * 4 / (1024**2)  # float32 = 4 bytes

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "model_size_mb": model_size_mb,
    }


def evaluate_model(model, dataloader, device="cuda"):
    """
    Evaluasi model dan kembalikan predictions dan true labels.

    Args:
        model: Model yang akan dievaluasi
        dataloader: DataLoader untuk validation set
        device: Device untuk inference

    Returns:
        tuple: (y_true, y_pred, y_prob)
    """
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    print("\nEvaluating model...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Inference"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def measure_inference_time(model, dataloader, device="cuda", warmup_iterations=10):
    """
    Mengukur waktu inferensi model.

    Args:
        model: Model yang akan diukur
        dataloader: DataLoader
        device: Device
        warmup_iterations: Jumlah iterasi warm-up

    Returns:
        dict: Dictionary dengan metrics waktu inferensi
    """
    model.eval()

    print("\nMeasuring inference time...")

    # Warm-up
    print("  Warming up...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= warmup_iterations:
                break
            inputs = inputs.to(device)
            _ = model(inputs)

    # Actual measurement
    print("  Measuring...")
    times = []
    total_images = 0

    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Timing"):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # Sinkronisasi CUDA jika menggunakan GPU
            if device == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()
            _ = model(inputs)

            if device == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()

            batch_time = (end_time - start_time) * 1000  # convert to ms
            times.append(batch_time)
            total_images += batch_size

    # Hitung statistik
    total_time = sum(times)
    avg_batch_time = np.mean(times)
    avg_image_time = total_time / total_images
    throughput = 1000 / avg_image_time  # images per second
    std_time = np.std(times)

    return {
        "total_time_ms": total_time,
        "avg_batch_time_ms": avg_batch_time,
        "avg_image_time_ms": avg_image_time,
        "throughput_imgs_per_sec": throughput,
        "std_time_ms": std_time,
        "total_images": total_images,
    }


def calculate_metrics(y_true, y_pred, model_name, class_names=None):
    """
    Menghitung berbagai metrics evaluasi.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Nama model
        class_names: Nama-nama kelas

    Returns:
        dict: Dictionary dengan semua metrics
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    return metrics


def evaluate_all_models(device="cuda"):
    """
    Evaluasi semua model (Swin dan DeiT).
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("Indonesian Food Classification")
    print("=" * 60)

    # Setup dataloader
    val_transform = get_val_transform()
    val_dataset = IndonesianFoodDataset(
        csv_file=DATASET_DIR / "val_split.csv",
        root_dir=DATASET_DIR / "train",
        transform=val_transform,
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Get class names from dataset
    class_names = val_dataset.class_names

    print(f"\n✓ Validation dataset loaded: {len(val_dataset)} samples")
    print(f"✓ Classes: {class_names}")

    # Models to evaluate
    models_info = [
        {"name": "swin", "weights": MODEL_DIR / "swin_best.pth"},
        {"name": "deit", "weights": MODEL_DIR / "deit_best.pth"},
    ]

    results = {}

    for model_info in models_info:
        model_name = model_info["name"]
        weights_path = model_info["weights"]

        print("\n" + "=" * 60)
        print(f"EVALUATING {model_name.upper()}")
        print("=" * 60)

        # Load model
        model = load_model(
            model_name, num_classes=5, weights_path=weights_path, device=device
        )

        # Count parameters
        param_info = count_parameters(model)
        print(f"\n✓ Model Parameters:")
        print(f"  Total: {param_info['total_params']:,}")
        print(f"  Trainable: {param_info['trainable_params']:,}")
        print(f"  Non-trainable: {param_info['non_trainable_params']:,}")
        print(f"  Model size: {param_info['model_size_mb']:.2f} MB")

        # Evaluate
        y_true, y_pred, y_prob = evaluate_model(model, val_loader, device)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, model_name, class_names=class_names)
        
        print(f"\n✓ Performance Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"  F1-Score (macro): {metrics['f1_macro']:.4f}")

        # Measure inference time
        inference_time = measure_inference_time(model, val_loader, device)

        print(f"\n✓ Inference Time:")
        print(f"  Average per image: {inference_time['avg_image_time_ms']:.2f} ms")
        print(
            f"  Throughput: {inference_time['throughput_imgs_per_sec']:.2f} images/sec"
        )
        print(f"  Total time: {inference_time['total_time_ms']:.2f} ms")

        # Combine all results
        results[model_name] = {
            "parameters": param_info,
            "metrics": metrics,
            "inference_time": inference_time,
            "predictions": {
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "y_prob": y_prob.tolist(),
            },
        }

        # Save individual results
        result_file = OUTPUT_DIR / f"{model_name}_evaluation.json"
        with open(result_file, "w") as f:
            json.dump(results[model_name], f, indent=4)
        print(f"\n✓ Results saved to: {result_file}")

    # Save comparison results
    comparison_file = OUTPUT_DIR / "model_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✓ Comparison results saved to: {comparison_file}")

    # Create comparison table
    create_comparison_table(results)

    return results


def create_comparison_table(results):
    """
    Membuat tabel perbandingan model.
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)

    # Prepare data for table
    comparison_data = []

    for model_name, result in results.items():
        row = {
            "Model": model_name.upper(),
            "Parameters": f"{result['parameters']['total_params']:,}",
            "Size (MB)": f"{result['parameters']['model_size_mb']:.2f}",
            "Accuracy": f"{result['metrics']['accuracy']:.4f}",
            "Precision": f"{result['metrics']['precision_macro']:.4f}",
            "Recall": f"{result['metrics']['recall_macro']:.4f}",
            "F1-Score": f"{result['metrics']['f1_macro']:.4f}",
            "Inf. Time (ms)": f"{result['inference_time']['avg_image_time_ms']:.2f}",
            "Throughput (img/s)": f"{result['inference_time']['throughput_imgs_per_sec']:.2f}",
        }
        comparison_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Print table
    print("\n" + df.to_string(index=False))

    # Save to CSV
    csv_path = OUTPUT_DIR / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Comparison table saved to: {csv_path}")

    return df


def main():
    """
    Fungsi utama untuk evaluasi.
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Evaluate all models
    results = evaluate_all_models(device=device)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
