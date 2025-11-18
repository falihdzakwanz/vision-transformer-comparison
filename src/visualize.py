"""
Visualization Script
===================
Script untuk memvisualisasikan hasil training dan evaluasi:
- Learning curves (loss & accuracy)
- Confusion matrices
- Comparison charts
- Sample predictions

Author: 122140132_FalihDzakwanZuhdi
Course: Deep Learning - Semester Ganjil 2025/2026
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Konfigurasi
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Style untuk plot
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def load_training_history(model_name):
    """
    Load training history dari file JSON.

    Args:
        model_name (str): Nama model ('swin' atau 'deit')

    Returns:
        dict: Training history
    """
    history_file = RESULTS_DIR / f"{model_name}_history.json"

    if not history_file.exists():
        print(f"⚠ Warning: History file not found for {model_name}")
        return None

    with open(history_file, "r") as f:
        history = json.load(f)

    return history


def load_evaluation_results(model_name):
    """
    Load evaluation results dari file JSON.

    Args:
        model_name (str): Nama model

    Returns:
        dict: Evaluation results
    """
    eval_file = RESULTS_DIR / f"{model_name}_evaluation.json"

    if not eval_file.exists():
        print(f"⚠ Warning: Evaluation file not found for {model_name}")
        return None

    with open(eval_file, "r") as f:
        results = json.load(f)

    return results


def plot_learning_curves(models=["swin", "deit"]):
    """
    Plot learning curves untuk semua model.

    Args:
        models (list): List nama model
    """
    print("\n" + "=" * 60)
    print("PLOTTING LEARNING CURVES")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = {"swin": "#FF6B6B", "deit": "#4ECDC4"}

    for model_name in models:
        history = load_training_history(model_name)

        if history is None:
            continue

        epochs = range(1, len(history["train_loss"]) + 1)
        color = colors.get(model_name, "blue")

        # Training Loss
        axes[0, 0].plot(
            epochs,
            history["train_loss"],
            label=f"{model_name.upper()} Train",
            marker="o",
            color=color,
            linewidth=2,
        )

        # Validation Loss
        axes[0, 1].plot(
            epochs,
            history["val_loss"],
            label=f"{model_name.upper()} Val",
            marker="s",
            color=color,
            linewidth=2,
        )

        # Training Accuracy
        axes[1, 0].plot(
            epochs,
            history["train_acc"],
            label=f"{model_name.upper()} Train",
            marker="o",
            color=color,
            linewidth=2,
        )

        # Validation Accuracy
        axes[1, 1].plot(
            epochs,
            history["val_acc"],
            label=f"{model_name.upper()} Val",
            marker="s",
            color=color,
            linewidth=2,
        )

    # Formatting
    axes[0, 0].set_title("Training Loss", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch", fontweight="bold")
    axes[0, 0].set_ylabel("Loss", fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Validation Loss", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch", fontweight="bold")
    axes[0, 1].set_ylabel("Loss", fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Training Accuracy", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch", fontweight="bold")
    axes[1, 0].set_ylabel("Accuracy", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch", fontweight="bold")
    axes[1, 1].set_ylabel("Accuracy", fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Learning Curves Comparison", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()

    save_path = FIGURES_DIR / "learning_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Learning curves saved to: {save_path}")
    plt.close()


def plot_confusion_matrices(models=["swin", "deit"], class_names=None):
    """
    Plot confusion matrices untuk semua model.

    Args:
        models (list): List nama model
        class_names (list): Nama-nama kelas
    """
    print("\n" + "=" * 60)
    print("PLOTTING CONFUSION MATRICES")
    print("=" * 60)

    if class_names is None:
        class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6))

    if len(models) == 1:
        axes = [axes]

    for idx, model_name in enumerate(models):
        results = load_evaluation_results(model_name)

        if results is None:
            continue

        cm = np.array(results["metrics"]["confusion_matrix"])

        # Normalize confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[idx],
            cbar_kws={"label": "Proportion"},
        )

        axes[idx].set_title(
            f"{model_name.upper()} Confusion Matrix", fontsize=14, fontweight="bold"
        )
        axes[idx].set_xlabel("Predicted Label", fontweight="bold")
        axes[idx].set_ylabel("True Label", fontweight="bold")

        # Add counts to cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = axes[idx].texts[i * cm.shape[1] + j]
                count = cm[i, j]
                text.set_text(f"{cm_normalized[i, j]:.2f}\n({count})")

    plt.tight_layout()

    save_path = FIGURES_DIR / "confusion_matrices.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Confusion matrices saved to: {save_path}")
    plt.close()


def plot_metrics_comparison(models=["swin", "deit"]):
    """
    Plot bar chart perbandingan metrics.

    Args:
        models (list): List nama model
    """
    print("\n" + "=" * 60)
    print("PLOTTING METRICS COMPARISON")
    print("=" * 60)

    metrics_data = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": [],
    }

    for model_name in models:
        results = load_evaluation_results(model_name)

        if results is None:
            continue

        metrics_data["Model"].append(model_name.upper())
        metrics_data["Accuracy"].append(results["metrics"]["accuracy"])
        metrics_data["Precision"].append(results["metrics"]["precision_macro"])
        metrics_data["Recall"].append(results["metrics"]["recall_macro"])
        metrics_data["F1-Score"].append(results["metrics"]["f1_macro"])

    df = pd.DataFrame(metrics_data)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df["Model"]))
    width = 0.2

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#FF6B6B", "#4ECDC4", "#95E1D3", "#F38181"]

    for i, metric in enumerate(metrics):
        offset = width * (i - len(metrics) / 2 + 0.5)
        bars = ax.bar(x + offset, df[metric], width, label=metric, color=colors[i])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Performance Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1.1])

    plt.tight_layout()

    save_path = FIGURES_DIR / "metrics_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Metrics comparison saved to: {save_path}")
    plt.close()


def plot_parameter_comparison(models=["swin", "deit"]):
    """
    Plot perbandingan jumlah parameter dan ukuran model.

    Args:
        models (list): List nama model
    """
    print("\n" + "=" * 60)
    print("PLOTTING PARAMETER COMPARISON")
    print("=" * 60)

    param_data = {"Model": [], "Parameters (M)": [], "Size (MB)": []}

    for model_name in models:
        results = load_evaluation_results(model_name)

        if results is None:
            continue

        param_data["Model"].append(model_name.upper())
        param_data["Parameters (M)"].append(results["parameters"]["total_params"] / 1e6)
        param_data["Size (MB)"].append(results["parameters"]["model_size_mb"])

    df = pd.DataFrame(param_data)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#FF6B6B", "#4ECDC4"]

    # Parameters
    bars1 = axes[0].bar(df["Model"], df["Parameters (M)"], color=colors)
    axes[0].set_ylabel("Parameters (Millions)", fontsize=12, fontweight="bold")
    axes[0].set_title("Model Parameters", fontsize=14, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}M",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Size
    bars2 = axes[1].bar(df["Model"], df["Size (MB)"], color=colors)
    axes[1].set_ylabel("Size (MB)", fontsize=12, fontweight="bold")
    axes[1].set_title("Model Size", fontsize=14, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f} MB",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()

    save_path = FIGURES_DIR / "parameter_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Parameter comparison saved to: {save_path}")
    plt.close()


def plot_inference_time_comparison(models=["swin", "deit"]):
    """
    Plot perbandingan waktu inferensi.

    Args:
        models (list): List nama model
    """
    print("\n" + "=" * 60)
    print("PLOTTING INFERENCE TIME COMPARISON")
    print("=" * 60)

    time_data = {"Model": [], "Avg Time (ms)": [], "Throughput (img/s)": []}

    for model_name in models:
        results = load_evaluation_results(model_name)

        if results is None:
            continue

        time_data["Model"].append(model_name.upper())
        time_data["Avg Time (ms)"].append(
            results["inference_time"]["avg_image_time_ms"]
        )
        time_data["Throughput (img/s)"].append(
            results["inference_time"]["throughput_imgs_per_sec"]
        )

    df = pd.DataFrame(time_data)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#FF6B6B", "#4ECDC4"]

    # Average time
    bars1 = axes[0].bar(df["Model"], df["Avg Time (ms)"], color=colors)
    axes[0].set_ylabel("Time (milliseconds)", fontsize=12, fontweight="bold")
    axes[0].set_title(
        "Average Inference Time per Image", fontsize=14, fontweight="bold"
    )
    axes[0].grid(axis="y", alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Throughput
    bars2 = axes[1].bar(df["Model"], df["Throughput (img/s)"], color=colors)
    axes[1].set_ylabel("Images per Second", fontsize=12, fontweight="bold")
    axes[1].set_title("Throughput", fontsize=14, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()

    save_path = FIGURES_DIR / "inference_time_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Inference time comparison saved to: {save_path}")
    plt.close()


def create_summary_figure(models=["swin", "deit"]):
    """
    Membuat summary figure dengan semua metrics penting.

    Args:
        models (list): List nama model
    """
    print("\n" + "=" * 60)
    print("CREATING SUMMARY FIGURE")
    print("=" * 60)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    colors = ["#FF6B6B", "#4ECDC4"]

    # Collect data
    data = {}
    for model_name in models:
        results = load_evaluation_results(model_name)
        if results:
            data[model_name] = results

    model_names = [m.upper() for m in data.keys()]

    # 1. Accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = [data[m]["metrics"]["accuracy"] for m in data.keys()]
    bars = ax1.bar(model_names, accuracies, color=colors[: len(model_names)])
    ax1.set_ylabel("Accuracy", fontweight="bold")
    ax1.set_title("Accuracy Comparison", fontweight="bold")
    ax1.set_ylim([0, 1])
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. F1-Score comparison
    ax2 = fig.add_subplot(gs[0, 1])
    f1_scores = [data[m]["metrics"]["f1_macro"] for m in data.keys()]
    bars = ax2.bar(model_names, f1_scores, color=colors[: len(model_names)])
    ax2.set_ylabel("F1-Score", fontweight="bold")
    ax2.set_title("F1-Score Comparison", fontweight="bold")
    ax2.set_ylim([0, 1])
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. Parameters comparison
    ax3 = fig.add_subplot(gs[0, 2])
    params = [data[m]["parameters"]["total_params"] / 1e6 for m in data.keys()]
    bars = ax3.bar(model_names, params, color=colors[: len(model_names)])
    ax3.set_ylabel("Parameters (M)", fontweight="bold")
    ax3.set_title("Model Parameters", fontweight="bold")
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}M",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 4. Inference time comparison
    ax4 = fig.add_subplot(gs[1, 0])
    inf_times = [data[m]["inference_time"]["avg_image_time_ms"] for m in data.keys()]
    bars = ax4.bar(model_names, inf_times, color=colors[: len(model_names)])
    ax4.set_ylabel("Time (ms)", fontweight="bold")
    ax4.set_title("Inference Time per Image", fontweight="bold")
    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 5. Model size comparison
    ax5 = fig.add_subplot(gs[1, 1])
    sizes = [data[m]["parameters"]["model_size_mb"] for m in data.keys()]
    bars = ax5.bar(model_names, sizes, color=colors[: len(model_names)])
    ax5.set_ylabel("Size (MB)", fontweight="bold")
    ax5.set_title("Model Size", fontweight="bold")
    for bar in bars:
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 6. Throughput comparison
    ax6 = fig.add_subplot(gs[1, 2])
    throughputs = [
        data[m]["inference_time"]["throughput_imgs_per_sec"] for m in data.keys()
    ]
    bars = ax6.bar(model_names, throughputs, color=colors[: len(model_names)])
    ax6.set_ylabel("Images/sec", fontweight="bold")
    ax6.set_title("Throughput", fontweight="bold")
    for bar in bars:
        height = bar.get_height()
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 7-8. Confusion matrices (bottom row)
    for idx, model_name in enumerate(data.keys()):
        ax = fig.add_subplot(gs[2, idx])
        cm = np.array(data[model_name]["metrics"]["confusion_matrix"])
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            ax=ax,
            cbar=False,
            square=True,
        )
        ax.set_title(f"{model_name.upper()} Confusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("True", fontweight="bold")

    plt.suptitle(
        "Vision Transformer Comparison: Swin vs DeiT",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    save_path = FIGURES_DIR / "summary_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Summary figure saved to: {save_path}")
    plt.close()


def main():
    """
    Fungsi utama untuk visualisasi.
    """
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("Indonesian Food Classification - Vision Transformer Comparison")
    print("=" * 60)

    models = ["swin", "deit"]
    
    # Load class names from label mapping
    import json
    label_mapping_path = BASE_DIR / "dataset" / "label_mapping.json"
    if label_mapping_path.exists():
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
        class_names = label_mapping['class_names']
    else:
        class_names = None

    # Plot semua visualisasi
    plot_learning_curves(models)
    plot_confusion_matrices(models, class_names=class_names)
    plot_metrics_comparison(models)
    plot_parameter_comparison(models)
    plot_inference_time_comparison(models)
    create_summary_figure(models)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETED!")
    print("=" * 60)
    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("\nGenerated figures:")
    print("  1. learning_curves.png")
    print("  2. confusion_matrices.png")
    print("  3. metrics_comparison.png")
    print("  4. parameter_comparison.png")
    print("  5. inference_time_comparison.png")
    print("  6. summary_comparison.png")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
