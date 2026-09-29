"""
visualize.py
============
Publication-grade visualizations for dataset inspection, training dynamics,
confusion matrix heatmaps, error analyses, and ablation comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List, Any

# Configure clean aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15
})


def plot_sample_dataset(raw_dataset, save_path: str = None, num_samples: int = 16):
    """Plots a clean grid of MNIST samples with their corresponding ground truth labels."""
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("MNIST Dataset Sample Images (28 x 28 Grayscale)", fontsize=14, fontweight="bold", y=0.95)

    for i, ax in enumerate(axes.flat):
        img, label = raw_dataset[i]
        # img is a torch tensor of shape (1, 28, 28)
        ax.imshow(img.squeeze().numpy(), cmap="gray")
        ax.set_title(f"Label: {label}", fontsize=11, fontweight="semibold")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved sample dataset plot to: {save_path}")
    plt.close()


def plot_learning_curves(history: Dict[str, List[float]], title: str = "Baseline Neural Network", save_path: str = None):
    """
    Plots Training Loss vs Validation Loss and Training Accuracy vs Validation Accuracy side-by-side.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], "o-", color="#1f77b4", linewidth=2, label="Training Loss")
    ax1.plot(epochs, history["val_loss"], "s--", color="#ff7f0e", linewidth=2, label="Validation Loss")
    ax1.set_title(f"{title} - Loss Dynamics", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Accuracy subplot
    ax2.plot(epochs, history["train_acc"], "o-", color="#2ca02c", linewidth=2, label="Training Accuracy")
    ax2.plot(epochs, history["val_acc"], "s--", color="#d62728", linewidth=2, label="Validation Accuracy")
    ax2.set_title(f"{title} - Accuracy Dynamics", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved learning curves to: {save_path}")
    plt.close()


def plot_confusion_matrix_heatmap(cm: np.ndarray, title: str = "Test Set Confusion Matrix", save_path: str = None):
    """
    Renders an annotated Seaborn heatmap of the confusion matrix with counts and colormap.
    """
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        linewidths=0.5,
        xticklabels=[str(i) for i in range(10)],
        yticklabels=[str(i) for i in range(10)]
    )
    plt.title(f"{title}\n(Overall Accuracy: {np.trace(cm)/np.sum(cm)*100:.2f}%)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=12, fontweight="semibold")
    plt.ylabel("True Ground Truth Class", fontsize=12, fontweight="semibold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved confusion matrix heatmap to: {save_path}")
    plt.close()


def plot_misclassified_gallery(misclassified: List[Dict[str, Any]], save_path: str = None, num_display: int = 10):
    """
    Renders a gallery of misclassified digits along with their true label,
    predicted label, and prediction confidence.
    """
    num_display = min(num_display, len(misclassified))
    cols = 5
    rows = (num_display + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.2 * rows))
    fig.suptitle("Sample Misclassified Test Digits (Error Analysis)", fontsize=14, fontweight="bold", y=0.98)

    for i in range(rows * cols):
        ax = axes.flat[i]
        if i < num_display:
            item = misclassified[i]
            img = item["image"]
            # Undo normalization for clean visual display
            img_disp = img * 0.3081 + 0.1307
            img_disp = np.clip(img_disp, 0.0, 1.0)

            ax.imshow(img_disp, cmap="magma")
            title_str = f"True: {item['true_label']}\nPred: {item['pred_label']} ({item['confidence']*100:.1f}%)"
            ax.set_title(title_str, fontsize=10, color="crimson", fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved misclassified gallery to: {save_path}")
    plt.close()


def plot_experiment_comparisons(experiments: Dict[str, Dict[str, Any]], save_path: str = None):
    """
    Plots multi-model validation accuracy trajectories and final test accuracy bar comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    model_names = []
    test_accuracies = []

    for idx, (name, data) in enumerate(experiments.items()):
        val_acc = data["history"]["val_acc"]
        epochs = range(1, len(val_acc) + 1)
        ax1.plot(epochs, val_acc, "o-", label=name, color=colors[idx % len(colors)], linewidth=2)
        model_names.append(name)
        test_accuracies.append(data["metrics"]["accuracy"] * 100.0)

    ax1.set_title("Validation Accuracy Trajectories Across Experiments", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True)

    # Bar chart of final test accuracies
    bars = ax2.bar(model_names, test_accuracies, color=colors[:len(model_names)], width=0.55, edgecolor="black", alpha=0.85)
    ax2.set_title("Final Test Set Accuracy Comparison", fontweight="bold")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_ylim(min(test_accuracies) - 2.0, 100.0)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=15, ha="right", fontweight="semibold")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.2f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved experiment comparison to: {save_path}")
    plt.close()
