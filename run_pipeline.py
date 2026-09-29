"""
run_pipeline.py
===============
Master pipeline execution script for Task 2: MNIST Neural Network.
Executes training, evaluation, comprehensive error analysis,
multi-model ablation studies, visual asset generation, and notebook rendering.
"""

import os
import sys
import json
import base64
import torch
import numpy as np

# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from dataset import get_mnist_dataloaders
from models import BaselineMLP, DeeperMLP, SigmoidMLP, DropoutMLP, count_parameters
from train import train_model
from evaluate import evaluate_comprehensive
from visualize import (
    plot_sample_dataset,
    plot_learning_curves,
    plot_confusion_matrix_heatmap,
    plot_misclassified_gallery,
    plot_experiment_comparisons
)


def set_seed(seed: int = 42):
    """Sets random seeds for full reproducibility across NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline():
    print("=" * 70)
    print("      TASK 2: MNIST HANDWRITTEN DIGIT CLASSIFIER PIPELINE")
    print("=" * 70)
    set_seed(42)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    assets_dir = os.path.join(base_dir, "assets")
    checkpoints_dir = os.path.join(base_dir, "checkpoints")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 1. Dataset Loading and Preprocessing
    print("\n[Step 1] Loading and preprocessing MNIST dataset...")
    train_loader, val_loader, test_loader, raw_test_dataset = get_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        val_split=0.1,  # 54,000 train, 6,000 val
        random_seed=42
    )
    print(f"  - Training batches: {len(train_loader)} (~54,000 samples)")
    print(f"  - Validation batches: {len(val_loader)} (~6,000 samples)")
    print(f"  - Test batches: {len(test_loader)} (~10,000 samples)")

    # 2. Save Sample Dataset Visual
    print("\n[Step 2] Generating sample digits grid visualization...")
    sample_img_path = os.path.join(assets_dir, "sample_digits.png")
    plot_sample_dataset(raw_test_dataset, save_path=sample_img_path)

    # 3. Train Baseline Model (Part C, D, E)
    print("\n[Step 3] Building and training Baseline Neural Network...")
    baseline_model = BaselineMLP(input_dim=784, hidden_dim=128, num_classes=10)
    print(f"  - Baseline parameter count: {count_parameters(baseline_model):,}")

    baseline_ckpt = os.path.join(checkpoints_dir, "baseline_best.pt")
    baseline_history, baseline_time = train_model(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        lr=0.001,
        optimizer_name="adam",
        device="cpu",
        checkpoint_path=baseline_ckpt,
        verbose=True
    )

    # Load best checkpoint for evaluation
    baseline_model.load_state_dict(torch.load(baseline_ckpt))

    # 4. Learning Curves
    print("\n[Step 4] Plotting baseline learning curves...")
    curves_path = os.path.join(assets_dir, "baseline_learning_curves.png")
    plot_learning_curves(baseline_history, title="Baseline Neural Network (128 ReLU)", save_path=curves_path)

    # 5. Comprehensive Evaluation (Part F)
    print("\n[Step 5] Evaluating Baseline Model on 10,000 unseen test samples...")
    test_results = evaluate_comprehensive(baseline_model, test_loader, device="cpu")
    print(f"  - Test Accuracy: {test_results['accuracy'] * 100:.2f}%")
    print(f"  - Macro Precision: {test_results['macro_precision'] * 100:.2f}%")
    print(f"  - Macro Recall: {test_results['macro_recall'] * 100:.2f}%")
    print(f"  - Macro F1-Score: {test_results['macro_f1'] * 100:.2f}%")
    print("\nClassification Report:\n", test_results["classification_report_str"])

    # Save Confusion Matrix Heatmap
    cm_path = os.path.join(assets_dir, "confusion_matrix.png")
    plot_confusion_matrix_heatmap(test_results["confusion_matrix"], title="Baseline Test Confusion Matrix", save_path=cm_path)

    # Save Misclassified Gallery
    misclassified_path = os.path.join(assets_dir, "misclassified_digits.png")
    plot_misclassified_gallery(test_results["misclassified_samples"], save_path=misclassified_path, num_display=10)
    print(f"  - Total misclassified samples: {test_results['total_misclassified']} out of {test_results['total_test_samples']}")

    # 6. Ablation Experimentation (Part G)
    print("\n[Step 6] Running Part G Ablation Experiments...")
    experiments = {}

    # Exp 1: Baseline
    experiments["Baseline (128 ReLU, Adam)"] = {
        "model": baseline_model,
        "history": baseline_history,
        "metrics": test_results,
        "train_time": baseline_time,
        "params": count_parameters(baseline_model),
        "notes": "Single hidden layer (128 neurons), ReLU, Adam (lr=1e-3)"
    }

    # Exp 2: Deeper Architecture (2 Hidden Layers: 256 -> 128)
    print("\n--- Running Experiment 2: Deeper Architecture (256 -> 128) ---")
    deeper_model = DeeperMLP(input_dim=784, hidden1=256, hidden2=128, num_classes=10)
    deeper_ckpt = os.path.join(checkpoints_dir, "deeper_best.pt")
    deeper_history, deeper_time = train_model(
        deeper_model, train_loader, val_loader, epochs=10, lr=0.001, optimizer_name="adam", checkpoint_path=deeper_ckpt, verbose=False
    )
    deeper_model.load_state_dict(torch.load(deeper_ckpt))
    deeper_results = evaluate_comprehensive(deeper_model, test_loader)
    experiments["Deeper MLP (256-128, Adam)"] = {
        "model": deeper_model,
        "history": deeper_history,
        "metrics": deeper_results,
        "train_time": deeper_time,
        "params": count_parameters(deeper_model),
        "notes": "2 hidden layers (256, 128), ReLU, Adam (lr=1e-3)"
    }
    print(f"  -> Deeper Model Test Accuracy: {deeper_results['accuracy'] * 100:.2f}%")

    # Exp 3: Activation Variant (Sigmoid instead of ReLU)
    print("\n--- Running Experiment 3: Activation Variant (Sigmoid MLP) ---")
    sigmoid_model = SigmoidMLP(input_dim=784, hidden_dim=128, num_classes=10)
    sigmoid_ckpt = os.path.join(checkpoints_dir, "sigmoid_best.pt")
    sigmoid_history, sigmoid_time = train_model(
        sigmoid_model, train_loader, val_loader, epochs=10, lr=0.001, optimizer_name="adam", checkpoint_path=sigmoid_ckpt, verbose=False
    )
    sigmoid_model.load_state_dict(torch.load(sigmoid_ckpt))
    sigmoid_results = evaluate_comprehensive(sigmoid_model, test_loader)
    experiments["Sigmoid MLP (128, Adam)"] = {
        "model": sigmoid_model,
        "history": sigmoid_history,
        "metrics": sigmoid_results,
        "train_time": sigmoid_time,
        "params": count_parameters(sigmoid_model),
        "notes": "Single hidden layer (128 neurons), Sigmoid activation, Adam"
    }
    print(f"  -> Sigmoid Model Test Accuracy: {sigmoid_results['accuracy'] * 100:.2f}%")

    # Exp 4: Optimizer Variant (Baseline with SGD + Momentum)
    print("\n--- Running Experiment 4: Optimizer Variant (SGD with Momentum 0.9) ---")
    sgd_model = BaselineMLP(input_dim=784, hidden_dim=128, num_classes=10)
    sgd_ckpt = os.path.join(checkpoints_dir, "sgd_best.pt")
    sgd_history, sgd_time = train_model(
        sgd_model, train_loader, val_loader, epochs=10, lr=0.01, optimizer_name="sgd", checkpoint_path=sgd_ckpt, verbose=False
    )
    sgd_model.load_state_dict(torch.load(sgd_ckpt))
    sgd_results = evaluate_comprehensive(sgd_model, test_loader)
    experiments["Baseline + SGD (lr=0.01, mom=0.9)"] = {
        "model": sgd_model,
        "history": sgd_history,
        "metrics": sgd_results,
        "train_time": sgd_time,
        "params": count_parameters(sgd_model),
        "notes": "Single hidden layer (128 neurons), ReLU, SGD (lr=0.01, momentum=0.9)"
    }
    print(f"  -> SGD Model Test Accuracy: {sgd_results['accuracy'] * 100:.2f}%")

    # Plot experiment comparison
    comp_path = os.path.join(assets_dir, "experiment_comparison.png")
    plot_experiment_comparisons(experiments, save_path=comp_path)

    # 7. Print Comparative Table
    print("\n" + "=" * 88)
    print("                            EXPERIMENT ABLATION SUMMARY")
    print("=" * 88)
    header = f"{'Model Experiment':<32} | {'Params':<9} | {'Val Acc (%)':<12} | {'Test Acc (%)':<12} | {'Time (s)':<8}"
    print(header)
    print("-" * 88)
    for name, exp in experiments.items():
        best_val = max(exp['history']['val_acc'])
        test_acc = exp['metrics']['accuracy'] * 100.0
        print(f"{name:<32} | {exp['params']:<9,d} | {best_val:<12.2f} | {test_acc:<12.2f} | {exp['train_time']:<8.1f}")
    print("=" * 88)

    # Save summary metrics to JSON
    summary_data = {
        "baseline": {
            "test_accuracy": float(test_results["accuracy"]),
            "macro_precision": float(test_results["macro_precision"]),
            "macro_recall": float(test_results["macro_recall"]),
            "macro_f1": float(test_results["macro_f1"]),
            "confusion_matrix": test_results["confusion_matrix"].tolist(),
            "history": baseline_history
        },
        "experiments": {
            name: {
                "params": exp["params"],
                "best_val_acc": float(max(exp["history"]["val_acc"])),
                "test_acc": float(exp["metrics"]["accuracy"]),
                "train_time": float(exp["train_time"]),
                "notes": exp["notes"]
            }
            for name, exp in experiments.items()
        }
    }

    with open(os.path.join(base_dir, "pipeline_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[+] Pipeline execution completed successfully! Summary saved to pipeline_summary.json")


if __name__ == "__main__":
    run_pipeline()
