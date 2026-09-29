"""
evaluate.py
===========
Evaluation metrics, confusion matrix computation, per-class classification
reports, and misclassification error analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from typing import Dict, Any, List, Tuple


@torch.no_grad()
def evaluate_comprehensive(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Computes comprehensive test metrics:
    - Global accuracy
    - Per-class precision, recall, f1-score
    - Confusion matrix (10x10)
    - Full classification report dictionary and string
    - Misclassified samples metadata (image index, true label, pred label, confidence)
    """
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(dev)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    misclassified = []
    global_idx = 0

    for images, targets in test_loader:
        images_dev = images.to(dev)
        logits = model(images_dev)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        targets_np = targets.numpy()

        for b in range(len(targets_np)):
            true_lbl = targets_np[b]
            pred_lbl = preds[b]
            prob_dist = probs[b]

            if true_lbl != pred_lbl:
                misclassified.append({
                    "sample_idx": global_idx,
                    "image": images[b].squeeze(0).numpy(),  # unflattened 28x28
                    "true_label": int(true_lbl),
                    "pred_label": int(pred_lbl),
                    "confidence": float(prob_dist[pred_lbl]),
                    "true_confidence": float(prob_dist[true_lbl])
                })
            global_idx += 1

        all_preds.extend(preds)
        all_targets.extend(targets_np)
        all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    prec, rec, f1, support = precision_recall_fscore_support(all_targets, all_preds, average=None)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro")

    report_str = classification_report(
        all_targets,
        all_preds,
        target_names=[f"Digit {i}" for i in range(10)],
        digits=4
    )

    # Sort misclassified samples by highest model confidence (most egregious errors)
    misclassified.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report_str": report_str,
        "per_class_precision": prec,
        "per_class_recall": rec,
        "per_class_f1": f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "misclassified_samples": misclassified,
        "total_misclassified": len(misclassified),
        "total_test_samples": len(all_targets)
    }
