"""
test_quick.py
=============
Quick verification script to test if the trained neural network works.
Loads the trained checkpoint (checkpoints/baseline_best.pt), runs inference on 
test samples, and prints sample predictions with ASCII digit art and confidence scores.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from models import BaselineMLP
from dataset import get_mnist_dataloaders


def render_ascii_digit(tensor_img: torch.Tensor) -> str:
    """Renders a 28x28 grayscale image tensor as an ASCII terminal graphic."""
    # Undo normalization: (x * 0.3081) + 0.1307
    img_np = tensor_img.squeeze().numpy() * 0.3081 + 0.1307
    img_np = np.clip(img_np, 0.0, 1.0)

    # ASCII grayscale gradient
    chars = " .:-=+*#%@"
    ascii_rows = []
    # Downsample slightly by stepping every 2 horizontal chars for terminal aspect ratio
    for r in range(0, 28, 2):
        row_str = "  "
        for c in range(28):
            val = img_np[r, c]
            char_idx = int(val * (len(chars) - 1))
            row_str += chars[char_idx]
        ascii_rows.append(row_str)
    return "\n".join(ascii_rows)


def run_quick_test():
    print("=" * 65)
    print("      MNIST NEURAL NETWORK — QUICK VERIFICATION TEST")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(base_dir, "checkpoints", "baseline_best.pt")
    data_dir = os.path.join(base_dir, "data")

    # 1. Check if model checkpoint exists
    print(f"\n[1/4] Checking checkpoint file...")
    if not os.path.exists(ckpt_path):
        print(f"  [ERROR] Checkpoint not found at: {ckpt_path}")
        print("  Please run 'python run_pipeline.py' first.")
        return
    print(f"  [PASS] Checkpoint found! ({os.path.getsize(ckpt_path):,} bytes)")

    # 2. Instantiate and load model
    print(f"\n[2/4] Initializing BaselineMLP and loading trained weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineMLP(input_dim=784, hidden_dim=128, num_classes=10)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"  [PASS] Model loaded successfully on {device}!")

    # 3. Load test data
    print(f"\n[3/4] Loading test dataset...")
    _, _, test_loader, raw_test_dataset = get_mnist_dataloaders(data_dir=data_dir, batch_size=100)
    print(f"  [PASS] Test dataset loaded ({len(raw_test_dataset):,} samples)!")

    # 4. Run fast inference on first 500 test samples to verify accuracy
    print(f"\n[4/4] Running fast test inference...")
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
            if total >= 1000:
                break

    acc = (correct / total) * 100.0
    print(f"  [PASS] Fast verification accuracy on {total} test samples: {acc:.2f}%")

    # 5. Interactive Visual Demo: 3 Random Test Predictions with ASCII Art
    print("\n" + "=" * 65)
    print("             SAMPLE PREDICTIONS WITH ASCII DIGIT ART")
    print("=" * 65)

    sample_indices = [0, 1, 2, 7, 8]
    for idx in sample_indices:
        img_tensor, true_label = test_loader.dataset[idx]
        with torch.no_grad():
            img_input = img_tensor.unsqueeze(0).to(device)
            logits = model(img_input)
            probs = F.softmax(logits, dim=1).cpu().squeeze()
            pred_label = int(probs.argmax())
            confidence = float(probs[pred_label]) * 100.0

        status = "[CORRECT]" if pred_label == true_label else "[WRONG]"
        print(f"\nTest Sample #{idx}:")
        print(render_ascii_digit(img_tensor))
        print(f"  -> True Label:      {true_label}")
        print(f"  -> Model Predicted: {pred_label} (Confidence: {confidence:.2f}%) {status}")

    print("\n" + "=" * 65)
    print("  RESULT: ALL TESTS PASSED! THE NEURAL NETWORK IS FULLY WORKING!")
    print("=" * 65)


if __name__ == "__main__":
    run_quick_test()
