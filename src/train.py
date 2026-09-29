"""
train.py
========
Training and validation routines for neural network models.
Tracks training/validation loss and accuracy across epochs.
"""

import time
import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Trains the model for a single epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Forward pass: compute predictions and batch loss
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass: compute gradients via backpropagation
        loss.backward()

        # Gradient descent optimization step
        optimizer.step()

        # Tracking metrics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_loss_acc(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluates the model on validation/test data."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    optimizer_name: str = "adam",
    device: str = "cpu",
    checkpoint_path: str = None,
    verbose: bool = True
) -> Tuple[Dict[str, List[float]], float]:
    """
    Full training pipeline with validation tracking and best model checkpointing.
    
    Args:
        model: PyTorch module
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        epochs: Number of complete passes over the training set
        lr: Learning rate
        optimizer_name: 'adam' or 'sgd'
        device: 'cpu' or 'cuda'
        checkpoint_path: Filepath to save best model weights
        verbose: Whether to print per-epoch progress
        
    Returns:
        tuple: (history_dict, total_training_time_seconds)
    """
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(dev)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    start_time = time.time()

    if verbose:
        print(f"[*] Training on device: {dev} | Optimizer: {optimizer_name.upper()} | LR: {lr} | Epochs: {epochs}")
        print("-" * 68)
        print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc (%)':<14} | {'Val Loss':<12} | {'Val Acc (%)':<12}")
        print("-" * 68)

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, dev)
        v_loss, v_acc = evaluate_loss_acc(model, val_loader, criterion, dev)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

        if verbose:
            print(f"{epoch:<8} | {t_loss:<12.4f} | {t_acc:<14.2f} | {v_loss:<12.4f} | {v_acc:<12.2f}")

    total_time = time.time() - start_time
    if verbose:
        print("-" * 68)
        print(f"[+] Training complete in {total_time:.2f}s | Best Val Acc: {best_val_acc:.2f}%\n")

    return history, total_time
