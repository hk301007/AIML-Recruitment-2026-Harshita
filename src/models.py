"""
models.py
=========
Neural network architectures for MNIST digit classification.
Includes:
- BaselineMLP: 1 Hidden Layer (784 -> 128 -> 10) with ReLU activation
- DeeperMLP: 2 Hidden Layers (784 -> 256 -> 128 -> 10) with ReLU
- SigmoidMLP: 1 Hidden Layer (784 -> 128 -> 10) with Sigmoid (for activation comparison)
- DropoutMLP: Regularized Architecture with Dropout (p=0.2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model: nn.Module) -> int:
    """Calculates total trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BaselineMLP(nn.Module):
    """
    Baseline Single-Hidden-Layer Feedforward Neural Network.
    
    Architecture:
    -------------
    1. Input Reshaping / Flatten: (Batch, 1, 28, 28) -> (Batch, 784)
    2. Hidden Linear Layer: 784 inputs -> 128 hidden neurons
       - Weight matrix: W1 in R^(128 x 784) -> 100,352 parameters
       - Bias vector:   b1 in R^(128)       -> 128 parameters
    3. Activation Function: Rectified Linear Unit (ReLU), f(x) = max(0, x)
       - Introduces non-linearity to learn arbitrary decision boundaries
       - Prevents vanishing gradient for positive activations
    4. Output Linear Layer: 128 inputs -> 10 class logits
       - Weight matrix: W2 in R^(10 x 128)  -> 1,280 parameters
       - Bias vector:   b2 in R^(10)        -> 10 parameters
    Total Trainable Parameters: 101,770
    
    Note on Softmax:
    PyTorch's `nn.CrossEntropyLoss` combines `nn.LogSoftmax()` and `nn.NLLLoss()`
    in a single numerically stable operation using the Log-Sum-Exp trick.
    Hence, raw unnormalized logits are emitted during training, and Softmax
    is applied during inference/probabilities calculation.
    """
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10):
        super(BaselineMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Softmax to raw logits to yield normalized class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class DeeperMLP(nn.Module):
    """
    Deeper Multi-Layer Perceptron (2 Hidden Layers).
    
    Architecture:
    -------------
    784 -> Linear(256) -> ReLU -> Linear(128) -> ReLU -> Linear(10)
    Total parameters: 235,146
    """
    def __init__(self, input_dim: int = 784, hidden1: int = 256, hidden2: int = 128, num_classes: int = 10):
        super(DeeperMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class SigmoidMLP(nn.Module):
    """
    Single-Hidden-Layer MLP using Sigmoid activation instead of ReLU.
    Used in Part G experimentation to demonstrate saturation and vanishing gradients.
    """
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10):
        super(SigmoidMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        logits = self.fc2(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class DropoutMLP(nn.Module):
    """
    Single-Hidden-Layer MLP with Dropout regularization (p=0.2).
    """
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, p_drop: float = 0.2, num_classes: int = 10):
        super(DropoutMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
