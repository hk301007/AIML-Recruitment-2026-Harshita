"""
dataset.py
==========
Data ingestion, preprocessing, normalization, and DataLoader preparation
for the MNIST handwritten digit classification task.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_datasets(data_dir: str = "./data"):
    """
    Downloads and prepares raw MNIST datasets.
    
    Transformations applied:
    - ToTensor(): Converts PIL Image (H x W x C) with values [0, 255] to 
      FloatTensor (C x H x W) with values scaled to [0.0, 1.0].
    - Normalize((0.1307,), (0.3081,)): Subtracts the MNIST dataset mean (0.1307)
      and divides by standard deviation (0.3081). This centers the pixel 
      distribution around 0 with unit variance, stabilizing gradient descent.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Unnormalized transform for clean plotting/visualizations
    raw_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_val_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    raw_test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=raw_transform
    )

    return train_val_dataset, test_dataset, raw_test_dataset


def get_mnist_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_split: float = 0.1,
    random_seed: int = 42
):
    """
    Creates reproducible DataLoader splits for training, validation, and testing.
    
    Args:
        data_dir: Root directory for MNIST storage
        batch_size: Mini-batch size for stochastic gradient descent
        val_split: Fraction of training set allocated to validation (default 10%)
        random_seed: Seed for split reproducibility
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, raw_test_dataset)
    """
    train_val_dataset, test_dataset, raw_test_dataset = get_mnist_datasets(data_dir)

    # Deterministic train/validation partition
    val_size = int(len(train_val_dataset) * val_split)
    train_size = len(train_val_dataset) - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        train_val_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, raw_test_dataset
