"""Data generation utilities for end-to-end pipeline testing."""

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def create_synthetic_dataset():
    """Create a synthetic dataset for end-to-end tests."""
    # Create synthetic data (random 16x16 images)
    num_samples = 20
    image_size = 16

    # Create random tensors for images and masks
    images = torch.rand(num_samples, 3, image_size, image_size)
    masks = torch.randint(
        0, 2, (num_samples, 1, image_size, image_size)
    ).float()

    # Create dataset and split into train/val/test
    dataset = TensorDataset(images, masks)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader
