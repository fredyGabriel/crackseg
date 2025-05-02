"""Test for training loop with a simple model and synthetic dataset."""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from unittest.mock import MagicMock

from src.training.losses import BCEDiceLoss
from src.training.metrics import IoUScore, F1Score
from src.training.trainer import Trainer


class SimpleSegmentationModel(nn.Module):
    """A very simple segmentation model for testing."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class SyntheticCrackDataset(Dataset):
    """Simple synthetic dataset for testing the training loop."""
    def __init__(self, size=100, image_size=64):
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """Get a random image and mask pair.

        Returns:
            tuple: (image, mask) tensors with correct types for training.
        """
        # Generate random image and mask
        image = torch.randn(1, self.image_size, self.image_size)
        mask = torch.randint(
            0, 2, (1, self.image_size, self.image_size),
            dtype=torch.float32
        )
        return image, mask


def test_training_loop():
    """Test a training loop with synthetic data."""
    # Create synthetic dataset and dataloaders
    train_dataset = SyntheticCrackDataset()
    val_dataset = SyntheticCrackDataset(size=20)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Create model and loss function
    model = SimpleSegmentationModel()
    loss_fn = BCEDiceLoss()
    # Configure mocks to return tensors
    iou_mock = MagicMock(spec=IoUScore)
    iou_mock.return_value = torch.tensor(0.75)  # Example value
    f1_mock = MagicMock(spec=F1Score)
    f1_mock.return_value = torch.tensor(0.85)  # Example value
    metrics = {
        "IoU": iou_mock,
        "F1": f1_mock
    }

    # Create trainer config
    trainer_cfg = OmegaConf.create({
        'trainer': {
            'epochs': 2,
            'device': 'cpu',
            'use_amp': False,
            'gradient_accumulation_steps': 1,
            'verbose': True,
            'optimizer': {
                '_target_': 'torch.optim.Adam',
                'lr': 0.001
            },
            'lr_scheduler': None
        }
    })

    # Create trainer
    logger_mock = MagicMock()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metrics_dict=metrics,
        cfg=trainer_cfg,
        logger_instance=logger_mock
    )

    # Run training and get results
    results = trainer.train()

    # Basic assertions
    assert isinstance(results, dict), \
        "Training should return a dict of metrics"
    assert 'loss' in results, "Results should include loss"
    assert 'IoU' in results, "Results should include IoU score"
    assert 'F1' in results, "Results should include F1 score"
