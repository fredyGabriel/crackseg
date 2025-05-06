"""Test that learning rate schedulers are correctly instantiated from Hydra
configs."""

import pytest
import torch
from omegaconf import OmegaConf
from src.training.factory import create_lr_scheduler


@pytest.fixture
def dummy_optimizer():
    model = torch.nn.Linear(10, 2)
    return torch.optim.Adam(model.parameters(), lr=0.001)


@pytest.mark.parametrize("config_path, expected_cls", [
    ("configs/training/lr_scheduler/step_lr.yaml",
     torch.optim.lr_scheduler.StepLR),
    ("configs/training/lr_scheduler/cosine.yaml",
     torch.optim.lr_scheduler.CosineAnnealingLR),
    ("configs/training/lr_scheduler/reduce_on_plateau.yaml",
     torch.optim.lr_scheduler.ReduceLROnPlateau),
])
def test_scheduler_instantiation_from_config(config_path, expected_cls,
                                             dummy_optimizer):
    cfg = OmegaConf.load(config_path)
    scheduler = create_lr_scheduler(dummy_optimizer, cfg)
    assert isinstance(scheduler, expected_cls)
