"""Test that learning rate schedulers are correctly instantiated from Hydra
configs."""

import os

import pytest
import torch
from omegaconf import OmegaConf

from src.training.factory import create_lr_scheduler


@pytest.fixture
def dummy_optimizer():
    """Create a dummy optimizer for scheduler testing."""
    # Use torch.nn.Linear as a simple model with parameters
    model = torch.nn.Linear(10, 5)
    return torch.optim.Adam(model.parameters(), lr=0.001)


# Helper to get absolute config path regardless of working directory
def get_absolute_config_path(relative_path: str) -> str:
    """Convert a relative config path to an absolute path regardless of
    working directory."""
    # Get the project root: if we're in scripts/, go up one level
    if os.path.basename(os.getcwd()) == "scripts":
        project_root = os.path.dirname(os.getcwd())
    else:
        project_root = os.getcwd()

    return os.path.join(project_root, relative_path)


@pytest.mark.parametrize(
    "config_path, expected_cls",
    [
        (
            "configs/training/lr_scheduler/step_lr.yaml",
            torch.optim.lr_scheduler.StepLR,
        ),
        (
            "configs/training/lr_scheduler/cosine.yaml",
            torch.optim.lr_scheduler.CosineAnnealingLR,
        ),
        (
            "configs/training/lr_scheduler/reduce_on_plateau.yaml",
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ),
    ],
)
def test_scheduler_instantiation_from_config(
    config_path: str,
    expected_cls: type,
    dummy_optimizer: torch.optim.Optimizer,
) -> None:
    # Use absolute path to find config regardless of working directory
    abs_config_path = get_absolute_config_path(config_path)
    cfg = OmegaConf.load(abs_config_path)

    # Create scheduler based on config
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Scheduler config must be a dict")
    scheduler = create_lr_scheduler(dummy_optimizer, dict(container))  # type: ignore[arg-type]

    assert isinstance(scheduler, expected_cls)
