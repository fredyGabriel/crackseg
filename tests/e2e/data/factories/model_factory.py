"""
Model data factory for E2E testing. This module provides PyTorch model
checkpoint generation for testing model loading and compatibility
scenarios.
"""

import tempfile
from pathlib import Path
from typing import Any

import torch

from .base import BaseDataFactory, TestData


class ModelDataFactory(BaseDataFactory):
    """Factory for generating test model data and checkpoints."""

    def generate(
        self,
        model_type: str = "simple",
        include_optimizer: bool = True,
        corrupt_data: bool = False,
        **kwargs: Any,
    ) -> TestData:
        """
        Generate test model checkpoint data. Args: model_type: Type of model
        checkpoint ('simple', 'complex') include_optimizer: Whether to include
        optimizer state corrupt_data: Whether to generate corrupted checkpoint
        **kwargs: Additional model parameters Returns: TestData containing
        generated model checkpoint
        """
        # Generate model state
        if model_type == "simple":
            model_state = {
                "conv1.weight": torch.randn(64, 3, 3, 3),
                "conv1.bias": torch.randn(64),
                "fc.weight": torch.randn(2, 128),
                "fc.bias": torch.randn(2),
            }
        else:  # complex
            model_state = {}
            for i in range(4):
                model_state[f"encoder.layer{i}.conv.weight"] = torch.randn(
                    64 * (2**i), 3 if i == 0 else 64 * (2 ** (i - 1)), 3, 3
                )
                model_state[f"encoder.layer{i}.conv.bias"] = torch.randn(
                    64 * (2**i)
                )

        # Build checkpoint
        checkpoint = {
            "model_state_dict": model_state,
            "epoch": kwargs.get("epoch", 1),
            "best_metric": kwargs.get("best_metric", 0.85),
            "config": {"model_name": model_type, "num_classes": 2},
        }

        if include_optimizer:
            checkpoint["optimizer_state_dict"] = {
                "state": {},
                "param_groups": [{"lr": 0.001}],
            }

        if corrupt_data:
            checkpoint["model_state_dict"] = {"corrupted": "data"}

        # Save checkpoint
        temp_dir = (
            self.environment_manager.state["artifacts_dir"]
            if self.environment_manager
            else Path(tempfile.gettempdir())
        )
        temp_dir.mkdir(exist_ok=True)

        checkpoint_file = (
            temp_dir / f"test_model_{model_type}_{id(checkpoint)}.pth"
        )

        torch.save(checkpoint, checkpoint_file)

        if self.environment_manager:
            self.environment_manager.register_temp_file(checkpoint_file)

        return {
            "data_type": "model",
            "file_path": checkpoint_file,
            "metadata": {
                "model_type": model_type,
                "include_optimizer": include_optimizer,
                "corrupt_data": corrupt_data,
                "format": "pytorch",
            },
            "cleanup_required": True,
        }
