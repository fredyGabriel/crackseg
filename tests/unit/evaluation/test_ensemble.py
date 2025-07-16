from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from crackseg.evaluation.ensemble import ensemble_evaluate


def test_ensemble_evaluate_empty_checkpoints():
    """Should raise ValueError if no checkpoints are provided."""
    empty_dataset = torch.utils.data.TensorDataset(torch.empty(0))
    with pytest.raises(ValueError):
        ensemble_evaluate(
            checkpoint_paths=[],
            config=OmegaConf.create({"model": {}}),
            dataloader=DataLoader(empty_dataset),
            metrics={},
        )


def test_ensemble_evaluate_incompatible_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise an error if models return incompatible outputs."""

    class DummyModelA(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones(x.shape[0], 1, 4, 4)

    class DummyModelB(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones(x.shape[0], 2, 4, 4)

    models: list[tuple[torch.nn.Module, dict[str, Any]]] = [
        (DummyModelA(), {}),
        (DummyModelB(), {}),
    ]

    def load_model_side_effect(
        *args: Any, **kwargs: Any
    ) -> tuple[torch.nn.Module, dict[str, Any]]:
        return models.pop(0)

    monkeypatch.setattr(
        "crackseg.evaluation.ensemble.load_model_from_checkpoint",
        load_model_side_effect,
    )

    class DummyDataset(Dataset[dict[str, torch.Tensor]]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {
                "image": torch.zeros(2, 3, 4, 4),
                "mask": torch.zeros(2, 1, 4, 4),
            }

    dataloader = DataLoader(DummyDataset(), batch_size=1)
    metrics = {"dummy": lambda o, t: torch.tensor(1.0)}
    with pytest.raises(ValueError):
        ensemble_evaluate(
            checkpoint_paths=["ckpt1.pth", "ckpt2.pth"],
            config=OmegaConf.create({"model": {}}),
            dataloader=dataloader,
            metrics=metrics,
        )
