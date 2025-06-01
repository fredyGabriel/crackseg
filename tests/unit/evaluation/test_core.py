import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.evaluation.core import evaluate_model


def test_evaluate_model_empty_dataloader():
    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(x[:, :1])

    model = DummyModel()

    class EmptyDataset(Dataset[torch.Tensor]):
        def __len__(self) -> int:
            return 0

        def __getitem__(self, idx: int) -> torch.Tensor:
            raise IndexError

    loader = DataLoader(EmptyDataset(), batch_size=1)
    metrics = {"dummy": lambda o, t: torch.tensor(1.0)}
    config = OmegaConf.create({})
    results, (inputs, targets, outputs) = evaluate_model(
        model, loader, metrics, torch.device("cpu"), config=config
    )
    assert isinstance(results, dict)
    assert len(results) == 0 or all(v in {0, 1.0} for v in results.values())
    assert inputs.shape[0] == 0


def test_evaluate_model_model_raises():
    class FailingModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("fail")

    model = FailingModel()

    class DummyDataset(Dataset[dict[str, torch.Tensor]]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {
                "image": torch.zeros(2, 3, 4, 4),
                "mask": torch.zeros(2, 1, 4, 4),
            }

    loader = DataLoader(DummyDataset(), batch_size=1)
    metrics = {"dummy": lambda o, t: torch.tensor(1.0)}
    config = OmegaConf.create({})
    with pytest.raises(RuntimeError):
        evaluate_model(
            model, loader, metrics, torch.device("cpu"), config=config
        )
