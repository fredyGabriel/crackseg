import pytest
import torch
from src.evaluation.core import evaluate_model


def test_evaluate_model_empty_dataloader():
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones_like(x[:, :1])
    model = DummyModel()
    loader = []  # Empty dataloader
    metrics = {'dummy': lambda o, t: torch.tensor(1.0)}
    results, (inputs, targets, outputs) = evaluate_model(
        model, loader, metrics, torch.device('cpu')
    )
    assert isinstance(results, dict)
    assert len(results) == 0 or all(
        v == 0 or v == 1.0 for v in results.values()
    )
    assert inputs.shape[0] == 0


def test_evaluate_model_model_raises():
    class FailingModel(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError('fail')
    model = FailingModel()
    batch = {'image': torch.zeros(2, 3, 4, 4), 'mask': torch.zeros(2, 1, 4, 4)}
    loader = [batch]
    metrics = {'dummy': lambda o, t: torch.tensor(1.0)}
    with pytest.raises(RuntimeError):
        evaluate_model(model, loader, metrics, torch.device('cpu'))
