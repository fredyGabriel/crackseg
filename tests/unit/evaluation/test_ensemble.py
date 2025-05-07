import pytest
import torch
from src.evaluation.ensemble import ensemble_evaluate


def test_ensemble_evaluate_empty_checkpoints():
    """Should raise ValueError if no checkpoints are provided."""
    with pytest.raises(ValueError):
        ensemble_evaluate(
            checkpoint_paths=[],
            config={'model': {}},
            dataloader=[],
            metrics={},
            device=torch.device('cpu'),
            output_dir='.'
        )


def test_ensemble_evaluate_incompatible_outputs(tmp_path, monkeypatch):
    """Should raise an error if models return incompatible outputs."""
    # Mock load_model_from_checkpoint to return models with different output
    # shapes
    class DummyModelA(torch.nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 1, 4, 4)

    class DummyModelB(torch.nn.Module):
        def forward(self, x):
            # Different channel count
            return torch.ones(x.shape[0], 2, 4, 4)

    # Helper to return models in order
    models = [(DummyModelA(), {}), (DummyModelB(), {})]

    def load_model_side_effect(*args, **kwargs):
        return models.pop(0)

    monkeypatch.setattr(
        'src.evaluation.ensemble.load_model_from_checkpoint',
        load_model_side_effect
    )
    batch = {'image': torch.zeros(2, 3, 4, 4), 'mask': torch.zeros(2, 1, 4, 4)}
    dataloader = [batch]
    metrics = {'dummy': lambda o, t: torch.tensor(1.0)}
    with pytest.raises(Exception):
        ensemble_evaluate(
            checkpoint_paths=['ckpt1.pth', 'ckpt2.pth'],
            config={'model': {}},
            dataloader=dataloader,
            metrics=metrics,
            device=torch.device('cpu'),
            output_dir=str(tmp_path)
        )
