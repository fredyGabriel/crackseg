import os
import pathlib
from collections.abc import Callable
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.evaluation.core import evaluate_model  # type: ignore
from src.evaluation.ensemble import ensemble_evaluate  # type: ignore
from src.evaluation.loading import load_model_from_checkpoint
from src.evaluation.results import save_evaluation_results


class DummyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, n: int = 2, shape: tuple[int, int, int] = (3, 4, 4)):
        self.n = n
        self.shape = shape

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.zeros(*self.shape),
            "mask": torch.zeros(1, self.shape[1], self.shape[2]),
        }


def test_evaluation_pipeline(tmp_path: pathlib.Path) -> None:
    """Integration: evaluate a dummy model and save results."""

    # 1. Dummy model
    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones(x.shape[0], 1, 4, 4)

    model = DummyModel()

    # 2. Realistic dataloader
    dataset = DummyDataset(n=2, shape=(3, 4, 4))
    dataloader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        dataset, batch_size=2
    )
    metrics: dict[
        str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = {"dummy": lambda o, t: torch.tensor(1.0)}

    # 3. Evaluate
    config = OmegaConf.create({"foo": "bar"})
    results, (inputs, _targets, _outputs) = evaluate_model(
        model, dataloader, metrics, torch.device("cpu"), config
    )
    assert isinstance(results, dict)
    assert "test_dummy" in results
    # Only the first batch is stored for visualization (by contract)
    assert inputs.shape[0] == 2  # 1 batch x 2 samples  # noqa: PLR2004

    # 4. Save results
    checkpoint = "dummy_ckpt.pth"
    save_evaluation_results(results, config, checkpoint, str(tmp_path))

    # 5. Check files
    metrics_dir = os.path.join(tmp_path, "metrics")
    assert os.path.exists(os.path.join(metrics_dir, "evaluation_results.yaml"))
    assert os.path.exists(os.path.join(metrics_dir, "evaluation_results.txt"))


def test_ensemble_evaluation_pipeline(tmp_path: pathlib.Path) -> None:
    """Integration: ensemble evaluation with two dummy models."""

    class DummyModelA(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones(x.shape[0], 1, 4, 4)

    class DummyModelB(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 1, 4, 4)

    # Simulate checkpoints: monkeypatch load_model_from_checkpoint
    from src.evaluation import ensemble as ensemble_mod

    models: list[torch.nn.Module] = [DummyModelA(), DummyModelB()]

    def fake_load_model_from_checkpoint(
        path: str, device: torch.device
    ) -> tuple[torch.nn.Module, dict[str, Any]]:
        return models.pop(0), {}

    ensemble_mod.load_model_from_checkpoint = fake_load_model_from_checkpoint

    dataset = DummyDataset(n=2, shape=(3, 4, 4))
    dataloader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        dataset, batch_size=2
    )
    metrics: dict[
        str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = {"dummy": lambda o, t: torch.tensor(1.0)}
    checkpoint_paths = ["ckptA.pth", "ckptB.pth"]
    config = OmegaConf.create({"model": {}})
    results = ensemble_evaluate(
        checkpoint_paths=checkpoint_paths,
        config=config,
        dataloader=dataloader,
        metrics=metrics,
    )
    assert isinstance(results, dict)
    assert "ensemble_dummy" in results
    # Check ensemble results file
    ensemble_dir = os.path.join(tmp_path, "metrics", "ensemble")
    assert os.path.exists(os.path.join(ensemble_dir, "ensemble_results.yaml"))


def test_load_model_from_checkpoint_corrupt(tmp_path: pathlib.Path) -> None:
    """Integration: should raise error when loading a corrupt checkpoint."""
    corrupt_ckpt = tmp_path / "corrupt.pth"
    corrupt_ckpt.write_bytes(b"not a real checkpoint")
    # Force torch.load to raise an error only in this context
    import torch as torch_mod

    orig_torch_load = torch_mod.load  # type: ignore

    def raise_runtime_error(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("corrupt")

    torch_mod.load = raise_runtime_error
    try:
        import pytest

        with pytest.raises(RuntimeError):
            load_model_from_checkpoint(str(corrupt_ckpt), torch.device("cpu"))
    finally:
        torch_mod.load = orig_torch_load


def test_evaluation_pipeline_incompatible_metric(
    tmp_path: pathlib.Path,
) -> None:
    """Integration: should raise error with incompatible metric."""

    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones(x.shape[0], 1, 4, 4)

    model = DummyModel()
    dataset = DummyDataset(n=2, shape=(3, 4, 4))
    dataloader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        dataset, batch_size=2
    )

    # Incompatible metric: expects a different shape
    def incompatible_metric(
        outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        # Tries to access a non-existent channel
        return outputs[:, 99].mean()

    metrics: dict[
        str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = {"incompatible": incompatible_metric}
    import pytest

    config = OmegaConf.create({})
    with pytest.raises(IndexError):
        evaluate_model(model, dataloader, metrics, torch.device("cpu"), config)


def test_evaluation_pipeline_multiple_batches_and_visualization(
    tmp_path: pathlib.Path,
) -> None:
    """Integration: evaluate with multiple batches and check visualizations."""

    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Returns a binary mask
            return (x[:, :1] > 0.5).float()  # noqa: PLR2004

    model = DummyModel()
    # Create dataset with 6 samples, batch size 2 (3 batches)
    dataset = DummyDataset(n=6, shape=(3, 4, 4))
    dataloader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        dataset, batch_size=2
    )
    metrics: dict[
        str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = {"dummy": lambda o, t: (o == t).float().mean()}
    config = OmegaConf.create({"foo": "bar"})
    results, (inputs, _targets, _outputs) = evaluate_model(
        model, dataloader, metrics, torch.device("cpu"), config
    )
    assert isinstance(results, dict)
    assert "test_dummy" in results
    # Only the first 2 batches are stored for visualization (by contract)
    assert inputs.shape[0] == 4  # 2 batches x 2 samples  # noqa: PLR2004
    # Save results and check files
    checkpoint = "dummy_ckpt.pth"
    save_evaluation_results(results, config, checkpoint, str(tmp_path))
    metrics_dir = os.path.join(tmp_path, "metrics")
    assert os.path.exists(os.path.join(metrics_dir, "evaluation_results.yaml"))
    assert os.path.exists(os.path.join(metrics_dir, "evaluation_results.txt"))
    # Check visualizations if they exist
    vis_dir = os.path.join(tmp_path, "visualizations")
    if os.path.exists(vis_dir):
        assert any(
            os.listdir(vis_dir)
        ), "Visualization directory should not be empty"
