import torch
import os
from src.evaluation.core import evaluate_model
from src.evaluation.results import save_evaluation_results
from src.evaluation.ensemble import ensemble_evaluate
from src.evaluation.loading import load_model_from_checkpoint


def test_evaluation_pipeline(tmp_path):
    """Integration: evaluate a dummy model and save results."""
    # 1. Dummy model
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 1, 4, 4)
    model = DummyModel()

    # 2. Realistic dataloader
    batch = {'image': torch.zeros(2, 3, 4, 4), 'mask': torch.zeros(2, 1, 4, 4)}
    dataloader = [batch]
    metrics = {'dummy': lambda o, t: torch.tensor(1.0)}

    # 3. Evaluate
    results, (inputs, targets, outputs) = evaluate_model(
        model, dataloader, metrics, torch.device('cpu')
    )
    assert isinstance(results, dict)
    assert 'test_dummy' in results
    # Only the first batch is stored for visualization (by contract)
    assert inputs.shape[0] == 2  # 1 batch x 2 samples

    # 4. Save results
    config = {'foo': 'bar'}
    checkpoint = 'dummy_ckpt.pth'
    save_evaluation_results(results, config, checkpoint, str(tmp_path))

    # 5. Check files
    metrics_dir = os.path.join(tmp_path, 'metrics')
    assert os.path.exists(os.path.join(metrics_dir, 'evaluation_results.yaml'))
    assert os.path.exists(os.path.join(metrics_dir, 'evaluation_results.txt'))


def test_ensemble_evaluation_pipeline(tmp_path):
    """Integration: ensemble evaluation with two dummy models."""
    class DummyModelA(torch.nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 1, 4, 4)

    class DummyModelB(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1, 4, 4)

    # Simulate checkpoints: monkeypatch load_model_from_checkpoint
    from src.evaluation import ensemble as ensemble_mod
    models = [DummyModelA(), DummyModelB()]

    def fake_load_model_from_checkpoint(path, device):
        return models.pop(0), {}

    ensemble_mod.load_model_from_checkpoint = fake_load_model_from_checkpoint

    batch = {'image': torch.zeros(2, 3, 4, 4), 'mask': torch.zeros(2, 1, 4, 4)}
    dataloader = [batch]
    metrics = {'dummy': lambda o, t: torch.tensor(1.0)}
    checkpoint_paths = ['ckptA.pth', 'ckptB.pth']
    config = {'model': {}}
    results = ensemble_evaluate(
        checkpoint_paths=checkpoint_paths,
        config=config,
        dataloader=dataloader,
        metrics=metrics,
        device=torch.device('cpu'),
        output_dir=str(tmp_path)
    )
    assert isinstance(results, dict)
    assert 'ensemble_dummy' in results
    # Check ensemble results file
    ensemble_dir = os.path.join(tmp_path, 'metrics', 'ensemble')
    assert os.path.exists(os.path.join(ensemble_dir, 'ensemble_results.yaml'))


def test_load_model_from_checkpoint_corrupt(tmp_path):
    """Integration: should raise error when loading a corrupt checkpoint."""
    corrupt_ckpt = tmp_path / 'corrupt.pth'
    corrupt_ckpt.write_bytes(b'not a real checkpoint')
    # Force torch.load to raise an error only in this context
    import torch as torch_mod
    orig_torch_load = torch_mod.load

    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("corrupt")

    torch_mod.load = raise_runtime_error
    try:
        import pytest
        with pytest.raises(RuntimeError):
            load_model_from_checkpoint(str(corrupt_ckpt), torch.device('cpu'))
    finally:
        torch_mod.load = orig_torch_load


def test_evaluation_pipeline_incompatible_metric(tmp_path):
    """Integration: should raise error with incompatible metric."""
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 1, 4, 4)
    model = DummyModel()
    batch = {'image': torch.zeros(2, 3, 4, 4), 'mask': torch.zeros(2, 1, 4, 4)}
    dataloader = [batch]

    # Incompatible metric: expects a different shape
    def incompatible_metric(outputs, targets):
        # Tries to access a non-existent channel
        return outputs[:, 99].mean()
    metrics = {'incompatible': incompatible_metric}
    import pytest
    with pytest.raises(Exception):
        evaluate_model(model, dataloader, metrics, torch.device('cpu'))


def test_evaluation_pipeline_multiple_batches_and_visualization(tmp_path):
    """Integration: evaluate with multiple batches and check visualizations."""
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # Returns a binary mask
            return (x[:, :1] > 0.5).float()
    model = DummyModel()
    # Create 3 batches
    dataloader = [
        {'image': torch.rand(2, 3, 4, 4),
         'mask': torch.randint(0, 2, (2, 1, 4, 4))}
        for _ in range(3)
    ]
    metrics = {'dummy': lambda o, t: (o == t).float().mean()}
    results, (inputs, targets, outputs) = evaluate_model(
        model, dataloader, metrics, torch.device('cpu')
    )
    assert isinstance(results, dict)
    assert 'test_dummy' in results
    # Only the first 2 batches are stored for visualization (by contract)
    assert inputs.shape[0] == 4  # 2 batches x 2 samples
    # Save results and check files
    config = {'foo': 'bar'}
    checkpoint = 'dummy_ckpt.pth'
    save_evaluation_results(results, config, checkpoint, str(tmp_path))
    metrics_dir = os.path.join(tmp_path, 'metrics')
    assert os.path.exists(os.path.join(metrics_dir, 'evaluation_results.yaml'))
    assert os.path.exists(os.path.join(metrics_dir, 'evaluation_results.txt'))
    # Check visualizations if they exist
    vis_dir = os.path.join(tmp_path, 'visualizations')
    if os.path.exists(vis_dir):
        assert any(os.listdir(vis_dir)), (
            "Visualization directory should not be empty"
        )
