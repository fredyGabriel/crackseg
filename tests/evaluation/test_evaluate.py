import os
import tempfile
import torch
from unittest.mock import MagicMock

import src.evaluate as evaluate


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(
        'sys.argv', ['evaluate.py', '--checkpoint', 'ckpt.pth.tar']
    )
    args = evaluate.parse_args()
    assert args.checkpoint == 'ckpt.pth.tar'
    assert args.visualize_samples == 5


def test_setup_output_directory_creates_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = evaluate.setup_output_directory(tmpdir)
        assert os.path.exists(out_dir)
        assert os.path.exists(os.path.join(out_dir, 'metrics'))
        assert os.path.exists(os.path.join(out_dir, 'visualizations'))


def test_load_model_from_checkpoint(monkeypatch):
    dummy_model = torch.nn.Linear(2, 2)
    # Creamos un state_dict compatible con Linear
    dummy_state = {
        'model_state_dict': {
            'weight': torch.randn(2, 2),
            'bias': torch.randn(2)
        },
        'config': {'model': {}}
    }
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: dummy_state)
    monkeypatch.setattr(evaluate, 'create_unet', lambda cfg: dummy_model)
    model, data = evaluate.load_model_from_checkpoint(
        'fake.pth', torch.device('cpu')
    )
    assert isinstance(model, torch.nn.Module)
    assert 'config' in data


def test_get_evaluation_dataloader(monkeypatch):
    dummy_loader = MagicMock()
    dummy_loader.dataset = [1, 2, 3]  # Simula un dataset con 3 muestras
    monkeypatch.setattr(
        evaluate,
        'create_dataloaders_from_config',
        lambda **kwargs: {'test': {'dataloader': dummy_loader}}
    )
    cfg = {'data': {}}
    loader = evaluate.get_evaluation_dataloader(cfg)
    assert loader is dummy_loader


def test_evaluate_model_basic():
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones_like(x[:, :1])
    model = DummyModel()
    batch = {'image': torch.zeros(2, 3, 4, 4),
             'mask': torch.zeros(2, 1, 4, 4)}
    loader = [batch]
    metrics = {'dummy': lambda o, t: torch.tensor(1.0)}
    results, (inputs, targets, outputs) = evaluate.evaluate_model(
        model, loader, metrics, torch.device('cpu')
    )
    assert 'test_dummy' in results
    assert inputs.shape[0] == 2


def test_visualize_predictions_creates_files(tmp_path):
    inputs = torch.zeros(2, 3, 4, 4)
    targets = torch.zeros(2, 1, 4, 4)
    outputs = torch.zeros(2, 1, 4, 4)
    evaluate.visualize_predictions(
        inputs, targets, outputs, str(tmp_path), num_samples=1
    )
    vis_dir = os.path.join(tmp_path, 'visualizations')
    assert any(f.endswith('.png') for f in os.listdir(vis_dir))


def test_save_evaluation_results_creates_files(tmp_path):
    results = {'test_metric': 1.0}
    config = {'foo': 'bar'}
    checkpoint = 'ckpt.pth.tar'
    evaluate.save_evaluation_results(
        results, config, checkpoint, str(tmp_path)
    )
    metrics_dir = os.path.join(tmp_path, 'metrics')
    assert os.path.exists(os.path.join(metrics_dir, 'evaluation_results.yaml'))
    assert os.path.exists(os.path.join(metrics_dir, 'evaluation_summary.txt'))
