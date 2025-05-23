import os
import tempfile
from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf

from src.evaluation.core import evaluate_model
from src.evaluation.data import get_evaluation_dataloader
from src.evaluation.ensemble import ensemble_evaluate
from src.evaluation.loading import load_model_from_checkpoint
from src.evaluation.results import save_evaluation_results
from src.evaluation.setup import parse_args, setup_output_directory
from src.utils.visualization import visualize_predictions


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["evaluate.py", "--checkpoint", "ckpt.pth.tar"]
    )
    args = parse_args()
    assert args.checkpoint == "ckpt.pth.tar"
    assert args.visualize_samples == 5  # noqa: PLR2004


def test_setup_output_directory_creates_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = setup_output_directory(tmpdir)
        assert os.path.exists(out_dir)
        assert os.path.exists(os.path.join(out_dir, "metrics"))
        assert os.path.exists(os.path.join(out_dir, "visualizations"))


@patch("src.utils.checkpointing.Path.exists", return_value=True)
@patch("src.utils.checkpointing.torch.load")
@patch("src.evaluation.loading.create_unet")
def test_load_model_from_checkpoint(
    mock_create_unet, mock_torch_load, mock_exists
):
    # Create a dummy model and test data
    dummy_model = torch.nn.Linear(2, 2)
    mock_create_unet.return_value = dummy_model

    # Simulate a checkpoint with required data using valid components
    encoder_out_channels = 512  # Typical output for CNNEncoder with depth 4
    encoder_skip_channels = [
        64,
        128,
        256,
        512,
    ]  # Skip channels for depth=4, init_features=64
    bottleneck_out_channels = encoder_out_channels * 2
    dummy_state = {
        "model_state_dict": {"weight": torch.randn(2, 2)},
        "config": {
            "model": {
                "encoder": {
                    "type": "CNNEncoder",
                    "in_channels": 3,
                    "init_features": 64,
                    "depth": 4,
                },
                "bottleneck": {
                    "type": "CNNBottleneckBlock",
                    "in_channels": encoder_out_channels,
                    "out_channels": bottleneck_out_channels,
                },
                "decoder": {
                    "type": "CNNDecoder",
                    "in_channels": bottleneck_out_channels,
                    "skip_channels_list": encoder_skip_channels,
                },
            }
        },
    }
    mock_torch_load.return_value = dummy_state

    # Also ensure load_state_dict does not fail
    with patch.object(torch.nn.Module, "load_state_dict"):
        # Call the function under test
        model, data = load_model_from_checkpoint(
            "fake.pth", torch.device("cpu")
        )

    # Check results
    assert model is dummy_model
    mock_create_unet.assert_called_once()
    mock_exists.assert_called()


def test_get_evaluation_dataloader(monkeypatch):
    dummy_loader = MagicMock()
    dummy_loader.dataset = [1, 2, 3]  # Simulate a dataset with 3 samples
    monkeypatch.setattr(
        "src.evaluation.data.create_dataloaders_from_config",
        lambda **kwargs: {"test": {"dataloader": dummy_loader}},
    )
    cfg: dict[str, object] = {"data": {}}
    loader = get_evaluation_dataloader(cfg)
    assert loader is dummy_loader


def test_evaluate_model_basic():
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones_like(x[:, :1])

    model = DummyModel()

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "image": torch.zeros(2, 3, 4, 4),
                "mask": torch.zeros(2, 1, 4, 4),
            }

    loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)
    metrics = {"dummy": lambda o, t: torch.tensor(1.0)}
    config = OmegaConf.create({})
    results, (inputs, targets, outputs) = evaluate_model(
        model, loader, metrics, torch.device("cpu"), config=config
    )
    assert "test_dummy" in results
    assert inputs.shape[0] == 2  # noqa: PLR2004


def test_visualize_predictions_creates_files(tmp_path):
    inputs = torch.zeros(2, 3, 4, 4)
    targets = torch.zeros(2, 1, 4, 4)
    outputs = torch.zeros(2, 1, 4, 4)
    visualize_predictions(
        inputs, targets, outputs, str(tmp_path), num_samples=1
    )
    vis_dir = os.path.join(tmp_path, "visualizations")
    assert any(f.endswith(".png") for f in os.listdir(vis_dir))


def test_save_evaluation_results_creates_files(tmp_path):
    results = {"test_metric": 1.0}
    config = {"foo": "bar"}
    checkpoint = "ckpt.pth.tar"
    save_evaluation_results(results, config, checkpoint, str(tmp_path))
    metrics_dir = os.path.join(tmp_path, "metrics")
    assert os.path.exists(os.path.join(metrics_dir, "evaluation_results.yaml"))
    txt_path = os.path.join(metrics_dir, "evaluation_results.txt")
    assert os.path.exists(txt_path)


@patch("pathlib.Path.exists", return_value=True)
@patch("torch.load")
def test_ensemble_evaluate_creates_results_and_files(
    mock_torch_load, mock_exists, tmp_path
):
    # Dummy model for ensemble
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones_like(x[:, :1])

    # Create a simulated checkpoint
    dummy_state = {
        "model_state_dict": {
            "weight": torch.randn(2, 2),
            "bias": torch.randn(2),
        },
        "config": {"model": {}},
    }
    mock_torch_load.return_value = dummy_state

    # Mock load_model_from_checkpoint for ensemble.py
    with patch(
        "src.evaluation.ensemble.load_model_from_checkpoint"
    ) as mock_load:
        # Prepare the mock
        mock_load.return_value = (DummyModel(), {"config": {"model": {}}})

        # Dummy dataloader: 2 batches of 2 samples
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return {
                    "image": torch.zeros(2, 3, 4, 4),
                    "mask": torch.zeros(2, 1, 4, 4),
                }

        dataloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)

        # Dummy metric
        metrics = {"dummy": lambda o, t: torch.tensor(1.0)}

        # Mock for visualize_predictions
        with patch("src.utils.visualization.visualize_predictions"):
            results = ensemble_evaluate(
                checkpoint_paths=["ckpt1.pth", "ckpt2.pth"],
                config=OmegaConf.create({"model": {}}),
                dataloader=dataloader,
                metrics=metrics,
            )

    # Check that the result key is present
    assert "ensemble_dummy" in results
    assert results["ensemble_dummy"] == 1.0

    # Check that the ensemble_results.yaml file was created
    ensemble_dir = os.path.join(tmp_path, "metrics", "ensemble")
    yaml_path = os.path.join(ensemble_dir, "ensemble_results.yaml")
    assert os.path.exists(yaml_path)
    # Check that the file contains the metric
    with open(yaml_path) as f:
        content = f.read()
        assert "ensemble_dummy" in content
