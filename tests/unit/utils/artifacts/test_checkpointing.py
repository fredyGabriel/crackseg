"""Unit tests for checkpointing utilities."""

# import os # Unused
# import sys # Unused
from pathlib import Path
from typing import Any

# import time  # Removed unused import
from unittest.mock import ANY, MagicMock, patch

import pytest
import torch

from crackseg.utils.checkpointing import (
    CheckpointSaveConfig,
    CheckpointSpec,
    adapt_legacy_checkpoint,
    create_standardized_filename,
    generate_checkpoint_metadata,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_completeness,
    verify_checkpoint_integrity,
)

# from torch.cuda.amp import GradScaler # Removed unused import


# --- Fixtures ---


@pytest.fixture
def mock_model() -> MagicMock:
    """Fixture to create a mock torch.nn.Module."""
    model = MagicMock(spec=torch.nn.Module)
    model.state_dict.return_value = {
        f"layer{i}.weight": torch.randn(5, 5) for i in range(1, 3)
    }
    # Add device attribute for load_checkpoint test compatibility if
    # device=None
    model.device = torch.device("cpu")  # Mock device attribute
    return model


@pytest.fixture
def mock_optimizer() -> MagicMock:
    """Fixture to create a mock torch.optim.Optimizer."""
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    optimizer.state_dict.return_value = {
        "param_groups": [],
        "state": {i: {"exp_avg": torch.randn(5, 5)} for i in range(2)},
    }
    return optimizer


# Removed mock_scaler fixture


@pytest.fixture
def mock_model_state(mock_model: MagicMock) -> dict[str, torch.Tensor]:
    """Fixture for a sample model state_dict."""
    return mock_model.state_dict()


@pytest.fixture
def mock_optimizer_state(mock_optimizer: MagicMock) -> dict[str, Any]:
    """Fixture for a sample optimizer state_dict."""
    return mock_optimizer.state_dict()


# Removed mock_scaler_state fixture

# --- Tests for save_checkpoint ---


@patch("torch.save")
def test_save_checkpoint_creates_file_and_calls_torch_save(
    mock_torch_save: MagicMock,
    tmp_path: Path,
    mock_model: MagicMock,
    mock_optimizer: MagicMock,
    mock_model_state: dict[str, torch.Tensor],
    mock_optimizer_state: dict[str, Any],
) -> None:
    """Test save_checkpoint creates dir/file and calls torch.save."""
    checkpoint_dir = tmp_path / "test_ckpts"
    epoch = 5
    metrics = {"loss": 0.1, "iou": 0.9}
    # Pass metrics via additional_data
    additional_data = {"metrics": metrics, "best_metric_value": 0.9}

    config = CheckpointSaveConfig(
        checkpoint_dir=checkpoint_dir,
        filename="test_last.pth",
        additional_data=additional_data,
    )
    save_checkpoint(
        model=mock_model,
        optimizer=mock_optimizer,
        epoch=epoch,
        config=config,
    )

    # Check directory was created
    assert checkpoint_dir.is_dir()

    # Check that torch.save was called once
    mock_torch_save.assert_called_once()

    # Check the arguments passed to torch.save
    saved_state = mock_torch_save.call_args[0][0]
    saved_path = mock_torch_save.call_args[0][1]

    assert saved_path == checkpoint_dir / "test_last.pth"
    assert saved_state["epoch"] == epoch
    assert saved_state["model_state_dict"] == mock_model_state
    assert saved_state["optimizer_state_dict"] == mock_optimizer_state
    assert "scaler_state_dict" not in saved_state  # Check scaler removed
    # Check additional_data was saved correctly
    assert saved_state["metrics"] == metrics
    assert saved_state["best_metric_value"] == 0.9  # noqa: PLR2004


def test_save_checkpoint_keep_last_n(
    tmp_path: Path,
    mock_model: MagicMock,
    mock_optimizer: MagicMock,
) -> None:
    """Test that only the last N checkpoints are kept
    (by epoch in filename)."""  # Updated docstring hint
    checkpoint_dir = tmp_path / "keep_last_n"
    checkpoint_dir.mkdir()
    keep_n = 3
    total_files_to_create = 5  # Creates epochs 0, 1, 2, 3, 4

    # Create dummy older checkpoints
    for i in range(total_files_to_create):
        filename = checkpoint_dir / f"ckpt_epoch_{i}.pth"
        # Save dummy content
        torch.save({"epoch": i}, filename)
        # No sleep needed if sorting by epoch number
        # time.sleep(0.02)  # Removed

    # Save the new checkpoint that triggers cleanup (epoch 5)
    new_epoch = 5
    new_filename = f"ckpt_epoch_{new_epoch}.pth"
    config = CheckpointSaveConfig(
        checkpoint_dir=checkpoint_dir,
        filename=new_filename,
        keep_last_n=keep_n,
    )
    save_checkpoint(
        model=mock_model,
        optimizer=mock_optimizer,
        epoch=new_epoch,
        config=config,
    )

    # Check the number of files remaining
    remaining_files = sorted(checkpoint_dir.glob("ckpt_epoch_*.pth"))
    assert len(remaining_files) == keep_n

    # Check that the correct files remain (the ones with the highest epochs)
    expected_remaining_epochs = list(
        range(new_epoch - keep_n + 1, new_epoch + 1)  # Should be epochs 3,4,5
    )
    remaining_epochs = []
    for f_path in remaining_files:
        try:
            # Extract epoch number from filename stem
            prefix = "_epoch_"
            epoch_str = f_path.stem.split(prefix)[-1]
            file_epoch = int(epoch_str)
            remaining_epochs.append(file_epoch)
        except (IndexError, ValueError):
            fail_msg = (
                f"Could not parse epoch from remaining file: {f_path.name}"
            )
            pytest.fail(fail_msg)

    assert sorted(remaining_epochs) == expected_remaining_epochs

    # Ensure the newly saved file still exists
    # (redundant if above check passes, but good sanity check)
    assert (checkpoint_dir / new_filename).exists()


@patch("torch.save")
def test_save_checkpoint_is_best(
    mock_torch_save: MagicMock,
    tmp_path: Path,
    mock_model: MagicMock,
    mock_optimizer: MagicMock,
) -> None:
    """Test save_checkpoint saves the normal checkpoint (best logic is
    external)."""
    checkpoint_dir = tmp_path / "test_ckpts_best"
    epoch = 10
    # Pass None for additional_data when metrics is None
    additional_data = None

    config = CheckpointSaveConfig(
        checkpoint_dir=checkpoint_dir,
        filename="last.pth",
        additional_data=additional_data,
    )
    save_checkpoint(
        model=mock_model,
        optimizer=mock_optimizer,
        epoch=epoch,
        config=config,
    )

    # Should be called ONCE for last.pth
    mock_torch_save.assert_called_once()
    mock_torch_save.assert_any_call(ANY, checkpoint_dir / "last.pth")


# --- Tests for load_checkpoint ---


def test_load_checkpoint_file_not_found(mock_model: MagicMock) -> None:
    """
    Test load_checkpoint raises FileNotFoundError when file doesn't exist.
    """
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(
            model=mock_model, checkpoint_path="nonexistent/path.pth"
        )


def test_load_checkpoint_loads_state(
    tmp_path: Path,
    mock_model_state: dict[str, torch.Tensor],
    mock_optimizer_state: dict[str, Any],
) -> None:
    """Test load_checkpoint correctly loads states into components."""
    checkpoint_dir = tmp_path / "load_test_ckpts"
    checkpoint_dir.mkdir()
    filepath = checkpoint_dir / "checkpoint_to_load.pth"
    epoch = 7
    best_metric = 0.85
    metrics_saved = {"iou": best_metric}

    # Create a dummy checkpoint file (without scaler state)
    state_to_save = {
        "epoch": epoch,
        "model_state_dict": mock_model_state,
        "optimizer_state_dict": mock_optimizer_state,
        # 'scaler_state_dict': mock_scaler_state, # Removed scaler state
        "best_metric_value": best_metric,
        "metrics": metrics_saved,  # Example additional data
    }
    torch.save(state_to_save, filepath)

    # Create new mocks to load into
    new_model = MagicMock(spec=torch.nn.Module)
    new_model.load_state_dict = MagicMock()
    new_model.device = torch.device("cpu")  # Add device attribute
    new_optimizer = MagicMock(spec=torch.optim.Optimizer)
    new_optimizer.load_state_dict = MagicMock()
    # new_scaler = MagicMock(spec=GradScaler) # Removed scaler
    # new_scaler.load_state_dict = MagicMock()

    # Load the checkpoint (explicitly pass device=None or a device)
    loaded_state_metadata = load_checkpoint(
        model=new_model,
        optimizer=new_optimizer,
        # scaler=new_scaler, # Removed scaler
        checkpoint_path=str(filepath),
        device=None,  # Explicitly pass None to use model.device
    )

    # Verify mock methods were called once
    new_model.load_state_dict.assert_called_once()
    new_optimizer.load_state_dict.assert_called_once()

    # Manually compare state dicts passed to mocks (NO assert_called_once_with)
    # Check model state
    call_args_model, _ = new_model.load_state_dict.call_args
    loaded_model_state = call_args_model[0]
    assert loaded_model_state.keys() == mock_model_state.keys()
    for key in mock_model_state:
        assert torch.equal(loaded_model_state[key], mock_model_state[key])

    # Check optimizer state
    call_args_opt, _ = new_optimizer.load_state_dict.call_args
    loaded_opt_state = call_args_opt[0]
    assert loaded_opt_state.keys() == mock_optimizer_state.keys()
    assert (
        loaded_opt_state["param_groups"]
        == mock_optimizer_state["param_groups"]
    )
    assert (
        loaded_opt_state["state"].keys()
        == mock_optimizer_state["state"].keys()
    )
    for key in mock_optimizer_state["state"]:
        assert torch.equal(
            loaded_opt_state["state"][key]["exp_avg"],
            mock_optimizer_state["state"][key]["exp_avg"],
        )

    # Verify metadata is returned
    assert loaded_state_metadata["epoch"] == epoch
    assert loaded_state_metadata["best_metric_value"] == best_metric
    assert loaded_state_metadata["metrics"] == metrics_saved


def test_load_checkpoint_handles_missing_keys(
    tmp_path: Path, mock_model_state: dict[str, torch.Tensor]
) -> None:
    """Test load_checkpoint handles checkpoints with missing optional keys."""
    checkpoint_dir = tmp_path / "missing_keys_ckpts"
    checkpoint_dir.mkdir()
    filepath = checkpoint_dir / "minimal_checkpoint.pth"
    epoch = 3

    # Create a checkpoint with only essential keys
    state_to_save = {
        "epoch": epoch,
        "model_state_dict": mock_model_state,
    }
    torch.save(state_to_save, filepath)

    # Create new mocks to load into
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.load_state_dict = MagicMock()
    mock_model.device = torch.device("cpu")  # Add device attribute
    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_optimizer.load_state_dict = MagicMock()
    # mock_scaler = MagicMock(spec=GradScaler) # Removed scaler
    # mock_scaler.load_state_dict = MagicMock()

    # Load the checkpoint (explicitly pass device=None)
    loaded_state_metadata = load_checkpoint(
        model=mock_model,
        optimizer=mock_optimizer,
        # scaler=mock_scaler, # Removed scaler
        checkpoint_path=str(filepath),
        device=None,  # Explicitly pass None
    )

    # Verify only model load_state_dict was called once
    mock_model.load_state_dict.assert_called_once()
    assert not mock_optimizer.load_state_dict.called

    # Manually compare model state dict (NO assert_called_once_with)
    call_args_model, _ = mock_model.load_state_dict.call_args
    loaded_model_state = call_args_model[0]
    assert loaded_model_state.keys() == mock_model_state.keys()
    for key in mock_model_state:
        assert torch.equal(loaded_model_state[key], mock_model_state[key])

    assert loaded_state_metadata["epoch"] == epoch
    assert "metrics" not in loaded_state_metadata
    assert "best_metric_value" not in loaded_state_metadata


def test_load_checkpoint_on_device(
    tmp_path: Path, mock_model_state: dict[str, torch.Tensor]
) -> None:
    """Test loading checkpoint onto a specific device."""
    checkpoint_dir = tmp_path / "device_ckpts"
    checkpoint_dir.mkdir()
    filepath = checkpoint_dir / "device_checkpoint.pth"
    torch.save({"model_state_dict": mock_model_state, "epoch": 1}, filepath)

    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.load_state_dict = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    # Use torch.device directly
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    load_checkpoint(
        model=mock_model, checkpoint_path=str(filepath), device=device
    )

    # Check if model.to(device) was called AFTER loading state (based on impl)
    mock_model.load_state_dict.assert_called_once()
    mock_model.to.assert_called_once_with(device)


# --- Tests for new standardized checkpoint functionality ---


def test_generate_checkpoint_metadata() -> None:
    """Test generation of automatic checkpoint metadata."""
    metadata = generate_checkpoint_metadata()

    assert "pytorch_version" in metadata
    assert "python_version" in metadata
    assert "platform" in metadata
    assert "timestamp" in metadata

    assert metadata["pytorch_version"] == torch.__version__
    assert "." in metadata["python_version"]  # Should have version format
    assert isinstance(metadata["timestamp"], str)


def test_validate_checkpoint_completeness() -> None:
    """Test checkpoint validation against specification."""
    spec = CheckpointSpec()

    # Complete checkpoint
    complete_checkpoint = {
        "epoch": 10,
        "model_state_dict": {"layer1.weight": torch.randn(5, 5)},
        "optimizer_state_dict": {"param_groups": []},
        "pytorch_version": "2.0.0",
        "timestamp": "2024-01-01T12:00:00",
        "config": {"lr": 0.001},
    }

    is_valid, missing = validate_checkpoint_completeness(
        complete_checkpoint, spec
    )
    assert is_valid
    assert len(missing) == 0

    # Incomplete checkpoint
    incomplete_checkpoint = {
        "epoch": 10,
        "model_state_dict": {"layer1.weight": torch.randn(5, 5)},
    }

    is_valid, missing = validate_checkpoint_completeness(
        incomplete_checkpoint, spec
    )
    assert not is_valid
    assert "optimizer_state_dict" in missing
    assert "pytorch_version" in missing
    assert "timestamp" in missing
    assert "config" in missing


def test_create_standardized_filename() -> None:
    """Test standardized filename generation."""
    # Regular checkpoint
    filename = create_standardized_filename("checkpoint", 5, "20240101_120000")
    assert filename == "checkpoint_epoch_005_20240101_120000.pth"

    # Best checkpoint
    filename = create_standardized_filename(
        "model", 10, "20240101_120000", is_best=True
    )
    assert filename == "model_best_epoch_010_20240101_120000.pth"

    # Auto-generated timestamp
    filename = create_standardized_filename("checkpoint", 1)
    assert filename.startswith("checkpoint_epoch_001_")
    assert filename.endswith(".pth")


@patch("torch.save")
def test_save_checkpoint_with_standardized_format(
    mock_torch_save: MagicMock,
    tmp_path: Path,
    mock_model: MagicMock,
    mock_optimizer: MagicMock,
) -> None:
    """Test save_checkpoint with new standardized format and validation."""
    checkpoint_dir = tmp_path / "standardized_ckpts"
    epoch = 15

    # Create mock scheduler
    mock_scheduler = MagicMock()
    mock_scheduler.state_dict.return_value = {"step_size": 10, "gamma": 0.9}

    config = CheckpointSaveConfig(
        checkpoint_dir=checkpoint_dir,
        filename="standardized_checkpoint.pth",
        include_scheduler=True,
        include_python_info=True,
        validate_completeness=True,
    )

    metrics = {"train_loss": 0.1, "val_iou": 0.85}
    training_config = {"lr": 0.001, "epochs": 20}

    save_checkpoint(
        model=mock_model,
        optimizer=mock_optimizer,
        epoch=epoch,
        config=config,
        scheduler=mock_scheduler,
        best_metric_value=0.85,
        metrics=metrics,
        training_config=training_config,
    )

    # Verify torch.save was called
    mock_torch_save.assert_called_once()

    # Check the saved data contains all expected fields
    saved_data = mock_torch_save.call_args[0][0]

    # Required fields
    assert saved_data["epoch"] == epoch
    assert "model_state_dict" in saved_data
    assert "optimizer_state_dict" in saved_data
    assert "scheduler_state_dict" in saved_data
    assert "pytorch_version" in saved_data
    assert "timestamp" in saved_data

    # Optional fields
    assert saved_data["best_metric_value"] == 0.85
    assert saved_data["metrics"] == metrics
    assert saved_data["config"] == training_config
    assert "python_version" in saved_data
    assert "platform" in saved_data


def test_verify_checkpoint_integrity(tmp_path: Path) -> None:
    """Test checkpoint integrity verification."""
    checkpoint_dir = tmp_path / "verify_ckpts"
    checkpoint_dir.mkdir()

    # Create a valid checkpoint
    valid_checkpoint = {
        "epoch": 5,
        "model_state_dict": {"layer1.weight": torch.randn(3, 3)},
        "optimizer_state_dict": {"param_groups": []},
        "pytorch_version": "2.0.0",
        "timestamp": "2024-01-01T12:00:00",
        "config": {"lr": 0.001},
    }

    valid_path = checkpoint_dir / "valid_checkpoint.pth"
    torch.save(valid_checkpoint, valid_path)

    # Test verification of valid checkpoint
    result = verify_checkpoint_integrity(valid_path)
    assert result["exists"]
    assert result["is_valid"]
    assert len(result["missing_fields"]) == 0  # Correct field name
    assert result["epoch"] == 5
    assert result["pytorch_version"] == "2.0.0"
    assert result["size_bytes"] > 0  # Correct field name
    assert result["error"] is None

    # Test verification of non-existent checkpoint
    nonexistent_path = checkpoint_dir / "nonexistent.pth"
    result = verify_checkpoint_integrity(nonexistent_path)
    assert not result["exists"]
    assert not result["is_valid"]
    assert "File does not exist" in result["error"]  # Correct error message


def test_adapt_legacy_checkpoint() -> None:
    """Test adaptation of legacy checkpoint format."""
    legacy_checkpoint = {
        "epoch": 10,
        "model_state_dict": {"layer1.weight": torch.randn(3, 3)},
        "optimizer_state_dict": {"param_groups": []},
        "best_metric_value": 0.9,
    }

    training_config = {"lr": 0.001, "epochs": 20}

    adapted = adapt_legacy_checkpoint(legacy_checkpoint, training_config)

    # Should preserve original data
    assert adapted["epoch"] == 10
    assert "model_state_dict" in adapted
    assert "optimizer_state_dict" in adapted
    assert adapted["best_metric_value"] == 0.9

    # Should add missing metadata
    assert "pytorch_version" in adapted
    assert "timestamp" in adapted
    assert "python_version" in adapted
    assert adapted["config"] == training_config


def test_load_checkpoint_with_scheduler(
    tmp_path: Path,
    mock_model_state: dict[str, torch.Tensor],
    mock_optimizer_state: dict[str, Any],
) -> None:
    """Test loading checkpoint with scheduler state."""
    checkpoint_dir = tmp_path / "scheduler_test"
    checkpoint_dir.mkdir()
    filepath = checkpoint_dir / "checkpoint_with_scheduler.pth"

    scheduler_state = {"step_size": 10, "gamma": 0.9, "last_epoch": 5}

    checkpoint_data = {
        "epoch": 10,
        "model_state_dict": mock_model_state,
        "optimizer_state_dict": mock_optimizer_state,
        "scheduler_state_dict": scheduler_state,
        "pytorch_version": "2.0.0",
        "timestamp": "2024-01-01T12:00:00",
    }
    torch.save(checkpoint_data, filepath)

    # Create mocks
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.load_state_dict = MagicMock()
    mock_model.device = torch.device("cpu")

    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_optimizer.load_state_dict = MagicMock()

    mock_scheduler = MagicMock()
    mock_scheduler.load_state_dict = MagicMock()

    # Load checkpoint
    loaded_data = load_checkpoint(
        checkpoint_path=filepath,
        model=mock_model,
        optimizer=mock_optimizer,
        scheduler=mock_scheduler,
    )

    # Verify all components were loaded
    mock_model.load_state_dict.assert_called_once()
    mock_optimizer.load_state_dict.assert_called_once()
    mock_scheduler.load_state_dict.assert_called_once()

    # Check scheduler state was loaded correctly
    scheduler_call_args = mock_scheduler.load_state_dict.call_args[0][0]
    assert scheduler_call_args == scheduler_state

    # Verify metadata is returned (excluding state dicts)
    assert loaded_data["epoch"] == 10
    assert loaded_data["pytorch_version"] == "2.0.0"
    assert "model_state_dict" not in loaded_data
    assert "scheduler_state_dict" not in loaded_data


def test_load_checkpoint_strict_validation(
    tmp_path: Path,
    mock_model_state: dict[str, torch.Tensor],
) -> None:
    """Test strict validation mode for loading checkpoints."""
    checkpoint_dir = tmp_path / "strict_validation"
    checkpoint_dir.mkdir()
    filepath = checkpoint_dir / "incomplete_checkpoint.pth"

    # Create incomplete checkpoint
    incomplete_data = {
        "epoch": 5,
        "model_state_dict": mock_model_state,
        # Missing required fields: optimizer_state_dict, pytorch_version, etc.
    }
    torch.save(incomplete_data, filepath)

    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.load_state_dict = MagicMock()
    mock_model.device = torch.device("cpu")

    # Should succeed without strict validation
    load_checkpoint(
        checkpoint_path=filepath,
        model=mock_model,
        strict_validation=False,
    )

    # Should fail with strict validation
    with pytest.raises(ValueError, match="Invalid checkpoint format"):
        load_checkpoint(
            checkpoint_path=filepath,
            model=mock_model,
            strict_validation=True,
        )
