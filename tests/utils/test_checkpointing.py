"""Tests for checkpointing utility functions."""

import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import functions to test
from src.utils.checkpointing import save_checkpoint, load_checkpoint
from torch.amp import GradScaler


# --- Fixtures ---

@pytest.fixture
def mock_model_state():
    """Returns a mock model state dict."""
    return {'layer1.weight': torch.randn(10, 5)}


@pytest.fixture
def mock_optimizer_state():
    """Returns a mock optimizer state dict."""
    return {'state': {0: {'exp_avg': torch.randn(10, 5)}}, 'param_groups': []}


@pytest.fixture
def mock_scaler_state():
    """Returns a mock scaler state dict."""
    return {'scale': torch.tensor(1024.0), '_growth_tracker': 0}


@pytest.fixture
def mock_model(mock_model_state):
    """Mock torch model with state_dict method."""
    model = MagicMock(spec=torch.nn.Module)
    model.state_dict.return_value = mock_model_state
    model.load_state_dict = MagicMock()
    return model


@pytest.fixture
def mock_optimizer(mock_optimizer_state):
    """Mock optimizer with state_dict method."""
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    optimizer.state_dict.return_value = mock_optimizer_state
    optimizer.load_state_dict = MagicMock()
    return optimizer


@pytest.fixture
def mock_scaler(mock_scaler_state):
    """Mock GradScaler with state_dict method."""
    scaler = MagicMock(spec=GradScaler)
    scaler.state_dict.return_value = mock_scaler_state
    scaler.load_state_dict = MagicMock()
    return scaler


# --- Test Cases ---

@patch('torch.save')
def test_save_checkpoint_creates_file_and_calls_torch_save(
    mock_torch_save,
    tmp_path: Path,
    mock_model,
    mock_optimizer,
    mock_scaler,
    mock_model_state,
    mock_optimizer_state,
    mock_scaler_state
):
    """Test save_checkpoint creates dir/file and calls torch.save."""
    checkpoint_dir = tmp_path / "test_ckpts"
    epoch = 5
    metrics = {"loss": 0.1, "iou": 0.9}

    save_checkpoint(
        epoch=epoch, model=mock_model, optimizer=mock_optimizer,
        scaler=mock_scaler, metrics=metrics, checkpoint_dir=checkpoint_dir,
        filename="test_last.pth"
    )

    # Check directory was created
    assert checkpoint_dir.is_dir()

    # Check torch.save was called
    expected_filepath = checkpoint_dir / "test_last.pth"
    mock_torch_save.assert_called_once()
    call_args, _ = mock_torch_save.call_args

    # Check the state dictionary saved
    saved_state = call_args[0]
    assert saved_state['epoch'] == epoch
    # Compare tensors explicitly for state dicts
    assert torch.equal(
        saved_state['model_state_dict']['layer1.weight'],
        mock_model_state['layer1.weight']
    )
    assert torch.equal(
        saved_state['optimizer_state_dict']['state'][0]['exp_avg'],
        mock_optimizer_state['state'][0]['exp_avg']
    )
    assert saved_state['optimizer_state_dict']['param_groups'] == \
        mock_optimizer_state['param_groups']
    assert torch.equal(
        saved_state['scaler_state_dict']['scale'],
        mock_scaler_state['scale']
    )
    assert saved_state['scaler_state_dict']['_growth_tracker'] == \
        mock_scaler_state['_growth_tracker']
    assert saved_state['metrics'] == metrics

    # Check the filepath saved to
    assert call_args[1] == expected_filepath


@patch('torch.save')
def test_save_checkpoint_is_best(
    mock_torch_save,
    tmp_path: Path,
    mock_model,
    mock_optimizer,
    mock_scaler
):
    """Test save_checkpoint saves a best checkpoint when is_best=True."""
    checkpoint_dir = tmp_path / "test_ckpts_best"
    epoch = 10

    save_checkpoint(
        epoch=epoch, model=mock_model, optimizer=mock_optimizer,
        scaler=mock_scaler, metrics=None, checkpoint_dir=checkpoint_dir,
        filename="last.pth", is_best=True, best_filename="best.pth"
    )

    # Expect torch.save to be called twice: once for last, once for best
    assert mock_torch_save.call_count == 2
    last_call_args, _ = mock_torch_save.call_args_list[0]
    best_call_args, _ = mock_torch_save.call_args_list[1]

    assert last_call_args[1] == checkpoint_dir / "last.pth"
    assert best_call_args[1] == checkpoint_dir / "best.pth"
    # State should be the same for both calls in this case
    # Check a few key parts of the state for equality
    assert last_call_args[0]['epoch'] == best_call_args[0]['epoch']
    # Compare tensors explicitly for state dicts (assuming mock state exists)
    # We might need to fetch the state from the mock objects if not passed in
    # For simplicity, let's assume the comparison logic shown in the other test
    # would apply here if the states were fully defined and passed.


def test_load_checkpoint_file_not_found(mock_model):
    """Test load_checkpoint returns default state when file doesn't exist."""
    result = load_checkpoint(
        model=mock_model, checkpoint_path="nonexistent/path.pth"
    )
    assert result == {"epoch": 0, "best_metric_value": None}
    mock_model.load_state_dict.assert_not_called()


def test_load_checkpoint_loads_state(
    tmp_path: Path,
    # Pass mocks for saving, not for loading verification
    mock_model_state,
    mock_optimizer_state,
    mock_scaler_state
):
    """Test load_checkpoint correctly loads states into components."""
    checkpoint_dir = tmp_path / "load_test_ckpts"
    checkpoint_dir.mkdir()
    filepath = checkpoint_dir / "checkpoint_to_load.pth"
    epoch = 7
    best_metric = 0.85

    # Create a dummy checkpoint file
    state_to_save = {
        'epoch': epoch,
        'model_state_dict': mock_model_state,
        'optimizer_state_dict': mock_optimizer_state,
        'scaler_state_dict': mock_scaler_state,
        'best_metric_value': best_metric,
        'metrics': {'iou': best_metric}
    }
    torch.save(state_to_save, filepath)

    # Create new mocks to load into
    new_model = MagicMock(spec=torch.nn.Module)
    new_model.load_state_dict = MagicMock()
    new_optimizer = MagicMock(spec=torch.optim.Optimizer)
    new_optimizer.load_state_dict = MagicMock()
    new_scaler = MagicMock(spec=GradScaler)
    new_scaler.load_state_dict = MagicMock()

    # Load the checkpoint
    loaded_state_metadata = load_checkpoint(
        model=new_model,
        optimizer=new_optimizer,
        scaler=new_scaler,
        checkpoint_path=str(filepath)
    )

    # Verify load_state_dict calls by checking the arguments manually
    new_model.load_state_dict.assert_called_once()
    model_call_args, _ = new_model.load_state_dict.call_args
    assert torch.equal(
        model_call_args[0]['layer1.weight'],
        mock_model_state['layer1.weight']
    )

    new_optimizer.load_state_dict.assert_called_once()
    opt_call_args, _ = new_optimizer.load_state_dict.call_args
    assert torch.equal(
        opt_call_args[0]['state'][0]['exp_avg'],
        mock_optimizer_state['state'][0]['exp_avg']
    )
    assert opt_call_args[0]['param_groups'] == \
        mock_optimizer_state['param_groups']

    new_scaler.load_state_dict.assert_called_once()
    scaler_call_args, _ = new_scaler.load_state_dict.call_args
    assert torch.equal(
        scaler_call_args[0]['scale'],
        mock_scaler_state['scale']
    )
    assert scaler_call_args[0]['_growth_tracker'] == \
        mock_scaler_state['_growth_tracker']

    # Verify returned metadata
    assert loaded_state_metadata['epoch'] == epoch
    assert loaded_state_metadata['best_metric_value'] == best_metric


def test_load_checkpoint_handles_missing_keys(
    tmp_path: Path,
    mock_model_state  # Only need model state for saving
):
    """Test load_checkpoint handles checkpoints with missing optional keys."""
    checkpoint_dir = tmp_path / "missing_keys_ckpts"
    checkpoint_dir.mkdir()
    filepath = checkpoint_dir / "minimal_checkpoint.pth"
    epoch = 3

    # Create a checkpoint with only essential keys
    state_to_save = {
        'epoch': epoch,
        'model_state_dict': mock_model_state,
    }
    torch.save(state_to_save, filepath)

    # Create new mocks to load into
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.load_state_dict = MagicMock()
    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_optimizer.load_state_dict = MagicMock()
    mock_scaler = MagicMock(spec=GradScaler)
    mock_scaler.load_state_dict = MagicMock()

    # Load the checkpoint
    loaded_state_metadata = load_checkpoint(
        model=mock_model,
        optimizer=mock_optimizer,
        scaler=mock_scaler,
        checkpoint_path=str(filepath)
    )

    # Verify only model state was loaded
    mock_model.load_state_dict.assert_called_once()
    model_call_args, _ = mock_model.load_state_dict.call_args
    assert torch.equal(
        model_call_args[0]['layer1.weight'],
        mock_model_state['layer1.weight']
    )

    mock_optimizer.load_state_dict.assert_not_called()
    mock_scaler.load_state_dict.assert_not_called()

    # Verify default metadata is returned for missing keys
    assert loaded_state_metadata['epoch'] == epoch
    assert loaded_state_metadata['best_metric_value'] is None
