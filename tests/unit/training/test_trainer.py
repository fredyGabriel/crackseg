"""Tests for the Trainer class."""

import pytest
import torch
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch

# Assuming the necessary modules exist in these paths
from src.training.trainer import Trainer
from src.utils.logging import NoOpLogger  # Use NoOpLogger for testing
from src.training.batch_processing import train_step, val_step


# --- Mocks and Fixtures ---

@pytest.fixture
def mock_model():
    """Mock torch model."""
    model = MagicMock(spec=torch.nn.Module)
    model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    # Mock the to() method to return self
    model.to.return_value = model
    return model


@pytest.fixture
def mock_dataloader():
    """Mock dataloader."""
    loader = MagicMock(spec=torch.utils.data.DataLoader)
    loader.__len__.return_value = 10  # Example length
    # Make it iterable (e.g., returning dummy data)
    loader.__iter__.return_value = iter([
        (torch.randn(2, 3, 4, 4), torch.randn(2, 1, 4, 4)) for _ in range(10)
    ])
    return loader


@pytest.fixture
def mock_loss_fn():
    """Mock loss function."""
    loss_fn = MagicMock(spec=torch.nn.Module)
    # Return a tensor
    loss_fn.return_value = torch.tensor(0.5, requires_grad=True)
    return loss_fn


@pytest.fixture
def mock_metrics_dict():
    """Mock metrics dictionary."""
    metric_mock = MagicMock()
    metric_mock.return_value = torch.tensor(0.8)  # Return a tensor
    return {"iou": metric_mock, "f1": metric_mock}


@pytest.fixture
def base_trainer_cfg():
    """Basic Hydra config for the trainer."""
    return OmegaConf.create({
        "training": {
            "epochs": 2,
            "device": "cpu",
            "use_amp": False,
            "gradient_accumulation_steps": 1,
            "verbose": False,
            "optimizer": {
                "_target_": "torch.optim.Adam",
                "lr": 1e-3
            },
            "lr_scheduler": None,
            "scheduler": None,
            # Add other necessary fields if init requires them
        }
        # Add other top-level keys if needed by mocks (e.g., logging)
    })


@pytest.fixture
def mock_logger_instance():
    """Mock logger instance."""
    return NoOpLogger()  # Use NoOpLogger, requires no setup


@pytest.fixture
def dummy_batch():
    """Returns a dummy batch of images and masks."""
    images = torch.randn(2, 3, 4, 4)
    masks = torch.randn(2, 1, 4, 4)
    return (images, masks)


@pytest.fixture
def dummy_data_loader():
    # Returns a DataLoader with 2 batches of dummy data
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 4

        def __getitem__(self, idx):
            return {
                'image': torch.randn(3, 4, 4),
                'mask': torch.randn(1, 4, 4)
            }
    return torch.utils.data.DataLoader(DummyDataset(), batch_size=2)


@pytest.fixture
def dummy_loss():
    return torch.nn.MSELoss()


@pytest.fixture
def dummy_metrics():
    return {}


# --- Test Cases ---

# Patch factory functions where they are used (in trainer module)
@patch('src.training.trainer.get_device', return_value=torch.device('cpu'))
@patch('src.training.trainer.create_lr_scheduler')
@patch('src.training.trainer.create_optimizer')
def test_trainer_initialization(
    mock_create_trainer_optimizer,
    mock_create_trainer_scheduler,
    mock_get_device,
    mock_model,
    mock_dataloader,
    mock_loss_fn,
    mock_metrics_dict,
    base_trainer_cfg,
    mock_logger_instance
):
    """Test if the Trainer class can be initialized correctly."""
    # Mock the return values of factory functions
    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_optimizer.param_groups = [{'lr': 1e-3}]
    mock_create_trainer_optimizer.return_value = mock_optimizer
    mock_create_trainer_scheduler.return_value = None

    try:
        # Instantiate Trainer
        # (it will internally try to call the patched factories)
        trainer = Trainer(
            model=mock_model,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            loss_fn=mock_loss_fn,
            metrics_dict=mock_metrics_dict,
            cfg=base_trainer_cfg,
            logger_instance=mock_logger_instance
        )

        # Assertions to check basic initialization
        assert trainer.model == mock_model
        assert trainer.train_loader == mock_dataloader
        # Assert that the patched factories were called and assigned
        assert trainer.optimizer is mock_optimizer
        assert trainer.scheduler is None

        mock_model.to.assert_called_once_with(torch.device('cpu'))
        # Check if the factory functions were called
        mock_create_trainer_optimizer.assert_called_once()
        # Scheduler factory might not be called if cfg.trainer.lr_scheduler
        # is None
        # Adjust assertion based on actual Trainer logic
        # (assuming it checks cfg)
        if base_trainer_cfg.training.lr_scheduler:
            mock_create_trainer_scheduler.assert_called_once()
        else:
            mock_create_trainer_scheduler.assert_called_once()
            assert trainer.scheduler is None

        mock_get_device.assert_called_once_with("cpu")

    except Exception as e:
        pytest.fail(f"Trainer initialization failed: {e}")


@patch('src.training.trainer.get_device', return_value=torch.device('cpu'))
@patch('src.training.trainer.create_lr_scheduler')
@patch('src.training.trainer.create_optimizer')
@patch('src.training.trainer.handle_epoch_checkpointing')
@patch('src.training.trainer.Trainer._step_scheduler')
@patch(
    'src.training.trainer.Trainer.validate',
    return_value={"loss": 0.4, "iou": 0.8}
)
@patch('src.training.trainer.Trainer._train_epoch', return_value=0.5)
def test_trainer_train_loop(
    mock_handle_checkpoint,
    mock_train_epoch,
    mock_validate,
    mock_step_scheduler,
    mock_create_trainer_optimizer,
    mock_create_trainer_scheduler,
    mock_get_device,
    mock_model,
    mock_dataloader,
    mock_loss_fn,
    mock_metrics_dict,
    base_trainer_cfg,
    mock_logger_instance
):
    """Test the main training loop orchestration in train()."""
    # Setup mocks for optimizer/scheduler creation
    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_optimizer.param_groups = [{'lr': 1e-3}]
    mock_create_trainer_optimizer.return_value = mock_optimizer
    mock_scheduler_instance = MagicMock()
    mock_create_trainer_scheduler.return_value = mock_scheduler_instance

    # El helper debe devolver un valor float simulado
    mock_handle_checkpoint.return_value = float('inf')

    # Modify config slightly for this test
    test_cfg = base_trainer_cfg.copy()
    test_cfg.training.epochs = 3  # Test with 3 epochs
    test_cfg.training.lr_scheduler = OmegaConf.create({
        "_target_": "torch.optim.lr_scheduler.StepLR",
        "step_size": 1
    })

    # Instanciar Trainer
    trainer = Trainer(
        model=mock_model,
        train_loader=mock_dataloader,
        val_loader=mock_dataloader,
        loss_fn=mock_loss_fn,
        metrics_dict=mock_metrics_dict,
        cfg=test_cfg,
        logger_instance=mock_logger_instance
    )

    # --- Ejecutar el método train ---
    final_results = trainer.train()

    # --- Assertions ---
    mock_create_trainer_optimizer.assert_called_once()
    mock_create_trainer_scheduler.assert_called_once()

    assert mock_train_epoch.call_count == test_cfg.training.epochs
    assert mock_validate.call_count == test_cfg.training.epochs

    if trainer.scheduler:
        assert mock_step_scheduler.call_count == test_cfg.training.epochs

    assert final_results == {"loss": 0.4, "iou": 0.8}

    for i in range(1, test_cfg.training.epochs + 1):
        mock_train_epoch.assert_any_call(i)

    if trainer.scheduler:
        assert mock_step_scheduler.call_count == test_cfg.training.epochs

    # Verificar que el helper de checkpointing se llama una vez por época
    assert mock_handle_checkpoint.call_count == test_cfg.training.epochs


@patch('src.training.trainer.get_device', return_value=torch.device('cpu'))
@patch('src.training.trainer.create_lr_scheduler')
@patch('src.training.trainer.create_optimizer')
def test_train_step_computes_loss_and_backward(
    mock_create_optimizer,
    mock_create_scheduler,
    mock_get_device,
    mock_model,
    mock_dataloader,
    mock_loss_fn,
    mock_metrics_dict,
    base_trainer_cfg,
    mock_logger_instance,
    dummy_batch
):
    # Configuración: grad_accum_steps = 2 para probar el escalado
    base_trainer_cfg.training.gradient_accumulation_steps = 2

    # Mock optimizer y scaler
    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_create_optimizer.return_value = mock_optimizer
    mock_create_scheduler.return_value = None

    # Mock scaler para capturar el backward
    mock_scaler = MagicMock()
    # El método scale() debe devolver un objeto con backward()
    mock_scaled_loss = MagicMock()
    mock_scaler.scale.return_value = mock_scaled_loss

    # Mock model: forward devuelve un tensor
    mock_model.return_value = torch.ones(2, 1, 4, 4)

    # Mock loss_fn: devuelve un tensor escalar
    mock_loss_fn.return_value = torch.tensor(0.8, requires_grad=True)

    # Instanciar Trainer y reemplazar scaler por el mock
    trainer = Trainer(
        model=mock_model,
        train_loader=mock_dataloader,
        val_loader=mock_dataloader,
        loss_fn=mock_loss_fn,
        metrics_dict=mock_metrics_dict,
        cfg=base_trainer_cfg,
        logger_instance=mock_logger_instance
    )
    trainer.scaler = mock_scaler  # Forzar el mock

    # Ejecutar _train_step
    result = train_step(
        model=mock_model,
        batch=dummy_batch,
        loss_fn=mock_loss_fn,
        optimizer=mock_optimizer,
        device=mock_get_device(),
        metrics_dict=mock_metrics_dict
    )

    # Verificaciones
    mock_model.assert_called_once()  # Se llama el forward
    mock_loss_fn.assert_called_once()  # Se calcula el loss
    # El valor devuelto debe ser el loss
    assert result["loss"].item() == pytest.approx(0.8)


@pytest.mark.cuda
@patch('src.training.trainer.get_device', return_value=torch.device('cuda:0'))
@patch('src.training.trainer.create_lr_scheduler')
@patch('src.training.trainer.create_optimizer')
def test_train_step_amp_cuda(
    mock_create_optimizer,
    mock_create_scheduler,
    mock_get_device,
    mock_model,
    mock_dataloader,
    mock_loss_fn,
    mock_metrics_dict,
    base_trainer_cfg,
    mock_logger_instance,
    dummy_batch
):
    # Habilitar AMP y usar CUDA
    base_trainer_cfg.training.use_amp = True
    base_trainer_cfg.training.device = "cuda:0"
    base_trainer_cfg.training.gradient_accumulation_steps = 1

    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_create_optimizer.return_value = mock_optimizer
    mock_create_scheduler.return_value = None

    mock_scaler = MagicMock()
    mock_scaled_loss = MagicMock()
    mock_scaler.scale.return_value = mock_scaled_loss

    mock_model.return_value = torch.ones(2, 1, 4, 4, device='cuda:0')
    mock_loss_fn.return_value = torch.tensor(0.5, device='cuda:0',
                                             requires_grad=True)

    trainer = Trainer(
        model=mock_model,
        train_loader=mock_dataloader,
        val_loader=mock_dataloader,
        loss_fn=mock_loss_fn,
        metrics_dict=mock_metrics_dict,
        cfg=base_trainer_cfg,
        logger_instance=mock_logger_instance
    )
    trainer.scaler = mock_scaler

    # Ejecutar _train_step
    result = train_step(
        model=mock_model,
        batch=dummy_batch,
        loss_fn=mock_loss_fn,
        optimizer=mock_optimizer,
        device=mock_get_device(),
        metrics_dict=mock_metrics_dict
    )

    mock_model.assert_called_once()
    mock_loss_fn.assert_called_once()
    # El valor devuelto debe ser el loss
    assert result["loss"].item() == pytest.approx(0.5)


@patch('src.training.trainer.get_device', return_value=torch.device('cpu'))
@patch('src.training.trainer.create_lr_scheduler')
@patch('src.training.trainer.create_optimizer')
def test_train_step_raises_on_forward_error(
    mock_create_optimizer,
    mock_create_scheduler,
    mock_get_device,
    mock_model,
    mock_dataloader,
    mock_loss_fn,
    mock_metrics_dict,
    base_trainer_cfg,
    mock_logger_instance,
    dummy_batch
):
    base_trainer_cfg.training.gradient_accumulation_steps = 1

    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_create_optimizer.return_value = mock_optimizer
    mock_create_scheduler.return_value = None

    mock_scaler = MagicMock()
    mock_scaler.scale.return_value = MagicMock()

    # Simular excepción en el forward
    mock_model.side_effect = RuntimeError("Forward error")

    trainer = Trainer(
        model=mock_model,
        train_loader=mock_dataloader,
        val_loader=mock_dataloader,
        loss_fn=mock_loss_fn,
        metrics_dict=mock_metrics_dict,
        cfg=base_trainer_cfg,
        logger_instance=mock_logger_instance
    )
    trainer.scaler = mock_scaler

    with pytest.raises(RuntimeError, match="Forward error"):
        train_step(
            model=mock_model,
            batch=dummy_batch,
            loss_fn=mock_loss_fn,
            optimizer=mock_optimizer,
            device=mock_get_device(),
            metrics_dict=mock_metrics_dict
        )


def test_epoch_level_logging(dummy_data_loader, dummy_loss,
                             dummy_metrics, tmp_path):
    # Configuración mínima para usar StepLR
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create({
        'training': {
            'epochs': 2,
            'device': 'cpu',
            'use_amp': False,
            'gradient_accumulation_steps': 1,
            'checkpoint_dir': str(checkpoint_dir),
            'optimizer': {'_target_': 'torch.optim.SGD', 'lr': 0.01},
            'lr_scheduler': {'_target_': 'torch.optim.lr_scheduler.StepLR',
                             'step_size': 1, 'gamma': 0.5},
            'scheduler': None,
            'verbose': False
        }
    })
    model = torch.nn.Conv2d(3, 1, 1)
    from unittest.mock import MagicMock
    logger = MagicMock()
    # Configurar el experiment_manager del logger mock para que
    # get_path retorne la ruta de checkpoint_dir correctamente
    exp_manager = MagicMock()
    exp_manager.get_path.return_value = str(checkpoint_dir)
    logger.experiment_manager = exp_manager

    trainer = Trainer(
        model=model,
        train_loader=dummy_data_loader,
        val_loader=dummy_data_loader,
        loss_fn=dummy_loss,
        metrics_dict=dummy_metrics,
        cfg=cfg,
        logger_instance=logger
    )
    trainer.train()
    # Verifica que logger.log_scalar fue llamado al menos una vez
    assert logger.log_scalar.call_count > 0


def test_batch_level_logging(dummy_data_loader, dummy_loss,
                             dummy_metrics, tmp_path):
    log_interval = 2  # Log every 2 batches
    num_epochs = 2
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create({
        'training': {
            'epochs': num_epochs,
            'device': 'cpu',
            'use_amp': False,
            'gradient_accumulation_steps': 1,
            'checkpoint_dir': str(checkpoint_dir),
            'optimizer': {'_target_': 'torch.optim.SGD', 'lr': 0.01},
            'lr_scheduler': None,
            'scheduler': None,
            'verbose': False,
            'log_interval_batches': log_interval
        }
    })
    model = torch.nn.Conv2d(3, 1, 1)
    from unittest.mock import MagicMock
    logger = MagicMock()
    # Configurar el experiment_manager del logger mock para que
    # get_path retorne la ruta de checkpoint_dir correctamente
    exp_manager = MagicMock()
    exp_manager.get_path.return_value = str(checkpoint_dir)
    logger.experiment_manager = exp_manager

    trainer = Trainer(
        model=model,
        train_loader=dummy_data_loader,
        val_loader=dummy_data_loader,
        loss_fn=dummy_loss,
        metrics_dict=dummy_metrics,
        cfg=cfg,
        logger_instance=logger
    )
    trainer.train()
    # Verifica que logger.log_scalar fue llamado al menos una vez
    assert logger.log_scalar.call_count > 0


def test_val_step_returns_metrics(
    mock_model, mock_loss_fn, mock_metrics_dict,
    base_trainer_cfg, mock_logger_instance, dummy_batch
):
    # No se necesita instanciar Trainer para testear val_step directamente
    metrics = val_step(
        model=mock_model,
        batch=dummy_batch,
        loss_fn=mock_loss_fn,
        device=torch.device('cpu'),
        metrics_dict=mock_metrics_dict
    )
    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "iou" in metrics
    assert "f1" in metrics
    assert all(isinstance(float(v), float) for v in metrics.values())


def test_validate_aggregates_metrics(
    mock_model, mock_loss_fn, mock_metrics_dict, base_trainer_cfg,
    mock_logger_instance, mock_dataloader
):
    """Test that validate correctly averages and returns metrics with val_
    prefix."""
    trainer = Trainer(
        model=mock_model,
        train_loader=None,
        val_loader=mock_dataloader,
        loss_fn=mock_loss_fn,
        metrics_dict=mock_metrics_dict,
        cfg=base_trainer_cfg,
        logger_instance=mock_logger_instance
    )
    val_metrics = trainer.validate(epoch=1)
    assert isinstance(val_metrics, dict)
    assert "val_loss" in val_metrics
    assert "val_iou" in val_metrics
    assert "val_f1" in val_metrics
    assert all(isinstance(v, float) for v in val_metrics.values())


def test_step_scheduler_reduce_on_plateau(monkeypatch):
    import torch
    from unittest.mock import MagicMock
    # Mock scheduler and optimizer
    scheduler = MagicMock(spec=torch.optim.lr_scheduler.ReduceLROnPlateau)
    optimizer = MagicMock()
    optimizer.param_groups = [{'lr': 0.01}]
    logger = MagicMock()
    # Case: metric present
    metrics = {'val_loss': 0.5}
    from src.utils.scheduler_helper import step_scheduler_helper
    lr = step_scheduler_helper(
        scheduler=scheduler,
        optimizer=optimizer,
        monitor_metric='val_loss',
        metrics=metrics,
        logger=logger
    )
    scheduler.step.assert_called_once_with(0.5)
    logger.info.assert_called()
    assert lr == 0.01
    # Case: metric missing
    scheduler.reset_mock()
    logger.reset_mock()
    lr = step_scheduler_helper(
        scheduler=scheduler,
        optimizer=optimizer,
        monitor_metric='val_loss',
        metrics={},
        logger=logger
    )
    scheduler.step.assert_not_called()
    logger.warning.assert_called()
    assert lr == 0.01


def test_step_scheduler_other_scheduler():
    from unittest.mock import MagicMock
    scheduler = MagicMock()
    optimizer = MagicMock()
    optimizer.param_groups = [{'lr': 0.02}]
    logger = MagicMock()
    from src.utils.scheduler_helper import step_scheduler_helper
    lr = step_scheduler_helper(
        scheduler=scheduler,
        optimizer=optimizer,
        monitor_metric='val_loss',
        metrics={'val_loss': 0.1},
        logger=logger
    )
    scheduler.step.assert_called_once()
    logger.info.assert_called()
    assert lr == 0.02


def test_trainer_early_stopping(
    monkeypatch, mock_model, mock_dataloader, mock_loss_fn,
    mock_metrics_dict, base_trainer_cfg, mock_logger_instance
):
    """Test that Trainer stops training when early_stopper.step returns True.
    """
    # Minimal config
    test_cfg = base_trainer_cfg.copy()
    test_cfg.training.epochs = 5

    # Mock EarlyStopping
    class DummyEarlyStopping:
        def __init__(self):
            self.monitor_metric = 'val_loss'
            self.calls = 0

        def step(self, value):
            self.calls += 1
            # Stop on third call
            return self.calls == 3

    early_stopper = DummyEarlyStopping()

    # Mock validate to always return the same value
    def fake_validate(self, epoch):
        return {'val_loss': 0.5}

    monkeypatch.setattr(
        'src.training.trainer.Trainer.validate',
        fake_validate
    )
    # Instantiate Trainer with early_stopper
    trainer = Trainer(
        model=mock_model,
        train_loader=mock_dataloader,
        val_loader=mock_dataloader,
        loss_fn=mock_loss_fn,
        metrics_dict=mock_metrics_dict,
        cfg=test_cfg,
        logger_instance=mock_logger_instance,
        early_stopper=early_stopper
    )
    # Force use of the mock early_stopper
    trainer.early_stopper = early_stopper
    trainer._train_epoch = lambda epoch: 0.0  # Mock to do nothing
    with patch(
        'src.training.trainer.handle_epoch_checkpointing',
        return_value=float('inf')
    ):
        trainer.train()
    # Should have called step 3 times and exited before completing all epochs
    assert early_stopper.calls == 3


def test_format_metrics():
    from src.utils.training_logging import format_metrics
    metrics = {"loss": 0.1234, "iou": 0.5678}
    formatted = format_metrics(metrics)
    assert "Loss: 0.1234" in formatted
    assert "Iou: 0.5678" in formatted
    assert "|" in formatted


def test_log_validation_results(caplog):
    from src.utils.training_logging import log_validation_results

    class DummyLogger:
        def __init__(self):
            self.last_msg = None

        def info(self, msg):
            self.last_msg = msg

    logger = DummyLogger()
    metrics = {"val_loss": 0.1, "val_iou": 0.9}
    log_validation_results(logger, 5, metrics)
    assert logger.last_msg.startswith("Epoch 5 | Validation Results | ")
    assert "Val_loss: 0.1000" in logger.last_msg
    assert "Val_iou: 0.9000" in logger.last_msg


def test_amp_autocast_context_manager():
    from src.utils.amp_utils import amp_autocast
    import torch
    with amp_autocast(False):
        x = torch.tensor([1.0], requires_grad=True)
        y = x * 2
        assert y.item() == 2.0
    # No error should occur
    with amp_autocast(True):
        x = torch.tensor([1.0], requires_grad=True)
        y = x * 2
        assert y.item() == 2.0


def test_optimizer_step_with_accumulation_no_amp():
    from src.utils.amp_utils import optimizer_step_with_accumulation
    import torch
    from unittest.mock import MagicMock
    optimizer = MagicMock()
    scaler = None
    loss = torch.tensor(1.0, requires_grad=True)
    grad_accum_steps = 2
    # batch_idx = 0: no step
    optimizer_step_with_accumulation(
        optimizer=optimizer,
        scaler=scaler,
        loss=loss,
        grad_accum_steps=grad_accum_steps,
        batch_idx=0,
        use_amp=False
    )
    optimizer.step.assert_not_called()
    # batch_idx = 1: step
    optimizer_step_with_accumulation(
        optimizer=optimizer,
        scaler=scaler,
        loss=loss,
        grad_accum_steps=grad_accum_steps,
        batch_idx=1,
        use_amp=False
    )
    optimizer.step.assert_called_once()
    optimizer.zero_grad.assert_called()


def test_optimizer_step_with_accumulation_amp():
    from src.utils.amp_utils import optimizer_step_with_accumulation
    import torch
    from unittest.mock import MagicMock
    optimizer = MagicMock()
    scaler = MagicMock()
    loss = torch.tensor(1.0, requires_grad=True)
    grad_accum_steps = 2
    # batch_idx = 0: no step
    optimizer_step_with_accumulation(
        optimizer=optimizer,
        scaler=scaler,
        loss=loss,
        grad_accum_steps=grad_accum_steps,
        batch_idx=0,
        use_amp=True
    )
    scaler.scale.assert_called_with(loss / grad_accum_steps)
    scaler.step.assert_not_called()
    # batch_idx = 1: step
    optimizer_step_with_accumulation(
        optimizer=optimizer,
        scaler=scaler,
        loss=loss,
        grad_accum_steps=grad_accum_steps,
        batch_idx=1,
        use_amp=True
    )
    scaler.step.assert_called_once_with(optimizer)
    scaler.update.assert_called()
    optimizer.zero_grad.assert_called()


def test_validate_trainer_config_valid():
    from src.training.config_validation import validate_trainer_config

    class DummyCfg:
        epochs = 5
        device = "cpu"
        optimizer = {"_target_": "torch.optim.Adam", "lr": 0.001}
        gradient_accumulation_steps = 1
    # No debe lanzar excepción
    validate_trainer_config(DummyCfg())


def test_validate_trainer_config_missing_field():
    from src.training.config_validation import validate_trainer_config

    class DummyCfg:
        device = "cpu"
        optimizer = {"_target_": "torch.optim.Adam", "lr": 0.001}
        gradient_accumulation_steps = 1
    import pytest
    with pytest.raises(ValueError):
        validate_trainer_config(DummyCfg())


def test_validate_trainer_config_invalid_type():
    from src.training.config_validation import validate_trainer_config

    class DummyCfg:
        epochs = "five"
        device = "cpu"
        optimizer = {"_target_": "torch.optim.Adam", "lr": 0.001}
        gradient_accumulation_steps = 1
    import pytest
    with pytest.raises(TypeError):
        validate_trainer_config(DummyCfg())


def test_validate_trainer_config_invalid_optimizer():
    from src.training.config_validation import validate_trainer_config

    class DummyCfg:
        epochs = 5
        device = "cpu"
        optimizer = {"lr": 0.001}  # Falta _target_
        gradient_accumulation_steps = 1
    import pytest
    with pytest.raises(ValueError):
        validate_trainer_config(DummyCfg())
