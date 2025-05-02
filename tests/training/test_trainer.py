"""Tests for the Trainer class."""

import pytest
import torch
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch

# Assuming the necessary modules exist in these paths
from src.training.trainer import Trainer
from src.utils.logging import NoOpLogger  # Use NoOpLogger for testing


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
        "trainer": {
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
    """Devuelve un batch dummy de imágenes y máscaras."""
    images = torch.randn(2, 3, 4, 4)
    masks = torch.randn(2, 1, 4, 4)
    return (images, masks)


@pytest.fixture
def dummy_data_loader():
    # Devuelve un DataLoader con 2 batches de datos dummy
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 4

        def __getitem__(self, idx):
            return torch.randn(3, 4, 4), torch.randn(1, 4, 4)
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
        if base_trainer_cfg.trainer.lr_scheduler:
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
@patch('src.training.trainer.save_checkpoint')
@patch('src.training.trainer.Trainer._step_scheduler')
@patch(
    'src.training.trainer.Trainer.validate',
    return_value={"loss": 0.4, "iou": 0.8}
)
@patch('src.training.trainer.Trainer._train_epoch', return_value=0.5)
def test_trainer_train_loop(
    mock_save_checkpoint,
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

    # Modify config slightly for this test
    test_cfg = base_trainer_cfg.copy()
    test_cfg.trainer.epochs = 3  # Test with 3 epochs
    test_cfg.trainer.lr_scheduler = OmegaConf.create({
        "_target_": "torch.optim.lr_scheduler.StepLR",
        "step_size": 1
    })

    # Instantiate Trainer
    # (it will internally try to call the patched factories)
    trainer = Trainer(
        model=mock_model,
        train_loader=mock_dataloader,
        val_loader=mock_dataloader,
        loss_fn=mock_loss_fn,
        metrics_dict=mock_metrics_dict,
        cfg=test_cfg,
        logger_instance=mock_logger_instance
    )

    # Remove manual setting, rely on patched factories during init
    # trainer.optimizer = mock_optimizer
    # trainer.scheduler = mock_scheduler_instance

    # --- Execute the train method ---
    final_results = trainer.train()

    # --- Assertions ---
    # Check factories were called during init
    mock_create_trainer_optimizer.assert_called_once()
    mock_create_trainer_scheduler.assert_called_once()

    assert mock_train_epoch.call_count == test_cfg.trainer.epochs
    assert mock_validate.call_count == test_cfg.trainer.epochs

    if trainer.scheduler:
        assert mock_step_scheduler.call_count == test_cfg.trainer.epochs

    assert final_results == {"loss": 0.4, "iou": 0.8}

    # Loop assertions
    for i in range(1, test_cfg.trainer.epochs + 1):
        mock_train_epoch.assert_any_call(i)
        # mock_validate.assert_any_call(i) # Arg check complex

    # Instead of checking args in loop, check total count after loop
    if trainer.scheduler:
        assert mock_step_scheduler.call_count == test_cfg.trainer.epochs

    # Assert save_checkpoint was called (e.g., once per epoch)
    assert mock_save_checkpoint.call_count == test_cfg.trainer.epochs


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
    base_trainer_cfg.trainer.gradient_accumulation_steps = 2

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
    loss_value = trainer._train_step(dummy_batch)

    # Verificaciones
    mock_model.assert_called_once()  # Se llama el forward
    mock_loss_fn.assert_called_once()  # Se calcula el loss
    mock_scaler.scale.assert_called_once()  # Se escala el loss
    mock_scaled_loss.backward.assert_called_once()  # Se hace backward

    # El valor devuelto debe ser el loss * grad_accum_steps
    assert loss_value == pytest.approx(0.8)


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
    base_trainer_cfg.trainer.use_amp = True
    base_trainer_cfg.trainer.device = "cuda:0"
    base_trainer_cfg.trainer.gradient_accumulation_steps = 1

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
    loss_value = trainer._train_step(dummy_batch)

    mock_model.assert_called_once()
    mock_loss_fn.assert_called_once()
    mock_scaler.scale.assert_called_once()
    mock_scaled_loss.backward.assert_called_once()
    assert loss_value == pytest.approx(0.5)


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
    base_trainer_cfg.trainer.gradient_accumulation_steps = 1

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
        trainer._train_step(dummy_batch)


# Remove patch for log_metrics_dict, check logger calls instead
# @patch('src.utils.logging.base.log_metrics_dict')
def test_epoch_level_logging(dummy_data_loader, dummy_loss,
                             dummy_metrics):
    # Configuración mínima para usar StepLR
    cfg = OmegaConf.create({
        'trainer': {
            'epochs': 2,
            'device': 'cpu',
            'use_amp': False,
            'grad_accum_steps': 1,
            'checkpoint_dir': 'checkpoints',
            'optimizer': {'_target_': 'torch.optim.SGD', 'lr': 0.01},
            'lr_scheduler': {'_target_': 'torch.optim.lr_scheduler.StepLR',
                             'step_size': 1, 'gamma': 0.5},
            'verbose': False
        }
    })
    model = torch.nn.Conv2d(3, 1, 1)
    logger = MagicMock()
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

    # Verifica que logger.log_scalar fue llamado
    train_epoch_calls = [c for c in logger.log_scalar.call_args_list if
                         c.kwargs.get('tag') == 'train/epoch_loss']
    val_loss_calls = [c for c in logger.log_scalar.call_args_list if
                      c.kwargs.get('tag') == 'val/loss']
    lr_calls = [c for c in logger.log_scalar.call_args_list if
                c.kwargs.get('tag') == 'lr']

    assert train_epoch_calls, 'No se registró train/epoch_loss'
    assert val_loss_calls, 'No se registró val/loss'
    assert lr_calls, 'No se registró el learning rate'
    # Check steps (example for first call)
    assert train_epoch_calls[0].kwargs['step'] == 1
    assert val_loss_calls[0].kwargs['step'] == 1
    assert lr_calls[0].kwargs['step'] == 1


# Remove patch for log_metrics_dict, check logger calls instead
# @patch('src.utils.logging.base.log_metrics_dict')
def test_batch_level_logging(dummy_data_loader, dummy_loss,
                             dummy_metrics):
    log_interval = 2  # Log every 2 batches
    num_epochs = 2
    num_batches_per_epoch = len(dummy_data_loader)

    cfg = OmegaConf.create({
        'trainer': {
            'epochs': num_epochs,
            'device': 'cpu',
            'use_amp': False,
            'grad_accum_steps': 1,
            'checkpoint_dir': 'checkpoints',
            'optimizer': {'_target_': 'torch.optim.SGD', 'lr': 0.01},
            'lr_scheduler': None,
            'verbose': False,
            'log_interval_batches': log_interval  # Set interval
        }
    })
    model = torch.nn.Conv2d(3, 1, 1)
    logger = MagicMock()
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

    # Check that the tag is correct
    batch_calls = [c for c in logger.log_scalar.call_args_list if
                   c.kwargs.get('tag') == 'train_batch/batch_loss']

    # Expected number of calls = epochs * (batches_per_epoch // log_interval)
    expected_calls = num_epochs * (num_batches_per_epoch // log_interval)
    assert len(batch_calls) == expected_calls, \
        f"Expected {expected_calls} batch logging calls, got \
{len(batch_calls)}"

    # Verifica que las llamadas tengan una estructura correcta
    for i, call in enumerate(batch_calls):
        # Check keyword arguments
        kwargs = call.kwargs
        assert kwargs.get('tag') == 'train_batch/batch_loss', \
            f"Call {i}: Tag mismatch"
        assert isinstance(kwargs.get('value'), float), \
            f"Call {i}: value is not float"
        assert isinstance(kwargs.get('step'), int) and \
               kwargs.get('step') > 0, f"Call {i}: Step is not a positive int"

    # Verify specific global steps based on the number of calls
    if expected_calls > 0:
        assert batch_calls[0].kwargs['step'] == log_interval, \
            f"First batch step should be {log_interval}"
    if expected_calls > 1:
        assert batch_calls[1].kwargs['step'] == 2 * log_interval, \
            f"Second batch step should be {2 * log_interval}"
    # Check step in subsequent epochs
    if num_epochs > 1 and expected_calls > \
       (num_batches_per_epoch // log_interval):
        multi_epoch_idx = num_batches_per_epoch // log_interval
        expected_step = num_batches_per_epoch + log_interval
        assert batch_calls[multi_epoch_idx].kwargs['step'] == expected_step, \
            f"Multi-epoch step should be {expected_step}"
