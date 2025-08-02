#!/usr/bin/env python3
"""
Main training pipeline for pavement crack segmentation. This module
provides the primary entry point for training crack segmentation
models using the U-Net architecture with configurable encoders and
decoders. It integrates with Hydra for configuration management and
supports features like: - Automated experiment tracking and logging -
Configurable data loading and augmentation - Model checkpointing and
resume functionality - Mixed precision training support -
Comprehensive error handling and validation The main training pipeline
consists of several stages: 1. Environment setup (device detection,
random seeds) 2. Data loading (train/validation dataloaders) 3. Model
creation and initialization 4. Training component setup (optimizer,
loss, metrics) 5. Checkpoint handling and resume logic 6. Training
execution via Trainer class 7. Cleanup and experiment finalization
Examples: Basic training with default configuration: ```bash python
run.py ``` Training with custom parameters: ```bash python run.py
training.epochs=100 data.batch_size=8 ``` Resume from checkpoint:
```bash python run.py training.checkpoints.resume_from_checkpoint=\\
path/to/checkpoint.pth ``` Configuration: This module uses Hydra
configuration management. The main config file is located at
'configs/config.yaml' with additional configs in subdirectories. Key
configuration sections: - model: Model architecture and parameters -
data: Dataset and dataloader configuration - training: Training
hyperparameters and settings - evaluation: Metrics and evaluation
settings Note: This module requires CUDA support for GPU training. CPU
training is supported but not recommended for large models due to
performance considerations.
"""

# In src/main.py (Skeleton and checkpointing logic)
import logging
import math  # For inf
import os
from typing import Any, cast

import hydra
import torch
from hydra import errors as hydra_errors  # Import Hydra errors
from omegaconf import DictConfig, OmegaConf
from omegaconf import errors as omegaconf_errors  # Import OmegaConf errors
from torch import optim  # For Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader  # Added for DataLoader

# Project import s
from crackseg.data.factory import (
    create_dataloaders_from_config,  # Import factory
)
from crackseg.training.trainer import Trainer, TrainingComponents
from crackseg.utils import (
    DataError,
    ModelError,
    ResourceError,
    get_device,
    load_checkpoint,
    set_random_seeds,
)
from crackseg.utils.experiment import initialize_experiment
from crackseg.utils.factory import (
    get_loss_fn,
    get_metrics_from_cfg,
    get_optimizer,
)

# Configure standard logger
log = logging.getLogger(__name__)


def _setup_environment(cfg: DictConfig) -> torch.device:
    """
    Set up the training environment with proper device selection and random
    seeds.

    This function configures the global training environment including:
    - Random seed initialization for reproducibility
    - CUDA availability validation
    - Device selection and configuration

    Args:
        cfg: Hydra configuration containing environment settings.
            Expected keys:
            - random_seed (int, optional): Random seed for reproducibility.
              Defaults to 42.
            - require_cuda (bool, optional): Whether CUDA is required.
              Defaults to True.

    Returns:
        torch.device: The selected device for training (e.g., 'cuda:0' or
        'cpu').

    Raises:
        ResourceError: If CUDA is required but not available on the system.

    Examples:
        ```python
        cfg = OmegaConf.create({"random_seed": 42, "require_cuda": True})
        device = _setup_environment(cfg)
        print(f"Training on device: {device}")
        ```

    Note:
        The function automatically detects the best available device (GPU vs
        CPU) and logs the selection for transparency.
    """
    log.info("Setting up environment...")
    set_random_seeds(cfg.get("random_seed", 42))

    if not torch.cuda.is_available() and cfg.get("require_cuda", True):
        log.error("CUDA is required but not available on this system.")
        raise ResourceError(
            "CUDA is required but not available on this system."
        )

    device = get_device()
    log.info("Using device: %s", device)
    return device


def _load_data(cfg: DictConfig) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """
    Load and create training and validation data loaders from configuration.

    This function handles the complete data loading pipeline including:
    - Configuration validation and path resolution
    - Transform pipeline creation
    - DataLoader instantiation with optimized settings
    - Error handling for missing or invalid data

    Args:
        cfg: Hydra configuration containing data settings.
            Expected structure:
            - cfg.data.data_root (str): Root directory for dataset
            - cfg.data.transform (DictConfig, optional): Transform
                configuration
            - cfg.data.dataloader (DictConfig, optional): DataLoader
                configuration

    Returns: tuple[DataLoader[Any], DataLoader[Any]]: A tuple containing:
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data

    Raises:
        DataError: If data loading fails due to:
            - Missing or invalid dataset files
            - Configuration errors
            - DataLoader creation failures
        OSError: If data directory is not accessible
        FileNotFoundError: If required data files are missing

    Examples:
        ```python
        cfg = OmegaConf.load("configs/data/default.yaml")
        train_loader, val_loader = _load_data(cfg)

        # Check data loader properties
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")

        # Sample a batch
        images, masks = next(iter(train_loader))
        print(f"Batch shape: {images.shape}, {masks.shape}")
        ```

    Note:
        The function automatically resolves relative paths using Hydra's
        original working directory and provides fallback configurations
        for missing transform or dataloader settings.
    """
    log.info("Loading data...")
    try:
        data_cfg = cfg.data
        transform_cfg = None
        if hasattr(cfg.data, "transform"):
            transform_cfg = cfg.data.transform
        elif "data/transform" in cfg:
            transform_cfg = cfg["data/transform"]
        else:
            log.warning(
                "Transform config not found in Hydra config. "
                "Using empty config."
            )
            transform_cfg = OmegaConf.create({})

        orig_cwd = hydra.utils.get_original_cwd()
        data_root = os.path.join(orig_cwd, data_cfg.get("data_root", "data/"))
        data_cfg["data_root"] = data_root

        dataloader_cfg = None
        if hasattr(cfg.data, "dataloader"):
            dataloader_cfg = cfg.data.dataloader
        elif "data/dataloader" in cfg:
            dataloader_cfg = cfg["data/dataloader"]
        else:
            log.warning(
                "Dataloader config not found in Hydra config. "
                "Using data config as fallback."
            )
            dataloader_cfg = data_cfg

        # Ensure dataloader_cfg is a DictConfig
        if not isinstance(dataloader_cfg, DictConfig):
            # Attempt to convert if it's a basic dict or list that OmegaConf
            # can handle
            try:
                converted_cfg = OmegaConf.create(dataloader_cfg)
                if isinstance(converted_cfg, DictConfig):
                    dataloader_cfg = converted_cfg
                else:
                    log.warning(
                        "Could not convert dataloader_cfg to DictConfig. "
                        "It is of type: %s. Using empty DictConfig.",
                        type(converted_cfg),
                    )
                    dataloader_cfg = OmegaConf.create({})
            except Exception as e:
                log.warning(
                    "Error converting dataloader_cfg to DictConfig: %s. "
                    "Using empty DictConfig.",
                    e,
                )
                dataloader_cfg = OmegaConf.create({})

        dataloaders_dict = create_dataloaders_from_config(
            data_config=data_cfg,
            transform_config=transform_cfg,
            dataloader_config=dataloader_cfg,
        )
        train_loader = dataloaders_dict.get("train", {}).get("dataloader")
        val_loader = dataloaders_dict.get("val", {}).get("dataloader")

        if not train_loader or not val_loader:
            log.error("Train or validation dataloader could not be created.")
            raise DataError(
                "Train or validation dataloader could not be created."
            )

        if not isinstance(train_loader, DataLoader) or not isinstance(
            val_loader, DataLoader
        ):
            raise DataError(
                "Train or validation loader is not a DataLoader instance"
            )

        log.info("Data loading complete.")
        return train_loader, val_loader
    except (
        OSError,
        DataError,
        omegaconf_errors.OmegaConfBaseException,
        FileNotFoundError,
        ImportError,
        ValueError,
        TypeError,
    ) as e:
        log.error("Error during data loading: %s", str(e))
        raise DataError(f"Error during data loading: {str(e)}") from e


def _create_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """
    Create and initialize the segmentation model from configuration.

    This function instantiates the neural network model using Hydra's
    instantiation system and prepares it for training by:
    - Creating the model from configuration
    - Moving the model to the specified device
    - Logging model information (type and parameter count)
    - Validating the model is properly initialized

    Args:
        cfg: Hydra configuration containing model settings.
            Expected structure:
            - cfg.model._target_: Full path to model class
                (e.g., "src.model.UNet")
            - cfg.model.**kwargs: Model-specific parameters

        device: Target device for model placement (e.g., 'cuda:0' or 'cpu').

    Returns:
        torch.nn.Module: The initialized model ready for training.

    Raises:
        ModelError: If model creation fails due to:
            - Invalid model configuration
            - Missing model dependencies
            - Instantiation errors
        ImportError: If the specified model class cannot be import ed
        AttributeError: If model configuration is malformed

    Examples:
        ```python
        cfg = OmegaConf.create({
            "model": {
                "_target_": "src.model.core.unet.UNet",
                "encoder_name": "resnet34",
                "encoder_weights": "imagenet",
                "in_channels": 3,
                "classes": 1
            }
        })
        device = torch.device("cuda:0")
        model = _create_model(cfg, device)

        # Check model properties
        print(f"Model type: {type(model).__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Device: {next(model.parameters()).device}")
        ```

    Note:
        The function uses Hydra's instantiate method which supports
        complex configuration patterns and automatic dependency injection.
    """
    log.info("Creating model...")
    try:
        model = hydra.utils.instantiate(cfg.model)
        model = cast(Module, model)
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        log.info(
            "Created %s model with %s parameters",
            type(model).__name__,
            num_params,
        )
        assert isinstance(model, Module)
        return model
    except (
        ModelError,
        hydra_errors.InstantiationException,
        AttributeError,
        ImportError,
        TypeError,
        ValueError,
    ) as e:
        log.error("Error creating model: %s", str(e))
        raise ModelError(f"Error creating model: {str(e)}") from e


def _setup_training_components(
    cfg: DictConfig, model: torch.nn.Module
) -> tuple[
    dict[str, Any],
    optim.Optimizer,
    torch.nn.Module,
]:
    """
    Set up training components including metrics, optimizer, and loss function.

    This function configures all components required for training:
    - Evaluation metrics from configuration
    - Optimizer with specified parameters
    - Loss function with appropriate configuration
    - Error handling and fallback mechanisms

    Args:
        cfg: Hydra configuration containing training settings.
            Expected structure:
            - cfg.evaluation.metrics (DictConfig, optional): Metrics
            configuration
            - cfg.training.optimizer (DictConfig): Optimizer configuration
            - cfg.training.loss (DictConfig): Loss function configuration

        model: The neural network model for parameter optimization.

    Returns:
        tuple containing:
            - dict[str, Any]: Dictionary of evaluation metrics
            - optim.Optimizer: Configured optimizer for training
            - torch.nn.Module: Loss function module

    Raises:
        ImportError: If specified components cannot be import ed
        AttributeError: If configuration is malformed
        ValueError: If configuration values are invalid
        TypeError: If component types are incompatible

    Examples:
        ```python
        cfg = OmegaConf.create({
            "evaluation": {
                "metrics": {
                    "iou": {"_target_": "src.metrics.IoU"},
                    "dice": {"_target_": "src.metrics.DiceScore"}
                }
            },
            "training": {
                "optimizer": {
                    "_target_": "torch.optim.Adam",
                    "lr": 0.001,
                    "weight_decay": 1e-4
                },
                "loss": {
                    "_target_": "src.training.losses.BCEDiceLoss",
                    "bce_weight": 0.5,
                    "dice_weight": 0.5
                }
            }
        })

        metrics, optimizer, loss_fn = _setup_training_components(cfg, model)
        print(f"Metrics: {list(metrics.keys())}")
        print(f"Optimizer: {type(optimizer).__name__}")
        print(f"Loss function: {type(loss_fn).__name__}")
        ```

    Note:
        The function provides robust fallback mechanisms:
        - Empty metrics dict if metrics config is missing
        - BCEWithLogitsLoss as fallback loss function
        - Comprehensive error logging for debugging
    """
    log.info("Setting up training components...")

    metrics: dict[str, Any] = {}
    if hasattr(cfg, "evaluation") and hasattr(cfg.evaluation, "metrics"):
        try:
            metrics = cast(
                dict[str, Any],
                get_metrics_from_cfg(cfg.evaluation.metrics),
            )
            log.info("Loaded metrics: %s", list(metrics.keys()))
        except (
            omegaconf_errors.OmegaConfBaseException,
            KeyError,
            AttributeError,
            ImportError,
            ValueError,
        ) as e:
            log.error("Error loading metrics: %s", e)
            metrics = {}
    else:
        log.warning("Evaluation metrics configuration not found.")

    optimizer_cfg = cfg.training.get("optimizer", {"type": "adam", "lr": 1e-3})
    # Convert model.parameters() to a list to satisfy get_optimizer
    optimizer = get_optimizer(list(model.parameters()), optimizer_cfg)
    log.info("Created optimizer: %s", type(optimizer).__name__)

    loss_fn_instance: torch.nn.Module
    if hasattr(cfg.training, "loss") and cfg.training.loss is not None:
        try:
            potential_loss_fn = get_loss_fn(cfg.training.loss)
            if isinstance(potential_loss_fn, torch.nn.Module):
                loss_fn_instance = potential_loss_fn
                log.info(
                    "Created loss function from config: %s",
                    type(loss_fn_instance).__name__,
                )
            else:
                log.error(
                    "get_loss_fn did not return an nn.Module. "
                    "Got: %s. Using fallback.",
                    str(type(potential_loss_fn)),
                )
                loss_fn_instance = torch.nn.BCEWithLogitsLoss()  # Fallback
        except (
            omegaconf_errors.OmegaConfBaseException,
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
        ) as e:
            log.error(
                "Error creating loss function from config: %s. "
                "Using fallback.",
                e,
            )
            loss_fn_instance = torch.nn.BCEWithLogitsLoss()  # Fallback
    else:
        log.warning(
            "Loss function configuration not found or is null. "
            "Using fallback: BCEWithLogitsLoss."
        )
        loss_fn_instance = torch.nn.BCEWithLogitsLoss()  # Fallback

    # scheduler and scaler removed for linter compliance
    log.info("Training setup complete.")
    return metrics, optimizer, loss_fn_instance


def _handle_checkpointing_and_resume(
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    experiment_logger: Any,
) -> tuple[int, float | None]:
    """
    Handle checkpoint loading and training resume functionality.

    This function manages the checkpointing system including:
    - Checkpoint directory setup and validation
    - Resume logic from existing checkpoints
    - Best model tracking initialization
    - Metric monitoring configuration

    Args:
        cfg: Hydra configuration containing checkpoint settings.
            Expected structure:
            - cfg.training.checkpoints.resume_from_checkpoint (str, optional):
              Path to checkpoint for resuming
            - cfg.training.checkpoints.save_best (DictConfig, optional):
              Best model saving configuration

        model: Neural network model for state loading.
        optimizer: Optimizer for state loading.
        device: Target device for checkpoint loading.
        experiment_logger: Experiment logger with checkpoint directory access.

    Returns:
        tuple containing:
            - int: Starting epoch number (0 for fresh start,
            epoch+1 for resume)
            - float | None: Best metric value from checkpoint (None for fresh
            start)

    Raises:
        FileNotFoundError: If specified checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
        KeyError: If checkpoint format is invalid

    Examples:
        ```python
        # Configuration for resuming from checkpoint
        cfg = OmegaConf.create({
            "training": {
                "checkpoints": {
                    "resume_from_checkpoint": "checkpoints/best_model.pth",
                    "save_best": {
                        "enabled": True,
                        "monitor_metric": "val_iou",
                        "monitor_mode": "max"
                    }
                }
            }
        })

        start_epoch, best_metric = _handle_checkpointing_and_resume(
            cfg, model, optimizer, device, experiment_logger
        )

        if start_epoch > 0:
            print(f"Resumed from epoch {start_epoch}, best metric: \
                {best_metric}")
        else:
            print("Starting fresh training")
        ```

    Note:
        - Supports both absolute and relative checkpoint paths
        - Automatically resolves paths relative to original working directory
        - Provides comprehensive logging for checkpoint operations
        - Initializes best metric tracking based on monitoring mode
    """
    log.info("Handling checkpointing and resume...")
    start_epoch = 0
    best_metric_value = None

    # Ensure experiment_logger and its manager are valid before use
    if not hasattr(experiment_logger, "experiment_manager"):
        log.error(
            "Experiment logger does not have 'experiment_manager'. "
            "Cannot determine checkpoint directory."
        )
        # Fallback or raise error depending on desired behavior
        # For now, log and continue, checkpointing might be disabled or fail
        checkpoint_dir = "checkpoints"  # Fallback
    else:
        experiment_manager = experiment_logger.experiment_manager
        checkpoint_dir = str(experiment_manager.get_path("checkpoints"))

    log.info("Using checkpoint directory: %s", checkpoint_dir)

    checkpoint_cfg = cfg.training.get("checkpoints", OmegaConf.create({}))
    resume_path = checkpoint_cfg.get("resume_from_checkpoint", None)

    if resume_path:
        msg = f"Attempting to resume from checkpoint: {resume_path}"
        log.info(msg)
        orig_cwd = hydra.utils.get_original_cwd()
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(orig_cwd, resume_path)

        if os.path.exists(resume_path):
            checkpoint_data = load_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=resume_path,
                device=device,
            )
            start_epoch = checkpoint_data.get("epoch", 0) + 1
            best_metric_value = checkpoint_data.get("best_metric_value", None)
            msg = (
                f"Resumed from epoch {start_epoch}. Best: {best_metric_value}"
            )
            log.info(msg)
        else:
            log.warning(
                f"Resume checkpoint not found: {resume_path}. Fresh start."
            )
    else:
        log.info("No checkpoint specified for resume, starting from scratch.")

    # Setup Best Model Tracking (initialization part)
    save_best_cfg = checkpoint_cfg.get("save_best", OmegaConf.create({}))
    monitor_metric = save_best_cfg.get("monitor_metric", None)
    monitor_mode = save_best_cfg.get("monitor_mode", "max")
    save_best_enabled = save_best_cfg.get("enabled", False)

    if save_best_enabled and not monitor_metric:
        msg = (
            "save_best enabled but monitor_metric not set. "
            "Disabling best model saving."
        )
        log.warning(msg)
        # save_best_enabled = False # This variable is local, trainer will
        # read from cfg
    elif save_best_enabled:
        log.info(
            f"Monitoring '{monitor_metric}' for best model (mode: "
            f"{monitor_mode})."
        )
        if best_metric_value is None:  # Initialize if not resuming
            best_metric_value = (
                -math.inf if monitor_mode == "max" else math.inf
            )
            log.info("Initializing best metric to %s", best_metric_value)

    log.info("Checkpointing and resume handling complete.")
    return start_epoch, best_metric_value


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """
Main training pipeline entry point for crack segmentation. This
function orchestrates the complete training workflow including: 1.
Environment setup and device configuration 2. Experiment
initialization and logging setup 3. Data loading and validation 4.
Model creation and initialization 5. Training component configuration
6. Checkpoint handling and resume logic 7. Training execution via
Trainer class 8. Cleanup and resource management The function is
decorated with Hydra's main decorator to enable configuration
management and CLI parameter overrides. Args: cfg: Complete Hydra
configuration object containing all settings. Key sections: - model:
Neural network architecture and parameters - data: Dataset and
dataloader configuration - training: Training hyperparameters and
settings - evaluation: Metrics and evaluation configuration -
experiment: Experiment tracking and logging settings Returns: None:
Function handles training execution and cleanup internally. Raises:
ResourceError: If required hardware resources are unavailable
DataError: If data loading or validation fails ModelError: If model
creation or initialization fails ConfigurationError: If configuration
is invalid or incomplete Exception: Any unhandled exception during
training execution Examples: Training with default configuration:
```bash python run.py ``` Training with parameter overrides: ```bash
python run.py training.epochs=100 \\ data.batch_size=8 \
model.encoder_name=resnet50 ``` Resume from checkpoint: ```bash python
run.py training.checkpoints.resume_from_checkpoint=\
path/to/checkpoint.pth ``` GPU-specific training: ```bash python
run.py training.device=cuda:1 \\ training.use_amp=true \
data.num_workers=8 ``` Configuration Examples: Minimal training
configuration: ```yaml model: _target_: src.model.core.unet.UNet
encoder_name: resnet34 classes: 1 data: data_root: data/ batch_size:
16 training: epochs: 100 optimizer: _target_: torch.optim.Adam lr:
0.001 ``` Production training configuration: ```yaml training: epochs:
200 use_amp: true checkpoints: save_freq: 10 save_best: enabled: true
monitor_metric: val_iou monitor_mode: max early_stopping: patience: 20
min_delta: 0.001 ``` Note: - Experiment tracking is automatically
initialized with unique timestamps - All training artifacts are saved
to structured output directories - Comprehensive error handling
ensures graceful failure recovery - Final evaluation should be
performed separately using evaluate.py See Also: - src.evaluate: For
model evaluation and inference - src.training.trainer: Core training
loop implementation - configs/: Configuration files and examples
"""
    experiment_logger = None
    try:
        # --- 1. Initial Setup ---
        log.info("Starting main execution...")
        device = _setup_environment(cfg)

        # Initialize experiment and logging
        experiment_dir, experiment_logger = initialize_experiment(cfg)
        log.info("Experiment initialized in: %s", experiment_dir)

        # --- 2. Data Loading ---
        train_loader, val_loader = _load_data(cfg)

        # --- 3. Model Creation ---
        model = _create_model(cfg, device)

        # --- 4. Training Setup ---
        metrics, optimizer, loss_fn = _setup_training_components(cfg, model)

        # --- 5. Checkpointing and Resume ---
        # Note: best_metric_value from here is mostly for initial logging.
        # The Trainer itself will manage and update the actual
        # best_metric_value based on its internal logic and config.
        _start_epoch, _ = _handle_checkpointing_and_resume(
            cfg, model, optimizer, device, experiment_logger
        )

        # --- 6. Training Loop (delegated to Trainer) ---
        log.info("Starting training loop...")
        components = TrainingComponents(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            metrics_dict=metrics,
        )
        trainer = Trainer(
            components=components,
            cfg=cfg,
            logger_instance=experiment_logger,
            # early_stopper can be passed if initialized separately
        )
        trainer.train()

        # --- 7. Final Evaluation ---
        log.info(
            "Final evaluation removed from main.py. "
            "Use evaluate.py for evaluation."
        )

    except Exception as e:
        # Log and properly handle the error
        if experiment_logger:
            experiment_logger.log_error(exception=e, context="Main execution")
            experiment_logger.close()

        raise e  # Re-raise to let Hydra handle it

    finally:
        # --- 8. Cleanup ---
        if experiment_logger:
            experiment_logger.close()
        log.info("Main execution finished.")


if __name__ == "__main__":
    # Hydra will automatically provide the cfg parameter when called from
    # command line
    # For direct execution, we need to handle this differently
    import sys

    if len(sys.argv) > 1:
        # If arguments are provided, let Hydra handle them
        main()
    else:
        # For direct execution without arguments, we need to provide a default
        # config
        from pathlib import Path

        from hydra import compose, initialize_config_dir

        config_dir = Path(__file__).parent.parent / "configs"
        with initialize_config_dir(
            config_dir=str(config_dir), version_base=None
        ):
            cfg = compose(config_name="base")
            main(cfg)
