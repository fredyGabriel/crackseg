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

import logging
import sys
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from crackseg.training.trainer import Trainer, TrainingComponents
from crackseg.utils.experiment import initialize_experiment

# Import specialized modules from training_pipeline package
from training_pipeline import (
    create_model,
    handle_checkpointing_and_resume,
    load_data,
    setup_environment,
    setup_training_components,
)

# Configure standard logger
log = logging.getLogger(__name__)


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
    model.encoder_name=resnet50 ``` Resume from checkpoint: ```bash
    python run.py training.checkpoints.resume_from_checkpoint=\
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
        device = setup_environment(cfg)

        # Initialize experiment and logging
        experiment_dir, experiment_logger = initialize_experiment(cfg)
        log.info("Experiment initialized in: %s", experiment_dir)

        # --- 2. Data Loading ---
        train_loader, val_loader = load_data(cfg)

        # --- 3. Model Creation ---
        model = create_model(cfg, device)

        # --- 4. Training Setup ---
        metrics, optimizer, loss_fn = setup_training_components(cfg, model)

        # --- 5. Checkpointing and Resume ---
        # Note: best_metric_value from here is mostly for initial logging.
        # The Trainer itself will manage and update the actual
        # best_metric_value based on its internal logic and config.
        _start_epoch, _ = handle_checkpointing_and_resume(
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
    if len(sys.argv) > 1:
        # If arguments are provided, let Hydra handle them
        main()
    else:
        # For direct execution without arguments, we need to provide a default
        # config
        config_dir = Path(__file__).parent.parent / "configs"
        with initialize_config_dir(
            config_dir=str(config_dir), version_base=None
        ):
            cfg = compose(config_name="base")
            main(cfg)
