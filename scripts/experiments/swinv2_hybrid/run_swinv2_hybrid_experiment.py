#!/usr/bin/env python3
"""
SwinV2 Hybrid Architecture Experiment Runner

This script runs the SwinV2 + ASPP + CNN hybrid architecture experiment
with Focal Dice Loss for crack segmentation using the standard project
Hydra configuration system.

Features:
- Hybrid architecture: SwinV2 Transformer + ASPP + CNN Decoder
- Optimized loss: Focal Dice Loss for class imbalance
- Hardware optimization: RTX 3070 Ti (8GB VRAM) settings
- Comprehensive monitoring and logging
- Reproducible training with fixed seeds
- Standard project Hydra configuration integration

Usage:
    # Use default SwinV2 hybrid configuration
    python scripts/experiments/swinv2_hybrid/run_swinv2_hybrid_experiment.py

    # Override specific parameters
    python scripts/experiments/swinv2_hybrid/run_swinv2_hybrid_experiment.py \
        training.batch_size=8 \
        training.learning_rate=0.0002
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="experiments/swinv2_hybrid/swinv2_hybrid_experiment",
)
def main(cfg: DictConfig) -> None:
    """Main experiment execution function using standard project configuration.

    Args:
        cfg: Hydra configuration object containing all experiment settings.
    """
    # Setup logging using standard project logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        # Validate environment using standard project utilities
        logger.info("Validating environment...")
        _validate_environment()

        # Validate configuration using standard project utilities
        logger.info("Validating configuration...")
        _validate_config(cfg)

        # Print experiment summary
        _print_experiment_summary(cfg)

        # Check for dry-run mode
        if cfg.get("dry_run", False):
            logger.info("Running dry-run validation...")
            _validate_components(cfg)
            logger.info("[SUCCESS] Dry run completed successfully.")
            return

        # Create training components using standard project factories
        logger.info("Creating training components...")
        components = _create_training_components(cfg)

        # Initialize trainer using standard project trainer
        logger.info("Initializing trainer...")
        from src.crackseg.training.trainer import Trainer, TrainingComponents

        training_components = TrainingComponents(
            model=components["model"],
            train_loader=components["train_dataloader"],
            val_loader=components["val_dataloader"],
            loss_fn=components["loss_fn"],
            metrics_dict=components["metrics"],
        )

        # Pass the full configuration directly to the Trainer
        # The Trainer will auto-detect the experiment configuration from the YAML
        trainer = Trainer(
            components=training_components,
            cfg=cfg,
        )

        # Start training
        logger.info("Starting SwinV2 hybrid experiment training...")
        trainer.train()

        logger.info(
            "[SUCCESS] SwinV2 hybrid experiment completed successfully!"
        )

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"SwinV2 experiment failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


def _validate_environment() -> None:
    """Validate environment using standard project utilities."""
    import torch

    # Check CUDA availability
    if not torch.cuda.is_available():
        logging.warning("CUDA not available. Training will use CPU")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")

    # Check required packages
    try:
        import timm

        logging.info(f"timm version: {timm.__version__}")
    except ImportError as e:
        raise ImportError("timm package required for SwinV2 models") from e

    try:
        import albumentations as A

        logging.info(f"albumentations version: {A.__version__}")
    except ImportError as e:
        raise ImportError(
            "albumentations package required for data augmentation"
        ) from e


def _validate_config(config: DictConfig) -> None:
    """Validate configuration using standard project patterns."""
    # Check required sections - use the nested configuration structure
    required_sections = ["model", "experiments"]
    for section in required_sections:
        if not hasattr(config, section):
            raise ValueError(f"{section} configuration missing")

    # Check that the specific experiment configuration exists
    if not hasattr(config.experiments, "swinv2_hybrid"):
        raise ValueError("swinv2_hybrid experiment configuration missing")

    # Check that training and data are available in the experiment config
    experiment_config = config.experiments.swinv2_hybrid
    if not hasattr(experiment_config, "training"):
        raise ValueError("training configuration missing in experiment")
    if not hasattr(experiment_config, "data"):
        raise ValueError("data configuration missing in experiment")

    # Validate batch size for memory constraints
    batch_size = experiment_config.training.get("batch_size", 4)
    if batch_size > 8:
        logging.warning(
            f"Large batch size ({batch_size}) may cause memory issues"
        )

    logging.info("Configuration validation passed")


def _validate_components(config: DictConfig) -> None:
    """Validate components using standard project factories."""
    try:
        # Get the experiment configuration
        experiment_config = config.experiments.swinv2_hybrid

        # Test model creation
        from crackseg.model.factory.config import create_model_from_config

        model = create_model_from_config(experiment_config.model)
        logging.info(f"[OK] Model created: {type(model).__name__}")

        # Test data loaders creation
        from crackseg.data.factory import create_dataloaders_from_config

        _ = create_dataloaders_from_config(
            data_config=experiment_config.data,
            transform_config=experiment_config.data.get("transform", {}),
            dataloader_config=experiment_config.data.get("dataloader", {}),
        )
        logging.info("[OK] Data loaders created successfully")

        # Test loss function creation using basic registry
        from crackseg.training.losses.registry.clean_registry import (
            CleanLossRegistry,
        )

        # Create a basic registry instance
        basic_registry = CleanLossRegistry()

        # Register the focal dice loss with basic registry
        def create_focal_dice_loss(**params):
            from crackseg.training.losses.focal_dice_loss import (
                FocalDiceLoss,
                FocalDiceLossConfig,
            )

            config = FocalDiceLossConfig(**params)
            return FocalDiceLoss(config=config)

        basic_registry.register_factory(
            "focal_dice_loss", create_focal_dice_loss
        )

        # Use the specific SwinV2 configuration
        swinv2_config = experiment_config.training.loss.config
        config_dict = dict(swinv2_config)
        config_dict.pop("_target_", None)
        loss_fn = basic_registry.instantiate("focal_dice_loss", **config_dict)
        logging.info(f"[OK] Loss function created: {type(loss_fn).__name__}")

        # Test optimizer creation
        import torch.optim

        optimizer_class = getattr(
            torch.optim,
            experiment_config.training.optimizer._target_.split(".")[-1],
        )
        optimizer_params = dict(experiment_config.training.optimizer)
        optimizer_params.pop("_target_", None)
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        logging.info(f"[OK] Optimizer created: {type(optimizer).__name__}")

        logging.info("[SUCCESS] All components validated successfully")

    except Exception as e:
        logging.error(f"Component validation failed: {e}")
        raise


def _print_experiment_summary(config: DictConfig) -> None:
    """Print experiment summary using standard project patterns."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("SwinV2 Hybrid Architecture Experiment")
    logger.info("=" * 80)

    # Get the experiment configuration
    experiment_config = config.experiments.swinv2_hybrid

    # Model architecture
    logger.info("Model Architecture:")
    model_config = experiment_config.model
    logger.info(f"  - Type: {model_config.get('type', 'Unknown')}")
    if hasattr(model_config, "encoder"):
        logger.info(
            f"  - Encoder: {model_config.encoder.get('_target_', 'Unknown')}"
        )

    # Training configuration
    logger.info("Training Configuration:")
    training_config = experiment_config.training
    logger.info(f"  - Epochs: {training_config.get('epochs', 'Unknown')}")
    logger.info(
        f"  - Batch Size: {training_config.get('batch_size', 'Unknown')}"
    )
    logger.info(
        f"  - Learning Rate: {training_config.get('learning_rate', 'Unknown')}"
    )

    # Data configuration
    logger.info("Data Configuration:")
    data_config = experiment_config.data
    logger.info(f"  - Data Root: {data_config.get('data_root', 'Unknown')}")
    logger.info(f"  - Image Size: {data_config.get('image_size', 'Unknown')}")

    logger.info("=" * 80)


def _create_training_components(config: DictConfig) -> dict:
    """Create training components using standard project factories."""
    # Get the experiment configuration
    experiment_config = config.experiments.swinv2_hybrid

    # Create model
    from crackseg.model.factory.config import create_model_from_config

    model = create_model_from_config(experiment_config.model)

    # Create data loaders
    from crackseg.data.factory import create_dataloaders_from_config

    dataloaders_dict = create_dataloaders_from_config(
        data_config=experiment_config.data,
        transform_config=experiment_config.data.get("transform", {}),
        dataloader_config=experiment_config.data.get("dataloader", {}),
    )

    # Create loss function using basic registry
    from crackseg.training.losses.registry.clean_registry import (
        CleanLossRegistry,
    )

    # Create a basic registry instance
    basic_registry = CleanLossRegistry()

    # Register the focal dice loss with basic registry
    def create_focal_dice_loss(**params):
        from crackseg.training.losses.focal_dice_loss import (
            FocalDiceLoss,
            FocalDiceLossConfig,
        )

        config = FocalDiceLossConfig(**params)
        return FocalDiceLoss(config=config)

    basic_registry.register_factory("focal_dice_loss", create_focal_dice_loss)

    # Use the specific SwinV2 configuration
    swinv2_config = experiment_config.training.loss.config
    config_dict = dict(swinv2_config)
    config_dict.pop("_target_", None)
    loss_fn = basic_registry.instantiate("focal_dice_loss", **config_dict)

    # Create optimizer
    import torch.optim

    optimizer_class = getattr(
        torch.optim,
        experiment_config.training.optimizer._target_.split(".")[-1],
    )
    optimizer_params = dict(experiment_config.training.optimizer)
    optimizer_params.pop("_target_", None)
    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    # Create scheduler
    from crackseg.training.factory import create_lr_scheduler

    scheduler = create_lr_scheduler(
        optimizer, experiment_config.training.scheduler
    )

    # Create metrics using the correct function
    from crackseg.utils.factory.factory import get_metrics_from_cfg

    metrics = get_metrics_from_cfg(experiment_config.evaluation.metrics)

    return {
        "model": model,
        "train_dataloader": dataloaders_dict["train"]["dataloader"],
        "val_dataloader": dataloaders_dict["val"]["dataloader"],
        "loss_fn": loss_fn,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "metrics": metrics,
    }


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

        config_dir = Path(__file__).parent.parent.parent.parent / "configs"
        with initialize_config_dir(
            config_dir=str(config_dir), version_base=None
        ):
            cfg = compose(
                config_name="experiments/swinv2_hybrid/swinv2_hybrid_experiment"
            )
            main(cfg)
