#!/usr/bin/env python3
"""
SwinV2 Hybrid Architecture Experiment Runner

This script runs the SwinV2 + ASPP + CNN hybrid architecture experiment
with Focal Dice Loss for crack segmentation.

Features:
- Hybrid architecture: SwinV2 Transformer + ASPP + CNN Decoder
- Optimized loss: Focal Dice Loss for class imbalance
- Hardware optimization: RTX 3070 Ti (8GB VRAM) settings
- Comprehensive monitoring and logging
- Reproducible training with fixed seeds

Usage:
    python scripts/experiments/run_swinv2_hybrid_experiment.py \
        [--config-override key=value]
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.crackseg.training.trainer import (  # noqa: E402
    Trainer,
    TrainingComponents,
)


def setup_logging() -> None:
    """Setup logging for the experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("swinv2_hybrid_experiment.log"),
        ],
    )


def validate_environment() -> None:
    """Validate that the environment is properly configured."""
    logger = logging.getLogger(__name__)

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA not available. Training will use CPU "
            "(not recommended for large models)"
        )
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")

        if gpu_memory < 7.0:
            logger.warning(
                f"GPU memory ({gpu_memory:.1f}GB) may be insufficient "
                f"for this model"
            )

    # Check required packages
    try:
        import timm

        logger.info(f"timm version: {timm.__version__}")
    except ImportError:
        logger.error("timm package not found. Install with: pip install timm")
        sys.exit(1)

    try:
        import albumentations as A

        logger.info(f"albumentations version: {A.__version__}")
    except ImportError:
        logger.error(
            "albumentations package not found. "
            "Install with: pip install albumentations"
        )
        sys.exit(1)


def load_experiment_config(
    config_overrides: list[str] | None = None,
) -> DictConfig:
    """Load and validate the experiment configuration."""
    logger = logging.getLogger(__name__)

    # Load base configuration
    config_path = (
        project_root
        / "configs"
        / "experiments"
        / "swinv2_hybrid"
        / "swinv2_hybrid_experiment.yaml"
    )

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # Load configuration
    config = OmegaConf.load(config_path)

    # Apply overrides if provided
    if config_overrides:
        for override in config_overrides:
            if "=" in override:
                key, value = override.split("=", 1)
                # Try to convert value to appropriate type
                try:
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace(".", "").replace("-", "").isdigit():
                        value = float(value)
                except ValueError:
                    pass  # Keep as string

                OmegaConf.update(config, key, value, merge=True)
                logger.info(f"Applied override: {key} = {value}")

    # Ensure config is DictConfig type
    if not isinstance(config, DictConfig):
        config = OmegaConf.create(config)

    # Validate critical configuration
    validate_config(config)  # type: ignore[arg-type]

    return config  # type: ignore[return-value]


def validate_config(config: DictConfig) -> None:
    """Validate the experiment configuration."""
    logger = logging.getLogger(__name__)

    # Check model configuration
    if not hasattr(config, "model"):
        logger.error("Model configuration missing")
        sys.exit(1)

    # Check training configuration
    if not hasattr(config, "training"):
        logger.error("Training configuration missing")
        sys.exit(1)

    # Check data configuration
    if not hasattr(config, "data"):
        logger.error("Data configuration missing")
        sys.exit(1)

    # Validate batch size for memory constraints
    batch_size = config.training.get("batch_size", 4)
    if batch_size > 8:
        logger.warning(
            f"Large batch size ({batch_size}) may cause memory issues "
            f"on RTX 3070 Ti"
        )

    logger.info("Configuration validation passed")


def validate_training_components(config: DictConfig) -> None:
    """
    Validate that all training components can be created and work correctly.
    """
    logger = logging.getLogger(__name__)

    logger.info("Validating training components creation...")

    try:
        # Test model creation
        logger.info("Testing model creation...")
        from crackseg.model.factory.config import create_model_from_config

        model = create_model_from_config(config.model)
        logger.info(f"✓ Model created successfully: {type(model).__name__}")

        # Test data loaders creation
        logger.info("Testing data loaders creation...")
        from crackseg.data.factory import create_dataloaders_from_config

        data_config = config.data
        transform_config = data_config.get("transform", {})
        dataloader_config = data_config.get("dataloader", {})

        dataloaders_dict = create_dataloaders_from_config(
            data_config=data_config,
            transform_config=transform_config,
            dataloader_config=dataloader_config,
        )

        train_loader = dataloaders_dict["train"]["dataloader"]
        val_loader = dataloaders_dict["val"]["dataloader"]

        # Ensure they are DataLoader instances
        from torch.utils.data import DataLoader

        if not isinstance(train_loader, DataLoader):
            raise TypeError(f"Expected DataLoader, got {type(train_loader)}")
        if not isinstance(val_loader, DataLoader):
            raise TypeError(f"Expected DataLoader, got {type(val_loader)}")

        logger.info(f"✓ Train dataloader created: {len(train_loader)} batches")
        logger.info(f"✓ Val dataloader created: {len(val_loader)} batches")

        # Test batch processing
        logger.info("Testing batch processing...")
        train_iter = iter(train_loader)
        batch = next(train_iter)

        # Verify batch structure
        if isinstance(batch, dict):
            if "image" not in batch or "mask" not in batch:
                raise ValueError("Batch dict missing 'image' or 'mask' keys")
            logger.info(f"✓ Batch structure correct: {list(batch.keys())}")
            logger.info(f"✓ Image shape: {batch['image'].shape}")
            logger.info(f"✓ Mask shape: {batch['mask'].shape}")
        else:
            raise TypeError(f"Expected dict batch, got {type(batch)}")

        # Test loss function creation
        logger.info("Testing loss function creation...")
        import torch.nn as nn

        from crackseg.utils.factory import get_loss_fn

        loss_fn = get_loss_fn(config.training.loss)

        # Ensure loss_fn is a Module
        if not isinstance(loss_fn, nn.Module):

            class LossWrapper(nn.Module):
                def __init__(self, fn):
                    super().__init__()
                    self.fn = fn

                def forward(self, *args, **kwargs):
                    return self.fn(*args, **kwargs)

            loss_fn = LossWrapper(loss_fn)

        logger.info(f"✓ Loss function created: {type(loss_fn).__name__}")

        # Test loss computation
        logger.info("Testing loss computation...")
        with torch.no_grad():
            # Create dummy predictions and targets
            pred = torch.randn(
                batch["image"].shape[0], 1, *batch["image"].shape[2:]
            )
            target = batch["mask"].float()

            loss_value = loss_fn(pred, target)
            logger.info(
                f"✓ Loss computation successful: {loss_value.item():.4f}"
            )

        # Test metrics creation
        logger.info("Testing metrics creation...")
        from crackseg.utils.factory import get_metrics_from_cfg

        metrics_dict = get_metrics_from_cfg(config.evaluation.metrics)
        logger.info(f"✓ Metrics created: {len(metrics_dict)} metrics")

        # Test optimizer creation
        logger.info("Testing optimizer creation...")
        from typing import Any, cast

        import torch.optim as optim
        from omegaconf import OmegaConf

        optimizer_config = config.training.optimizer
        # Resolve interpolations
        optimizer_params = OmegaConf.to_container(
            optimizer_config, resolve=True
        )
        if isinstance(optimizer_params, dict):
            # Remove _target_ from params
            optimizer_params.pop("_target_", None)
            optimizer_params_typed = cast(dict[str, Any], optimizer_params)
            optimizer_instance = optim.AdamW(
                model.parameters(), **optimizer_params_typed
            )
            logger.info(
                f"✓ Optimizer created: {type(optimizer_instance).__name__}"
            )
        else:
            raise ValueError("Failed to resolve optimizer configuration")

        # Test scheduler creation
        logger.info("Testing scheduler creation...")
        scheduler_config = config.training.scheduler
        # Resolve interpolations
        scheduler_params = OmegaConf.to_container(
            scheduler_config, resolve=True
        )
        if isinstance(scheduler_params, dict):
            # Remove _target_ from params
            scheduler_params.pop("_target_", None)
            scheduler_params_typed = cast(dict[str, Any], scheduler_params)
            scheduler_instance = (
                optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer_instance, **scheduler_params_typed
                )
            )
            logger.info(
                f"✓ Scheduler created: {type(scheduler_instance).__name__}"
            )
        else:
            raise ValueError("Failed to resolve scheduler configuration")

        logger.info("✓ All training components validated successfully!")

    except Exception as e:
        logger.error(f"✗ Training components validation failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


def create_training_components(config: DictConfig) -> TrainingComponents:
    """Create training components from configuration."""
    logger = logging.getLogger(__name__)

    logger.info("Creating training components...")

    # Create model from configuration
    from crackseg.model.factory.config import create_model_from_config

    model = create_model_from_config(config.model)

    # Create data loaders from configuration
    from crackseg.data.factory import create_dataloaders_from_config

    # Extract data configuration
    data_config = config.data
    transform_config = data_config.get("transform", {})
    dataloader_config = data_config.get("dataloader", {})

    # Create dataloaders
    dataloaders_dict = create_dataloaders_from_config(
        data_config=data_config,
        transform_config=transform_config,
        dataloader_config=dataloader_config,
    )

    train_loader = dataloaders_dict["train"]["dataloader"]
    val_loader = dataloaders_dict["val"]["dataloader"]

    # Ensure dataloaders are DataLoader instances
    from torch.utils.data import DataLoader

    if not isinstance(train_loader, DataLoader):
        raise TypeError(f"Expected DataLoader, got {type(train_loader)}")
    if not isinstance(val_loader, DataLoader):
        raise TypeError(f"Expected DataLoader, got {type(val_loader)}")

    # Create loss function from configuration
    import torch.nn as nn

    from crackseg.utils.factory import get_loss_fn

    loss_fn = get_loss_fn(config.training.loss)

    # Ensure loss_fn is a Module
    if not isinstance(loss_fn, nn.Module):
        # Wrap function in Module if needed
        class LossWrapper(nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, *args, **kwargs):
                return self.fn(*args, **kwargs)

        loss_fn = LossWrapper(loss_fn)

    # Create metrics dictionary
    from crackseg.utils.factory import get_metrics_from_cfg

    metrics_dict = get_metrics_from_cfg(config.evaluation.metrics)

    logger.info("Training components created successfully")

    return TrainingComponents(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metrics_dict=metrics_dict,
    )


def print_experiment_summary(config: DictConfig) -> None:
    """Print a summary of the experiment configuration."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("SWINV2 HYBRID ARCHITECTURE EXPERIMENT")
    logger.info("=" * 80)

    # Model architecture
    logger.info("Model Architecture:")
    logger.info("  - Type: SwinV2CnnAsppUNet")
    logger.info("  - Encoder: SwinV2 Tiny (swinv2_tiny_window16_256)")
    logger.info("  - Bottleneck: ASPP with rates [1, 6, 12, 18]")
    logger.info("  - Decoder: CNN with CBAM attention")
    logger.info(f"  - Classes: {config.model.num_classes}")

    # Loss function
    logger.info("Loss Function:")
    logger.info("  - Type: FocalDiceLoss")
    logger.info(
        f"  - Focal weight: {config.training.loss.config.focal_weight}"
    )
    logger.info(f"  - Dice weight: {config.training.loss.config.dice_weight}")
    logger.info(f"  - Focal alpha: {config.training.loss.config.focal_alpha}")
    logger.info(f"  - Focal gamma: {config.training.loss.config.focal_gamma}")

    # Training settings
    logger.info("Training Settings:")
    logger.info(f"  - Learning rate: {config.training.learning_rate}")
    logger.info(f"  - Batch size: {config.training.batch_size}")
    logger.info(
        f"  - Gradient accumulation: "
        f"{config.training.gradient_accumulation_steps}"
    )
    effective_batch_size = (
        config.training.batch_size
        * config.training.gradient_accumulation_steps
    )
    logger.info(f"  - Effective batch size: {effective_batch_size}")
    logger.info(f"  - Epochs: {config.training.epochs}")
    logger.info(f"  - Mixed precision: {config.training.use_amp}")

    # Hardware
    logger.info("Hardware Optimization:")
    if hasattr(config, "hardware"):
        logger.info(f"  - Device: {config.hardware.get('device', 'cuda')}")
        logger.info(
            "  - Memory efficient: "
            f"{config.hardware.get('memory_efficient', True)}"
        )
        logger.info(
            "  - Mixed precision: "
            f"{config.hardware.get('mixed_precision', True)}"
        )
    else:
        logger.warning("  - Hardware configuration section not found")

    # Reproducibility
    logger.info("Reproducibility:")
    if hasattr(config, "random_seed"):
        logger.info(f"  - Random seed: {config.random_seed}")
    else:
        logger.warning("  - Random seed not found in configuration")

    if hasattr(config, "deterministic_operations"):
        logger.info(f"  - Deterministic: {config.deterministic_operations}")
    else:
        logger.warning("  - Deterministic operations setting not found")

    logger.info("=" * 80)


def main() -> None:
    """Main experiment execution function."""
    parser = argparse.ArgumentParser(
        description="Run SwinV2 Hybrid Architecture Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config-override",
        action="append",
        help="Override configuration values (format: key=value)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without starting training",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Validate environment
        logger.info("Validating environment...")
        validate_environment()

        # Load configuration
        logger.info("Loading experiment configuration...")
        config = load_experiment_config(args.config_override)

        # Print experiment summary
        print_experiment_summary(config)

        if args.dry_run:
            logger.info("Running comprehensive dry-run validation...")
            validate_training_components(config)
            logger.info(
                "Dry run completed successfully. All components validated."
            )
            return

        # Create training components
        logger.info("Creating training components...")
        components = create_training_components(config)

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(components=components, cfg=config)

        # Start training
        logger.info("Starting training...")
        trainer.train()

        logger.info("Experiment completed successfully!")

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
