#!/usr/bin/env python3
"""
Example script demonstrating generalized experiment output organization.

This script shows how any experiment automatically gets its outputs
organized in a timestamped folder with the format "timestamp-experiment_name".

The system now automatically:
1. Creates experiment-specific directories
2. Saves metrics in experiment-specific folders
3. Saves configurations in experiment-specific folders
4. Saves checkpoints in experiment-specific folders
5. Avoids duplication in global folders

No manual configuration is required in individual experiment scripts.
"""

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from src.crackseg.training.trainer import Trainer, TrainingComponents
from src.crackseg.utils.logging.base import get_logger

logger = get_logger(__name__)


def _validate_config(cfg: DictConfig) -> None:
    """Validate that the configuration has required sections."""
    required_sections = ["training", "data", "model", "evaluation"]
    missing_sections = [
        section for section in required_sections if section not in cfg
    ]

    if missing_sections:
        raise ValueError(
            f"Missing required configuration sections: {missing_sections}"
        )

    logger.info("Configuration validation passed")


def _print_experiment_summary(cfg: DictConfig) -> None:
    """Print a summary of the experiment configuration."""
    logger.info("=" * 80)
    logger.info("Generalized Experiment Output Organization Demo")
    logger.info("=" * 80)

    # Extract experiment information
    experiment_name = cfg.get("experiment", {}).get(
        "name", "unknown_experiment"
    )
    logger.info(f"Experiment Name: {experiment_name}")

    # Model information
    model_type = cfg.get("model", {}).get("type", "unknown")
    logger.info(f"Model Type: {model_type}")

    # Training information
    training_cfg = cfg.get("training", {})
    epochs = training_cfg.get("epochs", "unknown")
    batch_size = training_cfg.get("batch_size", "unknown")
    learning_rate = training_cfg.get("optimizer", {}).get("lr", "unknown")

    logger.info("Training Configuration:")
    logger.info(f"  - Epochs: {epochs}")
    logger.info(f"  - Batch Size: {batch_size}")
    logger.info(f"  - Learning Rate: {learning_rate}")

    # Data information
    data_cfg = cfg.get("data", {})
    data_root = data_cfg.get("root", "unknown")
    image_size = data_cfg.get("image_size", "unknown")

    logger.info("Data Configuration:")
    logger.info(f"  - Data Root: {data_root}")
    logger.info(f"  - Image Size: {image_size}")

    logger.info("=" * 80)


def _create_training_components(cfg: DictConfig) -> dict[str, Any]:
    """Create training components from configuration."""
    logger.info("Creating training components...")

    # Import necessary modules
    from src.crackseg.data.factory import create_dataloaders
    from src.crackseg.evaluation.metrics.factory import create_metrics
    from src.crackseg.model.factory import create_model
    from src.crackseg.training.losses.factory import create_loss

    # Create model
    model = create_model(cfg.model)
    logger.info(f"Model created: {type(model).__name__}")

    # Create data loaders
    train_loader, val_loader = create_dataloaders(cfg.data)
    logger.info(
        f"Data loaders created - Train: {len(train_loader)}, Val: {len(val_loader)}"
    )

    # Create loss function
    loss_fn = create_loss(cfg.training.loss)
    logger.info(f"Loss function created: {type(loss_fn).__name__}")

    # Create metrics
    metrics = create_metrics(cfg.evaluation.metrics)
    logger.info(f"Metrics created: {len(metrics)} metrics")

    return {
        "model": model,
        "train_dataloader": train_loader,
        "val_dataloader": val_loader,
        "loss_fn": loss_fn,
        "metrics": metrics,
    }


def _validate_components(components: dict[str, Any]) -> None:
    """Validate that all components were created successfully."""
    required_components = [
        "model",
        "train_dataloader",
        "val_dataloader",
        "loss_fn",
        "metrics",
    ]
    missing_components = [
        comp for comp in required_components if comp not in components
    ]

    if missing_components:
        raise ValueError(f"Missing required components: {missing_components}")

    logger.info("Component validation passed")


def run_experiment(cfg: DictConfig) -> None:
    """Run the experiment with automatic output organization."""
    try:
        # Validate configuration
        _validate_config(cfg)

        # Print experiment summary
        _print_experiment_summary(cfg)

        # Create training components
        components = _create_training_components(cfg)

        # Validate components
        _validate_components(components)

        # Initialize trainer
        logger.info("Initializing trainer...")
        training_components = TrainingComponents(
            model=components["model"],
            train_loader=components["train_dataloader"],
            val_loader=components["val_dataloader"],
            loss_fn=components["loss_fn"],
            metrics_dict=components["metrics"],
        )

        # Create configuration with experiment information
        # The Trainer will automatically detect this and organize outputs
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = cfg.get("experiment", {}).get(
            "name", "example_experiment"
        )
        experiment_output_dir = (
            f"artifacts/experiments/{timestamp}-{experiment_name}"
        )

        # Create the experiment directory structure
        experiment_path = Path(experiment_output_dir)
        experiment_path.mkdir(parents=True, exist_ok=True)

        # Create modified configuration with experiment info at top level
        modified_cfg = OmegaConf.create(
            {
                "checkpoint_dir": experiment_output_dir,
                "experiment": {
                    "name": experiment_name,
                    "output_dir": experiment_output_dir,
                },
                "training": cfg.training,
                "data": cfg.data,
                "model": cfg.model,
                "evaluation": cfg.evaluation,
            }
        )

        # Initialize trainer - it will automatically organize outputs
        trainer = Trainer(
            components=training_components,
            cfg=modified_cfg,
        )

        # Start training
        logger.info("Starting experiment training...")
        final_metrics = trainer.train()

        logger.info("=" * 80)
        logger.info("[SUCCESS] Experiment completed successfully!")
        logger.info(f"Final metrics: {final_metrics}")
        logger.info(f"All outputs saved to: {experiment_output_dir}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="experiments/swinv2_hybrid/swinv2_hybrid_experiment",
)
def main(cfg: DictConfig) -> None:
    """Main function to run the experiment."""
    run_experiment(cfg)


if __name__ == "__main__":
    main()
