#!/usr/bin/env python3
"""
End-to-end test script to verify the full training pipeline.

This script performs a complete pipeline test:
1. Loads a small synthetic dataset
2. Configures a reduced UNet model
3. Trains for a few epochs
4. Saves checkpoints at intervals
5. Loads the model from checkpoint
6. Evaluates on the test set
7. Generates a results report
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Import local modules
from .modules.checkpointing import (
    _finalize_and_save_results,
    _load_best_checkpoint,
    _save_checkpoint_with_config,
)
from .modules.config import create_mini_config
from .modules.dataclasses import (
    EvaluationArgs,
    FinalResultsData,
    TrainingRunArgs,
)
from .modules.evaluation import _evaluate_model_on_test_set

# Import setup module first
from .modules.setup import (
    ExperimentLogger,
    create_dataloaders_from_config,
    get_logger,
    set_random_seeds,
)
from .modules.training import (
    _initialize_training_components,
    _run_train_epoch,
    _run_val_epoch,
)
from .modules.utils import (
    create_experiment_dir,
    save_config,
    visualize_results,
)

# Logging configuration
logger = get_logger("TestPipelineE2E") if get_logger else None

NO_CHANNEL_DIM = 3


def _setup_experiment_resources(
    base_cfg_callable: Callable[[], Any],
) -> tuple[Any, Path, torch.device, ExperimentLogger, Path, Path, Path]:
    """
    Handles creation of config, experiment dirs, logger, and seed/device setup.
    """
    cfg = OmegaConf.create(base_cfg_callable())
    # Forzamos el tipo a DictConfig para ExperimentLogger
    # (OmegaConf.create puede retornar ListConfig,
    # pero aquí debe ser DictConfig)
    cfg = cast(DictConfig, cfg)

    # Define project root for this function
    project_root = str(Path(__file__).resolve().parents[2])

    # Use the utility function from the separate module
    exp_dir_str = create_experiment_dir(project_root)
    exp_dir = Path(exp_dir_str)

    config_path = exp_dir / "config.yaml"
    save_config(cfg, str(config_path))  # save_config expects string path

    checkpoints_dir = exp_dir / "checkpoints"
    metrics_dir = exp_dir / "metrics"
    vis_dir = exp_dir / "visualizations"

    experiment_logger = ExperimentLogger(
        log_dir=str(exp_dir),  # ExperimentLogger might expect string
        experiment_name="e2e_test",
        config=cfg,
        log_level="INFO",
        log_to_file=True,
    )
    set_random_seeds(cfg.random_seed)
    # OmegaConf get workaround for require_cuda
    require_cuda = cfg.require_cuda if hasattr(cfg, "require_cuda") else False
    device = torch.device(
        "cuda" if torch.cuda.is_available() and require_cuda else "cpu"
    )
    if logger:
        logger.info(f"Using device: {device}")
    return (
        cfg,
        exp_dir,
        device,
        experiment_logger,
        checkpoints_dir,
        metrics_dir,
        vis_dir,
    )


def _prepare_dataloaders(
    cfg_data: Any,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Loads and returns train, val, and test dataloaders."""
    if logger:
        logger.info("Loading data...")
    dataloaders = create_dataloaders_from_config(
        data_config=cfg_data,
        transform_config=cfg_data.get("transforms", {}),
        # Pass the base data_config for dataloader params
        dataloader_config=cfg_data,
    )
    train_loader = dataloaders["train"]["dataloader"]
    val_loader = dataloaders["val"]["dataloader"]
    test_loader = dataloaders["test"]["dataloader"]

    def get_dataset_len(loader: Any) -> int:
        if isinstance(loader, DataLoader):
            # mypy: asegúrate de que loader.dataset es Sized
            from collections.abc import Sized

            dataset = loader.dataset
            if isinstance(dataset, Sized):
                return len(dataset)
            return -1
        elif hasattr(loader, "__len__"):
            return len(loader)
        return -1

    if logger:
        logger.info(
            f"Data loaded: {get_dataset_len(train_loader)} train, "
            f"{get_dataset_len(val_loader)} val, "
            f"{get_dataset_len(test_loader)} test"
        )
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    return train_loader, val_loader, test_loader


def _execute_training_and_validation(
    args: TrainingRunArgs,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
) -> tuple[float, float, dict[str, float], float, dict[str, float]]:
    """Runs the main training and validation loop for all epochs."""
    if logger:
        logger.info("Starting training loop...")
    best_metric_value = 0.0
    # Initialize return values for cases where training loop might not run
    # (e.g. 0 epochs)
    # These should be updated with actual final epoch values inside the loop.
    final_train_loss = float("nan")
    final_train_metrics = {k: float("nan") for k in args.metrics_dict}
    final_val_loss = float("nan")
    final_val_metrics = {k: float("nan") for k in args.metrics_dict}

    for epoch in range(args.cfg_training.epochs):
        if logger:
            logger.info(f"Epoch {epoch + 1}/{args.cfg_training.epochs}")

        # Training phase for one epoch
        final_train_loss, final_train_metrics = _run_train_epoch(
            args, train_loader
        )
        if logger:
            logger.info(
                f"Train Loss: {final_train_loss:.4f}, Metrics: "
                f"{final_train_metrics}"
            )
        args.experiment_logger.log_scalar(
            "train/loss", final_train_loss, epoch
        )
        for k, v in final_train_metrics.items():
            args.experiment_logger.log_scalar(f"train/{k}", v, epoch)

        # Validation phase for one epoch
        final_val_loss, final_val_metrics = _run_val_epoch(args, val_loader)
        if logger:
            logger.info(
                f"Validation Loss: {final_val_loss:.4f}, Metrics: "
                f"{final_val_metrics}"
            )
        args.experiment_logger.log_scalar("val/loss", final_val_loss, epoch)
        for k, v in final_val_metrics.items():
            args.experiment_logger.log_scalar(f"val/{k}", v, epoch)

        # Learning rate scheduler step
        if args.lr_scheduler:
            if isinstance(
                args.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                args.lr_scheduler.step(final_val_loss)
            else:
                args.lr_scheduler.step()

        current_monitor_metric = final_val_metrics.get(
            args.cfg_training.checkpoints.save_best.monitor_metric,
            (
                float("-inf")
                if args.cfg_training.checkpoints.save_best.monitor_mode
                == "max"
                else float("inf")
            ),
        )
        is_best = (
            args.cfg_training.checkpoints.save_best.monitor_mode == "max"
            and current_monitor_metric > best_metric_value
        ) or (
            args.cfg_training.checkpoints.save_best.monitor_mode == "min"
            and current_monitor_metric < best_metric_value
        )

        if is_best:
            best_metric_value = current_monitor_metric
            if logger:
                metric_name = (
                    args.cfg_training.checkpoints.save_best.monitor_metric
                )
                logger.info(f"New best {metric_name}: {best_metric_value:.4f}")
            # Guardar checkpoint usando la función del módulo separado
            _save_checkpoint_with_config(
                args=args,
                epoch=epoch,
                train_loss=final_train_loss,
                val_loss=final_val_loss,
                train_metrics=final_train_metrics,
                val_metrics=final_val_metrics,
                is_best=True,
            )

        # Save regular checkpoint if interval is reached
        if (
            args.cfg_training.checkpoints.save_interval_epochs > 0
            and (epoch + 1)
            % args.cfg_training.checkpoints.save_interval_epochs
            == 0
        ):
            _save_checkpoint_with_config(
                args=args,
                epoch=epoch,
                train_loss=final_train_loss,
                val_loss=final_val_loss,
                train_metrics=final_train_metrics,
                val_metrics=final_val_metrics,
                is_best=False,
            )

    if logger:
        logger.info("Training finished.")
    return (
        best_metric_value,
        final_train_loss,
        final_train_metrics,
        final_val_loss,
        final_val_metrics,
    )


def run_e2e_test() -> tuple[Any, Any]:
    """Run end-to-end test of the full pipeline."""
    exp_dir_final, results_final = None, None  # Initialize for finally block
    experiment_logger_instance = None

    try:
        # Setup experiment resources
        (
            cfg,
            exp_dir,
            device,
            experiment_logger_instance,
            checkpoints_dir,
            metrics_dir,
            vis_dir,
        ) = _setup_experiment_resources(create_mini_config)

        # Prepare dataloaders
        train_loader, val_loader, test_loader = _prepare_dataloaders(cfg.data)

        # Initialize training components
        model, loss_fn, optimizer, lr_scheduler, metrics_dict, scaler = (
            _initialize_training_components(cfg, device)
        )

        # Create training arguments
        training_args = TrainingRunArgs(
            cfg_training=cfg.training,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics_dict=metrics_dict,
            scaler=scaler,
            device=device,
            experiment_logger=experiment_logger_instance,
            checkpoints_dir=checkpoints_dir,
        )

        # Execute training and validation
        (
            best_metric_val,
            final_train_loss,
            final_train_metrics,
            final_val_loss,
            final_val_metrics,
        ) = _execute_training_and_validation(
            training_args, train_loader, val_loader
        )

        # Load best checkpoint for evaluation
        loaded_checkpoint_path = _load_best_checkpoint(training_args)

        # Create evaluation arguments
        evaluation_args = EvaluationArgs(
            model=model,
            loss_fn=loss_fn,
            metrics_dict=metrics_dict,
            device=device,
            experiment_logger=experiment_logger_instance,
            vis_dir=vis_dir,
            checkpoints_dir=checkpoints_dir,
        )

        # Evaluate on test set
        test_loss, test_metrics, sample_images, sample_masks, sample_preds = (
            _evaluate_model_on_test_set(evaluation_args, test_loader)
        )

        # Generate visualizations
        if sample_images:
            visualize_results(
                sample_images[:4],
                sample_masks[:4],
                sample_preds[:4],
                str(vis_dir / "test_predictions.png"),
            )

        # Create final results data
        final_results_data = FinalResultsData(
            exp_dir=exp_dir,
            metrics_dir=metrics_dir,
            final_train_loss=final_train_loss,
            final_train_metrics=final_train_metrics,
            final_val_loss=final_val_loss,
            final_val_metrics=final_val_metrics,
            test_loss=test_loss,
            test_metrics=test_metrics,
            epochs=cfg.training.epochs,
            best_metric_val=best_metric_val,
            loaded_checkpoint_path=loaded_checkpoint_path,
        )

        # Finalize and save results
        results_final = _finalize_and_save_results(final_results_data)
        exp_dir_final = exp_dir

        if logger:
            logger.info("End-to-end test completed successfully.")

    except Exception as e:
        if logger:
            logger.error(f"Error during e2e test: {e}")
        raise

    finally:
        # Cleanup
        if experiment_logger_instance:
            experiment_logger_instance.close()
        if logger:
            logger.info("Resources released.")

    return exp_dir_final, results_final


if __name__ == "__main__":
    run_e2e_test()
