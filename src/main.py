# In src/main.py (Skeleton and checkpointing logic)

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import os
import math  # For inf
from typing import Tuple

# Project imports
from src.utils import (
    ExperimentLogger,
    ConfigError,
    DataError,
    ModelError,
    TrainingError,
    ResourceError,
    set_random_seeds,
    get_device,
    save_checkpoint,
    load_checkpoint
)
from src.data.factory import create_dataloaders_from_config  # Import factory
from src.utils.factory import get_metrics_from_cfg
# from src.models.model_factory import create_model
# from src.training.optimizers import create_optimizer
# from src.training.schedulers import create_scheduler

# Configure standard logger
log = logging.getLogger(__name__)


def initialize_experiment(cfg: DictConfig) -> Tuple[str, ExperimentLogger]:
    """Initialize experiment directory and logging.

    Args:
        cfg: The configuration object

    Returns:
        Tuple containing:
        - experiment_dir: Path to the experiment directory
        - logger: Initialized logger instance

    Raises:
        ConfigError: If there are issues with the configuration
    """
    try:
        # Get experiment directory from Hydra's output dir
        experiment_dir = os.getcwd()  # Hydra changes working dir to output dir
        log.info(f"Experiment directory: {experiment_dir}")

        # Create additional output directories
        output_base = os.path.join(experiment_dir, "outputs")
        os.makedirs(os.path.join(output_base, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_base, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(output_base, "visualizations"), exist_ok=True)

        # Initialize experiment logger
        logger = ExperimentLogger(
            log_dir=experiment_dir,
            experiment_name=os.path.basename(experiment_dir),
            config=cfg,
            log_level=cfg.get("log_level", "INFO"),
            log_to_file=cfg.get("log_to_file", True)
        )

        # Log full configuration
        logger.log_config(OmegaConf.to_container(cfg))

        return experiment_dir, logger

    except Exception as e:
        raise ConfigError(
            f"Failed to initialize experiment: {str(e)}"
        ) from e


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training and evaluation entry point."""
    experiment_logger = None
    try:
        # --- 1. Initial Setup ---
        log.info("Starting main execution...")
        set_random_seeds(cfg.get('random_seed', 42))

        # Check CUDA availability
        if not torch.cuda.is_available() and cfg.get('require_cuda', True):
            raise ResourceError(
                "CUDA is required but not available on this system."
            )

        device = get_device()
        log.info(f"Using device: {device}")

        # Initialize experiment and logging
        experiment_dir, experiment_logger = initialize_experiment(cfg)
        log.info(f"Experiment initialized in: {experiment_dir}")

        # --- 2. Data Loading ---
        log.info("Loading data...")
        try:
            # Access configs safely using the correct structure
            data_cfg = cfg.data  # Direct access since it is in defaults
            transform_cfg = data_cfg.get("transforms", OmegaConf.create({}))
            dataloader_cfg = data_cfg  # Dataloader parameters are in data_cfg

            dataloaders_dict = create_dataloaders_from_config(
                data_config=data_cfg,
                transform_config=transform_cfg,
                dataloader_config=dataloader_cfg
            )
            train_loader = dataloaders_dict.get('train', {}).get('dataloader')
            val_loader = dataloaders_dict.get('val', {}).get('dataloader')
            test_loader = dataloaders_dict.get('test', {}).get('dataloader')

            if not train_loader or not val_loader:
                raise DataError(
                    "Train or validation dataloader could not be created."
                )
            log.info("Data loading complete.")

        except Exception as e:
            raise DataError(f"Error during data loading: {str(e)}") from e

        # --- 3. Model Creation --- Placeholder
        log.info("Creating model... (Placeholder)")
        try:
            model = torch.nn.Module()  # Simple placeholder
            model.to(device)
            log.info("Model creation complete.")
        except Exception as e:
            raise ModelError(f"Error creating model: {str(e)}") from e

        # --- 4. Training Setup --- Placeholder
        log.info("Setting up training components... (Partially Placeholder)")

        # Ensure evaluation metrics config exists
        metrics = {}
        if hasattr(cfg, 'evaluation') and hasattr(cfg.evaluation, 'metrics'):
            try:
                metrics = get_metrics_from_cfg(cfg.evaluation.metrics)
                log.info(f"Loaded metrics: {list(metrics.keys())}")
            except Exception as e:
                log.error(f"Error loading metrics: {e}")
                metrics = {}
        else:
            log.warning("Evaluation metrics configuration not found.")

        # Placeholder optimizer - Use config values
        optimizer_cfg = cfg.training.get(
            "optimizer", {"type": "adam", "lr": 1e-3})
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_cfg.get("lr", 1e-3)
        )

        # Placeholder scheduler - Use config values
        scheduler = None
        if cfg.training.get("scheduler", None):
            # TODO: Implement scheduler factory
            pass

        # Setup AMP if enabled
        use_amp = cfg.training.get("amp_enabled", False)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        log.info(f"AMP Enabled: {scaler is not None}")
        log.info("Training setup complete.")

        # --- 5. Checkpointing and Resume ---
        start_epoch = 0
        best_metric_value = None
        # Use .get to safely access nested config for checkpoints
        checkpoint_cfg = cfg.training.get("checkpoints", OmegaConf.create({}))
        default_ckpt_dir = "outputs/checkpoints/"
        checkpoint_dir = checkpoint_cfg.get("checkpoint_dir", default_ckpt_dir)
        resume_path = checkpoint_cfg.get("resume_from_checkpoint", None)

        if resume_path:
            msg = f"Attempting to resume from checkpoint: {resume_path}"
            log.info(msg)
            # Ensure path is interpretable (may be relative to original CWD)
            orig_cwd = hydra.utils.get_original_cwd()
            if not os.path.isabs(resume_path):
                resume_path = os.path.join(orig_cwd, resume_path)

            if os.path.exists(resume_path):
                checkpoint_data = load_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    checkpoint_path=resume_path,
                    device=device
                )
                start_epoch = checkpoint_data.get("epoch", 0) + 1
                best_metric_value = checkpoint_data.get("best_metric_value",
                                                        None)
                msg = (f"Resumed from epoch {start_epoch}. "
                       f"Best: {best_metric_value}")
                log.info(msg)
            else:
                msg = f"Resume checkpoint not found: {resume_path}. Fresh \
start."
                log.warning(msg)
        else:
            log.info(
                "No checkpoint specified for resume, starting from scratch.")

        # --- Setup Best Model Tracking ---
        save_best_cfg = checkpoint_cfg.get("save_best", OmegaConf.create({}))
        monitor_metric = save_best_cfg.get("monitor_metric", None)
        monitor_mode = save_best_cfg.get("monitor_mode", "max")
        save_best_enabled = save_best_cfg.get("enabled", False)
        best_filename = save_best_cfg.get("best_filename", "model_best.pth.tar"
                                          )
        fname_pattern = checkpoint_cfg.get(
            "filename",
            "checkpoint_epoch_{epoch:03d}.pth.tar"
        )
        save_interval = checkpoint_cfg.get("save_interval_epochs", 0)
        save_last = checkpoint_cfg.get("save_last", True)

        if save_best_enabled and not monitor_metric:
            msg = ("save_best enabled but monitor_metric not set. "
                   "Disabling best model saving.")
            log.warning(msg)
            save_best_enabled = False
        elif save_best_enabled:
            msg = (f"Monitoring '{monitor_metric}' for best model "
                   f"(mode: {monitor_mode}).")
            log.info(msg)
            if best_metric_value is None:  # Initialize if not resuming
                best_metric_value = -math.inf if monitor_mode == "max" else \
                    math.inf
                log.info(f"Initializing best metric to {best_metric_value}")

        # --- 6. Training Loop ---
        log.info(f"Starting training from epoch {start_epoch}...")
        num_epochs = cfg.training.get("epochs", 1)  # Use config value

        for epoch in range(start_epoch, num_epochs):
            log.info(f"--- Epoch {epoch}/{num_epochs - 1} ---")
            experiment_logger.log_hardware_stats()

            try:
                # Train - Placeholder Data
                log.warning("Using placeholder training data.")
                model.train()
                if optimizer:
                    optimizer.zero_grad()
                dummy_loss = torch.tensor(0.5, requires_grad=True)
                if scaler:
                    scaler.scale(dummy_loss).backward()
                else:
                    dummy_loss.backward()
                if optimizer:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                avg_train_loss = dummy_loss.item()
                if experiment_logger:
                    experiment_logger.log_scalar("train/epoch_loss",
                                                 avg_train_loss, epoch)

                # Evaluate - Placeholder Data
                log.warning("Using placeholder evaluation data.")
                eval_results = {"val_loss": 0.8}
                if metrics:
                    eval_results.update({f"val_{k}": 0.6 for k in metrics})
                if experiment_logger:
                    experiment_logger.log_scalar(
                        "val/epoch_loss",
                        eval_results["val_loss"],
                        epoch
                    )
                    for k, v in eval_results.items():
                        if k != "val_loss":
                            experiment_logger.log_scalar(
                                k.replace("val_", "val/"),
                                v,
                                epoch
                            )

                # Step LR Scheduler (if epoch-based) - Placeholder
                if scheduler and cfg.training.scheduler.get("step_per_epoch",
                                                            True):
                    if isinstance(scheduler,
                                  torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(eval_results.get(monitor_metric))
                    else:
                        scheduler.step()

                # --- Checkpoint Saving ---
                is_best = False
                if save_best_enabled and monitor_metric in eval_results:
                    current_metric = eval_results[monitor_metric]
                    if monitor_mode == "max" and current_metric > \
                            best_metric_value:
                        best_metric_value = current_metric
                        is_best = True
                        msg = (f"New best metric ({monitor_metric}): "
                               f"{best_metric_value:.4f}")
                        log.info(msg)
                    elif monitor_mode == "min" and current_metric < \
                            best_metric_value:
                        best_metric_value = current_metric
                        is_best = True
                        msg = (f"New best metric ({monitor_metric}): "
                               f"{best_metric_value:.4f}")
                        log.info(msg)

                # Prepare state dictionary
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(
                    ) if optimizer else None,
                    'scheduler_state_dict': scheduler.state_dict(
                    ) if scheduler else None,
                    'best_metric_value': best_metric_value,
                    'config': OmegaConf.to_container(cfg, resolve=True)
                }

                # Save interval checkpoint
                if save_interval > 0 and (epoch + 1) % save_interval == 0:
                    filename = fname_pattern.format(epoch=epoch + 1)
                    save_checkpoint(
                        state=state,
                        is_best=is_best,
                        checkpoint_dir=checkpoint_dir,
                        filename=filename,
                        best_filename=best_filename
                    )
                # Save last checkpoint
                elif save_last and epoch == num_epochs - 1:
                    filename = "last_checkpoint.pth.tar"
                    save_checkpoint(
                        state=state,
                        is_best=is_best,
                        checkpoint_dir=checkpoint_dir,
                        filename=filename,
                        best_filename=best_filename
                    )
                # Explicitly save if it's the best, even if not interval/last
                elif is_best:
                    filename = fname_pattern.format(epoch=epoch + 1)
                    save_checkpoint(
                        state=state,
                        is_best=True,
                        checkpoint_dir=checkpoint_dir,
                        filename=filename,
                        best_filename=best_filename
                    )

            except Exception as e:
                experiment_logger.log_error(
                    error=e,
                    context=f"Training epoch {epoch}"
                )
                raise TrainingError(
                    f"Error during training epoch {epoch}: {str(e)}"
                ) from e

        log.info("Training loop finished.")

        # --- 7. Final Evaluation --- Placeholder
        if test_loader is not None:
            log.info("Performing final evaluation on test set... \
(Placeholder Data)")
            best_model_path = os.path.join(checkpoint_dir, best_filename)
            test_model = model  # Default to last model state
            if save_best_enabled and os.path.exists(best_model_path):
                log.info(f"Loading best model from: {best_model_path}")
                load_checkpoint(
                    model=test_model,
                    checkpoint_path=best_model_path,
                    device=device
                )
            else:
                msg = (
                    f"Best model checkpoint not found at '{best_model_path}' "
                    "or save_best disabled. Evaluating with the last model "
                    "state."
                )
                log.warning(msg)

            # Dummy test results
            test_results = {"test_loss": 0.7}
            if metrics:
                test_results.update({f"test_{k}": 0.55 for k in metrics})
            if experiment_logger:
                experiment_logger.log_scalar(
                    "test/epoch_loss",
                    test_results["test_loss"],
                    -1
                )
                for k, v in test_results.items():
                    if k != "test_loss":
                        experiment_logger.log_scalar(
                            k.replace("test_", "test/"),
                            v,
                            -1
                        )

            log.info(f"Final Test Results (Placeholder): {test_results}")
        else:
            log.info("No test loader available, skipping final evaluation.")

    except Exception as e:
        if experiment_logger:
            experiment_logger.log_error(error=e, context="Main execution")
        log.exception("Fatal error during execution")
        raise

    finally:
        # --- 8. Cleanup ---
        if experiment_logger:
            experiment_logger.close()
        log.info("Main execution finished.")


if __name__ == "__main__":
    main()
