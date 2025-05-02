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
from src.utils.factory import get_metrics_from_cfg, get_optimizer, get_loss_fn
from src.model.factory import create_unet
from src.training.factory import create_lr_scheduler

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
            data_cfg = cfg.data
            transform_cfg = data_cfg.get("transforms", OmegaConf.create({}))
            dataloader_cfg = data_cfg
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

        # --- 3. Model Creation ---
        log.info("Creating model...")
        try:
            # Create model using the factory function
            model = create_unet(cfg.model)
            model.to(device)
            log.info(f"Created {type(model).__name__} model with \
{sum(p.numel() for p in model.parameters())} parameters")
        except Exception as e:
            raise ModelError(f"Error creating model: {str(e)}") from e

        # --- 4. Training Setup ---
        log.info("Setting up training components...")

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

        # Create optimizer using factory
        optimizer_cfg = cfg.training.get("optimizer",
                                         {"type": "adam", "lr": 1e-3})
        optimizer = get_optimizer(model.parameters(), optimizer_cfg)
        log.info(f"Created optimizer: {type(optimizer).__name__}")

        # Create loss function using factory
        loss_fn = None
        if hasattr(cfg.training, 'loss'):
            try:
                loss_fn = get_loss_fn(cfg.training.loss)
                log.info(f"Created loss function: {type(loss_fn).__name__}")
            except Exception as e:
                log.error(f"Error creating loss function: {e}")
                # Fallback to a simple loss if needed
                loss_fn = torch.nn.BCEWithLogitsLoss()
                log.info("Using fallback loss function: BCEWithLogitsLoss")

        # Create scheduler using factory
        scheduler = None
        if cfg.training.get("scheduler", None):
            try:
                scheduler = create_lr_scheduler(optimizer,
                                                cfg.training.scheduler)
                log.info(f"Created learning rate scheduler: \
{type(scheduler).__name__}")
            except Exception as e:
                log.error(f"Error creating scheduler: {e}")
                scheduler = None

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
                # Train
                model.train()
                train_loss = 0.0
                train_metrics = {k: 0.0 for k in metrics.keys()}
                train_batches = 0

                for batch_idx, batch in enumerate(train_loader):
                    inputs, targets = batch['image'].to(device), \
                        batch['mask'].to(device)

                    # Ensure inputs tensor has the correct shape (B, C, H, W)
                    # If channels are last (B, H, W, C)
                    if inputs.shape[-1] == 3:
                        # Change to (B, C, H, W)
                        inputs = inputs.permute(0, 3, 1, 2)

                    # Ensure targets tensor has a channel dimension
                    # If (B, H, W) without channel dimension
                    if len(targets.shape) == 3:
                        # Add channel dimension: (B,H,W)->(B,1,H,W)
                        targets = targets.unsqueeze(1)

                    # Clear gradients
                    optimizer.zero_grad()

                    # Forward pass
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            outputs = model(inputs)
                            batch_loss = loss_fn(outputs, targets)
                    else:
                        outputs = model(inputs)
                        batch_loss = loss_fn(outputs, targets)

                    # Backward pass
                    if use_amp:
                        scaler.scale(batch_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        batch_loss.backward()
                        optimizer.step()

                    train_loss += batch_loss.item()
                    with torch.no_grad():
                        for k, metric_fn in metrics.items():
                            train_metrics[k] += metric_fn(outputs,
                                                          targets).item()
                    train_batches += 1

                    # Log batch progress
                    if batch_idx % 10 == 0:
                        log.info(f"Batch {batch_idx}/{len(train_loader)}, \
Loss: {batch_loss.item():.4f}")

                # Calculate average training loss
                avg_train_loss = train_loss / train_batches if \
                    train_batches > 0 else 0
                for k in train_metrics:
                    train_metrics[k] /= train_batches if train_batches > 0 \
                        else 1
                if experiment_logger:
                    experiment_logger.log_scalar("train/epoch_loss",
                                                 avg_train_loss, epoch)
                    for k, v in train_metrics.items():
                        experiment_logger.log_scalar(f"train/{k}", v, epoch)
                log.info(
                    f"Training - Epoch: {epoch}, Loss: {avg_train_loss:.4f}, "
                    f"Metrics: {train_metrics}"
                )

                # Evaluate
                model.eval()
                eval_results = {"val_loss": 0.0}
                val_batches = 0

                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        inputs, targets = batch['image'].to(device), \
                            batch['mask'].to(device)

                        # Ensure inputs tensor has the correct shape
                        # (B, C, H, W)
                        # If channels are last (B, H, W, C)
                        if inputs.shape[-1] == 3:
                            # Change to (B, C, H, W)
                            inputs = inputs.permute(0, 3, 1, 2)

                        # Ensure targets tensor has a channel dimension
                        # If (B, H, W) without channel dimension
                        if len(targets.shape) == 3:
                            # Add channel dimension: (B,H,W)->(B,1,H,W)
                            targets = targets.unsqueeze(1)

                        # Forward pass
                        if use_amp:
                            with torch.amp.autocast('cuda'):
                                outputs = model(inputs)
                                batch_loss = loss_fn(outputs, targets)
                        else:
                            outputs = model(inputs)
                            batch_loss = loss_fn(outputs, targets)

                        eval_results["val_loss"] += batch_loss.item()
                        val_batches += 1

                        # Calculate metrics
                        if metrics:
                            for metric_name, metric_fn in metrics.items():
                                if metric_name not in eval_results:
                                    eval_results[f"val_{metric_name}"] = 0.0
                                eval_results[f"val_{metric_name}"] += \
                                    metric_fn(outputs, targets).item()

                # Calculate average validation metrics
                if val_batches > 0:
                    eval_results["val_loss"] /= val_batches
                    for metric_name in metrics:
                        if f"val_{metric_name}" in eval_results:
                            eval_results[f"val_{metric_name}"] /= val_batches

                # Log validation metrics
                if experiment_logger:
                    experiment_logger.log_scalar("val/epoch_loss",
                                                 eval_results["val_loss"],
                                                 epoch)
                    for k, v in eval_results.items():
                        if k != "val_loss":
                            experiment_logger.log_scalar(
                                k.replace("val_", "val/"), v, epoch)

                log.info(
                    f"Validation - Epoch: {epoch}, Loss: \
{eval_results['val_loss']:.4f}"
                )
                for k, v in eval_results.items():
                    if k != "val_loss":
                        log.info(f"Validation - {k}: {v:.4f}")

                # Step LR Scheduler (if epoch-based)
                if scheduler and cfg.training.scheduler.get("step_per_epoch",
                                                            True):
                    if isinstance(scheduler,
                                  torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(eval_results.get(
                            monitor_metric, eval_results["val_loss"]))
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

                # Save interval checkpoint
                if save_interval > 0 and (epoch + 1) % save_interval == 0:
                    filename = fname_pattern.format(epoch=epoch + 1)
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        checkpoint_dir=checkpoint_dir,
                        filename=filename,
                        additional_data={
                            'scheduler_state_dict': (
                                scheduler.state_dict() if scheduler else None
                            ),
                            'best_metric_value': best_metric_value,
                            'config': OmegaConf.to_container(cfg, resolve=True)
                        }
                    )
                # Save last checkpoint
                elif save_last and epoch == num_epochs - 1:
                    filename = "last_checkpoint.pth.tar"
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        checkpoint_dir=checkpoint_dir,
                        filename=filename,
                        additional_data={
                            'scheduler_state_dict': (
                                scheduler.state_dict() if scheduler else None
                            ),
                            'best_metric_value': best_metric_value,
                            'config': OmegaConf.to_container(cfg, resolve=True)
                        }
                    )
                # Explicitly save if it's the best, even if not interval/last
                elif is_best:
                    filename = fname_pattern.format(epoch=epoch + 1)
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        checkpoint_dir=checkpoint_dir,
                        filename=filename,
                        additional_data={
                            'scheduler_state_dict': (
                                scheduler.state_dict() if scheduler else None
                            ),
                            'best_metric_value': best_metric_value,
                            'config': OmegaConf.to_container(cfg, resolve=True)
                        }
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

        # --- 7. Final Evaluation ---
        if test_loader is not None:
            log.info("Performing final evaluation on test set...")
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

            # Perform test evaluation
            test_model.eval()
            test_results = {"test_loss": 0.0}
            test_batches = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    inputs, targets = batch['image'].to(device), batch[
                        'mask'].to(device)

                    # Ensure inputs tensor has the correct shape (B, C, H, W)
                    # If channels are last (B, H, W, C)
                    if inputs.shape[-1] == 3:
                        # Change to (B, C, H, W)
                        inputs = inputs.permute(0, 3, 1, 2)

                    # Ensure targets tensor has a channel dimension
                    # If (B, H, W) without channel dimension
                    if len(targets.shape) == 3:
                        # Add channel dimension -> (B, 1, H, W)
                        targets = targets.unsqueeze(1)

                    # Forward pass
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            outputs = test_model(inputs)
                            batch_loss = loss_fn(outputs, targets)
                    else:
                        outputs = test_model(inputs)
                        batch_loss = loss_fn(outputs, targets)

                    test_results["test_loss"] += batch_loss.item()
                    test_batches += 1

                    # Calculate metrics
                    if metrics:
                        for metric_name, metric_fn in metrics.items():
                            if metric_name not in test_results:
                                test_results[f"test_{metric_name}"] = 0.0
                            test_results[f"test_{metric_name}"] += \
                                metric_fn(outputs, targets).item()

            # Calculate average test metrics
            if test_batches > 0:
                test_results["test_loss"] /= test_batches
                for metric_name in metrics:
                    if f"test_{metric_name}" in test_results:
                        test_results[f"test_{metric_name}"] /= test_batches

            # Log test metrics
            if experiment_logger:
                experiment_logger.log_scalar("test/loss",
                                             test_results["test_loss"], -1)
                for k, v in test_results.items():
                    if k != "test_loss":
                        experiment_logger.log_scalar(
                            k.replace("test_", "test/"), v, -1)

            log.info(f"Test Results: Loss: {test_results['test_loss']:.4f}")
            for k, v in test_results.items():
                if k != "test_loss":
                    log.info(f"Test {k}: {v:.4f}")
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
