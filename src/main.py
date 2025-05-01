# In src/main.py (Skeleton and checkpointing logic)

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import os
import math  # For inf

# Project imports (assuming they exist)
from src.utils.seeds import set_random_seeds
from src.utils.device import get_device
from src.utils.checkpointing import save_checkpoint, load_checkpoint
from src.utils.loggers import BaseLogger, TensorBoardLogger, NoOpLogger
from src.data.factory import create_dataloaders_from_config  # Import factory
# from src.models.model_factory import create_model
from src.training.losses import get_loss_from_cfg  # Need the factory
from src.training.metrics import get_metrics_from_cfg  # Need the factory
# from src.training.optimizers import create_optimizer
# from src.training.schedulers import create_scheduler
from src.training.trainer import train_one_epoch, evaluate

# Configure standard logger
log = logging.getLogger(__name__)


def initialize_logger(cfg: DictConfig) -> BaseLogger:
    """Initializes the logger based on configuration."""
    logger_cfg = cfg.get('logging', {})  # Access logging section
    # Ensure logger_cfg is a DictConfig or dict
    if not isinstance(logger_cfg, (dict, DictConfig)):
        msg = "Logging configuration is not a dictionary. Disabling logging."
        log.warning(msg)
        return NoOpLogger()

    logger_type = logger_cfg.get("type", None)
    enabled = logger_cfg.get("enabled", False)
    log_dir = logger_cfg.get("log_dir", "outputs/tensorboard")  # config dir

    # Make sure log_dir is absolute or relative to hydra directory
    # Hydra by default makes paths relative to output directory
    # Use hydra.utils.get_original_cwd() or hydra.run.dir for absolute paths

    if not enabled:
        log.info("Logging disabled via configuration.")
        return NoOpLogger()

    if logger_type == "tensorboard":
        log.info(f"Initializing TensorBoardLogger in {log_dir}...")
        try:
            # Pass log_dir directly, Hydra will resolve relative path
            return TensorBoardLogger(log_dir=log_dir)
        except ImportError as e:
            log.error(f"Failed to initialize TensorBoardLogger: {e}")
            log.warning("Falling back to NoOpLogger.")
            return NoOpLogger()
    elif logger_type is None or logger_type == "none":
        log.info("Logging type is 'none'. Using NoOpLogger.")
        return NoOpLogger()
    else:
        msg = f"Unsupported logger type: {logger_type}. Using NoOpLogger."
        log.warning(msg)
        return NoOpLogger()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training and evaluation entry point."""
    # --- 1. Initial Setup ---
    log.info("Starting main execution...")
    set_random_seeds(cfg.get('seed', 42))  # Assuming seed config exists
    device = get_device()
    log.info(f"Using device: {device}")
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Initialize Logger
    # Use hydra.utils.get_original_cwd() if absolute path needed from start
    # hydra_output_dir = os.getcwd() # Hydra changes CWD to output directory
    logger_instance = initialize_logger(cfg)

    # --- 2. Data Loading ---
    log.info("Loading data...")
    try:
        # Access configs safely
        data_cfg = cfg.get("data", OmegaConf.create({}))
        transform_cfg = data_cfg.get("augmentation", OmegaConf.create({}))
        # Assuming dataloader section
        dataloader_cfg = cfg.get("dataloader", OmegaConf.create({}))

        dataloaders_dict = create_dataloaders_from_config(
            data_config=data_cfg,
            transform_config=transform_cfg,
            dataloader_config=dataloader_cfg
            # dataset_class=... # Could be configured if needed
        )
        train_loader = dataloaders_dict.get('train', {}).get('dataloader')
        val_loader = dataloaders_dict.get('val', {}).get('dataloader')
        # Optional test loader
        test_loader = dataloaders_dict.get('test', {}).get('dataloader')

        if not train_loader or not val_loader:
            raise ValueError("Train or Val dataloader could not be created.")
        log.info("Data loading complete.")

    except Exception as e:
        log.exception(f"Error during data loading: {e}. Exiting.")
        # Consider closing logger if initialized
        if logger_instance:
            logger_instance.close()
        return  # Exit if data loading fails

    # --- 3. Model Creation --- Placeholder
    log.info("Creating model... (Placeholder)")
    model = torch.nn.Module()  # Simple placeholder
    model.to(device)
    log.info("Model creation complete.")

    # --- 4. Training Setup --- Placeholder
    log.info("Setting up training components... (Partially Placeholder)")
    # Use factories for loss and metrics
    # Initialize loss function (will be used in training loop)
    loss_fn = get_loss_from_cfg(cfg.training.loss).to(device)
    # Ensure evaluation metrics config exists
    metrics = {}
    eval_cfg = cfg.get('evaluation', None)
    if eval_cfg and hasattr(eval_cfg, 'metrics'):
        metrics = get_metrics_from_cfg(eval_cfg.metrics)
        # Move metrics to device if needed (depends on Metric implementation)
        # metrics = {k: m.to(device) for k, m in metrics.items()}
    else:
        log.warning("Evaluation metrics configuration not found.")

    # Placeholder optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Placeholder
    scheduler = None  # Placeholder

    # Setup AMP if enabled
    use_amp = cfg.training.get("amp_enabled", False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    log.info(f"AMP Enabled: {scaler is not None}")
    log.info("Training setup complete.")

    # --- 5. Checkpointing and Resume ---
    start_epoch = 0
    best_metric_value = None
    # Use .get to safely access nested config for checkpoints
    checkpoint_cfg = cfg.logging.get("checkpoints", OmegaConf.create({}))
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
            # Get next epoch and best metric
            start_epoch = checkpoint_data.get("epoch", 0) + 1
            best_metric_value = checkpoint_data.get("best_metric_value", None)
            msg = f"Resumed from epoch {start_epoch}. Best: \
{best_metric_value}"
            log.info(msg)
        else:
            msg = f"Resume checkpoint not found: {resume_path}. Fresh start."
            log.warning(msg)
    else:
        log.info("No checkpoint specified for resume, starting from scratch.")

    # --- Setup Best Model Tracking ---
    # Use .get to safely access nested config for save_best
    save_best_cfg = checkpoint_cfg.get("save_best", OmegaConf.create({}))
    monitor_metric = save_best_cfg.get("monitor_metric", None)
    monitor_mode = save_best_cfg.get("monitor_mode", "max")
    save_best_enabled = save_best_cfg.get("enabled", False)
    best_filename = save_best_cfg.get("best_filename", "model_best.pth.tar")
    # Use .get for other checkpoint settings with defaults
    checkpoint_dir = checkpoint_cfg.get("checkpoint_dir", "checkpoints/")
    fname_pattern = checkpoint_cfg.get(
        "filename",
        "checkpoint_epoch_{epoch:03d}.pth.tar"
    )
    save_interval = checkpoint_cfg.get("save_interval_epochs", 0)
    save_last = checkpoint_cfg.get("save_last", True)

    if save_best_enabled and not monitor_metric:
        msg = "save_best enabled but monitor_metric not set. Disabling best \
model saving."
        log.warning(msg)
        save_best_enabled = False
    elif save_best_enabled:
        msg = f"Monitoring '{monitor_metric}' for best model (mode: \
{monitor_mode})."
        log.info(msg)
        if best_metric_value is None:  # Initialize if not resuming
            best_metric_value = -math.inf if monitor_mode == "max" else \
                math.inf
            log.info(f"Initializing best metric to {best_metric_value}")

    # --- 6. Training Loop ---
    log.info(f"Starting training from epoch {start_epoch}...")
    # Default to 1 epoch for placeholder
    num_epochs = cfg.training.get("epochs", 1)

    for epoch in range(start_epoch, num_epochs):
        log.info(f"--- Epoch {epoch}/{num_epochs - 1} ---")

        # Train - Placeholder Data
        log.warning("Using placeholder training data.")
        # Dummy training step if no loader
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
        if logger_instance:
            logger_instance.log_scalar("train/epoch_loss", avg_train_loss, epoch)

        # Evaluate - Placeholder Data
        log.warning("Using placeholder evaluation data.")
        # Dummy evaluation results if no loader
        eval_results = {"val_loss": 0.8}
        if metrics:
            eval_results.update({f"val_{k}": 0.6 for k in metrics})  # Placeholder value
        if logger_instance:
            logger_instance.log_scalar("val/epoch_loss",
                                       eval_results["val_loss"], epoch)
            for k, v in eval_results.items():
                if k != "val_loss":
                    logger_instance.log_scalar(k.replace("val_", "val/"), v,
                                               epoch)

        # Step LR Scheduler (if epoch-based) - Placeholder
        # if scheduler and cfg.training.scheduler.get("step_per_epoch", True):
        #    scheduler.step() # Pass metric if needed (e.g., ReduceLROnPlateau)

        # --- Checkpoint Saving ---
        is_best = False
        if save_best_enabled and monitor_metric in eval_results:
            current_metric = eval_results[monitor_metric]
            if monitor_mode == "max" and current_metric > best_metric_value:
                best_metric_value = current_metric
                is_best = True
                log.info(f"New best metric ({monitor_metric}): \
{best_metric_value:.4f}")
            elif monitor_mode == "min" and current_metric < best_metric_value:
                best_metric_value = current_metric
                is_best = True
                log.info(f"New best metric ({monitor_metric}): \
{best_metric_value:.4f}")

        # Prepare state dictionary
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else \
            None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else \
            None,
            'best_metric_value': best_metric_value,
            # Save config snapshot
            'config': OmegaConf.to_container(cfg, resolve=True)
        }

        # Save interval checkpoint
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            filename = fname_pattern.format(epoch=epoch + 1)
            save_checkpoint(
                state=state,
                # Pass is_best flag even for interval checkpoints
                is_best=is_best,
                checkpoint_dir=checkpoint_dir,  # Dir from config
                filename=filename,
                best_filename=best_filename
            )
        # Save last checkpoint
        elif save_last and epoch == num_epochs - 1:
            filename = "last_checkpoint.pth.tar"
            save_checkpoint(
                state=state,
                is_best=is_best,  # Also check if last is best
                checkpoint_dir=checkpoint_dir,
                filename=filename,
                best_filename=best_filename
            )
        # Explicitly save if it's the best, even if not interval/last
        elif is_best:
            # Use regular name format for the specific epoch checkpoint
            filename = fname_pattern.format(epoch=epoch + 1)
            save_checkpoint(
                state=state,
                is_best=True,  # Force saving best copy
                checkpoint_dir=checkpoint_dir,
                filename=filename,  # Still save the epoch-specific file
                best_filename=best_filename
            )

    log.info("Training loop finished.")

    # --- 7. Final Evaluation --- Placeholder
    if test_loader is not None:  # Check if test_loader was actually created
        log.info("Performing final evaluation on test set... \
(Placeholder Data)")
        # Load best model for final evaluation
        best_model_path = os.path.join(
            checkpoint_dir,
            best_filename
        )
        test_model = model  # Default to last model state
        if save_best_enabled and os.path.exists(best_model_path):
            log.info(f"Loading best model from: {best_model_path}")
            # Create a fresh model instance or ensure current one is clean
            # test_model = create_model(cfg.model).to(device) # Ideal case
            # For placeholder, re-use model instance is ok after loading state
            load_checkpoint(model=test_model, checkpoint_path=best_model_path,
                            device=device)
        else:
            log.warning(f"Best model checkpoint not found at \
'{best_model_path}' or save_best disabled. " "Evaluating with the last model \
state.")

        # Dummy test results
        test_results = {"test_loss": 0.7}
        if metrics:
            # Placeholder value
            test_results.update({f"test_{k}": 0.55 for k in metrics})
        if logger_instance:
            logger_instance.log_scalar("test/epoch_loss",
                                       test_results["test_loss"], -1)
            for k, v in test_results.items():
                if k != "test_loss":
                    logger_instance.log_scalar(k.replace("test_", "test/"), v,
                                               -1)

        log.info(f"Final Test Results (Placeholder): {test_results}")
        # Optionally save test results to a file
    else:
        log.info("No test loader available, skipping final evaluation.")

    # --- 8. Cleanup ---
    if logger_instance:
        logger_instance.close()
    log.info("Main execution finished.")


if __name__ == "__main__":
    main()
