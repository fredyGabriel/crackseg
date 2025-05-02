"""Handles the main training and evaluation loops using a Trainer class."""

import time
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
# from torch.optim import Optimizer # Not used directly in this module yet
# from torch.optim.lr_scheduler import _LRScheduler # Not used directly
from torch.utils.data import DataLoader
from hydra.utils import instantiate

# Import factory functions
from src.training.factory import create_optimizer, create_lr_scheduler
# Import get_scalar_metrics
# from src.training.metrics import get_scalar_metrics # Removed unused import
from src.utils.device import get_device
# Import log_metrics_dict - Removed, use logger instance methods
from src.utils import BaseLogger, get_logger
# Import checkpointing functions
from src.utils.checkpointing import load_checkpoint, save_checkpoint
# Import EarlyStopping
from src.utils.early_stopping import EarlyStopping


# Placeholder: Checkpointing
# from src.utils.checkpointing import save_checkpoint, load_checkpoint
# Placeholder: Early Stopping
# from src.evaluation.early_stopping import EarlyStopping


class Trainer:
    """Orchestrates the training and validation process."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: torch.nn.Module,
        metrics_dict: Dict[str, Any],  # Instantiated metrics
        cfg: DictConfig,
        logger_instance: Optional[BaseLogger] = None,
        # Early stopper can be instantiated outside and passed, or configured
        early_stopper: Optional[EarlyStopping] = None
    ):
        """Initializes the Trainer.

        Args:
            model: The model to train.
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            loss_fn: The loss function.
            metrics_dict: Dictionary of instantiated metric objects
                          {name: metric_fn}.
            cfg: Hydra configuration object (expecting cfg.trainer node).
            logger_instance: Optional logger for recording metrics.
            early_stopper: Optional EarlyStopping instance for early stopping.
        """
        # Store original full config if needed later
        self.full_cfg = cfg
        self.cfg = cfg.trainer  # Assume trainer config is under 'trainer' key
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metrics_dict = metrics_dict
        self.logger_instance = logger_instance
        # For internal messages
        self.internal_logger = get_logger(self.__class__.__name__)

        # --- Configuration Parsing ---
        self.epochs = self.cfg.get("epochs", 10)
        self.device_str = self.cfg.get("device", "auto")
        self.device = get_device(self.device_str)
        self.use_amp = self.cfg.get("use_amp", True)
        self.grad_accum_steps = self.cfg.get("gradient_accumulation_steps", 1)
        self.verbose = self.cfg.get("verbose", True)
        self.checkpoint_dir = self.cfg.get("checkpoint_dir", "checkpoints")
        self.checkpoint_load_path = self.cfg.get("checkpoint_load_path", None)
        # Add parsing for progress_bar, early_stopping, saving frequency later
        # Also add config for metric to monitor for best checkpoint
        self.monitor_metric = self.cfg.get("monitor_metric", "loss")
        self.monitor_mode = self.cfg.get("monitor_mode", "min")
        self.save_freq = self.cfg.get("save_freq", 0)

        self.model.to(self.device)

        # --- Setup Optimizer and Scheduler ---
        # Use factory functions based on config
        self.optimizer = create_optimizer(
            self.model.parameters(),
            self.full_cfg.trainer.optimizer  # Use original cfg path
        )
        self.scheduler = create_lr_scheduler(
            self.optimizer,
            self.full_cfg.trainer.lr_scheduler  # Use original cfg path
        )

        # --- Setup Mixed Precision ---
        scaler_enabled = self.use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=scaler_enabled)
        if self.use_amp and not scaler_enabled:
            self.internal_logger.warning("AMP requires CUDA, disabling AMP.")
            self.use_amp = False

        # --- Initialize Training State Variables ---
        self.start_epoch = 1
        # Initialize best metric based on mode
        self.best_metric_value = float('inf') if self.monitor_mode == "min" \
            else float('-inf')

        # --- Load Checkpoint if specified ---
        if self.checkpoint_load_path:
            loaded_state = load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                # scheduler=self.scheduler,  # Cannot load scheduler state yet
                # scaler=self.scaler, # Cannot load scaler state yet
                checkpoint_path=self.checkpoint_load_path,
                device=self.device
            )
            # Adjust start epoch (load_checkpoint returns the epoch *saved*)
            self.start_epoch = loaded_state.get('epoch', 0) + 1
            # Load best metric value if available in checkpoint
            # Using a different key to avoid conflict if 'metrics' dict exists
            saved_best_metric = loaded_state.get('best_metric_value', None)
            if saved_best_metric is not None:
                self.best_metric_value = saved_best_metric
                log_msg = (
                    f"Loaded best metric ({self.monitor_metric}): "
                    f"{self.best_metric_value:.4f}"
                )
                self.internal_logger.info(log_msg)
            # Note: Scaler state is loaded directly into self.scaler by
            # load_checkpoint if available

        # --- Setup Early Stopping --- #
        self.early_stopper = None
        early_stopping_cfg = self.cfg.get("early_stopping", None)
        if early_stopping_cfg:
            try:
                # Instantiate using Hydra, passing shared monitor params
                self.early_stopper = instantiate(
                    early_stopping_cfg,
                    monitor_metric=self.monitor_metric,
                    mode=self.monitor_mode,
                    verbose=self.verbose  # Reuse trainer verbosity
                )
                self.internal_logger.info("Early stopping enabled.")
            except Exception as e:
                self.internal_logger.error(
                    f"Error initializing EarlyStopping: {e}", exc_info=True
                )
                self.early_stopper = None  # Disable if init fails
        else:
            self.internal_logger.info("Early stopping disabled.")

        self.internal_logger.info(
            f"Trainer initialized. Device: {self.device}. Epochs: \
{self.epochs}. "
            f"AMP: {self.use_amp}. Grad Accum: {self.grad_accum_steps}. \
Starting Epoch: {self.start_epoch}"
        )
        if self.logger_instance:
            log_msg = f"Using logger: \
{self.logger_instance.__class__.__name__}"
            self.internal_logger.info(log_msg)
        # Log config if logger exists (implement log_config in logger later)
        # if self.logger_instance:
        #   from omegaconf import OmegaConf # Import here if used only here
        #   self.logger_instance.log_config(
        #        OmegaConf.to_container(cfg, resolve=True)
        #   )

    def train(self) -> Dict[str, float]:
        """Runs the full training loop starting from self.start_epoch."""
        start_msg = f"Starting training from epoch {self.start_epoch}..."
        self.internal_logger.info(start_msg)
        start_time = time.time()
        final_val_results = {}

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start_time = time.time()
            self.internal_logger.info(f"--- Epoch {epoch}/{self.epochs} ---")

            _ = self._train_epoch(epoch)  # train_loss is unused for now
            val_results = self.validate(epoch)
            final_val_results = val_results  # Keep track of last results

            # --- Handle LR Scheduling ---
            if self.scheduler:
                current_lr = self._step_scheduler(val_results)
                if current_lr is not None and self.logger_instance:
                    # Use log_scalar for individual values
                    # self.logger_instance.log_metrics({"lr": current_lr}, epoch)
                    self.logger_instance.log_scalar(
                        tag="lr", value=current_lr, step=epoch
                    )
            else:  # Added else block for clarity when no scheduler
                pass  # No scheduler step needed

            # --- Checkpointing Logic ---
            # is_best = self._check_if_best(val_results) # Removed unused var
            # Determine if this epoch's checkpoint should be saved based on
            # freq
            # save_epoch_ckpt = (
            #     self.save_freq > 0 and epoch % self.save_freq == 0
            # )

            # Always save last checkpoint
            save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                # scaler=self.scaler, # Removed scaler argument
                # Pass metrics as additional data
                additional_data={"metrics": val_results},
                checkpoint_dir=self.checkpoint_dir,
                keep_last_n=1,  # Example: keep only last 1 checkpoint
                # is_best=is_best, # Removed is_best
                # Save last checkpoint always
                filename="checkpoint_last.pth",
                # Optional epoch-wise checkpoint
                # filename=f"checkpoint_epoch_{epoch}.pth" if save_epoch_ckpt \
                # else None,
                # Save best checkpoint always if is_best
                # best_filename="checkpoint_best.pth" # Removed best_filename
            )

            # --- Early Stopping Check ---
            if self.early_stopper:
                current_metric = val_results.get(
                    self.early_stopper.monitor_metric)
                if self.early_stopper.step(current_metric):
                    self.internal_logger.info("Early stopping triggered.")
                    break  # Exit training loop

            epoch_duration = time.time() - epoch_start_time
            self.internal_logger.info(
                f"Epoch {epoch} finished in {epoch_duration:.2f}s"
            )

        total_time = time.time() - start_time
        end_msg = f"Training finished in {total_time:.2f}s"
        self.internal_logger.info(end_msg)
        # Return last validation results (or best results if tracking)
        return final_val_results

    def _train_step(self, batch: tuple) -> float:
        """Processes a single training batch.

        Args:
            batch: A tuple of (images, masks) tensors.

        Returns:
            float: The batch loss value, scaled for gradient accumulation and
                  ready for logging.
        """
        images, masks = batch
        images, masks = images.to(self.device), masks.to(self.device)

        with autocast('cuda', enabled=self.use_amp):
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)
            # Scale loss for accumulation
            loss = loss / self.grad_accum_steps

        self.scaler.scale(loss).backward()

        # Return unscaled loss for logging (multiply back by grad_accum_steps)
        return loss.item() * self.grad_accum_steps

    def _train_epoch(self, epoch: int) -> float:
        """Runs a single training epoch.

        Args:
            epoch: The current epoch number.

        Returns:
            float: Average loss value for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        # Get log interval from config
        log_interval = self.cfg.get("log_interval_batches", 0)

        # Reset gradients at the start of epoch
        self.optimizer.zero_grad()

        # TODO: Add progress bar (tqdm)

        for batch_idx, batch in enumerate(self.train_loader):
            batch_loss = self._train_step(batch)
            total_loss += batch_loss

            is_update_step = (batch_idx + 1) % self.grad_accum_steps == 0
            is_last_batch = (batch_idx + 1) == num_batches
            if is_update_step or is_last_batch:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # Reset gradients after step
                self.optimizer.zero_grad()

            # --- Batch Logging ---
            if self.logger_instance and log_interval > 0 and \
               (batch_idx + 1) % log_interval == 0:
                # Calculate global step (adjusting for 0-based batch_idx)
                global_step = (epoch - 1) * num_batches + batch_idx + 1
                # Use log_scalar for batch loss
                # self.logger_instance.log_metrics(
                #     {"batch_loss": batch_loss},
                #     global_step,
                #     prefix="train_batch/"  # Use distinct prefix
                # )
                self.logger_instance.log_scalar(
                    tag="train_batch/batch_loss",
                    value=batch_loss,
                    step=global_step
                )

        avg_loss = total_loss / num_batches
        log_msg = f"Epoch {epoch} | Average Training Loss: {avg_loss:.4f}"
        self.internal_logger.info(log_msg)

        # --- Log Epoch Metrics ---
        if self.logger_instance:
            # Use log_scalar for epoch loss
            # self.logger_instance.log_metrics(
            #     {"epoch_loss": avg_loss},
            #     epoch,
            #     prefix="train/"
            # )
            self.logger_instance.log_scalar(
                tag="train/epoch_loss", value=avg_loss, step=epoch
            )

        return avg_loss

    def _val_step(self, batch: tuple) -> Dict[str, float]:
        """Processes a single validation batch.

        Args:
            batch: A tuple of (images, masks) tensors.

        Returns:
            Dict[str, float]: Dictionary with loss and metric values for this
            batch.
        """
        images, masks = batch
        images, masks = images.to(self.device), masks.to(self.device)

        with autocast('cuda', enabled=self.use_amp):
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)

        # Calculate metrics
        metrics = {"loss": loss.item()}
        for name, metric_fn in self.metrics_dict.items():
            metric_val = metric_fn(outputs, masks)
            metrics[name] = metric_val.item()

        return metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """Runs validation on the validation dataset.

        Args:
            epoch: The current epoch number.

        Returns:
            Dict[str, float]: Average loss and metric values for the validation
                             dataset.
        """
        self.model.eval()
        total_metrics = {"loss": 0.0}
        total_metrics.update({name: 0.0 for name in self.metrics_dict})
        num_batches = len(self.val_loader)

        # TODO: Add progress bar (tqdm)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                batch_metrics = self._val_step(batch)

                # Accumulate metrics
                for name, value in batch_metrics.items():
                    total_metrics[name] += value

        # Calculate averages
        avg_metrics = {name: total / num_batches
                       for name, total in total_metrics.items()}

        # Format log message
        metrics_str = " | ".join(
            [f"{name.capitalize()}: {value:.4f}" for name, value in
             avg_metrics.items()]
        )
        log_msg = f"Epoch {epoch} | Validation Results | {metrics_str}"
        self.internal_logger.info(log_msg)

        # Ensure floats for logging consistency
        # scalar_results_for_logging = get_scalar_metrics(avg_metrics)

        if self.logger_instance:
            # Use log_scalar for each validation metric
            # self.logger_instance.log_metrics(
            #     avg_metrics, epoch, prefix="val/"
            # )
            for name, value in avg_metrics.items():
                self.logger_instance.log_scalar(
                    tag=f"val/{name}",
                    value=value,
                    step=epoch
                )

        # Return the dictionary with potentially non-scalar values if needed
        # upstream, otherwise return scalar_results_for_logging
        return avg_metrics

    # --- Helper Methods ---
    def _step_scheduler(self, metrics=None) -> Optional[float]:
        """Steps the learning rate scheduler."""
        current_lr = None
        if not self.scheduler:
            return current_lr

        # Logic depends on scheduler type
        if isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            monitor_metric = metrics.get(self.monitor_metric)
            if monitor_metric is None:
                self.internal_logger.warning(
                    f"ReduceLROnPlateau scheduler needs \
'{self.monitor_metric}' metric for step. Skipping scheduler step."
                )
            else:
                self.scheduler.step(monitor_metric)
        else:
            self.scheduler.step()  # Step per epoch for others

        # Get current LR after step
        if self.optimizer:  # Check if optimizer exists
            current_lr = self.optimizer.param_groups[0]['lr']
            msg = f"LR Scheduler step. Current LR: {current_lr:.6f}"
            self.internal_logger.info(msg)
            return current_lr
        return None  # Return None if no optimizer/scheduler

    def _check_if_best(self, metrics: Dict[str, float]) -> bool:
        """Checks if the current metrics represent the best seen so far."""
        current_metric = metrics.get(self.monitor_metric)
        if current_metric is None:
            self.internal_logger.warning(
                f"Monitor metric '{self.monitor_metric}' not found in \
validation results. Cannot determine if best."
            )
            return False

        is_improvement = False
        if self.monitor_mode == "min":
            if current_metric < self.best_metric_value:
                is_improvement = True
        elif self.monitor_mode == "max":
            if current_metric > self.best_metric_value:
                is_improvement = True

        if is_improvement:
            old_best = self.best_metric_value
            self.best_metric_value = current_metric
            self.internal_logger.info(
                f"Validation metric '{self.monitor_metric}' improved \
from {old_best:.4f} to {current_metric:.4f}. Saving best checkpoint."
            )
            return True
        return False

# --- Remove old standalone functions ---
# def train_one_epoch(...): ...
# def evaluate(...): ...
