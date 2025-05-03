"""Handles the main training and evaluation loops using a Trainer class."""

import time
import os
from typing import Any, Dict, Optional, Union

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
        # Usar 'training' como en la configuración real
        self.cfg = cfg.training
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

        # --- Checkpointing Configuration ---
        # Get the experiment manager from the logger
        if logger_instance and hasattr(logger_instance, 'experiment_manager'):
            self.experiment_manager = logger_instance.experiment_manager
            self.checkpoint_dir = str(
                self.experiment_manager.get_path("checkpoints")
            )
            self.internal_logger.info(
                f"Using checkpoint directory from ExperimentManager: "
                f"{self.checkpoint_dir}"
            )
        else:
            # Fallback to config or default
            default_checkpoint_dir = "outputs/checkpoints"
            checkpoint_dir_cfg = self.cfg.get(
                "checkpoint_dir", default_checkpoint_dir
            )

            # Asegurar que checkpoint_dir es una ruta absoluta
            if not os.path.isabs(checkpoint_dir_cfg):
                # Si es una ruta relativa, resolverla respecto al directorio
                # original
                try:
                    import hydra
                    orig_cwd = hydra.utils.get_original_cwd()
                    self.checkpoint_dir = os.path.join(
                        orig_cwd, checkpoint_dir_cfg
                    )
                    self.internal_logger.info(
                        f"Using absolute checkpoint directory: "
                        f"{self.checkpoint_dir}"
                    )
                except (ImportError, ValueError) as e:
                    # Fallback: usar la ruta relativa tal como está
                    self.checkpoint_dir = checkpoint_dir_cfg
                    self.internal_logger.warning(
                        f"Could not resolve absolute path: {e}. "
                        f"Using relative: {self.checkpoint_dir}"
                    )
            else:
                self.checkpoint_dir = checkpoint_dir_cfg

        # Create the checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.internal_logger.info(f"Checkpoint directory: \
{self.checkpoint_dir}")

        self.save_freq = self.cfg.get("save_freq", 0)
        self.checkpoint_load_path = self.cfg.get("checkpoint_load_path", None)

        # --- Save Best Configuration ---
        self.save_best_cfg = self.cfg.get("save_best", {})
        # Asegurarse de obtener enabled de la configuración correcta
        save_best_config = self.cfg.get("save_best", {})
        # Verificar si save_best existe directamente o dentro de checkpoints
        if not save_best_config and "checkpoints" in self.cfg:
            checkpoints_cfg = self.cfg.get("checkpoints", {})
            save_best_config = checkpoints_cfg.get("save_best", {})

        self.save_best_enabled = save_best_config.get("enabled", False)
        self.monitor_metric = self.save_best_cfg.get("monitor_metric",
                                                     "val_loss")
        self.monitor_mode = self.save_best_cfg.get("monitor_mode", "min")
        self.best_filename = self.save_best_cfg.get("best_filename",
                                                    "model_best.pth.tar")

        # Initialize best metric based on mode
        self.best_metric_value = float('inf') if self.monitor_mode == "min" \
            else float('-inf')

        self.model.to(self.device)

        # --- Setup Optimizer and Scheduler ---
        # Use factory functions based on config
        self.optimizer = create_optimizer(
            self.model.parameters(),
            self.full_cfg.training.optimizer  # Usar training según config
        )
        self.scheduler = create_lr_scheduler(
            self.optimizer,
            self.full_cfg.training.scheduler  # Usar training según config
        )

        # --- Setup Mixed Precision ---
        scaler_enabled = self.use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=scaler_enabled)
        if self.use_amp and not scaler_enabled:
            self.internal_logger.warning("AMP requires CUDA, disabling AMP.")
            self.use_amp = False

        # --- Initialize Training State Variables ---
        self.start_epoch = 1

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
                # Asegurarse de que monitor_metric sea consistente con
                # _check_if_best
                es_monitor = early_stopping_cfg.get("monitor",
                                                    self.monitor_metric)
                if not es_monitor.startswith("val_"):
                    es_monitor = f"val_{es_monitor}"

                # Instantiate using Hydra, passing shared monitor params
                self.early_stopper = instantiate(
                    early_stopping_cfg,
                    # Don't instantiate nested configs recursively
                    _recursive_=False,
                    monitor_metric=es_monitor,
                    mode=self.monitor_mode,
                    verbose=self.verbose  # Reuse trainer verbosity
                )
                self.internal_logger.info(
                    f"Early stopping enabled. Monitoring: {es_monitor}")
            except Exception as e:
                self.internal_logger.error(
                    f"Error initializing EarlyStopping: {e}", exc_info=True
                )
                self.early_stopper = None  # Disable if init fails
        else:
            self.internal_logger.info("Early stopping disabled.")

        # Log initialization summary
        config_summary = (
            f"Trainer initialized. Device: {self.device}. "
            f"Epochs: {self.epochs}. AMP: {self.use_amp}. "
            f"Grad Accum: {self.grad_accum_steps}. "
            f"Starting Epoch: {self.start_epoch}. "
            f"Save Best: {self.save_best_enabled}. "
            f"Early Stopping: {self.early_stopper is not None}."
        )
        self.internal_logger.info(config_summary)

        if self.logger_instance:
            log_msg = f"Using logger: \
{self.logger_instance.__class__.__name__}"
            self.internal_logger.info(log_msg)

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
                    self.logger_instance.log_scalar(
                        tag="lr", value=current_lr, step=epoch
                    )
            else:  # Added else block for clarity when no scheduler
                pass  # No scheduler step needed

            # --- Checkpointing Logic ---
            # Check if this is the best model so far
            is_best = self._check_if_best(val_results)

            # Always save last checkpoint
            save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                additional_data={
                    "metrics": val_results,
                    # Save current best value
                    "best_metric_value": self.best_metric_value
                },
                checkpoint_dir=self.checkpoint_dir,
                keep_last_n=1,  # Example: keep only last 1 checkpoint
                filename="checkpoint_last.pth",
            )

            # If this is the best model so far, save it separately
            if is_best:
                metric_val = val_results.get(self.monitor_metric, 'N/A')
                log_msg = (
                    f"Saving best model with "
                    f"{self.monitor_metric}={metric_val:.4f}"
                )
                self.internal_logger.info(log_msg)
                save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    additional_data={
                        "metrics": val_results,
                        "best_metric_value": self.best_metric_value
                    },
                    checkpoint_dir=self.checkpoint_dir,
                    keep_last_n=1,  # We only need to keep 1 best model
                    filename="model_best.pth.tar",
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

    def _train_step(
        self,
        batch: Union[tuple, Dict[str, torch.Tensor]]
    ) -> float:
        """Processes a single training batch.

        Args:
            batch: A tuple of (images, masks) tensors or
                a dictionary with 'image' and 'mask' keys.

        Returns:
            float: The batch loss value, scaled for gradient accumulation and
                  ready for logging.
        """
        # Handle both tuple format and dictionary format
        if isinstance(batch, tuple):
            images, masks = batch
        else:
            # Expect dictionary with 'image'/'mask' keys
            images = batch['image']
            masks = batch['mask']

        images, masks = images.to(self.device), masks.to(self.device)

        # Ensure masks have channel dimension if needed
        # Check if masks are 3D (missing channel dimension)
        if len(masks.shape) == 3:
            # Add channel dimension [B, H, W] -> [B, 1, H, W]
            masks = masks.unsqueeze(1)

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

    def _val_step(
        self,
        batch: Union[tuple, Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Processes a single validation batch.

        Args:
            batch: A tuple of (images, masks) tensors or
                a dictionary with 'image' and 'mask' keys.

        Returns:
            Dict[str, float]: Dictionary with loss and metric values for this
            batch.
        """
        # Handle both tuple format and dictionary format
        if isinstance(batch, tuple):
            images, masks = batch
        else:
            # Expect dictionary with 'image'/'mask' keys
            images = batch['image']
            masks = batch['mask']

        images, masks = images.to(self.device), masks.to(self.device)

        # Ensure masks have channel dimension if needed
        # Check if masks are 3D (missing channel dimension)
        if len(masks.shape) == 3:
            # Add channel dimension [B, H, W] -> [B, 1, H, W]
            masks = masks.unsqueeze(1)

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

        # Rename the loss metric to val_loss and add prefix to other metrics
        val_metrics = {}
        for name, value in avg_metrics.items():
            if name == "loss":
                val_metrics["val_loss"] = value
            else:
                val_metrics[f"val_{name}"] = value

        # Format log message
        metrics_str = " | ".join(
            [f"{name.capitalize()}: {value:.4f}" for name, value in
             val_metrics.items()]
        )
        log_msg = f"Epoch {epoch} | Validation Results | {metrics_str}"
        self.internal_logger.info(log_msg)

        # Log metrics to logger if available
        if self.logger_instance:
            for name, value in val_metrics.items():
                self.logger_instance.log_scalar(
                    tag=f"{name}",  # Already has val_ prefix
                    value=value,
                    step=epoch
                )

        # Return the validation metrics
        return val_metrics

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
        # Garantizar que usamos el nombre correcto de la métrica
        # Si el monitor_metric no tiene prefijo val_ pero buscamos en
        # val_metrics
        metric_name = self.monitor_metric
        all_val_metrics = all(k.startswith("val_") for k in metrics.keys())
        if not metric_name.startswith("val_") and all_val_metrics:
            metric_name = f"val_{self.monitor_metric}"

        current_metric = metrics.get(metric_name)
        if current_metric is None:
            available_metrics = ", ".join(metrics.keys())
            self.internal_logger.warning(
                f"Monitor metric '{metric_name}' not found in "
                f"validation results. Available: {available_metrics}. "
                f"Cannot determine if best."
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
                f"Validation metric '{metric_name}' improved from "
                f"{old_best:.4f} to {current_metric:.4f}. "
                f"Saving best checkpoint."
            )
            return True
        return False

# --- Remove old standalone functions ---
# def train_one_epoch(...): ...
# def evaluate(...): ...
