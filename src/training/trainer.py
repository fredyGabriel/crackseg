"""Trainer: orchestrates training, validation, checkpointing, and early
stopping."""

import time
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch.amp import GradScaler
from torch.utils.data import DataLoader

# Import factory functions
from src.training.factory import create_optimizer, create_lr_scheduler
from src.utils.device import get_device
from src.utils import BaseLogger
from src.utils.checkpointing import load_checkpoint
from src.utils.early_stopping import EarlyStopping
from src.utils.checkpointing_helper import handle_epoch_checkpointing
from src.utils.scheduler_helper import step_scheduler_helper
from src.training.batch_processing import train_step, val_step
from src.utils.training_logging import log_validation_results
from src.utils.amp_utils import amp_autocast, optimizer_step_with_accumulation
from src.training.config_validation import validate_trainer_config
from src.utils.checkpointing_setup import setup_checkpointing
from src.utils.early_stopping_setup import setup_early_stopping
from src.utils.logger_setup import setup_internal_logger, safe_log


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
        # --- Config validation ---
        validate_trainer_config(cfg.training)
        self.full_cfg = cfg
        self.cfg = cfg.training
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metrics_dict = metrics_dict
        self.logger_instance = logger_instance
        self.internal_logger = setup_internal_logger(logger_instance)

        # --- Configuration Parsing ---
        self.epochs = self.cfg.get("epochs", 10)
        self.device_str = self.cfg.get("device", "auto")
        self.device = get_device(self.device_str)
        self.use_amp = self.cfg.get("use_amp", True)
        self.grad_accum_steps = self.cfg.get("gradient_accumulation_steps", 1)
        self.verbose = self.cfg.get("verbose", True)

        # --- Setup Checkpointing ---
        self.checkpoint_dir, self.experiment_manager = setup_checkpointing(
            cfg,
            getattr(logger_instance, 'experiment_manager', None),
            self.internal_logger
        )
        self.save_freq = self.cfg.get("save_freq", 0)
        self.checkpoint_load_path = self.cfg.get("checkpoint_load_path", None)
        self.save_best_cfg = self.cfg.get("save_best", {})
        save_best_config = self.cfg.get("save_best", {})
        if not save_best_config and "checkpoints" in self.cfg:
            checkpoints_cfg = self.cfg.get("checkpoints", {})
            save_best_config = checkpoints_cfg.get("save_best", {})
        self.save_best_enabled = save_best_config.get("enabled", False)
        self.monitor_metric = self.save_best_cfg.get("monitor_metric",
                                                     "val_loss")
        self.monitor_mode = self.save_best_cfg.get("monitor_mode",
                                                   "min")
        self.best_filename = self.save_best_cfg.get("best_filename",
                                                    "model_best.pth.tar")
        self.best_metric_value = float('inf') if self.monitor_mode == "min" \
            else float('-inf')
        self.model.to(self.device)

        # --- Setup Optimizer and Scheduler ---
        self.optimizer = create_optimizer(
            self.model.parameters(),
            self.full_cfg.training.optimizer
        )
        self.scheduler = create_lr_scheduler(
            self.optimizer,
            self.full_cfg.training.scheduler
        )

        # --- Setup Mixed Precision ---
        scaler_enabled = self.use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=scaler_enabled)
        if self.use_amp and not scaler_enabled:
            safe_log(
                self.internal_logger, "warning",
                "AMP requires CUDA, disabling AMP."
            )
            self.use_amp = False

        # --- Initialize Training State Variables ---
        self.start_epoch = 1

        # --- Load Checkpoint if specified ---
        if self.checkpoint_load_path:
            loaded_state = load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                checkpoint_path=self.checkpoint_load_path,
                device=self.device
            )
            self.start_epoch = loaded_state.get('epoch', 0) + 1
            saved_best_metric = loaded_state.get('best_metric_value', None)
            if saved_best_metric is not None:
                self.best_metric_value = saved_best_metric
                log_msg = (
                    f"Loaded best metric ({self.monitor_metric}): "
                    f"{self.best_metric_value:.4f}"
                )
                safe_log(self.internal_logger, "info", log_msg)

        # --- Setup Early Stopping ---
        self.early_stopper = setup_early_stopping(
            cfg,
            monitor_metric="loss",
            monitor_mode="min",
            verbose=True,
            logger=self.internal_logger
        )

        # Log initialization summary
        config_summary = (
            f"Trainer initialized. Device: {self.device}. "
            f"Epochs: {self.epochs}. AMP: {self.use_amp}. "
            f"Grad Accum: {self.grad_accum_steps}. "
            f"Starting Epoch: {self.start_epoch}. "
            f"Save Best: {self.save_best_enabled}. "
            f"Early Stopping: {self.early_stopper is not None}."
        )
        safe_log(self.internal_logger, "info", config_summary)
        if self.logger_instance:
            log_msg = (
                f"Using logger: {self.logger_instance.__class__.__name__}"
            )
            safe_log(self.internal_logger, "info", log_msg)

    def train(self) -> Dict[str, float]:
        """Runs the full training loop starting from self.start_epoch."""
        start_msg = f"Starting training from epoch {self.start_epoch}..."
        safe_log(self.internal_logger, "info", start_msg)
        start_time = time.time()
        final_val_results = {}

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start_time = time.time()
            safe_log(
                self.internal_logger, "info",
                f"--- Epoch {epoch}/{self.epochs} ---"
            )

            _ = self._train_epoch(epoch)  # train_loss is unused for now
            val_results = self.validate(epoch)
            final_val_results = val_results  # Keep track of last results

            # --- Handle LR Scheduling ---
            if self.scheduler:
                current_lr = self._step_scheduler(val_results)
                if current_lr is not None and self.logger_instance:
                    self.logger_instance.log_scalar(
                        tag="lr", value=current_lr, step=epoch
                    )
            else:
                pass

            # --- Checkpointing Logic (refactorizado) ---
            self.best_metric_value = handle_epoch_checkpointing(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                val_results=val_results,
                monitor_metric=self.monitor_metric,
                monitor_mode=self.monitor_mode,
                best_metric_value=self.best_metric_value,
                checkpoint_dir=self.checkpoint_dir,
                logger=self.internal_logger,
                keep_last_n=1,
                last_filename="checkpoint_last.pth",
                best_filename=self.best_filename,
            )

            # --- Early Stopping Check ---
            if self.early_stopper:
                current_metric = val_results.get(
                    self.early_stopper.monitor_metric)
                if self.early_stopper.step(current_metric):
                    safe_log(
                        self.internal_logger, "info",
                        "Early stopping triggered."
                    )
                    break  # Exit training loop

            epoch_duration = time.time() - epoch_start_time
            safe_log(
                self.internal_logger, "info",
                f"Epoch {epoch} finished in {epoch_duration:.2f}s"
            )

        total_time = time.time() - start_time
        safe_log(
            self.internal_logger, "info",
            f"Training finished in {total_time:.2f}s"
        )
        return final_val_results

    def _train_epoch(self, epoch: int) -> float:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        log_interval = self.cfg.get("log_interval_batches", 0)
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(self.train_loader):
            with amp_autocast(self.use_amp):
                metrics = train_step(
                    model=self.model,
                    batch=batch,
                    loss_fn=self.loss_fn,
                    optimizer=self.optimizer,
                    device=self.device,
                    metrics_dict=self.metrics_dict
                )
                batch_loss = metrics["loss"]
            # Convert loss tensor to float for accumulation
            total_loss += batch_loss.item()
            optimizer_step_with_accumulation(
                optimizer=self.optimizer,
                scaler=self.scaler,
                loss=batch_loss,
                grad_accum_steps=self.grad_accum_steps,
                batch_idx=batch_idx,
                use_amp=self.use_amp
            )
            if self.logger_instance and log_interval > 0 and \
               (batch_idx + 1) % log_interval == 0:
                global_step = (epoch - 1) * num_batches + batch_idx + 1
                self.logger_instance.log_scalar(
                    tag="train_batch/batch_loss",
                    value=batch_loss.item(),
                    step=global_step
                )
        avg_loss = total_loss / num_batches
        log_msg = (
            f"Epoch {epoch} | Average Training Loss: {avg_loss:.4f}"
        )
        safe_log(self.internal_logger, "info", log_msg)
        if self.logger_instance:
            self.logger_instance.log_scalar(
                tag="train/epoch_loss", value=avg_loss, step=epoch
            )
        return avg_loss

    def validate(self, epoch: int) -> Dict[str, float]:
        """Runs validation over the val_loader and aggregates metrics."""
        self.model.eval()
        total_metrics = {"loss": 0.0}
        total_metrics.update({name: 0.0 for name in self.metrics_dict})
        num_batches = len(self.val_loader)
        with torch.no_grad():
            for batch in self.val_loader:
                metrics = val_step(
                    model=self.model,
                    batch=batch,
                    loss_fn=self.loss_fn,
                    device=self.device,
                    metrics_dict=self.metrics_dict
                )
                for name, value in metrics.items():
                    # Convert tensor to float for accumulation
                    total_metrics[name] += (
                        value.item() if hasattr(value, 'item')
                        else float(value)
                    )
        avg_metrics = {name: total / num_batches for name,
                       total in total_metrics.items()}
        val_metrics = {}
        for name, value in avg_metrics.items():
            if name == "loss":
                val_metrics["val_loss"] = value
            else:
                val_metrics[f"val_{name}"] = value
        # Usar el helper para logging
        log_validation_results(self.internal_logger, epoch, val_metrics)
        if self.logger_instance:
            for name, value in val_metrics.items():
                self.logger_instance.log_scalar(
                    tag=f"{name}",
                    value=value,
                    step=epoch
                )
        return val_metrics

    # --- Helper Methods ---
    def _step_scheduler(self, metrics=None) -> Optional[float]:
        """Steps the learning rate scheduler using a helper utility."""
        return step_scheduler_helper(
            scheduler=self.scheduler,
            optimizer=self.optimizer,
            monitor_metric=self.monitor_metric,
            metrics=metrics,
            logger=self.internal_logger
        )

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
            safe_log(
                self.internal_logger, "warning",
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
            safe_log(
                self.internal_logger, "info",
                f"Validation metric '{metric_name}' improved from "
                f"{old_best:.4f} to {current_metric:.4f}. "
                f"Saving best checkpoint."
            )
            return True
        return False

# --- Remove old standalone functions ---
# def train_one_epoch(...): ...
# def evaluate(...): ...
