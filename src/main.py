# In src/main.py (Skeleton and checkpointing logic)

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import os
import math  # For inf

# Project imports
from src.utils import (
    DataError,
    ModelError,
    ResourceError,
    set_random_seeds,
    get_device,
    load_checkpoint
)
from src.data.factory import create_dataloaders_from_config  # Import factory
from src.utils.factory import get_metrics_from_cfg, get_optimizer, get_loss_fn
from src.model.factory import create_unet
from src.training.factory import create_lr_scheduler
from src.utils.experiment import initialize_experiment
from src.training.trainer import Trainer

# Configure standard logger
log = logging.getLogger(__name__)


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
            log.error("CUDA is required but not available on this system.")
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
            # Obtener la configuración de transformaciones de Hydra
            # Puede estar en cfg.data.transform o cfg["data/transform"]
            transform_cfg = None
            if hasattr(cfg.data, "transform"):
                transform_cfg = cfg.data.transform
            elif "data/transform" in cfg:
                transform_cfg = cfg["data/transform"]
            else:
                log.warning(
                    "Transform config not found in Hydra config. "
                    "Using empty config."
                )
                transform_cfg = OmegaConf.create({})
            # Asegurar que data_root es absoluto
            orig_cwd = hydra.utils.get_original_cwd()
            data_root = os.path.join(
                orig_cwd,
                data_cfg.get("data_root", "data/")
            )
            data_cfg["data_root"] = data_root

            # Cargar la configuración específica del dataloader
            dataloader_cfg = None
            if hasattr(cfg.data, "dataloader"):
                dataloader_cfg = cfg.data.dataloader
            elif "data/dataloader" in cfg:
                dataloader_cfg = cfg["data/dataloader"]
            else:
                log.warning(
                    "Dataloader config not found in Hydra config. "
                    "Using data config as fallback."
                )
                dataloader_cfg = data_cfg

            # Depuración de configuración
            print("DEBUG - Hydra dataloader_cfg antes de conversión:")
            print(f"  Tipo: {type(dataloader_cfg)}")
            if hasattr(dataloader_cfg, "max_train_samples"):
                print(
                    f"  max_train_samples directamente: "
                    f"{dataloader_cfg.max_train_samples}"
                )
            elif (isinstance(dataloader_cfg, dict) and
                  "max_train_samples" in dataloader_cfg):
                print(
                    f"  max_train_samples en dict: "
                    f"{dataloader_cfg['max_train_samples']}"
                )
            else:
                print("  max_train_samples no encontrado en la configuración")

            dataloaders_dict = create_dataloaders_from_config(
                data_config=data_cfg,
                transform_config=transform_cfg,
                dataloader_config=dataloader_cfg
            )
            train_loader = dataloaders_dict.get('train', {}).get('dataloader')
            val_loader = dataloaders_dict.get('val', {}).get('dataloader')
            if not train_loader or not val_loader:
                log.error(
                    "Train or validation dataloader could not be created."
                )
                raise DataError(
                    "Train or validation dataloader could not be created."
                )
            log.info("Data loading complete.")
        except Exception as e:
            log.error(
                f"Error during data loading: {str(e)}"
            )
            raise DataError(
                f"Error during data loading: {str(e)}"
            ) from e

        # --- 3. Model Creation ---
        log.info("Creating model...")
        try:
            # Create model using the factory function
            model = create_unet(cfg._group_)
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

        # Get ExperimentManager instance from the logger
        experiment_manager = experiment_logger.experiment_manager

        # Use the checkpoints directory from the experiment manager
        checkpoint_dir = str(experiment_manager.get_path("checkpoints"))
        log.info(f"Using checkpoint directory: {checkpoint_dir}")

        # Get checkpoint configuration
        checkpoint_cfg = cfg.training.get("checkpoints", OmegaConf.create({}))
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
                best_metric_value = checkpoint_data.get(
                    "best_metric_value", None
                )
                msg = (
                    f"Resumed from epoch {start_epoch}. "
                    f"Best: {best_metric_value}"
                )
                log.info(msg)
            else:
                msg = (
                    f"Resume checkpoint not found: {resume_path}. "
                    f"Fresh start."
                )
                log.warning(msg)
        else:
            log.info(
                "No checkpoint specified for resume, starting from scratch."
            )

        # --- Setup Best Model Tracking ---
        save_best_cfg = checkpoint_cfg.get("save_best", OmegaConf.create({}))
        monitor_metric = save_best_cfg.get("monitor_metric", None)
        monitor_mode = save_best_cfg.get("monitor_mode", "max")
        save_best_enabled = save_best_cfg.get("enabled", False)

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

        # --- 6. Training Loop (delegado a Trainer) ---
        # Refactor: Usar Trainer para manejar el entrenamiento y validación
        trainer = Trainer(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        loss_fn=loss_fn,
                        metrics_dict=metrics,
                        cfg=cfg,
                        logger_instance=experiment_logger
        )
        trainer.train()

        # --- 7. Final Evaluation ---
        log.info(
            "Final evaluation removed from main.py. "
            "Use evaluate.py for evaluation."
        )

    except Exception as e:
        # Log and properly handle the error
        if experiment_logger:
            experiment_logger.log_error(exception=e, context="Main execution")
            experiment_logger.close()

        raise e  # Re-raise to let Hydra handle it

    finally:
        # --- 8. Cleanup ---
        if experiment_logger:
            experiment_logger.close()
        log.info("Main execution finished.")


if __name__ == "__main__":
    main()
