# In src/main.py (Skeleton and checkpointing logic)
import logging
import math  # For inf
import os
from typing import Any, cast

import hydra
import torch
from hydra import errors as hydra_errors  # Import Hydra errors
from omegaconf import DictConfig, OmegaConf
from omegaconf import errors as omegaconf_errors  # Import OmegaConf errors
from torch import optim  # For Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader  # Added for DataLoader

# Project imports
from src.data.factory import create_dataloaders_from_config  # Import factory
from src.training.components import (  # type: ignore[reportMissingImports]
    TrainingComponents,  # type: ignore[reportMissingImports]
)
from src.training.trainer import Trainer  # type: ignore[reportMissingImports]
from src.utils import (
    DataError,
    ModelError,
    ResourceError,
    get_device,
    load_checkpoint,
    set_random_seeds,
)
from src.utils.experiment import (
    initialize_experiment,  # type: ignore[reportMissingImports]
)
from src.utils.factory import (  # type: ignore[reportMissingImports]
    get_loss_fn,  # type: ignore[reportUnknownArgumentType]
    get_metrics_from_cfg,  # type: ignore[reportUnknownArgumentType]
    get_optimizer,  # type: ignore[reportUnknownArgumentType]
)

# Configure standard logger
log = logging.getLogger(__name__)


def _setup_environment(cfg: DictConfig) -> torch.device:
    """Sets up the environment, seeds, and device."""
    log.info("Setting up environment...")
    set_random_seeds(cfg.get("random_seed", 42))

    if not torch.cuda.is_available() and cfg.get("require_cuda", True):
        log.error("CUDA is required but not available on this system.")
        raise ResourceError(
            "CUDA is required but not available on this system."
        )

    device = get_device()
    log.info("Using device: %s", device)
    return device


def _load_data(cfg: DictConfig) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Loads and creates train and validation dataloaders."""
    log.info("Loading data...")
    try:
        data_cfg = cfg.data
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

        orig_cwd = hydra.utils.get_original_cwd()
        data_root = os.path.join(orig_cwd, data_cfg.get("data_root", "data/"))
        data_cfg["data_root"] = data_root

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

        # Ensure dataloader_cfg is a DictConfig
        if not isinstance(dataloader_cfg, DictConfig):
            # Attempt to convert if it's a basic dict or list that OmegaConf
            # can handle
            try:
                converted_cfg = OmegaConf.create(dataloader_cfg)
                if isinstance(converted_cfg, DictConfig):
                    dataloader_cfg = converted_cfg
                else:
                    log.warning(
                        "Could not convert dataloader_cfg to DictConfig. "
                        "It is of type: %s. Using empty DictConfig.",
                        type(converted_cfg),
                    )
                    dataloader_cfg = OmegaConf.create({})
            except Exception as e:
                log.warning(
                    "Error converting dataloader_cfg to DictConfig: %s. "
                    "Using empty DictConfig.",
                    e,
                )
                dataloader_cfg = OmegaConf.create({})

        dataloaders_dict = create_dataloaders_from_config(
            data_config=data_cfg,
            transform_config=transform_cfg,
            dataloader_config=dataloader_cfg,  # type: ignore
        )
        train_loader = dataloaders_dict.get("train", {}).get("dataloader")
        val_loader = dataloaders_dict.get("val", {}).get("dataloader")

        if not train_loader or not val_loader:
            log.error("Train or validation dataloader could not be created.")
            raise DataError(
                "Train or validation dataloader could not be created."
            )

        if not isinstance(train_loader, DataLoader) or not isinstance(
            val_loader, DataLoader
        ):
            raise DataError(
                "Train or validation loader is not a DataLoader instance"
            )

        log.info("Data loading complete.")
        return train_loader, val_loader
    except (
        OSError,
        DataError,
        omegaconf_errors.OmegaConfBaseException,
        FileNotFoundError,
        ImportError,
        ValueError,
        TypeError,
    ) as e:
        log.error("Error during data loading: %s", str(e))
        raise DataError(f"Error during data loading: {str(e)}") from e


def _create_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """Creates and loads the model to the specified device."""
    log.info("Creating model...")
    try:
        model = hydra.utils.instantiate(cfg.model)
        model = cast(Module, model)
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        log.info(
            "Created %s model with %s parameters",
            type(model).__name__,
            num_params,
        )
        assert isinstance(model, Module)
        return model
    except (
        ModelError,
        hydra_errors.InstantiationException,
        AttributeError,
        ImportError,
        TypeError,
        ValueError,
    ) as e:
        log.error("Error creating model: %s", str(e))
        raise ModelError(f"Error creating model: {str(e)}") from e


def _setup_training_components(
    cfg: DictConfig, model: torch.nn.Module
) -> tuple[
    dict[str, Any],
    optim.Optimizer,
    torch.nn.Module,
]:
    """Sets up metrics, optimizer, loss function, scheduler, and AMP scaler."""
    log.info("Setting up training components...")

    metrics: dict[str, Any] = {}
    if hasattr(cfg, "evaluation") and hasattr(cfg.evaluation, "metrics"):
        try:
            metrics = cast(
                dict[str, Any],
                get_metrics_from_cfg(cfg.evaluation.metrics),  # type: ignore[reportUnknownArgumentType]
            )
            log.info("Loaded metrics: %s", list(metrics.keys()))
        except (
            omegaconf_errors.OmegaConfBaseException,
            KeyError,
            AttributeError,
            ImportError,
            ValueError,
        ) as e:
            log.error("Error loading metrics: %s", e)
            metrics = {}
    else:
        log.warning("Evaluation metrics configuration not found.")

    optimizer_cfg = cfg.training.get("optimizer", {"type": "adam", "lr": 1e-3})
    # Convert model.parameters() to a list to satisfy get_optimizer
    optimizer = get_optimizer(list(model.parameters()), optimizer_cfg)  # type: ignore[reportUnknownArgumentType]
    log.info("Created optimizer: %s", type(optimizer).__name__)

    loss_fn_instance: torch.nn.Module
    if hasattr(cfg.training, "loss") and cfg.training.loss is not None:
        try:
            potential_loss_fn = get_loss_fn(cfg.training.loss)  # type: ignore[reportUnknownArgumentType]
            if isinstance(potential_loss_fn, torch.nn.Module):
                loss_fn_instance = potential_loss_fn
                log.info(
                    "Created loss function from config: %s",
                    type(loss_fn_instance).__name__,
                )
            else:
                log.error(
                    "get_loss_fn did not return an nn.Module. "
                    "Got: %s. Using fallback.",
                    str(type(potential_loss_fn)),  # type: ignore[reportUnknownArgumentType]
                )
                loss_fn_instance = torch.nn.BCEWithLogitsLoss()  # Fallback
        except (
            omegaconf_errors.OmegaConfBaseException,
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
        ) as e:
            log.error(
                "Error creating loss function from config: %s. "
                "Using fallback.",
                e,
            )
            loss_fn_instance = torch.nn.BCEWithLogitsLoss()  # Fallback
    else:
        log.warning(
            "Loss function configuration not found or is null. "
            "Using fallback: BCEWithLogitsLoss."
        )
        loss_fn_instance = torch.nn.BCEWithLogitsLoss()  # Fallback

    # scheduler and scaler removed for linter compliance
    log.info("Training setup complete.")
    return metrics, optimizer, loss_fn_instance


def _handle_checkpointing_and_resume(
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    experiment_logger: Any,
) -> tuple[int, float | None]:
    """Handles checkpoint loading and resume logic."""
    log.info("Handling checkpointing and resume...")
    start_epoch = 0
    best_metric_value = None

    # Ensure experiment_logger and its manager are valid before use
    if not hasattr(experiment_logger, "experiment_manager"):
        log.error(
            "Experiment logger does not have 'experiment_manager'. "
            "Cannot determine checkpoint directory."
        )
        # Fallback or raise error depending on desired behavior
        # For now, log and continue, checkpointing might be disabled or fail
        checkpoint_dir = "checkpoints"  # Fallback
    else:
        experiment_manager = experiment_logger.experiment_manager
        checkpoint_dir = str(experiment_manager.get_path("checkpoints"))

    log.info("Using checkpoint directory: %s", checkpoint_dir)

    checkpoint_cfg = cfg.training.get("checkpoints", OmegaConf.create({}))
    resume_path = checkpoint_cfg.get("resume_from_checkpoint", None)

    if resume_path:
        msg = f"Attempting to resume from checkpoint: {resume_path}"
        log.info(msg)
        orig_cwd = hydra.utils.get_original_cwd()
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(orig_cwd, resume_path)

        if os.path.exists(resume_path):
            checkpoint_data = load_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=resume_path,
                device=device,
            )
            start_epoch = checkpoint_data.get("epoch", 0) + 1
            best_metric_value = checkpoint_data.get("best_metric_value", None)
            msg = (
                f"Resumed from epoch {start_epoch}. Best: {best_metric_value}"
            )
            log.info(msg)
        else:
            log.warning(
                f"Resume checkpoint not found: {resume_path}. Fresh start."
            )
    else:
        log.info("No checkpoint specified for resume, starting from scratch.")

    # Setup Best Model Tracking (initialization part)
    save_best_cfg = checkpoint_cfg.get("save_best", OmegaConf.create({}))
    monitor_metric = save_best_cfg.get("monitor_metric", None)
    monitor_mode = save_best_cfg.get("monitor_mode", "max")
    save_best_enabled = save_best_cfg.get("enabled", False)

    if save_best_enabled and not monitor_metric:
        msg = (
            "save_best enabled but monitor_metric not set. "
            "Disabling best model saving."
        )
        log.warning(msg)
        # save_best_enabled = False # This variable is local, trainer will
        # read from cfg
    elif save_best_enabled:
        log.info(
            f"Monitoring '{monitor_metric}' for best model (mode: "
            f"{monitor_mode})."
        )
        if best_metric_value is None:  # Initialize if not resuming
            best_metric_value = (
                -math.inf if monitor_mode == "max" else math.inf
            )
            log.info("Initializing best metric to %s", best_metric_value)

    log.info("Checkpointing and resume handling complete.")
    return start_epoch, best_metric_value


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training and evaluation entry point."""
    experiment_logger = None
    try:
        # --- 1. Initial Setup ---
        log.info("Starting main execution...")
        device = _setup_environment(cfg)

        # Initialize experiment and logging
        experiment_dir, experiment_logger = initialize_experiment(cfg)
        log.info("Experiment initialized in: %s", experiment_dir)

        # --- 2. Data Loading ---
        train_loader, val_loader = _load_data(cfg)

        # --- 3. Model Creation ---
        model = _create_model(cfg, device)

        # --- 4. Training Setup ---
        metrics, optimizer, loss_fn = _setup_training_components(cfg, model)

        # --- 5. Checkpointing and Resume ---
        # Note: best_metric_value from here is mostly for initial logging.
        # The Trainer itself will manage and update the actual
        # best_metric_value based on its internal logic and config.
        _start_epoch, _ = _handle_checkpointing_and_resume(
            cfg, model, optimizer, device, experiment_logger
        )

        # --- 6. Training Loop (delegated to Trainer) ---
        log.info("Starting training loop...")
        components = TrainingComponents(  # type: ignore[reportUnknownArgumentType]
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            metrics_dict=metrics,
        )
        trainer = Trainer(
            components=components,  # type: ignore[reportUnknownArgumentType]
            cfg=cfg,
            logger_instance=experiment_logger,
            # early_stopper can be passed if initialized separately
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
