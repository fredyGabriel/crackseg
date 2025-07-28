#!/usr/bin/env python
"""
Evaluation script for trained crack segmentation models.

This script loads a trained model checkpoint and evaluates it on a provided
test dataset, generating comprehensive metrics and visualizations of the
results.

Usage:
    python -m src.evaluation --checkpoint /path/to/checkpoint.pth.tar --config
    /path/to/config.yaml
"""

import argparse  # Added for type hinting
import os
import sys
from dataclasses import dataclass  # Added dataclass
from pathlib import Path
from typing import Any  # Added for type hinting

import torch
import yaml
from omegaconf import OmegaConf
from omegaconf import errors as omegaconf_errors
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader

from crackseg.evaluation.core import evaluate_model  # type: ignore
from crackseg.evaluation.data import get_evaluation_dataloader
from crackseg.evaluation.ensemble import ensemble_evaluate
from crackseg.evaluation.loading import load_model_from_checkpoint
from crackseg.evaluation.results import save_evaluation_results

# Evaluation module imports
from crackseg.evaluation.setup import parse_args, setup_output_directory
from crackseg.training.metrics import F1Score, IoUScore

# Project imports
from crackseg.utils import get_device, set_random_seeds
from crackseg.utils.exceptions import ConfigError
from crackseg.utils.factory import get_metrics_from_cfg
from crackseg.utils.logging import get_logger
from crackseg.utils.logging.experiment import ExperimentLogger
from crackseg.utils.visualization import visualize_predictions

# Configure logger
log = get_logger("evaluation")


def _setup_evaluation_environment(
    args: argparse.Namespace,
) -> tuple[torch.device, Path, list[str]]:
    """Sets up the evaluation environment: seed, device, output dir,
    checkpoints."""
    set_random_seeds(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    log.info(f"Using device: {device}")

    output_dir_str = setup_output_directory(args.output_dir)
    output_dir = Path(output_dir_str)
    log.info(f"Output directory set to: {output_dir}")

    checkpoint_paths = (
        [cp.strip() for cp in args.checkpoint.split(",")]
        if args.ensemble
        else [args.checkpoint]
    )
    for cp_path in checkpoint_paths:
        if not os.path.exists(cp_path):
            raise FileNotFoundError(f"Checkpoint file not found: {cp_path}")
        log.info(f"Found checkpoint: {cp_path}")
    return device, output_dir, checkpoint_paths


def _load_and_prepare_config(
    args: argparse.Namespace, checkpoint_data: dict[str, Any], output_dir: Path
) -> DictConfig:
    """Loads, validates, overrides, and saves the configuration."""
    cfg = None
    if args.config:
        log.info(f"Loading configuration from: {args.config}")
        cfg = OmegaConf.load(args.config)
    elif "config" in checkpoint_data:
        log.info("Loading configuration from checkpoint.")
        cfg = checkpoint_data["config"]
        if isinstance(cfg, dict):  # Ensure it's an OmegaConf object
            cfg = OmegaConf.create(cfg)
    else:
        log.error(
            "No configuration found in checkpoint or provided via --config "
            "argument."
        )
        raise ConfigError("Missing model configuration.")

    if not isinstance(cfg, DictConfig):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        else:
            msg = "Loaded configuration is not a dictionary or DictConfig. "
            msg += f"Type: {type(cfg)}"
            log.error(msg)
            raise ConfigError(msg)

    if args.data_dir:
        log.info(f"Overriding data_dir with: {args.data_dir}")
        OmegaConf.update(cfg, "data.data_root", args.data_dir, merge=True)
    if args.batch_size is not None:
        log.info(f"Overriding batch_size with: {args.batch_size}")
        OmegaConf.update(
            cfg, "data.dataloader.batch_size", args.batch_size, merge=True
        )
    if args.num_workers is not None:
        log.info(f"Overriding num_workers with: {args.num_workers}")
        OmegaConf.update(
            cfg, "data.dataloader.num_workers", args.num_workers, merge=True
        )
    if args.visualize_samples is not None:  # Added for visualize_samples
        log.info(
            f"Overriding visualize_samples with: {args.visualize_samples}"
        )
        # Ensure the path in cfg exists or handle potential errors if it might
        # not.
        # For simplicity, assuming cfg.evaluation structure exists or is
        # created by Hydra defaults.
        OmegaConf.update(
            cfg,
            "evaluation.visualize_samples",
            args.visualize_samples,
            merge=True,
        )

    effective_config_path = output_dir / "effective_config.yaml"
    OmegaConf.save(config=cfg, f=str(effective_config_path))
    log.info(f"Effective configuration saved to: {effective_config_path}")
    return cfg


def _get_evaluation_components(
    cfg: DictConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[DataLoader[Any], dict[str, Any], ExperimentLogger]:
    """Gets dataloader, metrics, and experiment logger."""
    test_loader = get_evaluation_dataloader(
        config=cfg,
        # These might be redundant if cfg is already updated
        # but provides a clear pass-through if needed.
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    metrics = {}
    if hasattr(cfg, "evaluation") and hasattr(cfg.evaluation, "metrics"):
        try:
            metrics = get_metrics_from_cfg(cfg.evaluation.metrics)
            log.info(f"Loaded metrics: {list(metrics.keys())}")
        except (
            ConfigError,
            ImportError,
            AttributeError,
            KeyError,
            omegaconf_errors.OmegaConfBaseException,
        ) as e:
            log.error(f"Error loading metrics: {e}")
            metrics = {"dice": F1Score(), "iou": IoUScore()}
            log.info("Using default metrics: dice, iou")
    else:
        metrics = {"dice": F1Score(), "iou": IoUScore()}
        log.info("Using default metrics: dice, iou")

    experiment_logger = ExperimentLogger(
        log_dir=output_dir,
        experiment_name="evaluation",
        config=cfg,
        log_level="INFO",
        log_to_file=True,
    )

    # The effective_config.yaml is already saved by _load_and_prepare_config.
    # Saving another "config.yaml" might be redundant unless it serves a
    # different purpose.
    # For now, I'll keep it as it was in the original logic.
    config_path = output_dir / "config.yaml"  # Use Path object
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            OmegaConf.to_container(cfg, resolve=True),
            f,
            default_flow_style=False,
        )
    log.info(f"Configuration (possibly redundant) saved to: {config_path}")
    return test_loader, metrics, experiment_logger


@dataclass
class EvaluationRunParameters:
    is_ensemble: bool
    checkpoint_paths: list[str]
    cfg: DictConfig
    test_loader: DataLoader[Any]
    metrics_dict: dict[str, Any]
    model_for_single_eval: torch.nn.Module | None
    experiment_logger: ExperimentLogger


def _run_evaluation_and_log(params: EvaluationRunParameters) -> None:
    """Runs evaluation (single or ensemble) and logs results."""
    # Derive device, output_dir, and visualize_samples_count from params.cfg
    device = torch.device(params.cfg.device_str)
    output_dir = Path(params.cfg.output_dir_str)
    visualize_samples_count = params.cfg.evaluation.visualize_samples

    if params.is_ensemble and len(params.checkpoint_paths) > 1:
        ensemble_results = ensemble_evaluate(
            checkpoint_paths=params.checkpoint_paths,
            # ensemble_evaluate uses cfg for device & output_dir
            config=params.cfg,
            dataloader=params.test_loader,
            metrics=params.metrics_dict,
        )
        log.info("Ensemble Evaluation Results:")
        for metric_name, metric_value in ensemble_results.items():
            log.info(f"  {metric_name}: {metric_value}")
            params.experiment_logger.log_scalar(
                f"test/{metric_name}", metric_value, 0
            )
    else:
        if params.model_for_single_eval is None:
            log.error(
                "Model is None for single model evaluation. This should not "
                "happen."
            )
            raise ValueError(
                "Model cannot be None for single model evaluation."
            )

        current_checkpoint_path = params.checkpoint_paths[
            0
        ]  # For single model, there's one path
        log.info(
            f"Evaluating model from checkpoint: {current_checkpoint_path}"
        )
        results, (inputs, targets, outputs) = evaluate_model(
            model=params.model_for_single_eval,
            dataloader=params.test_loader,
            metrics=params.metrics_dict,
            device=device,
            config=params.cfg,
        )
        log.info("Evaluation Results:")
        for metric_name, metric_value in results.items():
            log.info(f"  {metric_name}: {metric_value}")
            params.experiment_logger.log_scalar(
                f"test/{metric_name.replace('test_', '')}", metric_value, 0
            )

        try:
            visualize_predictions(
                inputs=inputs,
                targets=targets,
                outputs=outputs,
                output_dir=str(
                    output_dir / "visualizations"
                ),  # Use derived output_dir
                num_samples=visualize_samples_count,  # Use derived count
            )
            log.info(
                f"Visualizations saved to: {output_dir / 'visualizations'}"
            )
        except Exception as e:
            log.error(f"Error during visualization: {e}", exc_info=True)

        save_evaluation_results(
            results=results,
            config=params.cfg,
            checkpoint_path=current_checkpoint_path,
            output_dir=str(output_dir / "metrics"),  # Use derived output_dir
        )


def main():
    """Main entry point for model evaluation."""
    args = parse_args()

    try:
        device, output_dir, checkpoint_paths = _setup_evaluation_environment(
            args
        )

        # Load first model for config and for single model evaluation case
        # Note: ensemble_evaluate re-loads all models internally based on its
        # config.
        model, checkpoint_data = load_model_from_checkpoint(
            checkpoint_paths[0], device
        )

        cfg = _load_and_prepare_config(args, checkpoint_data, output_dir)

        # Pass necessary parts of cfg for device and output_dir to
        # ensemble_evaluate
        # This aligns with how ensemble_evaluate was refactored.
        # Create a temporary config or ensure cfg directly has these.
        # For ensemble_evaluate, it expects 'device_str' and 'output_dir_str'
        # in the config.
        # Let's ensure they are present if not already.
        # This part is crucial for the interface with the refactored
        # ensemble_evaluate.
        if not hasattr(cfg, "device_str"):
            OmegaConf.update(cfg, "device_str", str(device), merge=True)
        if not hasattr(cfg, "output_dir_str"):
            OmegaConf.update(
                cfg, "output_dir_str", str(output_dir), merge=True
            )

        test_loader, metrics_dict, experiment_logger = (
            _get_evaluation_components(cfg, args, output_dir)
        )

        eval_params = EvaluationRunParameters(
            is_ensemble=args.ensemble,
            checkpoint_paths=checkpoint_paths,
            cfg=cfg,
            test_loader=test_loader,
            metrics_dict=metrics_dict,
            model_for_single_eval=model,
            experiment_logger=experiment_logger,
        )

        _run_evaluation_and_log(eval_params)

        experiment_logger.close()
        log.info(f"Evaluation complete. Results saved to: {output_dir}")

        # The final saving of 'evaluation_config.yaml' can be kept here
        # or moved into _finalize_script if we add such a function.
        # For now, keeping it simple.
        with open(
            output_dir / "evaluation_config.yaml", "w", encoding="utf-8"
        ) as f:
            OmegaConf.save(config=cfg, f=f)
        log.info(f"Final evaluation_config.yaml saved to {output_dir}")

    except FileNotFoundError as e_fnf:
        log.exception("File not found during evaluation: %s", str(e_fnf))
        sys.exit(1)
    except (
        ConfigError,
        yaml.YAMLError,
        omegaconf_errors.OmegaConfBaseException,
    ) as e_cfg:
        log.exception("Configuration error during evaluation: %s", str(e_cfg))
        sys.exit(1)
    except OSError as e_io:
        log.exception("I/O error during evaluation: %s", str(e_io))
        sys.exit(1)
    except (AttributeError, TypeError, ValueError, RuntimeError) as e_runtime:
        log.exception("Runtime error during evaluation: %s", str(e_runtime))
        sys.exit(1)
    except Exception as e_general:
        log.exception("Unexpected error during evaluation: %s", str(e_general))
        sys.exit(1)


if __name__ == "__main__":
    main()
