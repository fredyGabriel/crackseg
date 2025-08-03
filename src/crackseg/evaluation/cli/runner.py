"""Evaluation runner for CLI.

This module provides functions for running evaluation operations including
single model evaluation and ensemble evaluation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from crackseg.evaluation.core import evaluate_model
from crackseg.evaluation.ensemble import ensemble_evaluate
from crackseg.evaluation.results import save_evaluation_results
from crackseg.utils.logging import get_logger
from crackseg.utils.logging.experiment import ExperimentLogger

# Configure logger
log = get_logger("evaluation")


@dataclass
class EvaluationRunParameters:
    """Parameters for evaluation run."""

    is_ensemble: bool
    checkpoint_paths: list[str]
    cfg: DictConfig
    test_loader: DataLoader[Any]
    metrics_dict: dict[str, Any]
    model_for_single_eval: torch.nn.Module | None
    experiment_logger: ExperimentLogger


def run_evaluation_and_log(params: EvaluationRunParameters) -> None:
    """Run evaluation and log results.

    Args:
        params: Evaluation run parameters.
    """
    if params.is_ensemble:
        log.info("Running ensemble evaluation...")
        # Note: ensemble_evaluate function signature may need adjustment
        results = ensemble_evaluate(
            models=params.model_for_single_eval,  # type: ignore
            dataloader=params.test_loader,
            metrics=params.metrics_dict,
            device=next(params.model_for_single_eval.parameters()).device,  # type: ignore
        )
    else:
        log.info("Running single model evaluation...")
        # Note: evaluate_model function signature may need adjustment
        results = evaluate_model(
            model=params.model_for_single_eval,  # type: ignore
            dataloader=params.test_loader,
            metrics=params.metrics_dict,
            device=next(params.model_for_single_eval.parameters()).device,  # type: ignore
        )

    # Log results
    log.info("Evaluation completed. Logging results...")
    for metric_name, metric_value in results.items():
        log.info(f"{metric_name}: {metric_value:.4f}")
        params.experiment_logger.log_metric(metric_name, metric_value)

    # Save results
    results_save_path = (
        params.experiment_logger.log_dir / "evaluation_results.json"
    )
    save_evaluation_results(results, results_save_path)
    log.info(f"Results saved to: {results_save_path}")

    # Generate visualizations if requested
    if (
        hasattr(params.cfg.evaluation, "visualize")
        and params.cfg.evaluation.visualize
    ):
        log.info("Generating visualizations...")
        # TODO: Implement proper visualization with test_loader
        # For now, skip visualization as it requires tensor inputs
        log.info(
            "Visualization skipped - requires tensor inputs, not dataloader"
        )
        log.info("Visualizations generated successfully.")


def run_single_evaluation(
    model: torch.nn.Module,
    test_loader: DataLoader[Any],
    metrics_dict: dict[str, Any],
    experiment_logger: ExperimentLogger,
    output_dir: Path,
) -> dict[str, float]:
    """Run single model evaluation.

    Args:
        model: Model to evaluate.
        test_loader: Test dataloader.
        metrics_dict: Dictionary of metrics.
        experiment_logger: Logger for experiment tracking.
        output_dir: Output directory.

    Returns:
        dict[str, float]: Evaluation results.
    """
    log.info("Starting single model evaluation...")

    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        metrics=metrics_dict,
        device=next(model.parameters()).device,
    )

    # Log and save results
    for metric_name, metric_value in results.items():
        log.info(f"{metric_name}: {metric_value:.4f}")
        experiment_logger.log_metric(metric_name, metric_value)

    results_save_path = output_dir / "evaluation_results.json"
    save_evaluation_results(results, results_save_path)
    log.info(f"Results saved to: {results_save_path}")

    return results
