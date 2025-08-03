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

import sys

from crackseg.evaluation.cli.components import (
    get_evaluation_components,
    load_models_for_evaluation,
)
from crackseg.evaluation.cli.config import (
    load_and_prepare_config,
    validate_evaluation_config,
)
from crackseg.evaluation.cli.environment import setup_evaluation_environment
from crackseg.evaluation.cli.runner import (
    EvaluationRunParameters,
    run_evaluation_and_log,
)
from crackseg.evaluation.loading import load_model_from_checkpoint
from crackseg.evaluation.setup import parse_args
from crackseg.utils.logging import get_logger

# Configure logger
log = get_logger("evaluation")


def main() -> None:
    """Main evaluation function."""
    try:
        # Parse command line arguments
        args = parse_args()

        # Setup evaluation environment
        device, output_dir, checkpoint_paths = setup_evaluation_environment(
            args
        )

        # Load checkpoint data
        log.info(f"Loading checkpoint from: {checkpoint_paths[0]}")
        checkpoint_data = load_model_from_checkpoint(
            checkpoint_paths[0], device, return_data=True
        )

        # Load and prepare configuration
        cfg = load_and_prepare_config(args, checkpoint_data, output_dir)
        validate_evaluation_config(cfg)

        # Get evaluation components
        test_loader, metrics_dict, experiment_logger = (
            get_evaluation_components(cfg, args, output_dir)
        )

        # Load models for evaluation
        if args.ensemble:
            models = load_models_for_evaluation(checkpoint_paths, device)
            model_for_single_eval = models[
                0
            ]  # Use first model for compatibility
        else:
            model_for_single_eval = load_model_from_checkpoint(
                checkpoint_paths[0], device
            )
            models = [model_for_single_eval]

        # Prepare evaluation parameters
        params = EvaluationRunParameters(
            is_ensemble=args.ensemble,
            checkpoint_paths=checkpoint_paths,
            cfg=cfg,
            test_loader=test_loader,
            metrics_dict=metrics_dict,
            model_for_single_eval=model_for_single_eval,
            experiment_logger=experiment_logger,
        )

        # Run evaluation
        run_evaluation_and_log(params)

        log.info("Evaluation completed successfully!")

    except Exception as e:
        log.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
