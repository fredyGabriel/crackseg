#!/usr/bin/env python
"""
Evaluation script for trained crack segmentation models.

This script loads a trained model checkpoint and evaluates it on a provided
test dataset, generating comprehensive metrics and visualizations of the
results.

Usage:
    python -m src.evaluate --checkpoint /path/to/checkpoint.pth.tar --config
    /path/to/config.yaml
"""

import os
import sys
import yaml
import torch
from omegaconf import OmegaConf

# Project imports
from src.utils import set_random_seeds, get_device
from src.utils.exceptions import ConfigError
from src.utils.logging import get_logger
from src.utils.logging.experiment import ExperimentLogger
from src.utils.factory import get_metrics_from_cfg

# Evaluation module imports
from src.evaluation.setup import parse_args, setup_output_directory
from src.evaluation.data import get_evaluation_dataloader
from src.evaluation.core import evaluate_model
from src.evaluation.results import save_evaluation_results
from src.evaluation.loading import load_model_from_checkpoint
from src.evaluation.ensemble import ensemble_evaluate
from src.utils.visualization import visualize_predictions

# Configure logger
log = get_logger("evaluation")


def main():
    """Main entry point for model evaluation."""
    # Parse command-line arguments
    args = parse_args()

    try:
        # Set random seed for reproducibility
        set_random_seeds(args.seed)

        # Determine device
        if args.device:
            device = torch.device(args.device)
        else:
            device = get_device()
        log.info(f"Using device: {device}")

        # Split checkpoint paths if ensemble mode
        checkpoint_paths = ([cp.strip() for cp in args.checkpoint.split(",")]
                            if args.ensemble else [args.checkpoint])

        # Ensure checkpoint files exist
        for cp_path in checkpoint_paths:
            if not os.path.exists(cp_path):
                raise FileNotFoundError(f"Checkpoint file not found: {cp_path}"
                                        )
            log.info(f"Found checkpoint: {cp_path}")

        # Load first model and configuration
        model, checkpoint_data = load_model_from_checkpoint(
            checkpoint_paths[0], device)

        # Extract config from checkpoint or load from file
        config = checkpoint_data.get('config')

        if config is None and args.config:
            # Load config from file if not in checkpoint
            if args.config.endswith('.yaml'):
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                config = OmegaConf.load(args.config)
            log.info(f"Loaded configuration from: {args.config}")

        if config is None:
            raise ConfigError(
                "Configuration not found in checkpoint and not provided with \
--config"
            )

        # Convert to OmegaConf if it's a dict
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        # Setup output directory
        output_dir = setup_output_directory(args.output_dir)

        # Get dataloader for evaluation
        test_loader = get_evaluation_dataloader(
            config=config,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # Get evaluation metrics
        metrics = {}
        if hasattr(config, 'evaluation') and hasattr(config.evaluation,
                                                     'metrics'):
            try:
                metrics = get_metrics_from_cfg(config.evaluation.metrics)
                log.info(f"Loaded metrics: {list(metrics.keys())}")
            except Exception as e:
                log.error(f"Error loading metrics: {e}")
                # Load default metrics
                from src.training.metrics import IoUScore, F1Score
                metrics = {
                    'dice': F1Score(),
                    'iou': IoUScore()
                }
                log.info("Using default metrics: dice, iou")
        else:
            # Load default metrics
            from src.training.metrics import IoUScore, F1Score
            metrics = {
                'dice': F1Score(),
                'iou': IoUScore()
            }
            log.info("Using default metrics: dice, iou")

        # Create experiment logger
        experiment_logger = ExperimentLogger(
            log_dir=output_dir,
            experiment_name="evaluation",
            config=config,
            log_level='INFO',
            log_to_file=True
        )

        # Save config to output directory
        config_path = os.path.join(output_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(OmegaConf.to_container(config, resolve=True), f,
                      default_flow_style=False)
        log.info(f"Configuration saved to: {config_path}")

        # Perform evaluation
        if args.ensemble and len(checkpoint_paths) > 1:
            # Ensemble evaluation with multiple checkpoints
            ensemble_results = ensemble_evaluate(
                checkpoint_paths=checkpoint_paths,
                config=config,
                dataloader=test_loader,
                metrics=metrics,
                device=device,
                output_dir=output_dir
            )

            # Log ensemble results
            log.info("Ensemble Evaluation Results:")
            for metric_name, metric_value in ensemble_results.items():
                log.info(f"  {metric_name}: {metric_value:.4f}")
                experiment_logger.log_scalar(f"test/{metric_name}",
                                             metric_value, 0)

        else:
            # Single model evaluation
            log.info(f"Evaluating model from checkpoint: {args.checkpoint}")
            results, (inputs, targets, outputs) = evaluate_model(
                model=model,
                dataloader=test_loader,
                metrics=metrics,
                device=device
            )

            # Log results
            log.info("Evaluation Results:")
            for metric_name, metric_value in results.items():
                log.info(f"  {metric_name}: {metric_value:.4f}")
                experiment_logger.log_scalar(
                    f"test/{metric_name.replace('test_', '')}",
                    metric_value,
                    0
                )

            # Visualize predictions
            visualize_predictions(
                inputs=inputs,
                targets=targets,
                outputs=outputs,
                output_dir=output_dir,
                num_samples=args.visualize_samples
            )

            # Save evaluation results
            save_evaluation_results(
                results=results,
                config=config,
                checkpoint_path=args.checkpoint,
                output_dir=output_dir
            )

        # Close experiment logger
        experiment_logger.close()

        log.info(f"Evaluation complete. Results saved to: {output_dir}")

    except Exception as e:
        log.exception(f"Error during evaluation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
