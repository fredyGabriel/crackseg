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
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union

import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

# Project imports
from src.utils import set_random_seeds, get_device
from src.utils.exceptions import ConfigError, DataError, ModelError, \
    EvaluationError
from src.utils.logging import get_logger, ExperimentLogger
from src.data.factory import create_dataloaders_from_config
from src.model.factory import create_unet
from src.utils.factory import get_metrics_from_cfg

# Configure logger
log = get_logger("evaluation")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file (if not stored in checkpoint)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to test dataset (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results \
(default: ./outputs/evaluation/TIMESTAMP)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (overrides config)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers for data loading (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Enable ensemble evaluation with multiple checkpoints"
    )
    parser.add_argument(
        "--visualize_samples",
        type=int,
        default=5,
        help="Number of samples to visualize"
    )
    return parser.parse_args()


def setup_output_directory(base_dir: Optional[str] = None) -> str:
    """
    Create output directory for evaluation results.

    Args:
        base_dir: Base directory path (if None, uses ./outputs/evaluation/)

    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if base_dir is None:
        base_dir = os.path.join("outputs", "evaluation")

    output_dir = os.path.join(base_dir, timestamp)

    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

    log.info(f"Evaluation results will be saved to: {output_dir}")
    return output_dir


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model onto

    Returns:
        Tuple containing:
        - model: The loaded model
        - checkpoint_data: Additional data from the checkpoint
    """
    # Load checkpoint into memory first
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # Extract config from checkpoint if available
    if 'config' not in checkpoint:
        raise EvaluationError(
            f"Checkpoint at {checkpoint_path} does not contain model \
configuration. "
            "Please provide a configuration file with --config."
        )

    config = checkpoint['config']

    if isinstance(config, dict):
        config = OmegaConf.create(config)

    # Create model with the same architecture
    try:
        model = create_unet(config.model)
        model.to(device)
        log.info(f"Created model: {type(model).__name__}")
    except Exception as e:
        raise ModelError(f"Error creating model: {str(e)}") from e

    # Load weights into model
    # Note: We don't pass the model to load_checkpoint here because we already
    # have the checkpoint
    # loaded and just need to apply the state_dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        log.info(f"Model weights loaded from: {checkpoint_path}")
    except Exception as e:
        raise ModelError(f"Error loading model weights: {str(e)}") from e

    # Set model to evaluation mode
    model.eval()

    # Return model and checkpoint data
    checkpoint_data = {
        k: v for k, v in checkpoint.items()
        if k not in ['model_state_dict', 'optimizer_state_dict']
    }

    return model, checkpoint_data


def get_evaluation_dataloader(
    config: Union[Dict[str, Any], DictConfig],
    data_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None
) -> DataLoader:
    """
    Get dataloader for evaluation.

    Args:
        config: Configuration dictionary or DictConfig
        data_dir: Path to data directory (overrides config)
        batch_size: Batch size (overrides config)
        num_workers: Number of workers (overrides config)

    Returns:
        DataLoader for evaluation
    """
    # Clone config to avoid modifying the original
    if isinstance(config, DictConfig):
        data_config = OmegaConf.create(
            OmegaConf.to_container(config.data, resolve=True)
        )
    else:
        data_config = OmegaConf.create(config['data'])

    # Override data directory if provided
    if data_dir is not None:
        data_config.data_root = data_dir
        log.info(f"Using data directory: {data_dir}")

    # Override batch size if provided
    if batch_size is not None:
        data_config.batch_size = batch_size
        log.info(f"Using batch size: {batch_size}")

    # Override num_workers if provided
    if num_workers is not None:
        data_config.num_workers = num_workers
        log.info(f"Using num_workers: {num_workers}")

    # Get data loaders
    try:
        transform_config = data_config.get("transforms", OmegaConf.create({}))
        dataloaders_dict = create_dataloaders_from_config(
            data_config=data_config,
            transform_config=transform_config,
            dataloader_config=data_config
        )

        # We only need the test dataloader for evaluation
        test_loader = dataloaders_dict.get('test', {}).get('dataloader')

        if test_loader is None:
            raise DataError("Test dataloader could not be created")

        log.info(f"Test dataset loaded with {len(test_loader.dataset)} samples"
                 )
        return test_loader

    except Exception as e:
        raise DataError(f"Error during data loading: {str(e)}") from e


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    metrics: Dict[str, Any],
    device: torch.device
) -> Tuple[Dict[str, float], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Evaluate a model on a dataset.

    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        metrics: Dictionary of metric functions
        device: Device to use for evaluation

    Returns:
        Tuple containing:
        - Dictionary of metric results
        - Tuple of (inputs, targets, outputs) tensors for visualization
    """
    model.eval()
    results = {f"test_{name}": 0.0 for name in metrics.keys()}
    # Add loss placeholder even though we may not calculate it
    results["test_loss"] = 0.0

    # Store predictions and ground truth for visualization later
    all_inputs = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Extract inputs and targets
            if isinstance(batch, dict):
                inputs, targets = batch['image'], batch['mask']
            else:
                inputs, targets = batch

            # Ensure inputs tensor has the correct shape (B, C, H, W)
            # If channels are last
            if len(inputs.shape) == 4 and inputs.shape[-1] == 3:
                inputs = inputs.permute(0, 3, 1, 2)  # Change to (B, C, H, W)

            # Ensure targets tensor has a channel dimension
            # If (B, H, W) without channel dimension
            if len(targets.shape) == 3:
                targets = targets.unsqueeze(1)  # Add channel dimension

            # Handle numpy arrays which don't have .long() method
            if hasattr(targets, 'long'):
                targets = targets.long() if targets.dtype != torch.float32 \
                    else targets
            else:
                # Convert numpy array to tensor if needed
                targets = torch.tensor(targets, dtype=torch.float32)

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate metrics
            for name, metric_fn in metrics.items():
                metric_value = metric_fn(outputs, targets).item()
                results[f"test_{name}"] += metric_value

            # Store first few batches for visualization
            if batch_idx < 2:  # Limit number of stored batches to save memory
                all_inputs.append(inputs.cpu())
                all_targets.append(targets.cpu())
                all_outputs.append(outputs.cpu())

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                log.info(f"Evaluated {batch_idx + 1}/{len(dataloader)} batches"
                         )

    # Average metrics
    for key in results:
        results[key] /= len(dataloader)

    # Convert stored tensors to lists
    all_inputs = torch.cat(all_inputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)

    return results, (all_inputs, all_targets, all_outputs)


def visualize_predictions(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    outputs: torch.Tensor,
    output_dir: str,
    num_samples: int = 5
) -> None:
    """
    Create and save visualizations of model predictions.

    Args:
        inputs: Input images
        targets: Ground truth masks
        outputs: Model predictions
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    # Limit number of samples
    num_samples = min(num_samples, len(inputs))

    # Apply threshold to outputs (assuming sigmoid activation)
    binary_outputs = (outputs > 0.5).float()

    # Create directory if it doesn't exist
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Configure plotting
    plt.figure(figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        # Get current sample
        img = inputs[i].permute(1, 2, 0).numpy()
        mask = targets[i].squeeze().numpy()
        pred = outputs[i].squeeze().numpy()
        binary_pred = binary_outputs[i].squeeze().numpy()

        # Denormalize image if it's normalized to [-1, 1]
        if img.min() < 0:
            img = (img + 1) / 2
        # Clip to [0, 1] range
        img = np.clip(img, 0, 1)

        # Plot original image, ground truth, prediction, and binary prediction
        plt.subplot(num_samples, 4, i*4 + 1)
        plt.imshow(img)
        plt.title(f"Image {i+1}")
        plt.axis('off')

        plt.subplot(num_samples, 4, i*4 + 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(num_samples, 4, i*4 + 3)
        plt.imshow(pred, cmap='gray')
        plt.title("Prediction (Raw)")
        plt.axis('off')

        plt.subplot(num_samples, 4, i*4 + 4)
        plt.imshow(binary_pred, cmap='gray')
        plt.title("Prediction (Binary)")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "prediction_samples.png"), dpi=200)
    plt.close()

    # Save individual samples for detailed inspection
    for i in range(num_samples):
        img = inputs[i].permute(1, 2, 0).numpy()
        mask = targets[i].squeeze().numpy()
        pred = outputs[i].squeeze().numpy()

        # Denormalize image
        if img.min() < 0:
            img = (img + 1) / 2
        img = np.clip(img, 0, 1)

        # Create overlay visualization (prediction contour on image)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        # Create contour of the prediction
        masked_pred = np.ma.masked_where(pred < 0.5, pred)
        ax.imshow(masked_pred, cmap='jet', alpha=0.4)

        plt.title(f"Sample {i+1}: Image with crack overlay")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"overlay_sample_{i+1}.png"),
                    dpi=200)
        plt.close()

    log.info(f"Visualizations saved to {vis_dir}")


def save_evaluation_results(
    results: Dict[str, float],
    config: Any,
    checkpoint_path: str,
    output_dir: str
) -> None:
    """
    Save evaluation results to file.

    Args:
        results: Dictionary of evaluation results
        config: Configuration used for evaluation
        checkpoint_path: Path to the model checkpoint
        output_dir: Directory to save results
    """
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Prepare results dictionary
    eval_results = {
        "metrics": results,
        "checkpoint": checkpoint_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": (OmegaConf.to_container(config)
                   if hasattr(config, "to_container") else config)
    }

    # Save results as YAML
    yaml_path = os.path.join(metrics_dir, "evaluation_results.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(eval_results, f, default_flow_style=False)

    # Also save as text file for easy reading
    txt_path = os.path.join(metrics_dir, "evaluation_summary.txt")
    with open(txt_path, 'w') as f:
        f.write("CRACK SEGMENTATION MODEL EVALUATION\n")
        f.write("==================================\n\n")
        f.write(f"Date: {eval_results['timestamp']}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        f.write("Metrics:\n")
        for metric_name, metric_value in results.items():
            f.write(f"  {metric_name}: {metric_value:.4f}\n")

    log.info(f"Evaluation results saved to {yaml_path}")
    log.info(f"Evaluation summary saved to {txt_path}")


def ensemble_evaluate(
    checkpoint_paths: List[str],
    config: Union[Dict[str, Any], DictConfig],
    dataloader: DataLoader,
    metrics: Dict[str, Any],
    device: torch.device,
    output_dir: str
) -> Dict[str, float]:
    """
    Perform ensemble evaluation with multiple model checkpoints.

    Args:
        checkpoint_paths: List of paths to model checkpoints
        config: Configuration
        dataloader: DataLoader with evaluation data
        metrics: Dictionary of metric functions
        device: Device to use for evaluation
        output_dir: Directory to save results

    Returns:
        Dictionary of ensemble evaluation results
    """
    log.info(f"Performing ensemble evaluation with {len(checkpoint_paths)} \
models")

    # Load all models
    models = []
    for checkpoint_path in checkpoint_paths:
        model, _ = load_model_from_checkpoint(checkpoint_path, device)
        models.append(model)
        log.info(f"Loaded model from: {checkpoint_path}")

    # Initialize results
    ensemble_results = {f"ensemble_{name}": 0.0 for name in metrics.keys()}

    # Store predictions and ground truth for visualization
    all_inputs = []
    all_targets = []
    all_ensemble_outputs = []

    # Evaluate
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Extract inputs and targets
            if isinstance(batch, dict):
                inputs, targets = batch['image'], batch['mask']
            else:
                inputs, targets = batch

            # Ensure inputs tensor has the correct shape (B, C, H, W)
            if len(inputs.shape) == 4 and inputs.shape[-1] == 3:
                inputs = inputs.permute(0, 3, 1, 2)

            # Ensure targets tensor has a channel dimension
            if len(targets.shape) == 3:
                targets = targets.unsqueeze(1)

            # Handle numpy arrays which don't have .long() method
            if hasattr(targets, 'long'):
                targets = targets.long() if targets.dtype != torch.float32 \
                    else targets
            else:
                # Convert numpy array to tensor if needed
                targets = torch.tensor(targets, dtype=torch.float32)

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get predictions from all models
            ensemble_output = torch.zeros_like(
                models[0](inputs)  # Use first model to determine output shape
            )

            for model in models:
                model_output = model(inputs)
                ensemble_output += model_output

            # Average predictions
            ensemble_output /= len(models)

            # Calculate metrics
            for name, metric_fn in metrics.items():
                metric_value = metric_fn(ensemble_output, targets).item()
                ensemble_results[f"ensemble_{name}"] += metric_value

            # Store first few batches for visualization
            if batch_idx < 2:  # Limit number of stored batches to save memory
                all_inputs.append(inputs.cpu())
                all_targets.append(targets.cpu())
                all_ensemble_outputs.append(ensemble_output.cpu())

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                log.info(f"Ensemble evaluated {batch_idx + 1}/\
{len(dataloader)} batches")

    # Average metrics
    for key in ensemble_results:
        ensemble_results[key] /= len(dataloader)

    # Convert stored tensors for visualization
    if all_inputs:
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_ensemble_outputs = torch.cat(all_ensemble_outputs, dim=0)

        # Create ensemble visualization directory
        ensemble_vis_dir = os.path.join(output_dir, "visualizations",
                                        "ensemble")
        os.makedirs(ensemble_vis_dir, exist_ok=True)

        # Visualize ensemble predictions
        visualize_predictions(
            all_inputs,
            all_targets,
            all_ensemble_outputs,
            output_dir,
            num_samples=min(5, len(all_inputs))
        )

    # Save ensemble results
    ensemble_dir = os.path.join(output_dir, "metrics", "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)

    # Prepare ensemble results dictionary
    ensemble_eval_results = {
        "metrics": ensemble_results,
        "checkpoints": checkpoint_paths,
        "num_models": len(checkpoint_paths),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save results as YAML
    yaml_path = os.path.join(ensemble_dir, "ensemble_results.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(ensemble_eval_results, f, default_flow_style=False)

    log.info(f"Ensemble evaluation results saved to {yaml_path}")

    return ensemble_results


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
