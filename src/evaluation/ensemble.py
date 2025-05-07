import os
import yaml
from datetime import datetime
from typing import List, Dict, Any, Union
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from src.utils.visualization import visualize_predictions
from src.utils.logging import get_logger
from src.evaluation.loading import load_model_from_checkpoint

log = get_logger("evaluation.ensemble")


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
    if not checkpoint_paths:
        raise ValueError("No checkpoints provided for ensemble evaluation.")

    log.info(
        f"Performing ensemble evaluation with {len(checkpoint_paths)} models"
    )

    # Load all models
    models = []
    for checkpoint_path in checkpoint_paths:
        model, _ = load_model_from_checkpoint(checkpoint_path, device)
        models.append(model)
        log.info(f"Loaded model from: {checkpoint_path}")

    # Initialize results
    ensemble_results = {
        f"ensemble_{name}": 0.0 for name in metrics.keys()
    }

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
                targets = (
                    targets.long() if targets.dtype != torch.float32 else
                    targets
                )
            else:
                targets = torch.tensor(targets, dtype=torch.float32)

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get predictions from all models
            ensemble_output = torch.zeros_like(models[0](inputs))
            for model in models:
                model_output = model(inputs)
                ensemble_output += model_output
            ensemble_output /= len(models)

            # Calculate metrics
            for name, metric_fn in metrics.items():
                metric_value = metric_fn(ensemble_output, targets).item()
                ensemble_results[f"ensemble_{name}"] += metric_value

            # Store first few batches for visualization
            if batch_idx < 2:
                all_inputs.append(inputs.cpu())
                all_targets.append(targets.cpu())
                all_ensemble_outputs.append(ensemble_output.cpu())

            if (batch_idx + 1) % 10 == 0:
                log.info(
                    f"Ensemble evaluated {batch_idx + 1}/"
                    f"{len(dataloader)} batches"
                )

    # Average metrics
    for key in ensemble_results:
        ensemble_results[key] /= len(dataloader)

    # Convert stored tensors for visualization
    if all_inputs:
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_ensemble_outputs = torch.cat(all_ensemble_outputs, dim=0)

        ensemble_vis_dir = os.path.join(
            output_dir, "visualizations", "ensemble"
        )
        os.makedirs(ensemble_vis_dir, exist_ok=True)

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

    ensemble_eval_results = {
        "metrics": ensemble_results,
        "checkpoints": checkpoint_paths,
        "num_models": len(checkpoint_paths),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    yaml_path = os.path.join(
        ensemble_dir, "ensemble_results.yaml"
    )
    with open(yaml_path, 'w') as f:
        yaml.dump(ensemble_eval_results, f, default_flow_style=False)

    log.info(
        f"Ensemble evaluation results saved to {yaml_path}"
    )

    return ensemble_results
