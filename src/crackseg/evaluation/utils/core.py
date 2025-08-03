from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader[Any],
    metrics: dict[str, Any],
    device: torch.device,
    config: DictConfig,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Evaluate a model on a dataset.

    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        metrics: Dictionary of metric functions
        device: Device to use for evaluation
        config: Configuration object

    Returns:
        Tuple containing:
        - Dictionary of metric results
        - Tuple of (inputs, targets, outputs) tensors for visualization
    """
    if len(dataloader) == 0:
        results = {f"test_{name}": 0.0 for name in metrics.keys()}
        results["test_loss"] = 0.0
        empty = torch.empty(0)
        return results, (empty, empty, empty)

    model.eval()
    results = {f"test_{name}": 0.0 for name in metrics.keys()}
    # Add loss placeholder even though we may not calculate it
    results["test_loss"] = 0.0

    # Store predictions and ground truth for visualization later
    inputs_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    outputs_list: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Extract inputs and targets
            if isinstance(batch, dict):
                inputs, targets = batch["image"], batch["mask"]
            else:
                inputs, targets = batch

            # Ensure inputs tensor has the correct shape (B, C, H, W)
            # If channels are last
            if (
                len(inputs.shape) == config.data.num_dims_image
                and inputs.shape[-1] == config.data.num_channels_rgb
            ):
                inputs = inputs.permute(0, 3, 1, 2)  # Change to (B, C, H, W)

            # Ensure targets tensor has a channel dimension
            # If (B, H, W) without channel dimension
            if len(targets.shape) == config.data.num_dims_mask:
                targets = targets.unsqueeze(1)  # Add channel dimension

            # Handle numpy arrays which don't have .long() method
            if hasattr(targets, "long"):
                targets = (
                    targets.long()
                    if targets.dtype != torch.float32
                    else targets
                )
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
            if batch_idx < config.evaluation.num_batches_visualize:
                inputs_list.append(inputs.cpu())
                targets_list.append(targets.cpu())
                outputs_list.append(outputs.cpu())

            # Log progress (dejar a la función llamadora)

    # Average metrics
    for key in results:
        results[key] /= len(dataloader)

    # Convert stored tensors to tensors
    all_inputs = torch.cat(inputs_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    all_outputs = torch.cat(outputs_list, dim=0)

    return results, (all_inputs, all_targets, all_outputs)
