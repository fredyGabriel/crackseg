import torch
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader


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

            # Log progress (dejar a la función llamadora)

    # Average metrics
    for key in results:
        results[key] /= len(dataloader)

    # Convert stored tensors to lists
    all_inputs = torch.cat(all_inputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)

    return results, (all_inputs, all_targets, all_outputs)
