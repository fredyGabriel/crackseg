"""AMP and gradient accumulation utilities for training loops."""
from typing import Optional
import torch
from torch.cuda.amp import GradScaler
from torch import autocast


def amp_autocast(enabled: bool):
    """Context manager for autocast, enabled only if specified."""
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return autocast(device_type=device_type, enabled=enabled)


def optimizer_step_with_accumulation(
    *,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    loss: torch.Tensor,
    grad_accum_steps: int,
    batch_idx: int,
    use_amp: bool = False
) -> None:
    """Handles backward, step, and update for optimizer/scaler with
    accumulation.

    Args:
        optimizer: Optimizer instance.
        scaler: GradScaler instance or None.
        loss: Loss tensor.
        grad_accum_steps: Steps to accumulate gradients.
        batch_idx: Current batch index (0-based).
        use_amp: Whether to use AMP.
    """
    is_update_step = ((batch_idx + 1) % grad_accum_steps == 0)
    loss = loss / grad_accum_steps
    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
        if is_update_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    else:
        loss.backward()
        if is_update_step:
            optimizer.step()
            optimizer.zero_grad()
