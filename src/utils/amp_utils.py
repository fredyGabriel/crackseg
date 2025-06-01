"""AMP and gradient accumulation utilities for training loops."""

import torch
from torch import autocast

_gradscaler_device_value: str | None
try:
    from torch.amp import GradScaler  # type: ignore[reportMissingImports]

    _gradscaler_device_value = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler  # type: ignore[reportMissingImports]

    _gradscaler_device_value = None

_GRADSCALER_DEVICE: str | None = _gradscaler_device_value

__all__ = [
    "amp_autocast",
    "optimizer_step_with_accumulation",
    "GradScaler",
    "_GRADSCALER_DEVICE",
]


def amp_autocast(enabled: bool) -> autocast:
    """Context manager for autocast, enabled only if specified."""
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return autocast(device_type=device_type, enabled=enabled)


def optimizer_step_with_accumulation(  # noqa: PLR0913
    *,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    loss: torch.Tensor,
    grad_accum_steps: int,
    batch_idx: int,
    use_amp: bool = False,
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
    is_update_step: bool = (batch_idx + 1) % grad_accum_steps == 0
    loss = loss / grad_accum_steps
    if use_amp and scaler is not None:
        scaler.scale(loss).backward()  # type: ignore[reportUnknownMemberType]
        if is_update_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    else:
        loss.backward()  # type: ignore[reportUnknownMemberType]
        if is_update_step:
            optimizer.step()
            optimizer.zero_grad()
