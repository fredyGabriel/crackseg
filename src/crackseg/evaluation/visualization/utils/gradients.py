"""Utilities for gradient-flow visualization computations."""

from __future__ import annotations

from typing import Any


def compute_gradient_norms(gradients: list[Any]) -> list[float]:
    """Compute total gradient norm per epoch from mixed formats.

    Supports two cases:
    - Each epoch is a dict of name->value (numeric); sums numeric values excluding 'epoch'.
    - Each epoch is a list of dicts containing a 'norm' key; sums the 'norm' values.
    """
    gradient_norms: list[float] = []
    for epoch_grads in gradients:
        if isinstance(epoch_grads, dict):
            epoch_norm = sum(
                float(val)
                for key, val in epoch_grads.items()
                if key != "epoch" and isinstance(val, int | float)
            )
            gradient_norms.append(epoch_norm)
        else:
            epoch_norm = sum(
                float(grad.get("norm", 0.0)) for grad in epoch_grads
            )
            gradient_norms.append(epoch_norm)
    return gradient_norms
