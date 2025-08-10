"""Dynamic registration helpers for the loss registry."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from ...interfaces.loss_interface import ILossComponent  # type: ignore


def iter_loss_classes_from_module(
    module_path: str, loss_classes: list[str] | None
) -> list[tuple[str, type]]:
    """Return a list of (class_name, class_type) for loss classes in a module.

    If ``loss_classes`` is None, any attribute ending with "Loss" and being a class is returned.
    """
    module = importlib.import_module(module_path)

    if loss_classes is None:
        candidates = [
            name
            for name in dir(module)
            if isinstance(getattr(module, name), type)
            and name.endswith("Loss")
        ]
    else:
        candidates = loss_classes

    results: list[tuple[str, type]] = []
    for class_name in candidates:
        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            results.append((class_name, cls))
    return results


def build_factory_from_class(
    loss_class: type,
) -> Callable[..., ILossComponent]:
    """Create a simple factory function for a given loss class."""

    def factory(**params: Any) -> ILossComponent:
        return loss_class(**params)  # type: ignore[misc]

    return factory
