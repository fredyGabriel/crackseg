"""
Loss Function Registry Setup.

This module initializes and provides a global registry instance for loss
functions, leveraging the generic Registry system from
`src.model.factory.registry`. Loss functions registered here are expected to
be classes inheriting from `torch.nn.Module`.
"""

import logging

from torch import nn

from src.model.factory.registry import (
    Registry,  # Assuming this is the correct path to the generic Registry
)

log = logging.getLogger(__name__)

# Initialize a specific registry for loss functions.
# Loss functions should be nn.Module classes.
loss_registry = Registry(base_class=nn.Module, name="LossFunctions")

log.info(
    f"Initialized {loss_registry.name} registry for PyTorch "
    "nn.Module-based loss functions."
)

# Optional: Define a convenience decorator for registering losses if desired,
# though using loss_registry.register directly is also clear.
# from typing import List, Optional, Type
# def register_loss_module(name: Optional[str] = None,
# tags: Optional[List[str]] = None) -> Callable[[Type[nn.Module]],
# Type[nn.Module]]:
#     """
#     Convenience decorator to register a loss module.
#     """
#     return loss_registry.register(name=name, tags=tags)

# Example of how it would be used (in other files):
# from src.training.losses.loss_registry_setup import loss_registry # or
# register_loss_module
#
# @loss_registry.register(name="my_custom_loss", tags=["segmentation",
# "classification"])
# class MyLoss(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         # ...
#     def forward(self, predictions, targets):
#         # ...
#         return loss
