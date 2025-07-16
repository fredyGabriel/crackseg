"""
Interfaces and contracts for the loss system architecture.
This module defines the core interfaces that establish contracts
for loss components, factories, and registries.
"""

from .loss_interface import ILossCombinator, ILossComponent, ILossFactory

__all__ = ["ILossComponent", "ILossCombinator", "ILossFactory"]
