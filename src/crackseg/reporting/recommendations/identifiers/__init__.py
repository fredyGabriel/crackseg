"""Opportunity identifiers for recommendation engine.

This module provides specialized identifiers for different types of
optimization opportunities in experiments.
"""

from .architecture import ArchitectureIdentifier
from .opportunities import OpportunityIdentifier

__all__ = [
    "OpportunityIdentifier",
    "ArchitectureIdentifier",
]
