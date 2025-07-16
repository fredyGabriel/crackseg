"""Init file for components submodule."""

# Import components to ensure they are registered
from .cbam import CBAM  # noqa: F401

__all__ = ["CBAM"]
