"""Standard UNet implementation for crack segmentation.

This module provides a simple alias to the main BaseUNet implementation
located in crackseg.model.core.unet. This maintains backward compatibility
and provides a clean import path for the standard UNet architecture.
"""

from crackseg.model.core.unet import BaseUNet

# Alias for backward compatibility and clean imports
UNet = BaseUNet

__all__ = ["UNet"]
