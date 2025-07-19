"""Swin Transformer V2 Encoder for semantic segmentation tasks.

This module provides backward compatibility imports for the refactored
Swin Transformer encoder implementation. The actual implementation has been
modularized into specialized modules for better maintainability while
preserving the original API.

For new code, consider importing directly from the swin submodule:
    from crackseg.model.encoder.swin import SwinTransformerEncoder,
    SwinTransformerEncoderConfig

Key Features:
    - Swin Transformer V2 architecture with improved training stability
    - Multi-scale feature extraction for U-Net decoder compatibility
    - Flexible input size handling (resize, pad, or none)
    - Layer freezing and gradual unfreezing for transfer learning
    - Comprehensive error handling with ResNet fallback
    - Optimizer parameter grouping for fine-tuning strategies

References:
    - Swin Transformer V2: https://arxiv.org/abs/2111.09883
    - timm library: https://github.com/rwightman/pytorch-image-models
    - U-Net architecture: https://arxiv.org/abs/1505.04597
"""

# Backward compatibility imports - preserve original API
# For advanced usage, also expose the specialized components
from crackseg.model.encoder.swin import (
    SwinModelInitializer,
    SwinPreprocessor,
    SwinTransferLearning,
    SwinTransformerEncoder,
    SwinTransformerEncoderConfig,
)

__all__ = [
    "SwinTransformerEncoder",
    "SwinTransformerEncoderConfig",
    "SwinModelInitializer",
    "SwinPreprocessor",
    "SwinTransferLearning",
]
