"""Swin Transformer V2 Encoder module for semantic segmentation.

This module provides a complete implementation of the Swin Transformer V2
encoder architecture optimized for semantic segmentation tasks, with
comprehensive support for transfer learning, flexible input handling,
and robust error recovery.

Key Components:
    - SwinTransformerEncoder: Main encoder class with full functionality
    - SwinTransformerEncoderConfig: Configuration dataclass
    - Specialized utility classes for initialization, preprocessing, and
      transfer learning

The module is designed to maintain backward compatibility while providing
improved modularity and maintainability through separation of concerns.
"""

from crackseg.model.encoder.swin.config import SwinTransformerEncoderConfig
from crackseg.model.encoder.swin.core import SwinTransformerEncoder
from crackseg.model.encoder.swin.initialization import SwinModelInitializer
from crackseg.model.encoder.swin.preprocessing import SwinPreprocessor
from crackseg.model.encoder.swin.transfer_learning import SwinTransferLearning

__all__ = [
    "SwinTransformerEncoder",
    "SwinTransformerEncoderConfig",
    "SwinModelInitializer",
    "SwinPreprocessor",
    "SwinTransferLearning",
]
