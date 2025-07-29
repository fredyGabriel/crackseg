"""
Advanced Schema Validation System for Crack Segmentation Configurations.

This module provides comprehensive schema validation specifically tailored for
the CrackSeg pavement crack segmentation project. It includes deep validation
of model architectures, training parameters, data configurations, and ensures
compatibility with the project's specific requirements.

Key Features:
- Deep schema validation for crack segmentation models
- Architecture-specific validation (U-Net, CNN+LSTM, Swin Transformer)
- Hardware constraint validation (RTX 3070 Ti VRAM limits)
- Domain-specific validation (crack detection parameters)
- Type-safe configuration validation with detailed error reporting
"""

from .schema import CrackSegSchemaValidator, validate_crackseg_schema

# Re-export the main validation function and class for backward compatibility
__all__ = ["validate_crackseg_schema", "CrackSegSchemaValidator"]
