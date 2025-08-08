"""Model creation for training pipeline.

This module provides model creation functions for the training pipeline
including model instantiation, device placement, and validation.
"""

import logging
from typing import Any, cast

import hydra
import torch
from hydra import errors as hydra_errors
from omegaconf import DictConfig
from torch.nn import Module

from crackseg.utils import ModelError

# Configure standard logger
log = logging.getLogger(__name__)


def create_model(cfg: DictConfig, device: Any) -> torch.nn.Module:
    """
    Create and initialize the segmentation model from configuration.

    This function instantiates the neural network model using Hydra's
    instantiation system and prepares it for training by:
    - Creating the model from configuration
    - Moving the model to the specified device
    - Logging model information (type and parameter count)
    - Validating the model is properly initialized

    Args:
        cfg: Hydra configuration containing model settings.
            Expected structure:
            - cfg.model._target_: Full path to model class
                (e.g., "src.model.UNet")
            - cfg.model.**kwargs: Model-specific parameters

        device: Target device for model placement (e.g., 'cuda:0' or 'cpu').

    Returns:
        torch.nn.Module: The initialized model ready for training.

    Raises:
        ModelError: If model creation fails due to:
            - Invalid model configuration
            - Missing model dependencies
            - Instantiation errors
        ImportError: If the specified model class cannot be imported
        AttributeError: If model configuration is malformed

    Examples:
        ```python
        cfg = OmegaConf.create({
            "model": {
                "_target_": "src.model.core.unet.UNet",
                "encoder_name": "resnet34",
                "encoder_weights": "imagenet",
                "in_channels": 3,
                "classes": 1
            }
        })
        device = torch.device("cuda:0")
        model = create_model(cfg, device)

        # Check model properties
        print(f"Model type: {type(model).__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Device: {next(model.parameters()).device}")
        ```

    Note:
        The function uses Hydra's instantiate method which supports
        complex configuration patterns and automatic dependency injection.
    """
    log.info("Creating model...")
    try:
        # Configuration is now guaranteed to be at root level due to run.py restructuring
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model = cast(Module, model)
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        log.info(
            "Created %s model with %s parameters",
            type(model).__name__,
            num_params,
        )
        assert isinstance(model, Module)
        return model
    except (
        ModelError,
        hydra_errors.InstantiationException,
        AttributeError,
        ImportError,
        TypeError,
        ValueError,
    ) as e:
        log.error("Error creating model: %s", str(e))
        raise ModelError(f"Error creating model: {str(e)}") from e
