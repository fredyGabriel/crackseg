from typing import Any

import torch
from omegaconf import OmegaConf

from crackseg.model.factory import create_unet
from crackseg.utils.exceptions import EvaluationError, ModelError
from crackseg.utils.logging import get_logger
from crackseg.utils.storage import load_checkpoint_dict

log = get_logger("evaluation.loading")


def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Load a model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model onto

    Returns:
        Tuple containing:
        - model: The loaded model
        - checkpoint_data: Additional data from the checkpoint
    """
    # Load checkpoint dictionary from file
    checkpoint = load_checkpoint_dict(checkpoint_path, device=device)

    # Extract config from checkpoint if available
    if "config" not in checkpoint:
        raise EvaluationError(
            f"Checkpoint at {checkpoint_path} does not contain model "
            "configuration. Please provide a configuration file with --config."
        )

    config = checkpoint["config"]

    if isinstance(config, dict):
        config = OmegaConf.create(config)

    # Create model with the same architecture
    try:
        model = create_unet(config.model)
        model.to(device)
        log.info("Created model: %s", type(model).__name__)
    except Exception as e:
        raise ModelError(f"Error creating model: {str(e)}") from e

    # Load weights into model
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        log.info("Model weights loaded from: %s", checkpoint_path)
    except Exception as e:
        raise ModelError(f"Error loading model weights: {str(e)}") from e

    # Set model to evaluation mode
    model.eval()

    # Return model and checkpoint data
    checkpoint_data = {
        k: v
        for k, v in checkpoint.items()
        if k not in ["model_state_dict", "optimizer_state_dict"]
    }

    return model, checkpoint_data
