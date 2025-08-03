"""Configuration handling for evaluation CLI.

This module provides functions for loading, validating, and processing
configuration files for evaluation operations.
"""

import argparse
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from crackseg.utils.exceptions import ConfigError
from crackseg.utils.logging import get_logger

# Configure logger
log = get_logger("evaluation")


def load_and_prepare_config(
    args: argparse.Namespace, checkpoint_data: dict[str, Any], output_dir: Path
) -> DictConfig:
    """Loads, validates, overrides, and saves the configuration.

    Args:
        args: Command line arguments.
        checkpoint_data: Data loaded from checkpoint.
        output_dir: Output directory for saving config.

    Returns:
        DictConfig: Loaded and validated configuration.

    Raises:
        ConfigError: If configuration is invalid or missing.
    """
    cfg = None
    if args.config:
        log.info(f"Loading configuration from: {args.config}")
        cfg = OmegaConf.load(args.config)
    elif "config" in checkpoint_data:
        log.info("Loading configuration from checkpoint.")
        cfg = checkpoint_data["config"]
        if isinstance(cfg, dict):  # Ensure it's an OmegaConf object
            cfg = OmegaConf.create(cfg)
    else:
        log.error(
            "No configuration found in checkpoint or provided via --config "
            "argument."
        )
        raise ConfigError("Missing model configuration.")

    if not isinstance(cfg, DictConfig):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        else:
            raise ConfigError("Invalid configuration format.")

    # Override configuration with command line arguments
    if args.override:
        for override in args.override:
            try:
                key, value = override.split("=", 1)
                OmegaConf.update(cfg, key, value)
                log.info(f"Overriding config: {key} = {value}")
            except ValueError:
                log.warning(f"Invalid override format: {override}")

    # Save the final configuration
    config_save_path = output_dir / "evaluation_config.yaml"
    with open(config_save_path, "w") as f:
        OmegaConf.save(cfg, f)
    log.info(f"Configuration saved to: {config_save_path}")

    return cfg


def validate_evaluation_config(cfg: DictConfig) -> None:
    """Validates the evaluation configuration.

    Args:
        cfg: Configuration to validate.

    Raises:
        ConfigError: If configuration is invalid.
    """
    required_keys = ["model", "data", "evaluation"]
    for key in required_keys:
        if key not in cfg:
            raise ConfigError(f"Missing required configuration key: {key}")

    log.info("Configuration validation passed.")
