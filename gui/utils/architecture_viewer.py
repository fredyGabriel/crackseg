"""Architecture viewer utilities for the CrackSeg GUI.

This module provides functionality to instantiate models from Hydra
configurations and generate architecture visualizations using the model's
built-in visualization capabilities. Includes robust error handling for missing
Graphviz installation and temporary file management.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, cast

import streamlit as st
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from crackseg.model.factory.config import (
    InstantiationError,
    create_model_from_config,
)
from gui.utils.config.exceptions import ConfigError
from gui.utils.config.io import load_config_file

logger = logging.getLogger(__name__)


class ArchitectureViewerError(Exception):
    """Exception raised for architecture viewer errors."""

    pass


class GraphvizNotInstalledError(ArchitectureViewerError):
    """Exception raised when Graphviz is not installed."""

    pass


class ModelInstantiationError(ArchitectureViewerError):
    """Exception raised when model instantiation fails."""

    pass


class ArchitectureViewer:
    """Handles model instantiation and architecture visualization.

    This class provides methods to create models from configuration files
    and generate architecture diagrams with proper error handling and
    temporary file management.
    """

    def __init__(self) -> None:
        """Initialize the architecture viewer."""
        self._temp_files: list[Path] = []

    def __del__(self) -> None:
        """Clean up temporary files on destruction."""
        self.cleanup_temp_files()

    def cleanup_temp_files(self) -> None:
        """Clean up all temporary files created by this viewer."""
        for temp_file in self._temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(
                    f"Failed to remove temporary file {temp_file}: {e}"
                )
        self._temp_files.clear()

    def check_graphviz_available(self) -> bool:
        """Check if Graphviz is available on the system.

        Returns:
            True if Graphviz is available, False otherwise.
        """
        try:
            import graphviz  # type: ignore[import-untyped] # noqa: F401

            return True
        except ImportError:
            return False

    def instantiate_model_from_config_path(
        self, config_path: str | Path, device: str = "cpu"
    ) -> torch.nn.Module:
        """Instantiate a model from a configuration file path.

        Args:
            config_path: Path to the configuration file
            device: Device to instantiate the model on (default: "cpu")

        Returns:
            Instantiated PyTorch model

        Raises:
            ModelInstantiationError: If model instantiation fails
            ConfigError: If configuration loading fails
        """
        try:
            # Try to load with Hydra first for proper defaults resolution
            config_dict = self._load_config_with_hydra(config_path)
            if config_dict is None:
                # Fallback to simple loading
                config_dict = load_config_file(config_path)

            return self.instantiate_model_from_config(config_dict, device)

        except ConfigError as e:
            raise ModelInstantiationError(f"Failed to load config: {e}") from e
        except Exception as e:
            raise ModelInstantiationError(
                f"Unexpected error loading config: {e}"
            ) from e

    def _load_config_with_hydra(
        self, config_path: str | Path
    ) -> dict[str, Any] | None:
        """Load configuration using Hydra for proper defaults resolution.

        Args:
            config_path: Path to the configuration file

        Returns:
            Configuration dictionary or None if Hydra loading fails
        """
        try:
            config_path = Path(config_path)

            # Clear any existing Hydra instance
            GlobalHydra.instance().clear()

            # Get the config directory and filename
            config_dir = str(config_path.parent.absolute())
            config_name = config_path.stem

            # Initialize Hydra with the config directory
            with initialize_config_dir(
                config_dir=config_dir, version_base=None
            ):
                # Compose the configuration
                cfg = compose(config_name=config_name)

                # Convert to dictionary
                result = OmegaConf.to_container(cfg, resolve=True)
                if isinstance(result, dict):
                    return cast(dict[str, Any], result)
                else:
                    logger.warning(
                        f"Config is not a dictionary: {type(result)}"
                    )
                    return None

        except Exception as e:
            logger.warning(f"Hydra config loading failed: {e}")
            return None
        finally:
            # Clean up Hydra
            try:
                GlobalHydra.instance().clear()
            except Exception:
                pass

    def instantiate_model_from_config(
        self, config: dict[str, Any] | DictConfig, device: str = "cpu"
    ) -> torch.nn.Module:
        """Instantiate a model from a configuration dictionary.

        Args:
            config: Model configuration dictionary
            device: Device to instantiate the model on (default: "cpu")

        Returns:
            Instantiated PyTorch model

        Raises:
            ModelInstantiationError: If model instantiation fails
        """
        try:
            # Convert to DictConfig if needed
            if not isinstance(config, DictConfig):
                config_dict = config
                config = OmegaConf.create(config_dict)
            else:
                config_dict = cast(
                    dict[str, Any],
                    OmegaConf.to_container(config, resolve=True),
                )

            # Create model on CPU to avoid GPU memory issues during
            # instantiation
            model = create_model_from_config(config)

            # Move to specified device
            model = model.to(device)

            # Set to evaluation mode
            model.eval()

            logger.info(f"Successfully instantiated model on {device}")
            return model

        except InstantiationError as e:
            raise ModelInstantiationError(
                f"Model instantiation failed: {e}"
            ) from e
        except Exception as e:
            raise ModelInstantiationError(
                f"Unexpected error during instantiation: {e}"
            ) from e

    def generate_architecture_diagram(
        self, model: torch.nn.Module, filename: str | None = None
    ) -> Path:
        """Generate architecture diagram for a model.

        Args:
            model: PyTorch model with visualize_architecture method
            filename: Optional filename for the diagram (without extension)

        Returns:
            Path to the generated diagram file

        Raises:
            GraphvizNotInstalledError: If Graphviz is not available
            ArchitectureViewerError: If diagram generation fails
        """
        # Check if Graphviz is available
        if not self.check_graphviz_available():
            raise GraphvizNotInstalledError(
                "Graphviz is not installed. Please install it using:\n"
                "conda install graphviz python-graphviz"
            )

        # Check if model has visualize_architecture method
        if not hasattr(model, "visualize_architecture"):
            raise ArchitectureViewerError(
                f"Model {type(model).__name__} does not have "
                f"visualize_architecture method"
            )

        try:
            # Create temporary file if no filename provided
            if filename is None:
                temp_fd, temp_path = tempfile.mkstemp(
                    suffix=".png", prefix="architecture_"
                )
                temp_file = Path(temp_path)
                # Close the file descriptor as we only need the path
                import os

                os.close(temp_fd)
            else:
                temp_file = Path(tempfile.gettempdir()) / f"{filename}.png"

            # Track temporary file for cleanup
            self._temp_files.append(temp_file)

            # Generate architecture diagram
            # The model's visualize_architecture method will save to the file
            model.visualize_architecture(str(temp_file), view=False)

            # Verify the file was created
            if not temp_file.exists():
                raise ArchitectureViewerError(
                    "Architecture diagram was not generated"
                )

            logger.info(f"Architecture diagram generated: {temp_file}")
            return temp_file

        except Exception as e:
            if isinstance(
                e, GraphvizNotInstalledError | ArchitectureViewerError
            ):
                raise
            raise ArchitectureViewerError(
                f"Failed to generate architecture diagram: {e}"
            ) from e

    def get_model_summary(self, model: torch.nn.Module) -> dict[str, Any]:
        """Get comprehensive model summary information.

        Args:
            model: PyTorch model

        Returns:
            Dictionary containing model summary information
        """
        try:
            # Check if model has summary method
            if hasattr(model, "summary"):
                return model.summary()

            # Fallback: basic model information
            return {
                "model_type": type(model).__name__,
                "total_params": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "model_size_mb": sum(
                    p.numel() * p.element_size() for p in model.parameters()
                )
                / 1024
                / 1024,
            }

        except Exception as e:
            logger.warning(f"Failed to get model summary: {e}")
            return {"model_type": type(model).__name__, "error": str(e)}


@st.cache_resource
def get_architecture_viewer() -> ArchitectureViewer:
    """Get a cached architecture viewer instance.

    Returns:
        ArchitectureViewer instance cached by Streamlit.
    """
    return ArchitectureViewer()


def format_model_summary(summary: dict[str, Any]) -> str:
    """Format model summary for display.

    Args:
        summary: Model summary dictionary

    Returns:
        Formatted string representation of the summary
    """
    if "error" in summary:
        return f"Error getting summary: {summary['error']}"

    lines: list[str] = []
    lines.append(f"**Model Type:** {summary.get('model_type', 'Unknown')}")

    if "total_params" in summary:
        total_params = summary["total_params"]
        lines.append(f"**Total Parameters:** {total_params:,}")

    if "trainable_params" in summary:
        trainable_params = summary["trainable_params"]
        lines.append(f"**Trainable Parameters:** {trainable_params:,}")

    if "model_size_mb" in summary:
        size_mb = summary["model_size_mb"]
        lines.append(f"**Model Size:** {size_mb:.2f} MB")

    # Add architecture components if available
    if "encoder_type" in summary:
        lines.append(f"**Encoder:** {summary['encoder_type']}")

    if "bottleneck_type" in summary:
        lines.append(f"**Bottleneck:** {summary['bottleneck_type']}")

    if "decoder_type" in summary:
        lines.append(f"**Decoder:** {summary['decoder_type']}")

    return "\n".join(lines)


def display_graphviz_installation_help() -> None:
    """Display help information for installing Graphviz."""
    st.error("🚫 **Graphviz Not Installed**")

    st.markdown(
        """
    Architecture visualization requires Graphviz to be installed on your
    system.

    **Installation Instructions:**

    **Option 1: Using Conda (Recommended)**
    ```bash
    conda install graphviz python-graphviz
    ```

    **Option 2: Using pip + system package**
    ```bash
    # Install system package first
    # Ubuntu/Debian:
    sudo apt-get install graphviz

    # macOS (with Homebrew):
    brew install graphviz

    # Windows: Download from https://graphviz.org/download/

    # Then install Python package:
    pip install graphviz
    ```

    After installation, restart the application.
    """
    )
