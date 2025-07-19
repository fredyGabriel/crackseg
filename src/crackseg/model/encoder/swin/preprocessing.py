"""Input and output preprocessing for Swin Transformer V2 Encoder.

This module handles input tensor preprocessing (resizing, padding) and
output feature post-processing to ensure compatibility with different
input sizes and decoder expectations.
"""

import logging
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SwinPreprocessor:
    """
    Handles input preprocessing and output postprocessing for Swin models.
    """

    @staticmethod
    def preprocess_input(
        x: torch.Tensor,
        img_size: int,
        patch_size: int,
        handle_input_size: str,
        expected_input_dims: int,
        in_channels: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Preprocess input tensor based on configuration.

        Args:
            x: Input tensor of shape (B, C, H, W).
            img_size: Expected input image size.
            patch_size: Size of image patches.
            handle_input_size: Strategy for input size handling.
            expected_input_dims: Expected number of input dimensions.
            in_channels: Expected number of input channels.

        Returns:
            Tuple of (preprocessed tensor, metadata dictionary).

        Raises:
            ValueError: If input validation fails.
        """
        # Store original size for reference
        original_size = (x.shape[2], x.shape[3])
        metadata: dict[str, Any] = {"original_size": original_size}

        # Basic input validation
        if x.dim() != expected_input_dims:
            raise ValueError(
                f"Expected {expected_input_dims}D input (B,C,H,W), "
                f"got {x.dim()}D"
            )

        if x.size(1) != in_channels:
            raise ValueError(
                f"Expected {in_channels} input channels, got {x.size(1)}"
            )

        if (
            min(x.size(2), x.size(3)) < patch_size
            and handle_input_size == "none"
        ):
            raise ValueError(
                f"Input dimensions (H={x.size(2)}, W={x.size(3)}) must be at "
                f"least {patch_size}x{patch_size} with "
                f"handle_input_size='none'"
            )

        # Apply preprocessing based on configuration
        if handle_input_size == "resize":
            # Resize to the model's expected size
            if original_size != (img_size, img_size):
                x = F.interpolate(
                    x,
                    size=(img_size, img_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                metadata["resized"] = True
                logger.debug(
                    f"Resized input from {original_size} to "
                    f"({img_size}, {img_size})"
                )

        elif handle_input_size == "pad":
            # Calculate padding to make dimensions divisible by patch_size
            _, _, H, W = x.shape
            pad_h = (patch_size - H % patch_size) % patch_size
            pad_w = (patch_size - W % patch_size) % patch_size

            if pad_h > 0 or pad_w > 0:
                # Apply padding (pad_left, pad_right, pad_top, pad_bottom)
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
                metadata["padding"] = (0, pad_w, 0, pad_h)
                metadata["padded_size"] = (H + pad_h, W + pad_w)
                logger.debug(
                    f"Padded input from {(H, W)} to {(H + pad_h, W + pad_w)}"
                )

        return x, metadata

    @staticmethod
    def postprocess_features(
        features: list[torch.Tensor],
        metadata: dict[str, Any] | None,
        patch_size: int,
    ) -> list[torch.Tensor]:
        """Apply post-processing to features if needed.

        Args:
            features: List of feature tensors.
            metadata: Metadata from preprocessing (optional).
            patch_size: Size of image patches for scaling calculations.

        Returns:
            List of post-processed feature tensors.
        """
        # If no metadata or no modifications were made, return as is
        if metadata is None:
            return features

        processed_features: list[torch.Tensor] = []

        for feature_item in features:  # Renamed loop variable
            processed_feature_item: torch.Tensor = (
                feature_item  # Start with original
            )
            # Apply post-processing if needed
            if "resized" in metadata and metadata["resized"]:
                # Resize back to original spatial dimensions adjusted for the
                # reduction factor
                original_h, original_w = metadata["original_size"]
                feature_h = max(1, original_h // patch_size)
                feature_w = max(1, original_w // patch_size)

                # Only resize if dimensions are different
                current_h, current_w = (
                    processed_feature_item.shape[2],
                    processed_feature_item.shape[3],
                )
                if (current_h, current_w) != (feature_h, feature_w):
                    processed_feature_item = F.interpolate(
                        processed_feature_item,
                        size=(feature_h, feature_w),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )

            # If we need to handle padding, we could add logic here to crop
            # the features back to the unpadded size if needed

            processed_features.append(processed_feature_item)

        return processed_features

    @staticmethod
    def validate_config_parameters(handle_input_size: str) -> None:
        """Validate preprocessing configuration parameters.

        Args:
            handle_input_size: Strategy for input size handling.

        Raises:
            ValueError: If configuration is invalid.
        """
        valid_size_handlers = ["resize", "pad", "none"]
        if handle_input_size not in valid_size_handlers:
            raise ValueError(
                f"handle_input_size must be one of {valid_size_handlers}, "
                f"got {handle_input_size}"
            )
