"""Image processing utilities for crack segmentation evaluation."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from crackseg.utils.data.image_size import get_target_size_from_config

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image and mask processing for evaluation."""

    def __init__(self, config: DictConfig) -> None:
        """
        Initialize the image processor.

        Args:
            config: Model configuration containing image size
        """
        self.config = config
        self.target_size = self._get_target_size()

    def _get_target_size(self) -> tuple[int, int]:
        """Get target image size from config."""
        return get_target_size_from_config(self.config)

    def load_and_preprocess_image(
        self, image_path: str | Path
    ) -> torch.Tensor:
        """
        Load and preprocess image for model input.

        Args:
            image_path: Path to the input image

        Returns:
            Preprocessed image tensor
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        image = cv2.resize(image, self.target_size)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert to tensor and normalize with ImageNet stats
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW

        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        return image_tensor

    def load_mask(self, mask_path: str | Path) -> np.ndarray:
        """
        Load and preprocess ground truth mask.

        Args:
            mask_path: Path to the ground truth mask

        Returns:
            Preprocessed binary mask
        """
        mask_path = Path(mask_path)

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")

        # Resize to match prediction size
        mask = cv2.resize(mask, self.target_size)

        # Normalize to [0, 1]
        mask = mask.astype(np.float32) / 255.0

        # Binarize
        mask = (mask > 0.5).astype(np.float32)

        return mask

    def infer_mask_path(
        self, image_path: str | Path, mask_dir: str | Path
    ) -> Path | None:
        """
        Automatically infer mask path based on image path.

        Args:
            image_path: Path to the input image
            mask_dir: Directory containing ground truth masks

        Returns:
            Path to the inferred mask file, or None if not found
        """
        if not mask_dir:
            return None

        image_path = Path(image_path)
        image_name = image_path.stem  # Get filename without extension
        mask_dir = Path(mask_dir)

        # Common mask extensions to try
        mask_extensions = [".png", ".jpg", ".jpeg", ".tiff", ".tif"]

        for ext in mask_extensions:
            mask_path = mask_dir / f"{image_name}{ext}"
            if mask_path.exists():
                logger.info(f"Inferred mask path: {mask_path}")
                return mask_path

        logger.warning(
            f"No mask found for image {image_path.name} in {mask_dir}"
        )
        return None

    def find_image_files(self, image_dir: str | Path) -> list[Path]:
        """
        Find all image files in a directory.

        Args:
            image_dir: Directory containing images

        Returns:
            List of image file paths
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        # Supported image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        image_files = [
            f
            for f in image_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")

        return sorted(image_files)

    def find_corresponding_mask(
        self, image_file: Path, mask_dir: str | Path | None
    ) -> Path | None:
        """
        Find corresponding mask for an image file.

        Args:
            image_file: Path to the image file
            mask_dir: Directory containing masks

        Returns:
            Path to corresponding mask, or None if not found
        """
        if not mask_dir:
            return None

        mask_dir = Path(mask_dir)
        if not mask_dir.exists():
            return None

        # Try different mask extensions
        for ext in [".png", ".jpg", ".jpeg"]:
            potential_mask = mask_dir / f"{image_file.stem}{ext}"
            if potential_mask.exists():
                return potential_mask

        return None
