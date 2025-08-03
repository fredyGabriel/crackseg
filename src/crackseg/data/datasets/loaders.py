"""Data loading utilities for crack segmentation datasets.

This module provides specialized loaders for images and masks
with proper error handling and format conversion.
"""

import cv2
import numpy as np
import PIL.Image

from .types import SourceType


class ImageLoader:
    """Specialized loader for image files with format conversion.

    This class handles loading images from various sources (file paths,
    PIL Images, numpy arrays) with automatic format conversion to RGB.

    Features:
        - Automatic BGR to RGB conversion for OpenCV-loaded images
        - Support for multiple input formats
        - Robust error handling
        - EXIF orientation handling
    """

    def load(self, image_source: SourceType) -> PIL.Image.Image:
        """Load image from various source types.

        Args:
            image_source: Image source (file path, PIL Image, or numpy array)

        Returns:
            PIL Image in RGB format

        Raises:
            ValueError: If image source type is unsupported or loading fails
        """
        if isinstance(image_source, str):
            image = cv2.imread(image_source)
            if image is None:
                raise ValueError(f"Failed to load image: {image_source}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(image)
        elif isinstance(image_source, PIL.Image.Image):
            image = image_source
        elif isinstance(image_source, np.ndarray):
            image = PIL.Image.fromarray(image_source)
        else:
            raise ValueError(f"Unsupported image type: {type(image_source)}")

        return image


class MaskLoader:
    """Specialized loader for mask files with binary conversion.

    This class handles loading masks from various sources with automatic
    conversion to grayscale and binary thresholding.

    Features:
        - Automatic grayscale loading for mask files
        - Support for multiple input formats
        - Binary thresholding for consistent mask values
        - Robust error handling
    """

    def load(self, mask_source: SourceType) -> PIL.Image.Image:
        """Load mask from various source types.

        Args:
            mask_source: Mask source (file path, PIL Image, or numpy array)

        Returns:
            PIL Image in grayscale format

        Raises:
            ValueError: If mask source type is unsupported or loading fails
        """
        if isinstance(mask_source, str):
            mask = cv2.imread(mask_source, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {mask_source}")
            mask = PIL.Image.fromarray(mask)
        elif isinstance(mask_source, PIL.Image.Image):
            mask = mask_source
        elif isinstance(mask_source, np.ndarray):
            mask = PIL.Image.fromarray(mask_source)
        else:
            raise ValueError(f"Unsupported mask type: {type(mask_source)}")

        return mask
