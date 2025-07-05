"""Image data factory for E2E testing.

This module provides synthetic image generation with crack patterns
for testing crack segmentation models.
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import BaseDataFactory, TestData


class ImageDataFactory(BaseDataFactory):
    """Factory for generating test image data."""

    def generate(
        self,
        image_type: str = "crack",
        width: int = 512,
        height: int = 512,
        channels: int = 3,
        crack_density: float = 0.05,
        **kwargs: Any,
    ) -> TestData:
        """Generate test image data.

        Args:
            image_type: Type of image ('crack', 'no_crack', 'edge_case')
            width: Image width
            height: Image height
            channels: Number of channels
            crack_density: Density of crack pixels (for crack images)
            **kwargs: Additional image parameters

        Returns:
            TestData containing generated image
        """
        # Generate image based on type
        if image_type == "crack":
            image_array = self._generate_crack_image(
                width, height, channels, crack_density
            )
        elif image_type == "no_crack":
            image_array = self._generate_clean_image(width, height, channels)
        elif image_type == "edge_case":
            image_array = self._generate_edge_case_image(
                width, height, channels
            )
        else:
            image_array = np.random.randint(
                0, 256, (height, width, channels), dtype=np.uint8
            )

        # Save image
        temp_dir = (
            self.environment_manager.state["artifacts_dir"]
            if self.environment_manager
            else Path(tempfile.gettempdir())
        )
        temp_dir.mkdir(exist_ok=True)

        image_file = (
            temp_dir
            / f"test_image_{image_type}_{width}x{height}_{id(image_array)}.png"
        )

        if channels == 1:
            image_array = np.squeeze(image_array)

        image = Image.fromarray(image_array)
        image.save(image_file)

        if self.environment_manager:
            self.environment_manager.register_temp_file(image_file)

        return {
            "data_type": "image",
            "file_path": image_file,
            "metadata": {
                "image_type": image_type,
                "width": width,
                "height": height,
                "channels": channels,
                "crack_density": (
                    crack_density if image_type == "crack" else 0.0
                ),
                "format": "png",
            },
            "cleanup_required": True,
        }

    def _generate_crack_image(
        self, width: int, height: int, channels: int, density: float
    ) -> np.ndarray[Any, Any]:
        """Generate an image with crack-like patterns."""
        # Base pavement texture
        base_image = np.random.randint(
            100, 180, (height, width, channels), dtype=np.uint8
        )

        # Add crack patterns
        num_cracks = int(density * width * height / 1000)
        for _ in range(num_cracks):
            start_x = np.random.randint(0, width)
            start_y = np.random.randint(0, height)
            length = np.random.randint(10, min(width, height) // 4)
            angle = np.random.uniform(0, 2 * np.pi)

            for i in range(length):
                x = int(start_x + i * np.cos(angle))
                y = int(start_y + i * np.sin(angle))

                if 0 <= x < width and 0 <= y < height:
                    base_image[y, x] = [20, 20, 20]  # Dark crack

        return base_image

    def _generate_clean_image(
        self, width: int, height: int, channels: int
    ) -> np.ndarray[Any, Any]:
        """Generate a clean image without cracks."""
        base_value = np.random.randint(120, 200)
        image = np.full((height, width, channels), base_value, dtype=np.uint8)
        noise = np.random.randint(-20, 20, (height, width, channels))
        return np.clip(image + noise, 0, 255).astype(np.uint8)

    def _generate_edge_case_image(
        self, width: int, height: int, channels: int
    ) -> np.ndarray[Any, Any]:
        """Generate edge case images (very small, very large cracks, etc.)."""
        # Create base image
        image = np.random.randint(
            100, 200, (height, width, channels), dtype=np.uint8
        )

        # Add edge case patterns
        edge_case_type = np.random.choice(
            ["tiny_cracks", "large_cracks", "complex_pattern"]
        )

        if edge_case_type == "tiny_cracks":
            # Very thin cracks (1 pixel width)
            for _ in range(20):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                length = np.random.randint(5, 15)

                for _ in range(length):
                    if 0 <= x < width and 0 <= y < height:
                        image[y, x] = [10, 10, 10]
                    x += np.random.randint(-1, 2)
                    y += np.random.randint(-1, 2)

        elif edge_case_type == "large_cracks":
            # Large crack patterns
            for _ in range(3):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                length = np.random.randint(width // 4, width // 2)

                for _ in range(length):
                    if 0 <= x < width and 0 <= y < height:
                        # Wide crack
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < width and 0 <= ny < height:
                                    image[ny, nx] = [5, 5, 5]
                    x += np.random.randint(-2, 3)
                    y += np.random.randint(-2, 3)

        return image
