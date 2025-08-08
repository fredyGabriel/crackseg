"""Segmentation mask verification and visualization tool.

This module provides functionality to verify the accuracy of segmentation masks
through cross-review process and visual superposition onto original images.
Supports the CrackSeg project's quality assurance workflow for crack detection.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases for clarity
type ImageArray = np.ndarray  # Shape: (H, W, C) for RGB images
type MaskArray = np.ndarray  # Shape: (H, W) for binary masks
type OverlayArray = np.ndarray  # Shape: (H, W, C) for superimposed images


class SegmentationMaskVerifier:
    """Verifies segmentation mask accuracy through visual superposition.

    This class implements the cross-review process for segmentation masks by
    loading original images and their corresponding ground truth masks, then
    creating visual overlays to confirm alignment accuracy.

    Attributes:
        images_dir: Directory containing original images
        masks_dir: Directory containing ground truth masks
        output_dir: Directory to save verification results
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize the mask verifier with directory paths.

        Args:
            images_dir: Path to directory containing original images
            masks_dir: Path to directory containing ground truth masks
            output_dir: Path to directory for saving verification results.
                If None, creates 'verification_results' in current directory.
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.output_dir = (
            Path(output_dir) if output_dir else Path("verification_results")
        )

        # Validate directories exist
        self._validate_directories()

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)

        logger.info(f"Initialized verifier with images: {self.images_dir}")
        logger.info(f"Masks directory: {self.masks_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def _validate_directories(self) -> None:
        """Validate that required directories exist and are accessible."""
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}"
            )

        if not self.masks_dir.exists():
            raise FileNotFoundError(
                f"Masks directory not found: {self.masks_dir}"
            )

        if not self.images_dir.is_dir():
            raise NotADirectoryError(
                f"Images path is not a directory: {self.images_dir}"
            )

        if not self.masks_dir.is_dir():
            raise NotADirectoryError(
                f"Masks path is not a directory: {self.masks_dir}"
            )

    def load_image(self, image_path: Path) -> ImageArray:
        """Load and validate an image file.

        Args:
            image_path: Path to the image file

        Returns:
            Loaded image as numpy array with shape (H, W, C)

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded or has invalid format
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Convert to numpy array
                image_array = np.array(img)

                # Validate shape
                if image_array.ndim != 3 or image_array.shape[2] != 3:
                    raise ValueError(
                        f"Expected RGB image with shape (H, W, 3), got {image_array.shape}"
                    )

                return image_array

        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}") from e

    def load_mask(self, mask_path: Path) -> MaskArray:
        """Load and validate a binary mask file.

        Args:
            mask_path: Path to the mask file

        Returns:
            Loaded mask as binary numpy array with shape (H, W)

        Raises:
            FileNotFoundError: If mask file doesn't exist
            ValueError: If mask cannot be loaded or has invalid format
        """
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        try:
            with Image.open(mask_path) as img:
                # Convert to grayscale if necessary
                if img.mode != "L":
                    img = img.convert("L")

                # Convert to numpy array
                mask_array = np.array(img)

                # Validate shape
                if mask_array.ndim != 2:
                    raise ValueError(
                        f"Expected 2D mask array, got shape {mask_array.shape}"
                    )

                # Normalize to binary (0 or 255)
                mask_array = (mask_array > 127).astype(np.uint8) * 255

                return mask_array

        except Exception as e:
            raise ValueError(f"Failed to load mask {mask_path}: {e}") from e

    def create_overlay(
        self,
        image: ImageArray,
        mask: MaskArray,
        overlay_color: tuple[int, int, int] = (255, 0, 0),  # Red
    ) -> OverlayArray:
        """Create a visual overlay of mask on original image.

        Args:
            image: Original image array with shape (H, W, C)
            mask: Binary mask array with shape (H, W)
            overlay_color: RGB color for mask overlay (default: red)

        Returns:
            Image with mask overlaid in specified color
        """
        # Validate input shapes
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image and mask must have same spatial dimensions. "
                f"Image: {image.shape[:2]}, Mask: {mask.shape}"
            )

        # Create overlay image
        overlay = image.copy()

        # Convert mask to boolean for indexing
        mask_bool = mask > 127

        # Apply overlay color to masked regions
        overlay[mask_bool] = overlay_color

        return overlay

    def verify_single_pair(
        self, image_name: str, save_result: bool = True
    ) -> dict[str, Any]:
        """Verify a single image-mask pair.

        Args:
            image_name: Name of the image file (without extension)
            save_result: Whether to save the verification result

        Returns:
            Dictionary containing verification results and statistics
        """
        # Construct file paths
        image_path = self.images_dir / f"{image_name}.jpg"
        mask_path = self.masks_dir / f"{image_name}.png"

        try:
            # Load image and mask
            image = self.load_image(image_path)
            mask = self.load_mask(mask_path)

            # Create overlay
            overlay = self.create_overlay(image, mask)

            # Calculate verification statistics
            stats = self._calculate_verification_stats(image, mask)

            # Create visualization
            if save_result:
                self._save_verification_result(
                    image_name, image, mask, overlay
                )

            return {
                "image_name": image_name,
                "success": True,
                "image_shape": image.shape,
                "mask_shape": mask.shape,
                "statistics": stats,
            }

        except Exception as e:
            logger.error(f"Failed to verify {image_name}: {e}")
            return {
                "image_name": image_name,
                "success": False,
                "error": str(e),
            }

    def _calculate_verification_stats(
        self, image: ImageArray, mask: MaskArray
    ) -> dict[str, float]:
        """Calculate verification statistics for image-mask pair.

        Args:
            image: Original image array
            mask: Binary mask array

        Returns:
            Dictionary containing verification statistics
        """
        # Calculate mask coverage
        total_pixels = mask.size
        crack_pixels = np.sum(mask > 127)
        coverage_percentage = (crack_pixels / total_pixels) * 100

        # Calculate mask properties
        mask_bool = mask > 127
        if np.any(mask_bool):
            # Calculate average crack width (simplified)
            from scipy import ndimage

            _labeled, num_features = ndimage.label(mask_bool.astype(bool))  # type: ignore

            # Calculate basic statistics
            stats = {
                "total_pixels": int(total_pixels),
                "crack_pixels": int(crack_pixels),
                "coverage_percentage": float(coverage_percentage),
                "num_crack_regions": int(num_features),
                "image_height": image.shape[0],
                "image_width": image.shape[1],
            }
        else:
            stats = {
                "total_pixels": int(total_pixels),
                "crack_pixels": 0,
                "coverage_percentage": 0.0,
                "num_crack_regions": 0,
                "image_height": image.shape[0],
                "image_width": image.shape[1],
            }

        return stats

    def _save_verification_result(
        self,
        image_name: str,
        image: ImageArray,
        mask: MaskArray,
        overlay: OverlayArray,
    ) -> None:
        """Save verification result as a composite visualization.

        Args:
            image_name: Name of the image being verified
            image: Original image array
            mask: Binary mask array
            overlay: Overlay image array
        """
        # Create figure with three subplots
        _fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        # Mask
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Segmentation Mask", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title("Superposition", fontsize=14, fontweight="bold")
        axes[2].axis("off")

        # Adjust layout and save
        plt.tight_layout()

        output_path = self.output_dir / f"{image_name}_verification.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved verification result: {output_path}")

    def verify_dataset(self, max_samples: int | None = None) -> dict[str, Any]:
        """Verify multiple image-mask pairs in the dataset.

        Args:
            max_samples: Maximum number of samples to verify.
                If None, verifies all available pairs.

        Returns:
            Dictionary containing overall verification results
        """
        # Find all available image-mask pairs
        image_files = list(self.images_dir.glob("*.jpg"))
        mask_files = list(self.masks_dir.glob("*.png"))

        # Create mapping of available pairs
        image_names = {f.stem for f in image_files}
        mask_names = {f.stem for f in mask_files}
        available_pairs = image_names.intersection(mask_names)

        if not available_pairs:
            raise ValueError("No matching image-mask pairs found")

        # Limit samples if specified
        if max_samples:
            available_pairs = set(list(available_pairs)[:max_samples])

        logger.info(f"Found {len(available_pairs)} image-mask pairs to verify")

        # Verify each pair
        results = []
        successful_verifications = 0

        for image_name in sorted(available_pairs):
            result = self.verify_single_pair(image_name)
            results.append(result)

            if result["success"]:
                successful_verifications += 1
                logger.info(f"✅ Verified {image_name}")
            else:
                logger.error(
                    f"❌ Failed to verify {image_name}: {result.get('error', 'Unknown error')}"
                )

        # Calculate overall statistics
        overall_stats = self._calculate_overall_stats(results)

        return {
            "total_pairs": len(available_pairs),
            "successful_verifications": successful_verifications,
            "failed_verifications": len(available_pairs)
            - successful_verifications,
            "success_rate": successful_verifications / len(available_pairs),
            "individual_results": results,
            "overall_statistics": overall_stats,
        }

    def _calculate_overall_stats(
        self, results: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate overall statistics from verification results.

        Args:
            results: List of individual verification results

        Returns:
            Dictionary containing overall statistics
        """
        successful_results = [r for r in results if r["success"]]

        if not successful_results:
            return {
                "average_coverage": 0.0,
                "min_coverage": 0.0,
                "max_coverage": 0.0,
                "total_crack_pixels": 0,
            }

        # Extract coverage percentages
        coverages = [
            r["statistics"]["coverage_percentage"] for r in successful_results
        ]
        crack_pixels = [
            r["statistics"]["crack_pixels"] for r in successful_results
        ]

        return {
            "average_coverage": float(np.mean(coverages)),
            "min_coverage": float(np.min(coverages)),
            "max_coverage": float(np.max(coverages)),
            "total_crack_pixels": int(np.sum(crack_pixels)),
        }


def main() -> None:
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Verify segmentation mask accuracy through visual superposition"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing original images",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        required=True,
        help="Directory containing ground truth masks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save verification results",
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to verify"
    )
    parser.add_argument(
        "--single-image", type=str, help="Verify only a single image-mask pair"
    )

    args = parser.parse_args()

    # Initialize verifier
    verifier = SegmentationMaskVerifier(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
    )

    # Perform verification
    if args.single_image:
        # Verify single image
        result = verifier.verify_single_pair(args.single_image)
        if result["success"]:
            logger.info(f"✅ Successfully verified {args.single_image}")
            logger.info(
                f"Coverage: {result['statistics']['coverage_percentage']:.2f}%"
            )
        else:
            logger.error(f"❌ Failed to verify {args.single_image}")
    else:
        # Verify entire dataset
        results = verifier.verify_dataset(max_samples=args.max_samples)

        logger.info("=" * 50)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total pairs: {results['total_pairs']}")
        logger.info(f"Successful: {results['successful_verifications']}")
        logger.info(f"Failed: {results['failed_verifications']}")
        logger.info(f"Success rate: {results['success_rate']:.2%}")

        if results["overall_statistics"]:
            stats = results["overall_statistics"]
            logger.info(f"Average coverage: {stats['average_coverage']:.2f}%")
            logger.info(
                f"Coverage range: {stats['min_coverage']:.2f}% - {stats['max_coverage']:.2f}%"
            )
            logger.info(f"Total crack pixels: {stats['total_crack_pixels']:,}")


if __name__ == "__main__":
    main()
