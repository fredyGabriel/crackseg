#!/usr/bin/env python3
"""
Specialized script to process PY-CrackDB images from 351x500 to 320x320.

This script analyzes binarized masks in 4 quadrants to determine the optimal
cropping strategy that preserves the maximum amount of crack pixels.
The algorithm considers both horizontal and vertical cuts simultaneously.

Author: CrackSeg Project
Date: 2024
"""

import argparse
import logging
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


class QuadrantDensity(NamedTuple):
    """Density information for a quadrant."""

    density: float
    crack_pixels: int
    total_pixels: int


class CropDecision(NamedTuple):
    """Decision for cropping strategy."""

    horizontal_cut: str  # "left" or "right"
    vertical_cut: str  # "top" or "bottom"
    total_crack_pixels: int
    overall_density: float


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("crop_py_crackdb_images.log"),
            logging.StreamHandler(),
        ],
    )


def analyze_quadrant_density(mask: np.ndarray) -> dict[str, QuadrantDensity]:
    """
    Analyzes crack density in all 4 quadrants of the image.

    Args:
        mask: Binarized mask (0 = background, 255 = crack)

    Returns:
        Dictionary with density information for each quadrant
    """
    # Verify that the mask is binarized
    unique_values = np.unique(mask)
    if not (set(unique_values) <= {0, 255}):
        logging.warning(
            f"Mask is not completely binarized. Unique values: {unique_values}"
        )

    # Convert to binary (0 and 1)
    binary_mask = (mask > 127).astype(np.uint8)

    # Get dimensions
    height, width = binary_mask.shape

    # Calculate quadrant boundaries
    mid_width = width // 2
    mid_height = height // 2

    # Define quadrants: top-left, top-right, bottom-left, bottom-right
    quadrants = {
        "top_left": binary_mask[:mid_height, :mid_width],
        "top_right": binary_mask[:mid_height, mid_width:],
        "bottom_left": binary_mask[mid_height:, :mid_width],
        "bottom_right": binary_mask[mid_height:, mid_width:],
    }

    # Calculate density for each quadrant
    quadrant_densities = {}
    for name, quadrant in quadrants.items():
        crack_pixels = np.sum(quadrant)
        total_pixels = quadrant.size
        density = crack_pixels / total_pixels if total_pixels > 0 else 0.0

        quadrant_densities[name] = QuadrantDensity(
            density=float(density),
            crack_pixels=int(crack_pixels),
            total_pixels=total_pixels,
        )

        logging.debug(
            f"{name}: density={density:.4f}, "
            f"crack_pixels={crack_pixels}, total_pixels={total_pixels}"
        )

    return quadrant_densities


def determine_optimal_crop(
    quadrant_densities: dict[str, QuadrantDensity],
) -> CropDecision:
    """
    Determines the optimal cropping strategy based on quadrant densities.

    Args:
        quadrant_densities: Density information for all 4 quadrants

    Returns:
        CropDecision with optimal horizontal and vertical cuts
    """
    # Calculate horizontal preference (left vs right)
    left_density = (
        quadrant_densities["top_left"].density
        + quadrant_densities["bottom_left"].density
    ) / 2
    right_density = (
        quadrant_densities["top_right"].density
        + quadrant_densities["bottom_right"].density
    ) / 2

    horizontal_cut = "left" if left_density >= right_density else "right"

    # Calculate vertical preference (top vs bottom)
    top_density = (
        quadrant_densities["top_left"].density
        + quadrant_densities["top_right"].density
    ) / 2
    bottom_density = (
        quadrant_densities["bottom_left"].density
        + quadrant_densities["bottom_right"].density
    ) / 2

    vertical_cut = "top" if top_density >= bottom_density else "bottom"

    # Calculate total crack pixels for the selected region
    if horizontal_cut == "left" and vertical_cut == "top":
        selected_quadrant = quadrant_densities["top_left"]
    elif horizontal_cut == "right" and vertical_cut == "top":
        selected_quadrant = quadrant_densities["top_right"]
    elif horizontal_cut == "left" and vertical_cut == "bottom":
        selected_quadrant = quadrant_densities["bottom_left"]
    else:  # right and bottom
        selected_quadrant = quadrant_densities["bottom_right"]

    # Calculate overall density for the selected region
    overall_density = selected_quadrant.density

    logging.info(
        f"Crop decision: horizontal={horizontal_cut}, vertical={vertical_cut}, "
        f"density={overall_density:.4f}, crack_pixels={selected_quadrant.crack_pixels}"
    )

    return CropDecision(
        horizontal_cut=horizontal_cut,
        vertical_cut=vertical_cut,
        total_crack_pixels=selected_quadrant.crack_pixels,
        overall_density=overall_density,
    )


def crop_image_bidirectional(
    image: np.ndarray,
    decision: CropDecision,
    target_width: int = 320,
    target_height: int = 320,
) -> np.ndarray:
    """
    Crops the image according to the bidirectional decision.

    Args:
        image: Input image (351x500) or mask (351x500)
        decision: CropDecision with horizontal and vertical preferences
        target_width: Target width (320)
        target_height: Target height (320)

    Returns:
        Cropped image (320x320) or mask (320x320)
    """
    height, width = image.shape[:2]

    # Calculate crop boundaries
    if decision.horizontal_cut == "left":
        x_start = 0
        x_end = target_width
    else:  # right
        x_start = width - target_width
        x_end = width

    if decision.vertical_cut == "top":
        y_start = 0
        y_end = target_height
    else:  # bottom
        y_start = height - target_height
        y_end = height

    # Perform the crop
    cropped = image[y_start:y_end, x_start:x_end]

    # Verify output dimensions based on input type
    if len(image.shape) == 3:  # Image with channels
        expected_shape = (target_height, target_width, 3)
    else:  # Mask (2D)
        expected_shape = (target_height, target_width)

    if cropped.shape != expected_shape:
        raise ValueError(
            f"Crop error: expected shape {expected_shape}, got {cropped.shape}"
        )

    return cropped


def validate_input_dimensions(image: np.ndarray, mask: np.ndarray) -> bool:
    """
    Validates that input dimensions are correct for PY-CrackDB.

    Args:
        image: Input image
        mask: Input mask

    Returns:
        True if dimensions are correct
    """
    img_height, img_width = image.shape[:2]
    mask_height, mask_width = mask.shape[:2]

    # Verify expected dimensions for PY-CrackDB
    if img_width != 351 or img_height != 500:
        logging.error(
            f"Incorrect image dimensions: {img_width}x{img_height}, expected 351x500"
        )
        return False

    if mask_width != 351 or mask_height != 500:
        logging.error(
            f"Incorrect mask dimensions: {mask_width}x{mask_height}, expected 351x500"
        )
        return False

    # Verify that image and mask have the same dimensions
    if img_width != mask_width or img_height != mask_height:
        logging.error(
            f"Dimensions don't match: image {img_width}x{img_height}, "
            f"mask {mask_width}x{mask_height}"
        )
        return False

    return True


def find_matching_files(
    image_dir: Path, mask_dir: Path
) -> list[tuple[Path, Path]]:
    """
    Finds corresponding image-mask file pairs.

    Args:
        image_dir: Images directory
        mask_dir: Masks directory

    Returns:
        List of tuples (image_path, mask_path)
    """
    pairs = []

    # Get image files
    image_files = list(image_dir.glob("*.jpg")) + list(
        image_dir.glob("*.jpeg")
    )

    for image_path in image_files:
        # Look for corresponding mask
        mask_name = image_path.stem + ".png"
        mask_path = mask_dir / mask_name

        if mask_path.exists():
            pairs.append((image_path, mask_path))
        else:
            logging.warning(f"No mask found for {image_path.name}")

    return pairs


def process_single_pair(
    image_path: Path,
    mask_path: Path,
    output_image_dir: Path,
    output_mask_dir: Path,
    file_number: int,
) -> dict[str, object]:
    """
    Processes a single image-mask file pair.

    Args:
        image_path: Input image path
        mask_path: Input mask path
        output_image_dir: Output directory for images
        output_mask_dir: Output directory for masks
        file_number: Sequential number for renaming

    Returns:
        Dictionary with processing information
    """
    try:
        # Load image and mask
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")

        # Validate dimensions
        if not validate_input_dimensions(image, mask):
            return {
                "success": False,
                "error": "Incorrect dimensions",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }

        # Analyze quadrant densities
        quadrant_densities = analyze_quadrant_density(mask)

        # Determine optimal crop
        decision = determine_optimal_crop(quadrant_densities)

        # Crop image and mask
        cropped_image = crop_image_bidirectional(image, decision)
        cropped_mask = crop_image_bidirectional(mask, decision)

        # Verify output dimensions
        if cropped_image.shape != (320, 320, 3):
            raise ValueError(
                f"Incorrect output image dimensions: {cropped_image.shape}"
            )

        if cropped_mask.shape != (320, 320):
            raise ValueError(
                f"Incorrect output mask dimensions: {cropped_mask.shape}"
            )

        # Generate output filenames
        output_image_name = f"{file_number}.jpg"
        output_mask_name = f"{file_number}.png"

        output_image_path = output_image_dir / output_image_name
        output_mask_path = output_mask_dir / output_mask_name

        # Save files
        cv2.imwrite(str(output_image_path), cropped_image)
        cv2.imwrite(str(output_mask_path), cropped_mask)

        logging.info(
            f"Processed: {image_path.name} -> {output_image_name} "
            f"(h_cut: {decision.horizontal_cut}, v_cut: {decision.vertical_cut}, "
            f"density: {decision.overall_density:.4f})"
        )

        return {
            "success": True,
            "horizontal_cut": decision.horizontal_cut,
            "vertical_cut": decision.vertical_cut,
            "crack_pixels": decision.total_crack_pixels,
            "density": decision.overall_density,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "output_image": str(output_image_path),
            "output_mask": str(output_mask_path),
        }

    except Exception as e:
        logging.error(f"Error processing {image_path.name}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def process_dataset(
    input_dir: str,
    output_dir: str,
    image_dir: str = "images",
    mask_dir: str = "masks",
) -> dict[str, object]:
    """
    Processes the entire PY-CrackDB dataset.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        image_dir: Images subdirectory
        mask_dir: Masks subdirectory

    Returns:
        Dictionary with processing statistics
    """
    # Configure paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    input_image_dir = input_path / image_dir
    input_mask_dir = input_path / mask_dir
    output_image_dir = output_path / image_dir
    output_mask_dir = output_path / mask_dir

    # Create output directories
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    # Verify input directories
    if not input_image_dir.exists():
        raise ValueError(f"Images directory not found: {input_image_dir}")

    if not input_mask_dir.exists():
        raise ValueError(f"Masks directory not found: {input_mask_dir}")

    # Find file pairs
    file_pairs = find_matching_files(input_image_dir, input_mask_dir)

    if not file_pairs:
        raise ValueError("No image-mask file pairs found")

    logging.info(f"Found {len(file_pairs)} file pairs to process")
    logging.info("Expected input dimensions: 351x500")
    logging.info("Target output dimensions: 320x320")

    # Statistics
    stats = {
        "total_files": len(file_pairs),
        "processed_successfully": 0,
        "errors": 0,
        "crop_decisions": {
            "left_top": 0,
            "left_bottom": 0,
            "right_top": 0,
            "right_bottom": 0,
        },
        "total_crack_pixels_preserved": 0,
        "average_density": 0.0,
        "error_details": [],
    }

    # Process each file pair
    start_time = time.time()

    for i, (image_path, mask_path) in enumerate(file_pairs, 1):
        result = process_single_pair(
            image_path, mask_path, output_image_dir, output_mask_dir, i
        )

        if result["success"]:
            stats["processed_successfully"] += 1

            # Track crop decisions
            decision_key = (
                f"{result['horizontal_cut']}_{result['vertical_cut']}"
            )
            stats["crop_decisions"][decision_key] += 1

            # Track crack preservation
            stats["total_crack_pixels_preserved"] += result["crack_pixels"]
            stats["average_density"] += result["density"]
        else:
            stats["errors"] += 1
            stats["error_details"].append(result)

    # Calculate final statistics
    total_time = time.time() - start_time

    if stats["processed_successfully"] > 0:
        stats["average_density"] /= stats["processed_successfully"]

    # Generate report
    logging.info("=" * 60)
    logging.info("PY-CRACKDB PROCESSING REPORT")
    logging.info("=" * 60)
    logging.info(f"Total files: {stats['total_files']}")
    logging.info(f"Successfully processed: {stats['processed_successfully']}")
    logging.info(f"Errors: {stats['errors']}")
    logging.info(
        f"Total crack pixels preserved: {stats['total_crack_pixels_preserved']:,}"
    )
    logging.info(f"Average density: {stats['average_density']:.4f}")

    logging.info("\nCrop decisions:")
    for decision, count in stats["crop_decisions"].items():
        percentage = (
            (count / stats["processed_successfully"] * 100)
            if stats["processed_successfully"] > 0
            else 0
        )
        logging.info(f"  {decision}: {count} ({percentage:.1f}%)")

    logging.info(f"\nTotal time: {total_time:.2f} seconds")
    logging.info(
        f"Average time per file: {total_time / stats['total_files']:.2f} seconds"
    )

    if stats["errors"] > 0:
        logging.warning("\nFiles with errors:")
        for error in stats["error_details"]:
            logging.warning(f"  - {error['image_path']}: {error['error']}")

    return stats


def main():
    """Main function of the script."""
    parser = argparse.ArgumentParser(
        description="Process PY-CrackDB images from 351x500 to 320x320 with bidirectional cropping"
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input directory with 'images' and 'masks' subdirectories",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory where processed images will be saved",
    )

    parser.add_argument(
        "--image_dir",
        default="images",
        help="Images subdirectory name (default: 'images')",
    )

    parser.add_argument(
        "--mask_dir",
        default="masks",
        help="Masks subdirectory name (default: 'masks')",
    )

    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    setup_logging(args.log_level)

    logging.info("Starting PY-CrackDB image processing")
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info("Processing: 351x500 -> 320x320 with bidirectional cropping")

    try:
        # Process dataset
        process_dataset(
            args.input_dir, args.output_dir, args.image_dir, args.mask_dir
        )

        logging.info("Processing completed successfully")

    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
