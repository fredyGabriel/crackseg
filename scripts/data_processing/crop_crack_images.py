#!/usr/bin/env python3
"""
Script to process pavement crack images from 640x360 to 360x360.

This script analyzes binarized masks to determine which side of the image
contains more cracks, crops the image and corresponding mask, and renames
files sequentially.

Author: CrackSeg Project
Date: 2024
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("crop_crack_images.log"),
            logging.StreamHandler(),
        ],
    )


def analyze_crack_density(mask: np.ndarray) -> str:
    """
    Analyzes crack density in each half of the image.

    Args:
        mask: Binarized mask (0 = background, 255 = crack)

    Returns:
        "left" if the left half has more cracks
        "right" if the right half has more cracks
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
    _, width = binary_mask.shape

    # Divide into two halves
    left_half = binary_mask[:, : width // 2]
    right_half = binary_mask[:, width // 2 :]

    # Calculate crack density (crack pixels / total pixels)
    left_density = np.sum(left_half) / left_half.size
    right_density = np.sum(right_half) / right_half.size

    # Count crack pixels for logging
    left_crack_pixels = np.sum(left_half)
    right_crack_pixels = np.sum(right_half)

    logging.debug(
        f"Left density: {left_density:.4f} ({left_crack_pixels} pixels)"
    )
    logging.debug(
        f"Right density: {right_density:.4f} ({right_crack_pixels} pixels)"
    )

    # Return the side with higher density
    # In case of tie, prefer left side
    return "left" if left_density >= right_density else "right"


def crop_image(
    image: np.ndarray, side: str, target_width: int = 360
) -> np.ndarray:
    """
    Crops the image according to the selected side.

    Args:
        image: Input image (640x360)
        side: "left" or "right"
        target_width: Target width (360)

    Returns:
        Cropped image (360x360)
    """
    _, width = image.shape[:2]

    if side == "left":
        # Keep pixels 0-360 (left side)
        cropped = image[:, :target_width]
    else:
        # Keep pixels 280-640 (last 360 pixels from right side)
        cropped = image[:, width - target_width :]

    # Verify output dimensions
    if cropped.shape[1] != target_width:
        raise ValueError(
            f"Crop error: expected width {target_width}, got {cropped.shape[1]}"
        )

    return cropped


def validate_input_dimensions(image: np.ndarray, mask: np.ndarray) -> bool:
    """
    Validates that input dimensions are correct.

    Args:
        image: Input image
        mask: Input mask

    Returns:
        True if dimensions are correct
    """
    img_height, img_width = image.shape[:2]
    mask_height, mask_width = mask.shape[:2]

    # Verify expected dimensions
    if img_width != 640 or img_height != 360:
        logging.error(
            f"Incorrect image dimensions: {img_width}x{img_height}, expected 640x360"
        )
        return False

    if mask_width != 640 or mask_height != 360:
        logging.error(
            f"Incorrect mask dimensions: {mask_width}x{mask_height}, expected 640x360"
        )
        return False

    # Verify that image and mask have the same dimensions
    if img_width != mask_width or img_height != mask_height:
        logging.error(
            f"Dimensions don't match: image {img_width}x{img_height}, mask {mask_width}x{mask_height}"
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

        # Analyze crack density
        side = analyze_crack_density(mask)

        # Crop image and mask
        cropped_image = crop_image(image, side)
        cropped_mask = crop_image(mask, side)

        # Verify output dimensions
        if cropped_image.shape != (360, 360, 3):
            raise ValueError(
                f"Incorrect output image dimensions: {cropped_image.shape}"
            )

        if cropped_mask.shape != (360, 360):
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
            f"Processed: {image_path.name} -> {output_image_name} (crop: {side})"
        )

        return {
            "success": True,
            "side": side,
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
    Processes the entire crack image dataset.

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

    # Statistics
    stats = {
        "total_files": len(file_pairs),
        "processed_successfully": 0,
        "errors": 0,
        "left_cuts": 0,
        "right_cuts": 0,
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
            if result["side"] == "left":
                stats["left_cuts"] += 1
            else:
                stats["right_cuts"] += 1
        else:
            stats["errors"] += 1
            stats["error_details"].append(result)

    # Calculate total time
    total_time = time.time() - start_time

    # Generate report
    logging.info("=" * 50)
    logging.info("PROCESSING REPORT")
    logging.info("=" * 50)
    logging.info(f"Total files: {stats['total_files']}")
    logging.info(f"Successfully processed: {stats['processed_successfully']}")
    logging.info(f"Errors: {stats['errors']}")
    logging.info(f"Left cuts: {stats['left_cuts']}")
    logging.info(f"Right cuts: {stats['right_cuts']}")
    logging.info(f"Total time: {total_time:.2f} seconds")
    logging.info(
        f"Average time per file: {total_time / stats['total_files']:.2f} seconds"
    )

    if stats["errors"] > 0:
        logging.warning("Files with errors:")
        for error in stats["error_details"]:
            logging.warning(f"  - {error['image_path']}: {error['error']}")

    return stats


def main():
    """Main function of the script."""
    parser = argparse.ArgumentParser(
        description="Process crack images from 640x360 to 360x360 with intelligent cropping"
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

    logging.info("Starting crack image processing")
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")

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
