#!/usr/bin/env python3
"""
Convenience script to process CFD dataset from 480x320 to 320x320.

This script is a wrapper around crop_crack_images_configurable.py
specifically for the CFD dataset processing.

Author: CrackSeg Project
Date: 2024
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def validate_cfd_structure(input_dir: Path) -> bool:
    """
    Validate that the CFD dataset has the expected structure.

    Args:
        input_dir: Path to the CFD dataset directory

    Returns:
        True if structure is valid
    """
    images_dir = input_dir / "images"
    masks_dir = input_dir / "masks"

    if not images_dir.exists():
        logging.error(f"Images directory not found: {images_dir}")
        return False

    if not masks_dir.exists():
        logging.error(f"Masks directory not found: {masks_dir}")
        return False

    # Check if there are image files
    image_files = list(images_dir.glob("*.jpg"))
    if not image_files:
        logging.error(f"No JPG files found in {images_dir}")
        return False

    # Check if there are mask files
    mask_files = list(masks_dir.glob("*.png"))
    if not mask_files:
        logging.error(f"No PNG files found in {masks_dir}")
        return False

    logging.info(
        f"Found {len(image_files)} images and {len(mask_files)} masks"
    )
    return True


def process_cfd_dataset(
    input_dir: str,
    output_dir: str,
    log_level: str = "INFO",
    force: bool = False,
) -> int:
    """
    Process CFD dataset from 480x320 to 320x320.

    Args:
        input_dir: Input directory containing CFD dataset
        output_dir: Output directory for processed images
        log_level: Logging level
        force: Force overwrite of existing output directory

    Returns:
        0 on success, 1 on error
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Validate input structure
    if not validate_cfd_structure(input_path):
        return 1

    # Check if output directory exists
    if output_path.exists() and not force:
        logging.error(
            f"Output directory {output_dir} already exists. "
            "Use --force to overwrite."
        )
        return 1

    # Build command for configurable script
    cmd = [
        sys.executable,
        "crop_crack_images_configurable.py",
        "--input_dir",
        input_dir,
        "--output_dir",
        output_dir,
        "--expected_width",
        "480",
        "--expected_height",
        "320",
        "--image_dir",
        "images",
        "--mask_dir",
        "masks",
        "--log_level",
        log_level,
    ]

    logging.info("Starting CFD dataset processing")
    logging.info(f"Input: {input_dir}")
    logging.info(f"Output: {output_dir}")
    logging.info("Transformation: 480x320 → 320x320")

    try:
        # Run the configurable script
        subprocess.run(cmd, check=True, capture_output=False)
        logging.info("CFD dataset processing completed successfully")
        return 0

    except subprocess.CalledProcessError as e:
        logging.error(f"Error during processing: {e}")
        return 1
    except FileNotFoundError:
        logging.error(
            "crop_crack_images_configurable.py not found. "
            "Make sure you're running this script from the correct directory."
        )
        return 1


def main():
    """Main function of the script."""
    parser = argparse.ArgumentParser(
        description="Process CFD dataset from 480x320 to 320x320"
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input directory containing CFD dataset with 'images' and 'masks' subdirectories",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory where processed images will be saved",
    )

    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing output directory",
    )

    args = parser.parse_args()

    # Configure logging
    setup_logging(args.log_level)

    # Process the dataset
    return process_cfd_dataset(
        args.input_dir,
        args.output_dir,
        args.log_level,
        args.force,
    )


if __name__ == "__main__":
    exit(main())
