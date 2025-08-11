#!/usr/bin/env python3
"""
Example script to process the complete PY-CrackDB dataset.

This script demonstrates how to use the bidirectional cropping algorithm
to process all 369 images from 351x500 to 320x320.

Author: CrackSeg Project
Date: 2024
"""

import logging
from pathlib import Path

from crop_py_crackdb_images import process_dataset


def main() -> None:
    """Process the complete PY-CrackDB dataset."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Define paths
    input_dir = "data/PY-CrackBD"
    output_dir = "data/PY-CrackBD_processed"

    # Verify input directory exists
    if not Path(input_dir).exists():
        logging.error(f"Input directory not found: {input_dir}")
        logging.error("Please ensure the PY-CrackDB dataset is available")
        return 1

    logging.info("Starting PY-CrackDB dataset processing")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("Processing: 351x500 -> 320x320 with bidirectional cropping")

    try:
        # Process the dataset
        stats = process_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            image_dir="images",
            mask_dir="masks",
        )

        # Display summary
        logging.info("\n" + "=" * 60)
        logging.info("PROCESSING SUMMARY")
        logging.info("=" * 60)
        logging.info(
            f"✅ Successfully processed: {stats['processed_successfully']}/{stats['total_files']} images"
        )
        logging.info(f"❌ Errors: {stats['errors']}")
        logging.info(
            f"📊 Total crack pixels preserved: {stats['total_crack_pixels_preserved']:,}"
        )
        logging.info(f"📈 Average density: {stats['average_density']:.4f}")

        logging.info("\nCrop decision distribution:")
        for decision, count in stats["crop_decisions"].items():
            processed_count = stats["processed_successfully"]
            percentage = (
                (count / processed_count * 100)
                if isinstance(processed_count, int | float)
                and processed_count > 0
                else 0
            )
            logging.info(f"  {decision}: {count} ({percentage:.1f}%)")

        logging.info(f"\n📁 Output saved to: {output_dir}")
        logging.info("✅ Processing completed successfully!")

        return 0

    except Exception as e:
        logging.error(f"❌ Error during processing: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
