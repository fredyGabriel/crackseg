"""Example script demonstrating segmentation mask verification.

This script shows how to use the SegmentationMaskVerifier to verify
the accuracy of segmentation masks through visual superposition,
as required for the CrackSeg project quality assurance workflow.
"""

import logging
from pathlib import Path

from segmentation_mask_verifier import SegmentationMaskVerifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Demonstrate mask verification with PY-CrackBD dataset."""

    # Define dataset paths
    dataset_root = Path("data/PY-CrackBD/Segmentation")
    images_dir = dataset_root / "Original image"
    masks_dir = dataset_root / "Ground truth"
    output_dir = Path("artifacts/verification_results")

    # Validate paths exist
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return

    if not masks_dir.exists():
        logger.error(f"Masks directory not found: {masks_dir}")
        return

    # Initialize verifier
    try:
        verifier = SegmentationMaskVerifier(
            images_dir=images_dir, masks_dir=masks_dir, output_dir=output_dir
        )
        logger.info("✅ Verifier initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize verifier: {e}")
        return

    # Find available image-mask pairs
    image_files = list(images_dir.glob("*.jpg"))
    mask_files = list(masks_dir.glob("*.png"))

    image_names = {f.stem for f in image_files}
    mask_names = {f.stem for f in mask_files}
    available_pairs = image_names.intersection(mask_names)

    if not available_pairs:
        logger.error("No matching image-mask pairs found")
        return

    logger.info(f"Found {len(available_pairs)} image-mask pairs")

    # Select a sample image for demonstration
    sample_image = sorted(available_pairs)[0]  # First available image
    logger.info(f"Using sample image: {sample_image}")

    # Verify single image-mask pair
    logger.info("=" * 50)
    logger.info("VERIFYING SINGLE IMAGE-MASK PAIR")
    logger.info("=" * 50)

    result = verifier.verify_single_pair(sample_image)

    if result["success"]:
        logger.info(f"✅ Successfully verified {sample_image}")
        stats = result["statistics"]
        logger.info(f"Image shape: {result['image_shape']}")
        logger.info(f"Mask shape: {result['mask_shape']}")
        logger.info(f"Crack coverage: {stats['coverage_percentage']:.2f}%")
        logger.info(f"Crack pixels: {stats['crack_pixels']:,}")
        logger.info(f"Number of crack regions: {stats['num_crack_regions']}")
        logger.info(f"Verification result saved to: {output_dir}")
    else:
        logger.error(
            f"❌ Failed to verify {sample_image}: {result.get('error', 'Unknown error')}"
        )

    # Verify multiple images (limited sample)
    logger.info("\n" + "=" * 50)
    logger.info("VERIFYING MULTIPLE IMAGE-MASK PAIRS")
    logger.info("=" * 50)

    try:
        # Verify first 5 images for demonstration
        results = verifier.verify_dataset(max_samples=5)

        logger.info(f"Total pairs processed: {results['total_pairs']}")
        logger.info(
            f"Successful verifications: {results['successful_verifications']}"
        )
        logger.info(f"Failed verifications: {results['failed_verifications']}")
        logger.info(f"Success rate: {results['success_rate']:.2%}")

        if results["overall_statistics"]:
            stats = results["overall_statistics"]
            logger.info(f"Average coverage: {stats['average_coverage']:.2f}%")
            logger.info(
                f"Coverage range: {stats['min_coverage']:.2f}% - {stats['max_coverage']:.2f}%"
            )
            logger.info(f"Total crack pixels: {stats['total_crack_pixels']:,}")

        logger.info(f"All verification results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"❌ Failed to verify dataset: {e}")

    logger.info("\n" + "=" * 50)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 50)
    logger.info(
        "The verification process has created visual overlays showing:"
    )
    logger.info("1. Original image")
    logger.info("2. Segmentation mask")
    logger.info("3. Superposition (mask overlaid on original image)")
    logger.info("")
    logger.info("This demonstrates the cross-review process for confirming")
    logger.info("segmentation mask accuracy through visual inspection.")


if __name__ == "__main__":
    main()
