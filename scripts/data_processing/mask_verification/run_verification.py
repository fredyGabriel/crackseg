#!/usr/bin/env python3
"""Command-line script for running segmentation mask verification.

This script provides a convenient command-line interface for running
the segmentation mask verification process with various options.
"""

import argparse
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from segmentation_mask_verifier import SegmentationMaskVerifier


def main() -> None:
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Verify segmentation mask accuracy through visual superposition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify a single image-mask pair
  python run_verification.py --single-image 125

  # Verify multiple images (max 10)
  python run_verification.py --max-samples 10

  # Verify all available pairs
  python run_verification.py

  # Use custom output directory
  python run_verification.py --output-dir ./my_results --max-samples 5
        """,
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/PY-CrackBD/Segmentation/Original image"),
        help="Directory containing original images (default: data/PY-CrackBD/Segmentation/Original image)",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=Path("data/PY-CrackBD/Segmentation/Ground truth"),
        help="Directory containing ground truth masks (default: data/PY-CrackBD/Segmentation/Ground truth)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/verification_results"),
        help="Directory to save verification results (default: artifacts/verification_results)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to verify (default: all available)",
    )
    parser.add_argument(
        "--single-image",
        type=str,
        help="Verify only a single image-mask pair by name",
    )
    parser.add_argument(
        "--list-available",
        action="store_true",
        help="List all available image-mask pairs and exit",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Adjust paths to work from project root
    project_root = Path(__file__).parent.parent.parent
    images_dir = project_root / args.images_dir
    masks_dir = project_root / args.masks_dir
    output_dir = project_root / args.output_dir

    # Check if directories exist
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        sys.exit(1)

    if not masks_dir.exists():
        print(f"❌ Masks directory not found: {masks_dir}")
        sys.exit(1)

    # Find available image-mask pairs
    image_files = list(images_dir.glob("*.jpg"))
    mask_files = list(masks_dir.glob("*.png"))

    image_names = {f.stem for f in image_files}
    mask_names = {f.stem for f in mask_files}
    available_pairs = image_names.intersection(mask_names)

    if not available_pairs:
        print("❌ No matching image-mask pairs found")
        sys.exit(1)

    # List available pairs if requested
    if args.list_available:
        print(f"📋 Available image-mask pairs ({len(available_pairs)} total):")
        for pair in sorted(available_pairs):
            print(f"  - {pair}")
        sys.exit(0)

    # Initialize verifier
    try:
        verifier = SegmentationMaskVerifier(
            images_dir=images_dir, masks_dir=masks_dir, output_dir=output_dir
        )
        print("✅ Verifier initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize verifier: {e}")
        sys.exit(1)

    # Perform verification
    if args.single_image:
        # Verify single image
        if args.single_image not in available_pairs:
            print(
                f"❌ Image '{args.single_image}' not found in available pairs"
            )
            print(f"Available pairs: {sorted(available_pairs)[:10]}...")
            sys.exit(1)

        print(f"🔍 Verifying single image: {args.single_image}")
        result = verifier.verify_single_pair(args.single_image)

        if result["success"]:
            print("✅ Verification completed successfully!")
            stats = result["statistics"]
            print("\n📊 RESULTS:")
            print(f"Image: {args.single_image}")
            print(f"Image shape: {result['image_shape']}")
            print(f"Mask shape: {result['mask_shape']}")
            print(f"Crack coverage: {stats['coverage_percentage']:.2f}%")
            print(f"Crack pixels: {stats['crack_pixels']:,}")
            print(f"Number of crack regions: {stats['num_crack_regions']}")
            print(f"Verification result saved to: {output_dir}")
        else:
            print(
                f"❌ Verification failed: {result.get('error', 'Unknown error')}"
            )
            sys.exit(1)

    else:
        # Verify multiple images
        max_samples = args.max_samples if args.max_samples else None
        if max_samples:
            print(f"🔄 Verifying up to {max_samples} image-mask pairs...")
        else:
            print(
                f"🔄 Verifying all {len(available_pairs)} image-mask pairs..."
            )

        try:
            results = verifier.verify_dataset(max_samples=max_samples)

            print("✅ Verification completed successfully!")
            print("\n📊 SUMMARY:")
            print(f"Total pairs processed: {results['total_pairs']}")
            print(
                f"Successful verifications: {results['successful_verifications']}"
            )
            print(f"Failed verifications: {results['failed_verifications']}")
            print(f"Success rate: {results['success_rate']:.2%}")

            if results["overall_statistics"]:
                stats = results["overall_statistics"]
                print("\n📈 STATISTICS:")
                print(f"Average coverage: {stats['average_coverage']:.2f}%")
                print(
                    f"Coverage range: {stats['min_coverage']:.2f}% - {stats['max_coverage']:.2f}%"
                )
                print(f"Total crack pixels: {stats['total_crack_pixels']:,}")

            print(f"\n💾 All verification results saved to: {output_dir}")

            # Show failed verifications if any
            failed_results = [
                r for r in results["individual_results"] if not r["success"]
            ]
            if failed_results:
                print("\n❌ FAILED VERIFICATIONS:")
                for result in failed_results:
                    print(
                        f"  - {result['image_name']}: {result.get('error', 'Unknown error')}"
                    )

        except Exception as e:
            print(f"❌ Verification failed: {e}")
            sys.exit(1)

    print("\n🎯 VERIFICATION PROCESS COMPLETED")
    print("The verification process has created visual overlays showing:")
    print("1. Original image")
    print("2. Segmentation mask")
    print("3. Superposition (mask overlaid on original image)")
    print("\nThis demonstrates the cross-review process for confirming")
    print("segmentation mask accuracy through visual inspection.")


if __name__ == "__main__":
    main()
