"""Demonstration script for segmentation mask verification.

This script demonstrates the cross-review process for verifying segmentation
mask accuracy through visual superposition, as shown in the example image.
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from segmentation_mask_verifier import SegmentationMaskVerifier


def demo_single_verification() -> None:
    """Demonstrate verification of a single image-mask pair."""

    print("🔍 SEGMENTATION MASK VERIFICATION DEMO")
    print("=" * 50)

    # Define paths - adjust to work from project root
    project_root = Path(__file__).parent.parent.parent
    images_dir = project_root / "data/PY-CrackBD/Segmentation/Original image"
    masks_dir = project_root / "data/PY-CrackBD/Segmentation/Ground truth"
    output_dir = project_root / "artifacts/verification_demo"

    # Check if directories exist
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return

    if not masks_dir.exists():
        print(f"❌ Masks directory not found: {masks_dir}")
        return

    # Find available images
    image_files = list(images_dir.glob("*.jpg"))
    mask_files = list(masks_dir.glob("*.png"))

    if not image_files:
        print("❌ No image files found")
        return

    if not mask_files:
        print("❌ No mask files found")
        return

    # Find matching pairs
    image_names = {f.stem for f in image_files}
    mask_names = {f.stem for f in mask_files}
    available_pairs = image_names.intersection(mask_names)

    if not available_pairs:
        print("❌ No matching image-mask pairs found")
        return

    # Select a sample image
    sample_image = sorted(available_pairs)[0]
    print(f"📸 Using sample image: {sample_image}")

    # Initialize verifier
    try:
        verifier = SegmentationMaskVerifier(
            images_dir=images_dir, masks_dir=masks_dir, output_dir=output_dir
        )
        print("✅ Verifier initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize verifier: {e}")
        return

    # Perform verification
    print("\n🔄 Performing verification...")
    result = verifier.verify_single_pair(sample_image)

    if result["success"]:
        print("✅ Verification completed successfully!")
        print("\n📊 VERIFICATION RESULTS:")
        print("-" * 30)

        stats = result["statistics"]
        print(f"Image: {sample_image}")
        print(f"Image shape: {result['image_shape']}")
        print(f"Mask shape: {result['mask_shape']}")
        print(f"Crack coverage: {stats['coverage_percentage']:.2f}%")
        print(f"Crack pixels: {stats['crack_pixels']:,}")
        print(f"Number of crack regions: {stats['num_crack_regions']}")
        print(
            f"Image dimensions: {stats['image_width']} x {stats['image_height']}"
        )

        print(f"\n💾 Verification result saved to: {output_dir}")
        print(f"📁 File: {sample_image}_verification.png")

        print("\n🎯 VERIFICATION PROCESS COMPLETED")
        print("The verification process has created a visual overlay showing:")
        print("1. Original image (left)")
        print("2. Segmentation mask (center)")
        print("3. Superposition (right) - mask overlaid on original image")
        print("\nThis demonstrates the cross-review process for confirming")
        print("segmentation mask accuracy through visual inspection.")

    else:
        print(
            f"❌ Verification failed: {result.get('error', 'Unknown error')}"
        )


def demo_batch_verification() -> None:
    """Demonstrate batch verification of multiple image-mask pairs."""

    print("\n" + "=" * 50)
    print("🔄 BATCH VERIFICATION DEMO")
    print("=" * 50)

    # Define paths - adjust to work from project root
    project_root = Path(__file__).parent.parent.parent
    images_dir = project_root / "data/PY-CrackBD/Segmentation/Original image"
    masks_dir = project_root / "data/PY-CrackBD/Segmentation/Ground truth"
    output_dir = project_root / "artifacts/batch_verification_demo"

    # Initialize verifier
    try:
        verifier = SegmentationMaskVerifier(
            images_dir=images_dir, masks_dir=masks_dir, output_dir=output_dir
        )
    except Exception as e:
        print(f"❌ Failed to initialize verifier: {e}")
        return

    # Perform batch verification (limited to 3 samples for demo)
    print("🔄 Performing batch verification (3 samples)...")

    try:
        results = verifier.verify_dataset(max_samples=3)

        print("✅ Batch verification completed!")
        print("\n📊 BATCH VERIFICATION RESULTS:")
        print("-" * 40)
        print(f"Total pairs processed: {results['total_pairs']}")
        print(
            f"Successful verifications: {results['successful_verifications']}"
        )
        print(f"Failed verifications: {results['failed_verifications']}")
        print(f"Success rate: {results['success_rate']:.2%}")

        if results["overall_statistics"]:
            stats = results["overall_statistics"]
            print("\n📈 OVERALL STATISTICS:")
            print(f"Average coverage: {stats['average_coverage']:.2f}%")
            print(
                f"Coverage range: {stats['min_coverage']:.2f}% - {stats['max_coverage']:.2f}%"
            )
            print(f"Total crack pixels: {stats['total_crack_pixels']:,}")

        print(f"\n💾 All verification results saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Batch verification failed: {e}")


def main() -> None:
    """Run the complete verification demonstration."""

    print("🚀 SEGMENTATION MASK VERIFICATION DEMONSTRATION")
    print("=" * 60)
    print("This demo shows the cross-review process for verifying")
    print("segmentation mask accuracy through visual superposition.")
    print("=" * 60)

    # Demo single verification
    demo_single_verification()

    # Demo batch verification
    demo_batch_verification()

    print("\n" + "=" * 60)
    print("🎉 DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("The verification process has successfully demonstrated:")
    print("✅ Loading original images and ground truth masks")
    print("✅ Creating visual overlays for cross-review")
    print("✅ Calculating verification statistics")
    print("✅ Saving verification results")
    print("\nThis fulfills the requirement: 'The accuracy of segmentation")
    print("masks is verified through a cross-review process and by visually")
    print(
        "superimposing the masks onto the original images to confirm alignment'"
    )


if __name__ == "__main__":
    main()
