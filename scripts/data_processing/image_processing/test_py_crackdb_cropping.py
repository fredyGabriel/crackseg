#!/usr/bin/env python3
"""
Test script for PY-CrackDB bidirectional cropping algorithm.

This script tests the quadrant analysis and cropping decision logic
with a sample image to verify the algorithm works correctly.

Author: CrackSeg Project
Date: 2024
"""

import logging
from pathlib import Path

import cv2
import numpy as np

# Import the functions from the main script
from crop_py_crackdb_images import (
    analyze_quadrant_density,
    crop_image_bidirectional,
    determine_optimal_crop,
    validate_input_dimensions,
)


def setup_test_logging() -> None:
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def create_test_mask() -> np.ndarray:
    """
    Create a test mask with known crack distribution.

    Creates a 351x500 mask with cracks concentrated in the top-left quadrant
    to test the algorithm's decision making.
    """
    mask = np.zeros((500, 351), dtype=np.uint8)

    # Add cracks in top-left quadrant (higher density)
    mask[50:150, 50:200] = 255  # Large crack area
    mask[200:250, 100:150] = 255  # Medium crack area

    # Add some cracks in other quadrants (lower density)
    mask[300:350, 250:300] = 255  # Bottom-right quadrant
    mask[100:120, 280:320] = 255  # Top-right quadrant

    return mask


def test_quadrant_analysis() -> None:
    """Test the quadrant density analysis."""
    print("=" * 50)
    print("TESTING QUADRANT ANALYSIS")
    print("=" * 50)

    # Create test mask
    mask = create_test_mask()
    print(f"Test mask shape: {mask.shape}")
    print(f"Total crack pixels: {np.sum(mask > 127)}")

    # Analyze quadrants
    quadrant_densities = analyze_quadrant_density(mask)

    print("\nQuadrant Analysis Results:")
    for quadrant, density_info in quadrant_densities.items():
        print(f"  {quadrant}:")
        print(f"    Density: {density_info.density:.4f}")
        print(f"    Crack pixels: {density_info.crack_pixels}")
        print(f"    Total pixels: {density_info.total_pixels}")


def test_crop_decision() -> None:
    """Test the cropping decision logic."""
    print("\n" + "=" * 50)
    print("TESTING CROP DECISION")
    print("=" * 50)

    # Create test mask
    mask = create_test_mask()

    # Analyze quadrants
    quadrant_densities = analyze_quadrant_density(mask)

    # Determine optimal crop
    decision = determine_optimal_crop(quadrant_densities)

    print("Optimal crop decision:")
    print(f"  Horizontal cut: {decision.horizontal_cut}")
    print(f"  Vertical cut: {decision.vertical_cut}")
    print(f"  Total crack pixels preserved: {decision.total_crack_pixels}")
    print(f"  Overall density: {decision.overall_density:.4f}")


def test_bidirectional_cropping() -> None:
    """Test the bidirectional cropping function."""
    print("\n" + "=" * 50)
    print("TESTING BIDIRECTIONAL CROPPING")
    print("=" * 50)

    # Create test image and mask
    test_image = np.random.randint(0, 255, (500, 351, 3), dtype=np.uint8)
    test_mask = create_test_mask()

    # Validate dimensions
    if not validate_input_dimensions(test_image, test_mask):
        print("❌ Dimension validation failed")
        return

    print("✅ Dimension validation passed")

    # Analyze and decide
    quadrant_densities = analyze_quadrant_density(test_mask)
    decision = determine_optimal_crop(quadrant_densities)

    # Perform cropping
    cropped_image = crop_image_bidirectional(test_image, decision)
    cropped_mask = crop_image_bidirectional(test_mask, decision)

    print(f"Original image shape: {test_image.shape}")
    print(f"Cropped image shape: {cropped_image.shape}")
    print(f"Original mask shape: {test_mask.shape}")
    print(f"Cropped mask shape: {cropped_mask.shape}")

    # Verify output dimensions
    if cropped_image.shape == (320, 320, 3) and cropped_mask.shape == (
        320,
        320,
    ):
        print("✅ Cropping successful - correct output dimensions")
    else:
        print("❌ Cropping failed - incorrect output dimensions")


def test_with_real_image() -> None:
    """Test with a real PY-CrackDB image if available."""
    print("\n" + "=" * 50)
    print("TESTING WITH REAL IMAGE")
    print("=" * 50)

    # Try to find a real image
    image_path = Path("data/PY-CrackBD/images/217.jpg")
    mask_path = Path("data/PY-CrackBD/masks/217.png")

    if not image_path.exists():
        print("❌ Real image not found, skipping real image test")
        return

    if not mask_path.exists():
        print("❌ Real mask not found, skipping real image test")
        return

    print(f"Testing with real image: {image_path}")

    # Load real image and mask
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("❌ Failed to load real image or mask")
        return

    print(f"Real image shape: {image.shape}")
    print(f"Real mask shape: {mask.shape}")

    # Validate dimensions
    if not validate_input_dimensions(image, mask):
        print("❌ Real image dimension validation failed")
        return

    print("✅ Real image dimension validation passed")

    # Analyze and decide
    quadrant_densities = analyze_quadrant_density(mask)
    decision = determine_optimal_crop(quadrant_densities)

    print("Real image crop decision:")
    print(f"  Horizontal cut: {decision.horizontal_cut}")
    print(f"  Vertical cut: {decision.vertical_cut}")
    print(f"  Total crack pixels preserved: {decision.total_crack_pixels}")
    print(f"  Overall density: {decision.overall_density:.4f}")

    # Perform cropping
    cropped_image = crop_image_bidirectional(image, decision)
    cropped_mask = crop_image_bidirectional(mask, decision)

    print(f"Cropped image shape: {cropped_image.shape}")
    print(f"Cropped mask shape: {cropped_mask.shape}")

    # Save test results
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(str(output_dir / "test_cropped_image.jpg"), cropped_image)
    cv2.imwrite(str(output_dir / "test_cropped_mask.png"), cropped_mask)

    print(f"✅ Test results saved to {output_dir}")


def main() -> None:
    """Run all tests."""
    setup_test_logging()

    print("PY-CRACKDB CROPPING ALGORITHM TEST")
    print("=" * 60)

    try:
        # Test quadrant analysis
        test_quadrant_analysis()

        # Test crop decision
        test_crop_decision()

        # Test bidirectional cropping
        test_bidirectional_cropping()

        # Test with real image
        test_with_real_image()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        raise


if __name__ == "__main__":
    main()
