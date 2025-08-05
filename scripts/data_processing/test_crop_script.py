#!/usr/bin/env python3
"""
Test script to verify the functionality of crop_crack_images.py

This script generates test data and executes the processing to verify
that everything works correctly.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def create_test_data(input_dir: Path, num_samples: int = 10):
    """Create test data with synthetic images and masks."""

    # Create directories
    images_dir = input_dir / "images"
    masks_dir = input_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {num_samples} test file pairs...")

    for i in range(num_samples):
        # Create synthetic 640x360 image
        image = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

        # Create synthetic mask with cracks
        mask = np.zeros((360, 640), dtype=np.uint8)

        # Simulate cracks in different positions
        if i % 3 == 0:
            # More cracks on the left side
            mask[100:300, 50:300] = 255
            mask[150:250, 100:250] = 0  # Holes in cracks
        elif i % 3 == 1:
            # More cracks on the right side
            mask[100:300, 340:590] = 255
            mask[150:250, 390:540] = 0  # Holes in cracks
        else:
            # Uniformly distributed cracks
            mask[100:300, 200:440] = 255
            mask[150:250, 250:390] = 0  # Holes in cracks

        # Add noise to simulate real cracks
        noise = np.random.randint(0, 50, mask.shape, dtype=np.uint8)
        mask = np.clip(mask + noise, 0, 255)

        # Save files
        image_path = images_dir / f"test_image_{i + 1}.jpg"
        mask_path = masks_dir / f"test_image_{i + 1}.png"

        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(mask_path), mask)

        print(f"  Created: {image_path.name} and {mask_path.name}")

    print("Test data created successfully.")


def verify_output(output_dir: Path, num_samples: int = 10):
    """Verify that output files are correct."""

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"

    print("\nVerifying output files...")

    # Verify that directories exist
    if not images_dir.exists():
        print("❌ ERROR: Output images directory does not exist")
        return False

    if not masks_dir.exists():
        print("❌ ERROR: Output masks directory does not exist")
        return False

    # Verify files
    for i in range(1, num_samples + 1):
        image_path = images_dir / f"{i}.jpg"
        mask_path = masks_dir / f"{i}.png"

        # Verify that files exist
        if not image_path.exists():
            print(f"❌ ERROR: Image {i}.jpg does not exist")
            return False

        if not mask_path.exists():
            print(f"❌ ERROR: Mask {i}.png does not exist")
            return False

        # Verify dimensions
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"❌ ERROR: Could not load image {i}.jpg")
            return False

        if mask is None:
            print(f"❌ ERROR: Could not load mask {i}.png")
            return False

        if image.shape != (360, 360, 3):
            print(
                f"❌ ERROR: Image {i}.jpg has incorrect dimensions: {image.shape}"
            )
            return False

        if mask.shape != (360, 360):
            print(
                f"❌ ERROR: Mask {i}.png has incorrect dimensions: {mask.shape}"
            )
            return False

        print(f"  ✅ File {i}: Image {image.shape}, Mask {mask.shape}")

    print("✅ Verification completed successfully.")
    return True


def run_crop_script(input_dir: Path, output_dir: Path):
    """Execute the processing script."""

    print("\nExecuting processing script...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    # Command to execute the script
    cmd = [
        sys.executable,
        "crop_crack_images.py",
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--log_level",
        "INFO",
    ]

    try:
        # Execute the script
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent
        )

        # Show output
        if result.stdout:
            print("Script output:")
            print(result.stdout)

        if result.stderr:
            print("Script errors:")
            print(result.stderr)

        # Verify exit code
        if result.returncode == 0:
            print("✅ Script executed successfully.")
            return True
        else:
            print(f"❌ Script failed with exit code: {result.returncode}")
            return False

    except Exception as e:
        print(f"❌ Error executing script: {e}")
        return False


def main():
    """Main test function."""

    print("=" * 60)
    print("IMAGE PROCESSING SCRIPT TEST")
    print("=" * 60)

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Configure paths
        input_dir = temp_path / "test_input"
        output_dir = temp_path / "test_output"

        print(f"Temporary directory: {temp_path}")

        try:
            # Step 1: Create test data
            print("\n1. Creating test data...")
            create_test_data(input_dir, num_samples=10)

            # Step 2: Execute processing script
            print("\n2. Executing processing script...")
            success = run_crop_script(input_dir, output_dir)

            if not success:
                print("❌ Script failed. Aborting test.")
                return 1

            # Step 3: Verify results
            print("\n3. Verifying results...")
            verification_success = verify_output(output_dir, num_samples=10)

            if verification_success:
                print("\n✅ TEST COMPLETED SUCCESSFULLY")
                print("The processing script works correctly.")
                return 0
            else:
                print("\n❌ TEST FAILED")
                print("Errors were found during verification.")
                return 1

        except Exception as e:
            print(f"\n❌ ERROR DURING TEST: {e}")
            return 1


if __name__ == "__main__":
    exit(main())
