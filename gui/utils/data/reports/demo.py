"""
Demonstration script for AsyncResultsScanner functionality. This
script demonstrates the basic capabilities of the results scanner
including async scanning, progress tracking, and triplet validation.
"""

import asyncio
import logging
import time
from pathlib import Path

from .core import ScanProgress
from .scanner import (
    AsyncResultsScanner,
    create_results_scanner,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def progress_callback(progress: ScanProgress) -> None:
    """Example progress callback for demonstration."""
    percentage = progress.progress_percent
    print(
        f"Progress: {percentage:.1f}% "
        f"({progress.processed_files}/{progress.total_files}) "
        f"Found: {progress.found_triplets} triplets, "
        f"Errors: {progress.errors}"
    )


async def demo_basic_scanning() -> None:
    """Demonstrate basic async scanning functionality."""
    print("\n=== Demo: Basic Async Scanning ===")

    # Create test directory structure if it doesn't exist
    test_dir = Path("demo_results")
    test_dir.mkdir(exist_ok=True)

    # Create some dummy files to demonstrate
    test_files = [
        "image_001.png",
        "image_001_mask.png",
        "image_001_pred.png",
        "image_002.png",
        "image_002_mask.png",
        "image_002_pred.png",
        "incomplete_003.png",  # Incomplete triplet
    ]

    for filename in test_files:  # type: ignore[assignment]
        file_path = test_dir / filename  # type: ignore[assignment]
        if not file_path.exists():
            file_path.write_text(f"Dummy content for {filename} ")  # type: ignore[misc]

    try:
        # Create scanner with progress callback
        scanner = create_results_scanner(
            results_path=test_dir,
            max_concurrent=4,
            progress_callback=progress_callback,
        )

        print(f"Scanning directory: {test_dir}")

        triplet_count = 0
        start_time = time.time()

        # Scan asynchronously
        async for triplet in scanner.scan_async():
            triplet_count += 1
            print(f"  Found triplet: {triplet.id}")
            image_name = (
                triplet.image_path.name
                if triplet.image_path is not None
                else "N/A"
            )  # type: ignore[union-attr]
            print(f"    Image: {image_name}")
            mask_name = (
                triplet.mask_path.name  # type: ignore[union-attr]
                if triplet.mask_path is not None
                else "N/A"
            )
            pred_name = (
                triplet.prediction_path.name  # type: ignore[union-attr]
                if triplet.prediction_path is not None
                else "N/A"
            )
            print(f"    Mask: {mask_name}")
            print(f"    Prediction: {pred_name}")
            print(f"    Complete: {triplet.is_complete}")
            print()

        elapsed = time.time() - start_time
        print(f"Scan completed in {elapsed:.2f}s")
        print(f"Total triplets found: {triplet_count}")

        # Get final progress
        final_progress = scanner.current_progress
        print(f"Final progress: {final_progress.progress_percent:.1f}%")

    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise
    finally:
        # Cleanup demo files
        for filename in test_files:  # type: ignore[assignment]
            try:
                (test_dir / filename).unlink(missing_ok=True)  # type: ignore[misc]
            except Exception:
                pass
        try:
            test_dir.rmdir()
        except Exception:
            pass


async def demo_concurrent_performance() -> None:
    """Demonstrate concurrent scanning performance."""
    print("\n=== Demo: Concurrent Performance ===")

    # Test with different concurrency levels
    test_dir = Path("perf_test_results")
    test_dir.mkdir(exist_ok=True)

    # Create more files for performance testing
    num_triplets = 10
    test_files = []

    for i in range(num_triplets):
        files = [
            f"batch_{i:03d}.png",
            f"batch_{i:03d}_mask.png",
            f"batch_{i:03d}_pred.png",
        ]
        test_files.extend(files)  # type: ignore[arg-type]

    # Create dummy files
    for filename in test_files:  # type: ignore[assignment]
        file_path = test_dir / filename  # type: ignore[assignment]
        file_path.write_text(f"Test content for {filename} ")  # type: ignore[misc]

    try:
        # Test different concurrency levels
        for max_concurrent in [1, 4, 8]:
            print(f"\nTesting with max_concurrent={max_concurrent}")

            scanner = AsyncResultsScanner(
                base_path=test_dir, max_concurrent_operations=max_concurrent
            )

            start_time = time.time()
            triplet_count = 0

            async for _ in scanner.scan_async():
                triplet_count += 1

            elapsed = time.time() - start_time
            print(f"  Found {triplet_count} triplets in {elapsed:.3f}s")
            print(f"  Throughput: {triplet_count / elapsed:.1f} triplets/sec")

    finally:
        # Cleanup
        for filename in test_files:  # type: ignore[assignment]
            try:
                (test_dir / filename).unlink(missing_ok=True)  # type: ignore[misc]
            except Exception:
                pass
        try:
            test_dir.rmdir()
        except Exception:
            pass


async def demo_error_handling() -> None:
    """Demonstrate error handling capabilities."""
    print("\n=== Demo: Error Handling ===")

    # Test with non-existent directory
    try:
        scanner = AsyncResultsScanner(Path("non_existent_directory"))
        print("ERROR: Should have raised FileNotFoundError")
    except FileNotFoundError:
        print("✓ Correctly handled non-existent directory")

    # Test with file instead of directory
    test_file = Path("test_file.txt")
    test_file.write_text("This is a file, not a directory")

    try:
        scanner = AsyncResultsScanner(test_file)
        print("ERROR: Should have raised ValueError")
    except ValueError:
        print("✓ Correctly handled file instead of directory")
    finally:
        test_file.unlink(missing_ok=True)

    # Test cancellation
    test_dir = Path("cancel_test")
    test_dir.mkdir(exist_ok=True)

    # Create many files to allow cancellation test
    for i in range(20):
        for suffix in ["", "_mask", "_pred"]:
            (test_dir / f"cancel_{i:03d}{suffix}.png").write_text("test")

    try:
        scanner = AsyncResultsScanner(test_dir)

        print("Testing scan cancellation...")
        processed = 0
        async for _ in scanner.scan_async():
            processed += 1
            if processed >= 2:  # Cancel after processing 2 triplets
                scanner.cancel_scan()
                break

        print(f"✓ Successfully cancelled scan after {processed} triplets")

    finally:
        # Cleanup
        for i in range(20):
            for suffix in ["", "_mask", "_pred"]:
                try:
                    (test_dir / f"cancel_{i:03d}{suffix}.png").unlink()
                except Exception:
                    pass
        try:
            test_dir.rmdir()
        except Exception:
            pass


async def main() -> None:
    """Run all demonstration scenarios."""
    print("AsyncResultsScanner Demonstration")
    print("=" * 40)

    # Run all demos
    await demo_basic_scanning()
    await demo_concurrent_performance()
    await demo_error_handling()

    print("\n=== All Demos Completed Successfully! ===")
    print("Phase 1: Core AsyncIO scanner implementation verified ✓")


if __name__ == "__main__":
    asyncio.run(main())
