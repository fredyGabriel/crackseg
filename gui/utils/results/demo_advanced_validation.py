"""Demo script for advanced triplet validation system.

This script demonstrates the enhanced validation capabilities including:
- Multi-level validation (basic to paranoid)
- Error detection and recovery
- Performance monitoring
- Event-driven notifications

Phase 3: Advanced validation demonstration.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

from .advanced_validation import (
    AdvancedTripletValidator,
    ValidationLevel,
)
from .cache import TripletCache
from .events import EventManager, EventType, ScanEvent

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ValidationDemoRunner:
    """Comprehensive demo of advanced validation features."""

    def __init__(self) -> None:
        """Initialize the demo runner."""
        self.event_manager = EventManager()
        self.cache = TripletCache(capacity=100)
        self.demo_results: dict[str, Any] = {}

        # Register event handlers for demo
        self._setup_event_handlers()

    def _setup_event_handlers(self) -> None:
        """Setup event handlers to monitor validation events."""

        def on_triplet_found(event: ScanEvent) -> None:
            """Handle triplet found events."""
            triplet = event.data.get("triplet")
            if triplet:
                logger.info(f"✅ Triplet validated: {triplet.id}")

        def on_scan_error(event: ScanEvent) -> None:
            """Handle scan error events."""
            error = event.data.get("error", "Unknown error")
            context = event.data.get("context", "Unknown context")
            logger.warning(f"❌ Validation error in {context}: {error}")

        # Register handlers
        self.event_manager.subscribe(EventType.TRIPLET_FOUND, on_triplet_found)
        self.event_manager.subscribe(EventType.SCAN_ERROR, on_scan_error)

    async def run_comprehensive_demo(self) -> dict[str, Any]:
        """Run comprehensive validation demo with all features."""
        logger.info("🚀 Starting Advanced Validation Demo")
        logger.info("=" * 60)

        # Demo phases
        demo_phases = [
            ("Basic Validation Level", self._demo_basic_validation),
            ("Standard Validation Level", self._demo_standard_validation),
            ("Thorough Validation Level", self._demo_thorough_validation),
            ("Paranoid Validation Level", self._demo_paranoid_validation),
            ("Error Recovery Demo", self._demo_error_recovery),
            ("Performance Comparison", self._demo_performance_comparison),
            ("Cache Integration", self._demo_cache_integration),
        ]

        # Run all demo phases
        for phase_name, phase_func in demo_phases:
            logger.info(f"\n📋 Phase: {phase_name}")
            logger.info("-" * 40)

            try:
                phase_results = await phase_func()
                self.demo_results[phase_name] = phase_results
                logger.info(f"✅ {phase_name} completed successfully")
            except Exception as e:
                logger.error(f"❌ {phase_name} failed: {e}")
                self.demo_results[phase_name] = {"error": str(e)}

        # Generate final report
        final_report = self._generate_demo_report()
        logger.info("\n📊 Demo Summary")
        logger.info("=" * 60)
        self._print_demo_summary(final_report)

        return final_report

    async def _demo_basic_validation(self) -> dict[str, Any]:
        """Demonstrate basic validation level."""
        validator = AdvancedTripletValidator(
            validation_level=ValidationLevel.BASIC,
            enable_recovery=False,
            event_manager=self.event_manager,
        )

        # Create test files
        test_files = await self._create_test_triplet("basic_test")

        # Validate triplet
        semaphore = asyncio.Semaphore(2)
        result = await validator.validate_triplet_advanced(
            "basic_test_triplet", test_files, semaphore
        )

        stats = validator.get_stats()

        # Cleanup
        validator.cleanup()
        await self._cleanup_test_files(test_files)

        return {
            "validation_result": {
                "is_valid": result.is_valid,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
                "validation_time": result.validation_time,
            },
            "stats": stats,
        }

    async def _demo_standard_validation(self) -> dict[str, Any]:
        """Demonstrate standard validation level."""
        validator = AdvancedTripletValidator(
            validation_level=ValidationLevel.STANDARD,
            enable_recovery=True,
            event_manager=self.event_manager,
        )

        # Create test files with some issues
        test_files = await self._create_test_triplet_with_issues(
            "standard_test"
        )

        # Validate triplet
        semaphore = asyncio.Semaphore(2)
        result = await validator.validate_triplet_advanced(
            "standard_test_triplet", test_files, semaphore
        )

        stats = validator.get_stats()

        # Cleanup
        validator.cleanup()
        await self._cleanup_test_files(test_files)

        return {
            "validation_result": {
                "is_valid": result.is_valid,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
                "validation_time": result.validation_time,
                "has_recoverable_errors": result.has_recoverable_errors,
            },
            "stats": stats,
        }

    async def _demo_thorough_validation(self) -> dict[str, Any]:
        """Demonstrate thorough validation level."""
        validator = AdvancedTripletValidator(
            validation_level=ValidationLevel.THOROUGH,
            enable_recovery=True,
            event_manager=self.event_manager,
        )

        # Create test files
        test_files = await self._create_test_triplet("thorough_test")

        # Validate triplet
        semaphore = asyncio.Semaphore(2)
        result = await validator.validate_triplet_advanced(
            "thorough_test_triplet", test_files, semaphore
        )

        stats = validator.get_stats()

        # Cleanup
        validator.cleanup()
        await self._cleanup_test_files(test_files)

        return {
            "validation_result": {
                "is_valid": result.is_valid,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
                "validation_time": result.validation_time,
                "metadata_keys": list(result.metadata.keys()),
            },
            "stats": stats,
        }

    async def _demo_paranoid_validation(self) -> dict[str, Any]:
        """Demonstrate paranoid validation level."""
        validator = AdvancedTripletValidator(
            validation_level=ValidationLevel.PARANOID,
            enable_recovery=True,
            event_manager=self.event_manager,
        )

        # Create test files
        test_files = await self._create_test_triplet("paranoid_test")

        # Validate triplet
        semaphore = asyncio.Semaphore(2)
        result = await validator.validate_triplet_advanced(
            "paranoid_test_triplet", test_files, semaphore
        )

        stats = validator.get_stats()

        # Cleanup
        validator.cleanup()
        await self._cleanup_test_files(test_files)

        return {
            "validation_result": {
                "is_valid": result.is_valid,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
                "validation_time": result.validation_time,
                "has_checksums": any(
                    "checksum" in key for key in result.metadata.keys()
                ),
            },
            "stats": stats,
        }

    async def _demo_error_recovery(self) -> dict[str, Any]:
        """Demonstrate error recovery mechanisms."""
        validator = AdvancedTripletValidator(
            validation_level=ValidationLevel.STANDARD,
            enable_recovery=True,
            event_manager=self.event_manager,
        )

        # Create problematic test files
        test_files = await self._create_problematic_triplet("recovery_test")

        # Validate triplet
        semaphore = asyncio.Semaphore(2)
        result = await validator.validate_triplet_advanced(
            "recovery_test_triplet", test_files, semaphore
        )

        stats = validator.get_stats()

        # Cleanup
        validator.cleanup()
        await self._cleanup_test_files(test_files)

        return {
            "validation_result": {
                "is_valid": result.is_valid,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
                "validation_time": result.validation_time,
                "recovery_attempted": result.has_recoverable_errors,
            },
            "stats": stats,
        }

    async def _demo_performance_comparison(self) -> dict[str, Any]:
        """Compare performance across validation levels."""
        validation_levels = [
            ValidationLevel.BASIC,
            ValidationLevel.STANDARD,
            ValidationLevel.THOROUGH,
            ValidationLevel.PARANOID,
        ]

        performance_results = {}

        for level in validation_levels:
            validator = AdvancedTripletValidator(
                validation_level=level,
                enable_recovery=False,
                event_manager=None,  # Disable events for pure performance test
            )

            # Create test files
            test_files = await self._create_test_triplet(
                f"perf_test_{level.name}"
            )

            # Validate triplet multiple times for average
            times = []
            semaphore = asyncio.Semaphore(2)

            for i in range(5):  # 5 iterations for average
                result = await validator.validate_triplet_advanced(
                    f"perf_test_{level.name}_{i}", test_files, semaphore
                )
                times.append(result.validation_time)

            avg_time = sum(times) / len(times)
            stats = validator.get_stats()

            performance_results[level.name] = {
                "avg_validation_time": avg_time,
                "min_time": min(times),
                "max_time": max(times),
                "total_validated": stats["total_validated"],
                "success_rate": stats["success_rate"],
            }

            # Cleanup
            validator.cleanup()
            await self._cleanup_test_files(test_files)

        return performance_results

    async def _demo_cache_integration(self) -> dict[str, Any]:
        """Demonstrate integration with caching system."""
        validator = AdvancedTripletValidator(
            validation_level=ValidationLevel.STANDARD,
            enable_recovery=True,
            event_manager=self.event_manager,
        )

        # Create test files
        test_files = await self._create_test_triplet("cache_test")

        # First validation (cache miss)
        semaphore = asyncio.Semaphore(2)
        result1 = await validator.validate_triplet_advanced(
            "cache_test_triplet", test_files, semaphore
        )

        # Cache the result if valid
        if result1.is_valid and result1.triplet:
            self.cache.cache_triplet(result1.triplet)

        # Simulate second validation (cache hit)
        cached_triplet = self.cache.get_triplet("cache_test_triplet")

        # Second validation for comparison
        result2 = await validator.validate_triplet_advanced(
            "cache_test_triplet", test_files, semaphore
        )

        cache_stats = self.cache.get_stats()
        validator_stats = validator.get_stats()

        # Cleanup
        validator.cleanup()
        await self._cleanup_test_files(test_files)

        return {
            "first_validation": {
                "is_valid": result1.is_valid,
                "validation_time": result1.validation_time,
            },
            "second_validation": {
                "is_valid": result2.is_valid,
                "validation_time": result2.validation_time,
            },
            "cache_hit": cached_triplet is not None,
            "cache_stats": cache_stats,
            "validator_stats": validator_stats,
        }

    async def _create_test_triplet(self, base_name: str) -> list[Path]:
        """Create a valid test triplet."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create PNG signature for valid image files
        png_signature = b"\x89PNG\r\n\x1a\n" + b"fake_png_data" * 100

        files = []
        for suffix in ["", "_mask", "_pred"]:
            file_path = temp_dir / f"{base_name}{suffix}.png"
            with open(file_path, "wb") as f:
                f.write(png_signature)
            files.append(file_path)

        return files

    async def _create_test_triplet_with_issues(
        self, base_name: str
    ) -> list[Path]:
        """Create a test triplet with some issues."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create PNG signature for valid image files
        png_signature = b"\x89PNG\r\n\x1a\n" + b"fake_png_data" * 100

        files = []
        for i, suffix in enumerate(["", "_mask", "_pred"]):
            file_path = temp_dir / f"{base_name}{suffix}.png"

            if i == 1:  # Make mask file suspiciously small
                with open(file_path, "wb") as f:
                    f.write(b"small")
            else:
                with open(file_path, "wb") as f:
                    f.write(png_signature)

            files.append(file_path)

        return files

    async def _create_problematic_triplet(self, base_name: str) -> list[Path]:
        """Create a problematic test triplet for error recovery demo."""
        temp_dir = Path(tempfile.mkdtemp())

        files = []
        # Only create image and mask, missing prediction
        for suffix in ["", "_mask"]:
            file_path = temp_dir / f"{base_name}{suffix}.png"
            with open(file_path, "wb") as f:
                f.write(b"invalid_image_data")
            files.append(file_path)

        return files

    async def _cleanup_test_files(self, files: list[Path]) -> None:
        """Clean up test files."""
        for file_path in files:
            try:
                if file_path.exists():
                    file_path.unlink()
                # Remove parent directory if empty
                if file_path.parent.exists() and not any(
                    file_path.parent.iterdir()
                ):
                    file_path.parent.rmdir()
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")

    def _generate_demo_report(self) -> dict[str, Any]:
        """Generate comprehensive demo report."""
        return {
            "demo_phases_completed": len(self.demo_results),
            "successful_phases": sum(
                1
                for result in self.demo_results.values()
                if "error" not in result
            ),
            "failed_phases": sum(
                1 for result in self.demo_results.values() if "error" in result
            ),
            "detailed_results": self.demo_results,
            "cache_stats": self.cache.get_stats(),
        }

    def _print_demo_summary(self, report: dict[str, Any]) -> None:
        """Print formatted demo summary."""
        logger.info(f"📊 Phases completed: {report['demo_phases_completed']}")
        logger.info(f"✅ Successful: {report['successful_phases']}")
        logger.info(f"❌ Failed: {report['failed_phases']}")

        # Performance comparison summary
        if "Performance Comparison" in self.demo_results:
            perf_data = self.demo_results["Performance Comparison"]
            logger.info("\n⚡ Performance Summary:")
            for level, data in perf_data.items():
                logger.info(
                    f"  {level}: {data['avg_validation_time']:.4f}s avg, "
                    f"{data['success_rate']:.1f}% success"
                )

        # Cache performance summary
        cache_stats = report.get("cache_stats", {})
        if cache_stats:
            logger.info("\n💾 Cache Performance:")
            logger.info(f"  Hit rate: {cache_stats.get('hit_rate', 0):.1f}%")
            logger.info(
                f"  Total operations: {cache_stats.get('total_operations', 0)}"
            )


async def main() -> dict[str, Any]:
    """Run the advanced validation demo."""
    demo_runner = ValidationDemoRunner()

    try:
        results = await demo_runner.run_comprehensive_demo()
        logger.info("\n🎉 Advanced Validation Demo completed successfully!")
        return results
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
