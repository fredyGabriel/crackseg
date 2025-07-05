"""Visual regression testing utilities for E2E testing.

This module provides screenshot comparison and visual regression testing
capabilities, enabling detection of UI changes and layout regressions
across test runs.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np
from selenium.webdriver.remote.webdriver import WebDriver

logger = logging.getLogger(__name__)


class ComparisonMode(Enum):
    """Visual comparison modes."""

    EXACT = "exact"
    PIXEL_DIFF = "pixel_diff"
    STRUCTURAL = "structural"
    PERCEPTUAL = "perceptual"


class ComparisonResult(NamedTuple):
    """Result of visual comparison between two images.

    Attributes:
        passed: Whether the comparison passed (within tolerance)
        difference_percentage: Percentage of different pixels
        similarity_score: Similarity score (0.0 to 1.0)
        diff_image_path: Path to difference image, if created
        baseline_path: Path to baseline image
        current_path: Path to current screenshot
        details: Additional comparison details
    """

    passed: bool
    difference_percentage: float
    similarity_score: float
    diff_image_path: Path | None
    baseline_path: Path
    current_path: Path
    details: dict[str, Any]


@dataclass
class VisualRegressionConfig:
    """Configuration for visual regression testing.

    Attributes:
        enabled: Whether visual regression testing is enabled
        tolerance_percentage: Maximum allowed difference percentage
        pixel_tolerance: Tolerance for individual pixel differences
        create_diff_images: Create visual difference images
        save_baselines: Automatically save new baselines
        artifacts_dir: Directory to store comparison artifacts
        baselines_dir: Directory to store baseline images
        comparison_mode: Mode for image comparison
        ignore_antialiasing: Ignore antialiasing differences
        ignore_transparency: Ignore alpha channel differences
        min_similarity_score: Minimum required similarity score
    """

    enabled: bool = True
    tolerance_percentage: float = 0.1
    pixel_tolerance: int = 10
    create_diff_images: bool = True
    save_baselines: bool = False
    artifacts_dir: Path = field(
        default_factory=lambda: Path("test-artifacts/visual-regression")
    )
    baselines_dir: Path = field(
        default_factory=lambda: Path("test-artifacts/baselines")
    )
    comparison_mode: ComparisonMode = ComparisonMode.PIXEL_DIFF
    ignore_antialiasing: bool = True
    ignore_transparency: bool = False
    min_similarity_score: float = 0.95


class HasVisualRegression(Protocol):
    """Protocol for classes that support visual regression testing."""

    def compare_visual(
        self,
        driver: WebDriver,
        test_name: str,
        element_selector: str | None = None,
    ) -> ComparisonResult:
        """Compare current state with baseline for visual regression."""
        ...

    def update_baseline(
        self,
        driver: WebDriver,
        test_name: str,
        element_selector: str | None = None,
    ) -> Path:
        """Update baseline image for future comparisons."""
        ...


class VisualRegression:
    """Core visual regression testing functionality.

    Provides methods for comparing screenshots against baselines to detect
    visual regressions in UI components and layouts.
    """

    def __init__(self, config: VisualRegressionConfig | None = None) -> None:
        """Initialize visual regression testing with configuration.

        Args:
            config: Visual regression configuration
        """
        self.config = config or VisualRegressionConfig()
        self._comparison_results: list[ComparisonResult] = []

        # Ensure directories exist
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.config.baselines_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"Visual regression initialized: {self.config.artifacts_dir}"
        )

    def compare_visual(
        self,
        driver: WebDriver,
        test_name: str,
        element_selector: str | None = None,
    ) -> ComparisonResult:
        """Compare current state with baseline for visual regression.

        Args:
            driver: WebDriver instance
            test_name: Name of the test for baseline identification
            element_selector: Optional CSS selector for specific element

        Returns:
            ComparisonResult with comparison details
        """
        if not self.config.enabled:
            logger.debug("Visual regression testing disabled")
            return self._create_disabled_result(test_name)

        try:
            # Take current screenshot
            current_path = self._capture_current_screenshot(
                driver, test_name, element_selector
            )

            # Find baseline image
            baseline_path = self._get_baseline_path(
                test_name, element_selector
            )

            if not baseline_path.exists():
                if self.config.save_baselines:
                    # Create new baseline
                    baseline_path = self._create_baseline(
                        current_path, baseline_path
                    )
                    logger.info(f"Created new baseline: {baseline_path}")

                    result = ComparisonResult(
                        passed=True,
                        difference_percentage=0.0,
                        similarity_score=1.0,
                        diff_image_path=None,
                        baseline_path=baseline_path,
                        current_path=current_path,
                        details={"reason": "baseline_created"},
                    )
                else:
                    logger.warning(f"No baseline found for: {test_name}")
                    result = ComparisonResult(
                        passed=False,
                        difference_percentage=100.0,
                        similarity_score=0.0,
                        diff_image_path=None,
                        baseline_path=baseline_path,
                        current_path=current_path,
                        details={"reason": "no_baseline"},
                    )

                self._comparison_results.append(result)
                return result

            # Perform comparison
            result = self._perform_comparison(
                baseline_path, current_path, test_name
            )

            self._comparison_results.append(result)
            return result

        except Exception as e:
            logger.error(f"Visual regression comparison failed: {e}")
            return self._create_error_result(test_name, str(e))

    def update_baseline(
        self,
        driver: WebDriver,
        test_name: str,
        element_selector: str | None = None,
    ) -> Path:
        """Update baseline image for future comparisons.

        Args:
            driver: WebDriver instance
            test_name: Name of the test
            element_selector: Optional CSS selector for specific element

        Returns:
            Path to updated baseline image
        """
        logger.info(f"Updating baseline for: {test_name}")

        # Take current screenshot
        current_path = self._capture_current_screenshot(
            driver, test_name, element_selector
        )

        # Get baseline path
        baseline_path = self._get_baseline_path(test_name, element_selector)

        # Update baseline
        baseline_path = self._create_baseline(current_path, baseline_path)

        logger.info(f"Baseline updated: {baseline_path}")
        return baseline_path

    def get_comparison_results(self) -> list[ComparisonResult]:
        """Get all comparison results from current test session.

        Returns:
            List of comparison results
        """
        return self._comparison_results.copy()

    def clear_comparison_results(self) -> None:
        """Clear stored comparison results."""
        self._comparison_results.clear()

    def _capture_current_screenshot(
        self,
        driver: WebDriver,
        test_name: str,
        element_selector: str | None,
    ) -> Path:
        """Capture current screenshot for comparison.

        Args:
            driver: WebDriver instance
            test_name: Name of the test
            element_selector: Optional element selector

        Returns:
            Path to captured screenshot
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"current_{test_name}_{timestamp}.png"

        if element_selector:
            # Sanitize selector for filename
            sanitized_selector = "".join(
                c for c in element_selector if c.isalnum() or c in "._-"
            )[:50]
            filename = (
                f"current_{test_name}_{sanitized_selector}_{timestamp}.png"
            )

        screenshot_path = self.config.artifacts_dir / filename

        if element_selector:
            # Screenshot specific element
            element = driver.find_element("css selector", element_selector)
            element.screenshot(str(screenshot_path))
        else:
            # Full page screenshot
            driver.save_screenshot(str(screenshot_path))

        return screenshot_path

    def _get_baseline_path(
        self,
        test_name: str,
        element_selector: str | None,
    ) -> Path:
        """Get path to baseline image.

        Args:
            test_name: Name of the test
            element_selector: Optional element selector

        Returns:
            Path to baseline image
        """
        filename = f"baseline_{test_name}.png"

        if element_selector:
            sanitized_selector = "".join(
                c for c in element_selector if c.isalnum() or c in "._-"
            )[:50]
            filename = f"baseline_{test_name}_{sanitized_selector}.png"

        return self.config.baselines_dir / filename

    def _create_baseline(self, source_path: Path, baseline_path: Path) -> Path:
        """Create baseline image from source.

        Args:
            source_path: Path to source image
            baseline_path: Path where baseline should be created

        Returns:
            Path to created baseline
        """
        import shutil

        # Ensure baseline directory exists
        baseline_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy source to baseline location
        shutil.copy2(source_path, baseline_path)

        return baseline_path

    def _perform_comparison(
        self,
        baseline_path: Path,
        current_path: Path,
        test_name: str,
    ) -> ComparisonResult:
        """Perform visual comparison between images.

        Args:
            baseline_path: Path to baseline image
            current_path: Path to current screenshot
            test_name: Name of the test

        Returns:
            ComparisonResult with comparison details
        """
        try:
            # Use PIL for image comparison
            from PIL import Image

            # Load images
            baseline_img = Image.open(baseline_path).convert("RGB")
            current_img = Image.open(current_path).convert("RGB")

            # Ensure images have same size
            if baseline_img.size != current_img.size:
                logger.warning(
                    f"Image size mismatch: baseline {baseline_img.size} "
                    f"vs current {current_img.size}"
                )
                # Resize current to match baseline
                current_img = current_img.resize(
                    baseline_img.size, Image.Resampling.LANCZOS
                )

            # Calculate difference
            diff_percentage, similarity_score = self._calculate_difference(
                baseline_img, current_img
            )

            # Create difference image if requested
            diff_image_path = None
            if self.config.create_diff_images and diff_percentage > 0:
                diff_image_path = self._create_diff_image(
                    baseline_img, current_img, test_name
                )

            # Determine if comparison passed
            passed = (
                diff_percentage <= self.config.tolerance_percentage
                and similarity_score >= self.config.min_similarity_score
            )

            details = {
                "mode": self.config.comparison_mode.value,
                "tolerance": self.config.tolerance_percentage,
                "pixel_tolerance": self.config.pixel_tolerance,
                "baseline_size": baseline_img.size,
                "current_size": current_img.size,
            }

            return ComparisonResult(
                passed=passed,
                difference_percentage=diff_percentage,
                similarity_score=similarity_score,
                diff_image_path=diff_image_path,
                baseline_path=baseline_path,
                current_path=current_path,
                details=details,
            )

        except ImportError:
            logger.error("PIL (Pillow) not available for image comparison")
            return self._create_error_result(test_name, "PIL not available")
        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            return self._create_error_result(test_name, str(e))

    def _calculate_difference(
        self,
        baseline_img: Any,
        current_img: Any,
    ) -> tuple[float, float]:
        """Calculate difference percentage and similarity score.

        Args:
            baseline_img: Baseline PIL Image
            current_img: Current PIL Image

        Returns:
            Tuple of (difference_percentage, similarity_score)
        """
        # Convert to numpy arrays for comparison
        baseline_array = np.array(baseline_img)
        current_array = np.array(current_img)

        # Calculate pixel differences
        diff_array = np.abs(
            baseline_array.astype(int) - current_array.astype(int)
        )

        # Apply pixel tolerance
        significant_diff = diff_array > self.config.pixel_tolerance

        # Calculate percentage of different pixels
        total_pixels = baseline_array.shape[0] * baseline_array.shape[1]
        different_pixels = np.sum(np.any(significant_diff, axis=2))
        diff_percentage = (different_pixels / total_pixels) * 100

        # Calculate similarity score
        similarity_score = 1.0 - (diff_percentage / 100.0)

        return diff_percentage, max(0.0, similarity_score)

    def _create_diff_image(
        self,
        baseline_img: Any,
        current_img: Any,
        test_name: str,
    ) -> Path:
        """Create visual difference image.

        Args:
            baseline_img: Baseline PIL Image
            current_img: Current PIL Image
            test_name: Name of the test

        Returns:
            Path to created difference image
        """
        from PIL import ImageChops, ImageEnhance

        # Create difference image
        diff_img = ImageChops.difference(baseline_img, current_img)

        # Enhance difference visibility
        enhancer = ImageEnhance.Brightness(diff_img)
        diff_img = enhancer.enhance(3.0)  # Make differences more visible

        # Save difference image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        diff_filename = f"diff_{test_name}_{timestamp}.png"
        diff_path = self.config.artifacts_dir / diff_filename

        diff_img.save(diff_path)

        logger.debug(f"Difference image created: {diff_path}")
        return diff_path

    def _create_disabled_result(self, test_name: str) -> ComparisonResult:
        """Create result for disabled visual regression testing."""
        return ComparisonResult(
            passed=True,
            difference_percentage=0.0,
            similarity_score=1.0,
            diff_image_path=None,
            baseline_path=Path("disabled"),
            current_path=Path("disabled"),
            details={"reason": "disabled"},
        )

    def _create_error_result(
        self, test_name: str, error_msg: str
    ) -> ComparisonResult:
        """Create result for comparison errors."""
        return ComparisonResult(
            passed=False,
            difference_percentage=100.0,
            similarity_score=0.0,
            diff_image_path=None,
            baseline_path=Path("error"),
            current_path=Path("error"),
            details={"reason": "error", "error": error_msg},
        )


class VisualRegressionMixin:
    """Mixin providing visual regression testing capabilities for test classes.

    Integrates with the existing BaseE2ETest mixin pattern to provide
    visual regression testing functionality.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize visual regression mixin."""
        super().__init__(*args, **kwargs)
        self._visual_regression = VisualRegression()

        if hasattr(self, "_test_logger"):
            self._test_logger.debug("Visual regression mixin initialized")  # type: ignore[attr-defined]

    def compare_visual(
        self,
        driver: WebDriver,
        test_name: str,
        element_selector: str | None = None,
    ) -> ComparisonResult:
        """Compare current state with baseline for visual regression.

        Args:
            driver: WebDriver instance
            test_name: Name of the test for baseline identification
            element_selector: Optional CSS selector for specific element

        Returns:
            ComparisonResult with comparison details
        """
        if hasattr(self, "log_test_step"):
            step_msg = f"Visual regression comparison: {test_name}"
            if element_selector:
                step_msg += f" (element: {element_selector})"
            self.log_test_step(step_msg)  # type: ignore[attr-defined]

        result = self._visual_regression.compare_visual(
            driver, test_name, element_selector
        )

        if hasattr(self, "log_assertion"):
            self.log_assertion(  # type: ignore[attr-defined]
                f"Visual regression check: {test_name}",
                result.passed,
                f"Difference: {result.difference_percentage:.2f}%, "
                f"Similarity: {result.similarity_score:.2f}",
            )

        return result

    def update_baseline(
        self,
        driver: WebDriver,
        test_name: str,
        element_selector: str | None = None,
    ) -> Path:
        """Update baseline image for future comparisons.

        Args:
            driver: WebDriver instance
            test_name: Name of the test
            element_selector: Optional CSS selector for specific element

        Returns:
            Path to updated baseline image
        """
        if hasattr(self, "log_test_step"):
            self.log_test_step(f"Updating visual baseline: {test_name}")  # type: ignore[attr-defined]

        baseline_path = self._visual_regression.update_baseline(
            driver, test_name, element_selector
        )

        if hasattr(self, "log_test_step"):
            self.log_test_step(f"Baseline updated: {baseline_path.name}")  # type: ignore[attr-defined]

        return baseline_path

    def get_comparison_results(self) -> list[ComparisonResult]:
        """Get all comparison results from current test session.

        Returns:
            List of comparison results
        """
        return self._visual_regression.get_comparison_results()

    def clear_comparison_results(self) -> None:
        """Clear stored comparison results."""
        self._visual_regression.clear_comparison_results()
