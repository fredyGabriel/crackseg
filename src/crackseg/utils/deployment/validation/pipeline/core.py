"""Core validation pipeline functionality.

This module contains the main ValidationPipeline class that orchestrates
the validation process using specialized components.
"""

import logging
from typing import TYPE_CHECKING, Any

from .compatibility import CompatibilityChecker
from .config import ValidationThresholds
from .functional import FunctionalTestRunner
from .performance import PerformanceBenchmarker
from .reporting import ValidationReporter
from .security import SecurityScanner

if TYPE_CHECKING:
    from ..config import DeploymentConfig

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """Comprehensive validation pipeline for deployment packages.

    Handles functional testing, performance benchmarking, compatibility
    checks, and security scanning.
    """

    def __init__(self) -> None:
        """Initialize validation pipeline."""
        self.thresholds = ValidationThresholds()

        # Initialize specialized components
        self.functional_runner = FunctionalTestRunner()
        self.performance_benchmarker = PerformanceBenchmarker()
        self.security_scanner = SecurityScanner()
        self.compatibility_checker = CompatibilityChecker()
        self.reporter = ValidationReporter()

        logger.info("ValidationPipeline initialized")

    def validate_deployment(
        self, packaging_result: dict[str, Any], config: "DeploymentConfig"
    ) -> dict[str, Any]:
        """Validate deployment package.

        Args:
            packaging_result: Result from packaging system
            config: Deployment configuration

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating deployment package for {config.artifact_id}")

        try:
            validation_results = {}

            # 1. Functional testing
            if config.run_functional_tests:
                logger.info("Running functional tests...")
                functional_result = self.functional_runner.run_tests(
                    packaging_result, config
                )
                validation_results.update(functional_result)

            # 2. Performance benchmarking
            if config.run_performance_tests:
                logger.info("Running performance benchmarks...")
                performance_result = (
                    self.performance_benchmarker.run_benchmarks(
                        packaging_result, config
                    )
                )
                validation_results.update(performance_result)

            # 3. Security scanning
            if config.run_security_scan:
                logger.info("Running security scan...")
                security_result = self.security_scanner.run_scan(
                    packaging_result, config
                )
                validation_results.update(security_result)

            # 4. Compatibility checks
            logger.info("Running compatibility checks...")
            compatibility_result = self.compatibility_checker.run_checks(
                packaging_result, config
            )
            validation_results.update(compatibility_result)

            # 5. Overall validation assessment
            overall_success = self._assess_validation_results(
                validation_results
            )

            result = {
                "success": overall_success,
                "functional_tests_passed": validation_results.get(
                    "functional_tests_passed", False
                ),
                "performance_score": validation_results.get(
                    "performance_score", 0.0
                ),
                "security_scan_passed": validation_results.get(
                    "security_scan_passed", False
                ),
                "inference_time_ms": validation_results.get(
                    "inference_time_ms", 0.0
                ),
                "memory_usage_mb": validation_results.get(
                    "memory_usage_mb", 0.0
                ),
                "throughput_requests_per_second": validation_results.get(
                    "throughput_rps", 0.0
                ),
                "vulnerabilities_found": validation_results.get(
                    "vulnerabilities_found", 0
                ),
                "security_score": validation_results.get(
                    "security_score", 0.0
                ),
                "validation_details": validation_results,
            }

            # Generate validation report
            report = self.reporter.generate_report(validation_results, config)
            result["validation_report"] = report

            logger.info(
                f"Validation completed: {'SUCCESS' if overall_success else 'FAILED'}"
            )
            return result

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "functional_tests_passed": False,
                "performance_score": 0.0,
                "security_scan_passed": False,
            }

    def _assess_validation_results(self, results: dict[str, Any]) -> bool:
        """Assess overall validation results.

        Args:
            results: Validation results dictionary

        Returns:
            True if all critical validations pass
        """
        try:
            # Check functional tests
            if results.get("functional_tests_passed") is False:
                logger.error("Functional tests failed")
                return False

            # Check performance thresholds
            inference_time = results.get("inference_time_ms", 0.0)
            if inference_time > self.thresholds.max_inference_time_ms:
                logger.error(
                    f"Inference time {inference_time}ms exceeds threshold "
                    f"{self.thresholds.max_inference_time_ms}ms"
                )
                return False

            memory_usage = results.get("memory_usage_mb", 0.0)
            if memory_usage > self.thresholds.max_memory_usage_mb:
                logger.error(
                    f"Memory usage {memory_usage}MB exceeds threshold "
                    f"{self.thresholds.max_memory_usage_mb}MB"
                )
                return False

            throughput = results.get("throughput_rps", 0.0)
            if throughput < self.thresholds.min_throughput_rps:
                logger.error(
                    f"Throughput {throughput} RPS below threshold "
                    f"{self.thresholds.min_throughput_rps} RPS"
                )
                return False

            # Check security scan
            if results.get("security_scan_passed") is False:
                logger.error("Security scan failed")
                return False

            vulnerabilities = results.get("vulnerabilities_found", 0)
            if vulnerabilities > self.thresholds.max_vulnerabilities:
                logger.error(
                    f"Found {vulnerabilities} vulnerabilities, maximum allowed "
                    f"is {self.thresholds.max_vulnerabilities}"
                )
                return False

            security_score = results.get("security_score", 0.0)
            if security_score < self.thresholds.min_security_score:
                logger.error(
                    f"Security score {security_score} below threshold "
                    f"{self.thresholds.min_security_score}"
                )
                return False

            logger.info("All validation checks passed")
            return True

        except Exception as e:
            logger.error(f"Validation assessment failed: {e}")
            return False
