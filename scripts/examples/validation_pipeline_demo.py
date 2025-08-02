#!/usr/bin/env python3
"""
Validation Pipeline Demo.

This script demonstrates the usage of the ValidationPipeline
to validate artifacts for deployment readiness.
"""

import logging
import sys
from pathlib import Path

from crackseg.utils.deployment.config import DeploymentConfig
from crackseg.utils.deployment.validation_pipeline import ValidationPipeline

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def create_mock_config() -> DeploymentConfig:
    """Create a mock deployment configuration."""
    return DeploymentConfig(
        artifact_id="crackseg-model-v2.1",
        target_environment="docker",
        target_format="pytorch",
        run_functional_tests=True,
        run_performance_tests=True,
        run_security_scan=True,
    )


def demonstrate_validation_pipeline() -> None:
    """Demonstrate the validation pipeline."""
    print("\n🔍 VALIDATION PIPELINE DEMO")
    print("=" * 50)
    print("Running comprehensive validation pipeline...")

    config = create_mock_config()
    pipeline = ValidationPipeline()

    # Run validation
    validation_results = pipeline.validate_artifact(config)

    print("\n📊 Validation Results:")
    logger.info(f"  Success: {validation_results.get('success', False)}")
    logger.info(
        f"  Functional Tests: "
        f"{validation_results.get('functional_tests_passed', False)}"
    )
    logger.info(
        f"  Performance Score: "
        f"{validation_results.get('performance_score', 0.0):.2f}"
    )
    logger.info(
        f"  Security Scan: "
        f"{validation_results.get('security_scan_passed', False)}"
    )
    logger.info(
        f"  Inference Time: "
        f"{validation_results.get('inference_time_ms', 0.0):.2f} ms"
    )
    logger.info(
        f"  Memory Usage: "
        f"{validation_results.get('memory_usage_mb', 0.0):.2f} MB"
    )
    throughput = validation_results.get("throughput_requests_per_second", 0.0)
    logger.info(f"  Throughput: {throughput:.2f} RPS")

    if validation_results.get("success", False):
        print("✅ Validation pipeline completed successfully!")
        print("🚀 Artifact is ready for deployment")
    else:
        print("❌ Validation pipeline failed!")
        print("🔧 Please address validation issues before deployment")


def demonstrate_performance_benchmark() -> None:
    """Demonstrate performance benchmarking."""
    print("\n⚡ PERFORMANCE BENCHMARK DEMO")
    print("=" * 50)
    print("Running performance benchmark tests...")

    config = create_mock_config()
    pipeline = ValidationPipeline()

    # Run performance benchmark
    performance_results = pipeline.benchmark_performance(config)

    print("\n📈 Performance Benchmark Results:")
    logger.info(
        f"  Performance Score: "
        f"{performance_results.get('performance_score', 0.0):.2f}"
    )
    logger.info(
        f"  Inference Time: "
        f"{performance_results.get('inference_time_ms', 0.0):.2f} ms"
    )
    logger.info(
        f"  Memory Usage: "
        f"{performance_results.get('memory_usage_mb', 0.0):.2f} MB"
    )
    logger.info(
        f"  Throughput: "
        f"{performance_results.get('throughput_rps', 0.0):.2f} RPS"
    )
    logger.info(
        f"  CPU Usage: "
        f"{performance_results.get('cpu_usage_percent', 0.0):.1f}%"
    )
    logger.info(
        f"  GPU Usage: "
        f"{performance_results.get('gpu_usage_percent', 0.0):.1f}%"
    )
    logger.info(
        f"  Throughput Score: "
        f"{performance_results.get('throughput_score', 0.0):.2f}"
    )

    if performance_results.get("performance_score", 0.0) >= 0.8:
        print("✅ Performance benchmark passed!")
        print("⚡ Model meets performance requirements")
    else:
        print("❌ Performance benchmark failed!")
        print("🔧 Consider model optimization")


def demonstrate_security_scan() -> None:
    """Demonstrate security scanning."""
    print("\n🛡️ SECURITY SCAN DEMO")
    print("=" * 50)
    print("Running security vulnerability scan...")

    config = create_mock_config()
    pipeline = ValidationPipeline()

    # Run security scan
    security_results = pipeline.scan_security(config)

    print("\n🛡️ Security Scan Results:")
    logger.info(
        f"  Security Scan Passed: "
        f"{security_results.get('security_scan_passed', False)}"
    )
    logger.info(
        f"  Security Score: "
        f"{security_results.get('security_score', 0.0):.1f}/10.0"
    )
    vulnerabilities = security_results.get("vulnerabilities_found", 0)
    logger.info(f"  Vulnerabilities Found: {vulnerabilities}")
    logger.info(
        f"  Security Level: "
        f"{security_results.get('security_level', 'unknown')}"
    )

    if security_results.get("security_scan_passed", False):
        print("✅ Security scan passed!")
        print("🛡️ No critical vulnerabilities found")
    else:
        print("❌ Security scan failed!")
        print("🔧 Please address security vulnerabilities")


def demonstrate_compatibility_check() -> None:
    """Demonstrate compatibility checking."""
    print("\n🔧 COMPATIBILITY CHECK DEMO")
    print("=" * 50)
    print("Running compatibility validation...")

    config = create_mock_config()
    pipeline = ValidationPipeline()

    # Run compatibility check
    compatibility_results = pipeline.check_compatibility(config)

    print("\n🔧 Compatibility Check Results:")
    logger.info(
        f"  Python Compatible: "
        f"{compatibility_results.get('python_compatible', False)}"
    )
    logger.info(
        f"  Dependencies Compatible: "
        f"{compatibility_results.get('dependencies_compatible', False)}"
    )
    logger.info(
        f"  Environment Compatible: "
        f"{compatibility_results.get('environment_compatible', False)}"
    )
    logger.info(
        f"  Compatibility Score: "
        f"{compatibility_results.get('compatibility_score', 0.0):.2f}"
    )

    if compatibility_results.get("compatibility_score", 0.0) >= 0.9:
        print("✅ Compatibility check passed!")
        print("🔧 Environment is compatible")
    else:
        print("❌ Compatibility check failed!")
        print("🔧 Please verify environment compatibility")


def main() -> None:
    """Main function to demonstrate validation pipeline."""
    print("🔍 CRACKSEG VALIDATION PIPELINE DEMO")
    print("=" * 60)
    print("This demo showcases the comprehensive validation pipeline")
    print("for ML model artifacts before deployment.")
    print("=" * 60)

    try:
        # Demonstrate validation pipeline
        demonstrate_validation_pipeline()

        # Demonstrate individual validation components
        demonstrate_performance_benchmark()
        demonstrate_security_scan()
        demonstrate_compatibility_check()

        print("\n✅ Demo completed successfully!")
        print("\n🎯 Key Takeaways:")
        print("- Validation pipeline ensures deployment readiness")
        print("- Performance benchmarking validates model efficiency")
        print("- Security scanning prevents vulnerabilities")
        print("- Compatibility checking ensures environment fit")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
