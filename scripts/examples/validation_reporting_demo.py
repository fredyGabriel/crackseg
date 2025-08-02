#!/usr/bin/env python3
"""
Validation Reporting Demo.

This script demonstrates the usage of the ValidationReporter
to generate comprehensive validation reports.
"""

import logging
import sys
from pathlib import Path

from crackseg.utils.deployment.config import DeploymentConfig
from crackseg.utils.deployment.validation_pipeline import ValidationPipeline
from crackseg.utils.deployment.validation_reporter import ValidationReporter

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


def demonstrate_validation_reporting() -> None:
    """Demonstrate validation reporting capabilities."""
    print("\n📊 VALIDATION REPORTING DEMO")
    print("=" * 50)
    print("Generating comprehensive validation reports...")

    config = create_mock_config()
    pipeline = ValidationPipeline()
    reporter = ValidationReporter()

    # Run validation
    validation_results = pipeline.validate_artifact(config)

    # Generate comprehensive report
    report_data = reporter.generate_comprehensive_report(
        validation_results, config
    )

    # Save reports in different formats
    markdown_path = reporter.save_report(report_data, "markdown")
    json_path = reporter.save_report(report_data, "json")

    print("\n📄 Generated Reports:")
    logger.info(f"  Markdown Report: {markdown_path}")
    logger.info(f"  JSON Report: {json_path}")

    if validation_results.get("success", False):
        print("✅ Validation reporting completed successfully!")
        print("📊 Comprehensive reports generated")
    else:
        print("❌ Validation reporting completed with issues!")
        print("📋 Reports include failure details")


def demonstrate_chart_generation() -> None:
    """Demonstrate chart generation capabilities."""
    print("\n📈 CHART GENERATION DEMO")
    print("=" * 50)
    print("Generating validation metric charts...")

    config = create_mock_config()
    pipeline = ValidationPipeline()
    reporter = ValidationReporter()

    # Run validation
    validation_results = pipeline.validate_artifact(config)

    # Generate comprehensive report
    report_data = reporter.generate_comprehensive_report(
        validation_results, config
    )

    # Generate performance charts
    chart_paths = reporter.generate_performance_charts(report_data)

    print(f"\n📊 Generated Charts ({len(chart_paths)} total):")
    for chart_path in chart_paths:
        logger.info(f"  Chart: {chart_path}")

    print("✅ Chart generation completed!")
    print("📊 Performance and summary charts created")


def demonstrate_risk_assessment() -> None:
    """Demonstrate risk assessment capabilities."""
    print("\n⚠️ RISK ASSESSMENT DEMO")
    print("=" * 50)
    print("Assessing deployment risk levels...")

    config = create_mock_config()
    pipeline = ValidationPipeline()
    reporter = ValidationReporter()

    # Run validation
    validation_results = pipeline.validate_artifact(config)

    # Generate comprehensive report
    report_data = reporter.generate_comprehensive_report(
        validation_results, config
    )
    risk_level = report_data.risk_level
    deployment_ready = report_data.deployment_ready

    print("\n📊 Risk Assessment Results:")
    logger.info(f"  Risk Level: {risk_level}")
    logger.info(f"  Deployment Ready: {deployment_ready}")
    logger.info(
        f"  Estimated Deployment Time: {report_data.estimated_deployment_time}"
    )

    if deployment_ready:
        print("✅ Risk assessment passed!")
        print("🚀 Deployment is safe to proceed")
    else:
        print("❌ Risk assessment failed!")
        print("🔧 Address issues before deployment")


def demonstrate_recommendation_engine() -> None:
    """Demonstrate recommendation engine capabilities."""
    print("\n💡 RECOMMENDATION ENGINE DEMO")
    print("=" * 50)
    print("Generating actionable recommendations...")

    config = create_mock_config()
    pipeline = ValidationPipeline()
    reporter = ValidationReporter()

    # Run validation
    validation_results = pipeline.validate_artifact(config)

    # Generate comprehensive report
    report_data = reporter.generate_comprehensive_report(
        validation_results, config
    )
    recommendations = report_data.recommendations

    print(f"\n💡 Recommendations ({len(recommendations)} total):")
    for i, recommendation in enumerate(recommendations, 1):
        logger.info(f"  {i}. {recommendation}")

    print("✅ Recommendation engine completed!")
    print("💡 Actionable advice generated")


def main() -> None:
    """Main function to demonstrate validation reporting."""
    print("📊 CRACKSEG VALIDATION REPORTING DEMO")
    print("=" * 60)
    print("This demo showcases comprehensive validation reporting")
    print("for ML model artifacts with detailed analysis.")
    print("=" * 60)

    try:
        # Demonstrate validation reporting
        demonstrate_validation_reporting()

        # Demonstrate individual reporting components
        demonstrate_chart_generation()
        demonstrate_risk_assessment()
        demonstrate_recommendation_engine()

        print("\n✅ Demo completed successfully!")
        print("\n🎯 Key Takeaways:")
        print("- Comprehensive reporting provides detailed insights")
        print("- Multiple report formats support different needs")
        print("- Risk assessment prevents deployment issues")
        print("- Recommendation engine guides improvements")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
