#!/usr/bin/env python3
"""
Example script demonstrating the CrackSeg deployment system.

This script shows how to use the DeploymentManager to deploy
artifacts through the complete pipeline.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crackseg.utils.deployment import DeploymentManager
from crackseg.utils.traceability import TraceabilityStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Demonstrate deployment system usage."""
    logger.info("🚀 Starting CrackSeg Deployment System Demo")

    try:
        # 1. Initialize traceability storage
        storage_path = Path("traceability_storage")
        storage = TraceabilityStorage(storage_path=storage_path)

        # 2. Initialize deployment manager
        output_dir = Path("infrastructure/deployment/packages")
        deployment_manager = DeploymentManager(
            storage=storage, output_dir=output_dir
        )

        # 3. Deploy artifact
        logger.info("📦 Starting artifact deployment...")
        result = deployment_manager.deploy_artifact(
            artifact_id="swin-unet-v1",
            target_environment="production",
            deployment_type="container",
            enable_quantization=True,
            target_format="onnx",
        )

        # 5. Display results
        logger.info("📊 Deployment Results:")
        logger.info(f"  Success: {result.success}")
        logger.info(f"  Deployment ID: {result.deployment_id}")
        logger.info(f"  Artifact ID: {result.artifact_id}")
        logger.info(f"  Target Environment: {result.target_environment}")

        if result.success:
            logger.info("✅ Deployment completed successfully!")
            logger.info(f"  Original Size: {result.original_size_mb:.2f} MB")
            logger.info(f"  Optimized Size: {result.optimized_size_mb:.2f} MB")
            logger.info(
                f"  Compression Ratio: {result.compression_ratio:.2f}x"
            )
            functional_status = (
                "✅ PASSED" if result.functional_tests_passed else "❌ FAILED"
            )
            security_status = (
                "✅ PASSED" if result.security_scan_passed else "❌ FAILED"
            )

            logger.info(f"  Functional Tests: {functional_status}")
            logger.info(
                f"  Performance Score: "
                f"{result.performance_benchmark_score:.2f}"
            )
            logger.info(f"  Security Scan: {security_status}")

            if result.deployment_url:
                logger.info(f"  Deployment URL: {result.deployment_url}")
            if result.health_check_url:
                logger.info(f"  Health Check URL: {result.health_check_url}")
            if result.monitoring_dashboard_url:
                logger.info(
                    f"  Monitoring Dashboard: "
                    f"{result.monitoring_dashboard_url}"
                )
        else:
            logger.error("❌ Deployment failed!")
            if result.error_message:
                logger.error(f"  Error: {result.error_message}")

        logger.info("🎉 Deployment demo completed!")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
