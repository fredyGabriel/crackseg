#!/usr/bin/env python3
"""
Example script demonstrating artifact selection and environment configuration.

This script shows how to use the ArtifactSelector and EnvironmentConfigurator
to select appropriate artifacts and configure deployment environments.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crackseg.utils.deployment import (
    ArtifactSelector,
    EnvironmentConfigurator,
    SelectionCriteria,
)
from crackseg.utils.traceability import TraceabilityStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Demonstrate artifact selection and environment configuration."""
    logger.info(
        "🔍 Starting Artifact Selection and Environment Configuration Demo"
    )

    try:
        # 1. Initialize traceability storage
        storage_path = Path("traceability_storage")
        storage = TraceabilityStorage(storage_path=storage_path)

        # 2. Initialize components
        artifact_selector = ArtifactSelector(storage.query_interface)
        environment_configurator = EnvironmentConfigurator()

        # 3. Demonstrate artifact selection for different environments
        environments = ["production", "staging", "development"]
        deployment_types = ["container", "serverless", "edge"]

        for env in environments:
            for deploy_type in deployment_types:
                logger.info(
                    f"\n📋 Environment: {env}, Deployment: {deploy_type}"
                )

                # Get recommendations
                recommendations = (
                    artifact_selector.get_artifact_recommendations(
                        env, deploy_type
                    )
                )

                logger.info(f"  Criteria: {recommendations['criteria']}")
                logger.info(
                    "  Selection Result: "
                    f"{recommendations['selection_result'].selection_reason}"
                )

                if recommendations["recommendations"]:
                    for rec in recommendations["recommendations"]:
                        logger.info(
                            f"  Recommended Artifact: {rec['artifact_id']}"
                        )
                        logger.info(f"  Reason: {rec['reason']}")
                else:
                    logger.info("  No artifacts found matching criteria")

        # 4. Demonstrate custom selection criteria
        logger.info("\n🎯 Custom Selection Criteria Demo")

        custom_criteria = SelectionCriteria(
            min_accuracy=0.90,
            max_inference_time_ms=300.0,
            max_memory_usage_mb=512.0,
            max_model_size_mb=100.0,
            preferred_format="onnx",
            target_environment="production",
            deployment_type="container",
            model_family="swin-unet",
            tags=["optimized", "quantized"],
        )

        selection_result = artifact_selector.select_artifacts(custom_criteria)
        logger.info(f"  Custom Selection: {selection_result.selection_reason}")
        logger.info(f"  Candidates: {selection_result.total_candidates}")
        logger.info(f"  Filtered: {selection_result.filtered_candidates}")
        logger.info(f"  Selected: {len(selection_result.selected_artifacts)}")

        # 5. Demonstrate environment configuration
        logger.info("\n⚙️ Environment Configuration Demo")

        from crackseg.utils.deployment import DeploymentConfig

        configs = [
            DeploymentConfig(
                artifact_id="swin-unet-v1",
                target_environment="production",
                deployment_type="container",
                enable_quantization=True,
                target_format="onnx",
            ),
            DeploymentConfig(
                artifact_id="swin-unet-v1",
                target_environment="staging",
                deployment_type="serverless",
                enable_quantization=False,
                target_format="pytorch",
            ),
            DeploymentConfig(
                artifact_id="swin-unet-v1",
                target_environment="development",
                deployment_type="edge",
                enable_quantization=True,
                target_format="torchscript",
            ),
        ]

        for config in configs:
            logger.info(
                f"\n🔧 Configuring for {config.target_environment}-"
                f"{config.deployment_type}"
            )

            env_result = environment_configurator.configure_environment(config)

            if env_result.success:
                env_config = env_result.environment_config
                if env_config:
                    summary = environment_configurator.get_environment_summary(
                        env_config
                    )
                else:
                    summary = {}

                logger.info(f"  Environment: {summary['environment_name']}")
                logger.info(f"  Resources: {summary['resources']}")
                logger.info(f"  Python: {summary['python_version']}")
                logger.info(
                    f"  Packages: {summary['required_packages_count']}"
                )
                logger.info(
                    "  System Dependencies: "
                    f"{summary['system_dependencies_count']}"
                )
                logger.info(f"  Replicas: {summary['replicas']}")
                logger.info(f"  Autoscaling: {summary['autoscaling']}")
                logger.info(f"  Log Level: {summary['log_level']}")

                logger.info(
                    "  Configuration Files: "
                    f"{len(env_result.configuration_files)}"
                )
                for file_path in env_result.configuration_files:
                    logger.info(f"    - {file_path}")
            else:
                logger.error(
                    f"  Configuration failed: {env_result.error_message}"
                )

        logger.info(
            "\n🎉 Artifact Selection and Environment Configuration demo "
            "completed!"
        )

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
