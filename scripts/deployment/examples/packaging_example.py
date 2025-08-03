"""Example script demonstrating advanced packaging and containerization.

This script showcases the enhanced PackagingSystem capabilities including
automated containerization, security scanning, and multi-target deployment.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path modification
from src.crackseg.utils.deployment import (  # noqa: E402
    DeploymentConfig,
    PackagingSystem,
)
from src.crackseg.utils.traceability import ArtifactEntity  # noqa: E402
from src.crackseg.utils.traceability.enums import ArtifactType  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_artifact() -> ArtifactEntity:
    """Create a sample artifact for packaging demonstration."""
    return ArtifactEntity(
        artifact_id="sample-crackseg-model",
        artifact_type=ArtifactType.MODEL,
        file_path=Path("models/sample_model.pth"),
        file_size=1024000,  # 1MB
        checksum="sample_checksum_1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        name="Sample CrackSeg Model",
        description="Sample model for packaging demonstration",
        tags=["crack-detection", "production-ready"],
        owner="demo_user",
        experiment_id="demo_experiment_123",
        metadata={
            "model_architecture": "unet",
            "training_epochs": 100,
            "accuracy": 0.85,
            "framework": "pytorch",
        },
    )


def create_deployment_configs() -> list[DeploymentConfig]:
    """Create different deployment configurations for testing."""
    configs = []

    # Production configuration
    production_config = DeploymentConfig(
        artifact_id="sample-crackseg-model",
        deployment_type="kubernetes",
        target_environment="production",
        target_format="pytorch",
        enable_quantization=True,
        enable_pruning=True,
        enable_health_checks=True,
        enable_metrics_collection=True,
    )
    configs.append(production_config)

    # Staging configuration
    staging_config = DeploymentConfig(
        artifact_id="sample-crackseg-model",
        deployment_type="container",
        target_environment="staging",
        target_format="pytorch",
        enable_quantization=True,
        enable_pruning=False,
        enable_health_checks=True,
        enable_metrics_collection=True,
    )
    configs.append(staging_config)

    # Development configuration
    development_config = DeploymentConfig(
        artifact_id="sample-crackseg-model",
        deployment_type="container",
        target_environment="development",
        target_format="pytorch",
        enable_quantization=False,
        enable_pruning=False,
        enable_health_checks=False,
        enable_metrics_collection=False,
    )
    configs.append(development_config)

    return configs


def demonstrate_packaging_system() -> None:
    """Demonstrate the enhanced packaging system capabilities."""
    logger.info("🚀 Starting Packaging System Demonstration")

    # Initialize packaging system
    packaging_system = PackagingSystem()

    # Create sample artifact
    artifact = create_sample_artifact()
    logger.info(f"📦 Created sample artifact: {artifact.artifact_id}")

    # Create deployment configurations
    configs = create_deployment_configs()
    logger.info(f"⚙️ Created {len(configs)} deployment configurations")

    # Simulate optimization results
    optimization_result = {
        "artifact_id": artifact.artifact_id,
        "optimized_model_path": (
            "deployments/sample-crackseg-model/optimized_model.pth"
        ),
        "compression_ratio": 2.5,
        "optimization_strategy": "production",
    }

    # Test packaging for each configuration
    for i, config in enumerate(configs, 1):
        logger.info(
            f"\n🔧 Testing Configuration {i}: {config.target_environment}"
        )
        logger.info(f"   Deployment Type: {config.deployment_type}")
        logger.info(f"   Target Environment: {config.target_environment}")
        logger.info(f"   Quantization: {config.enable_quantization}")
        logger.info(f"   Pruning: {config.enable_pruning}")

        try:
            # Package artifact
            result = packaging_system.package_artifact(
                optimization_result, config
            )

            if result.success:
                logger.info("✅ Packaging completed successfully!")
                logger.info(
                    f"   Package Size: {result.package_size_mb:.2f} MB"
                )
                logger.info(f"   Dependencies: {result.dependencies_count}")
                logger.info(f"   Build Time: {result.build_time_seconds:.2f}s")

                if result.container_image_name:
                    logger.info(
                        f"   Container Image: {result.container_image_name}"
                    )
                    logger.info(
                        f"   Image Size: {result.image_size_mb:.2f} MB"
                    )
                    logger.info(f"   Layer Count: {result.layer_count}")

                if result.kubernetes_manifests:
                    logger.info(
                        f"   Kubernetes Manifests: "
                        f"{len(result.kubernetes_manifests)}"
                    )

                if result.docker_compose_path:
                    logger.info(
                        f"   Docker Compose: {result.docker_compose_path}"
                    )

                # Security scan results
                if result.security_scan_results:
                    scan_results = result.security_scan_results
                    if "error" not in scan_results:
                        logger.info("🔒 Security scan completed:")
                        logger.info(
                            f"   Total Vulnerabilities: "
                            f"{scan_results.get('total_vulnerabilities', 0)}"
                        )
                        critical_vulns = scan_results.get(
                            "critical_vulnerabilities", 0
                        )
                        logger.info(f"   Critical: {critical_vulns}")
                        logger.info(
                            f"   High: "
                            f"{scan_results.get('high_vulnerabilities', 0)}"
                        )
                        logger.info(
                            f"   Medium: "
                            f"{scan_results.get('medium_vulnerabilities', 0)}"
                        )
                        logger.info(
                            f"   Low: "
                            f"{scan_results.get('low_vulnerabilities', 0)}"
                        )
                    else:
                        logger.warning(
                            f"⚠️ Security scan error: {scan_results['error']}"
                        )

            else:
                logger.error(f"❌ Packaging failed: {result.error_message}")

        except Exception as e:
            logger.error(f"❌ Error during packaging: {e}")

    # Demonstrate packaging recommendations
    logger.info("\n📋 Packaging Recommendations:")
    for config in configs:
        recommendations = packaging_system.get_packaging_recommendations(
            config
        )
        logger.info(
            f"\n   Environment: {recommendations['target_environment']}"
        )
        logger.info(
            f"   Deployment Type: {recommendations['deployment_type']}"
        )
        logger.info(
            f"   Base Image: {recommendations['recommended_base_image']}"
        )

        if recommendations["optimization_suggestions"]:
            logger.info("   Optimization Suggestions:")
            for suggestion in recommendations["optimization_suggestions"]:
                logger.info(f"     • {suggestion}")

        if recommendations["security_recommendations"]:
            logger.info("   Security Recommendations:")
            for rec in recommendations["security_recommendations"]:
                logger.info(f"     • {rec}")

        if recommendations["performance_tips"]:
            logger.info("   Performance Tips:")
            for tip in recommendations["performance_tips"]:
                logger.info(f"     • {tip}")


def demonstrate_containerization_features() -> None:
    """Demonstrate advanced containerization features."""
    logger.info("\n🐳 Advanced Containerization Features:")

    packaging_system = PackagingSystem()

    # Show available base images
    logger.info("📦 Available Base Images:")
    for image_type, image_name in packaging_system.base_images.items():
        logger.info(f"   {image_type}: {image_name}")

    # Show containerization configurations
    logger.info("\n⚙️ Containerization Configurations:")
    for env, config in packaging_system.container_configs.items():
        logger.info(f"   {env.upper()}:")
        logger.info(f"     Base Image: {config.base_image}")
        logger.info(f"     Multi-stage: {config.multi_stage}")
        logger.info(f"     Non-root User: {config.non_root_user}")
        logger.info(f"     Security Scan: {config.security_scan}")
        logger.info(f"     Layer Caching: {config.layer_caching}")
        logger.info(f"     Compression: {config.compression}")


def demonstrate_package_structure() -> None:
    """Demonstrate the comprehensive package structure."""
    logger.info("\n📁 Package Structure Demonstration:")

    # Create a sample package structure
    package_dir = Path(
        "infrastructure/deployment/packages/test-crackseg-model/package"
    )
    package_dir.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    directories = ["app", "config", "scripts", "tests", "docs", "k8s"]

    for dir_name in directories:
        (package_dir / dir_name).mkdir(exist_ok=True)
        logger.info(f"   📂 {dir_name}/")

    # Create sample files
    sample_files = {
        "app/main.py": "FastAPI application entry point",
        "app/streamlit_app.py": "Streamlit web interface",
        "scripts/health_check.py": "Health check script",
        "scripts/deploy_docker.sh": "Docker deployment script",
        "scripts/deploy_kubernetes.sh": "Kubernetes deployment script",
        "config/app_config.json": "Application configuration",
        "config/environment.json": "Environment configuration",
        "k8s/deployment.yaml": "Kubernetes deployment manifest",
        "k8s/service.yaml": "Kubernetes service manifest",
        "k8s/ingress.yaml": "Kubernetes ingress manifest",
        "docker-compose.yml": "Docker Compose configuration",
        "Dockerfile": "Multi-stage Dockerfile",
        "requirements.txt": "Python dependencies",
    }

    for file_path, description in sample_files.items():
        full_path = package_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"# {description}\n")
        logger.info(f"   📄 {file_path} - {description}")

    logger.info(f"\n✅ Package structure created at: {package_dir}")


def main() -> None:
    """Main demonstration function."""
    logger.info("🎯 CrackSeg Packaging System Demonstration")
    logger.info("=" * 50)

    try:
        # Demonstrate packaging system
        demonstrate_packaging_system()

        # Demonstrate containerization features
        demonstrate_containerization_features()

        # Demonstrate package structure
        demonstrate_package_structure()

        logger.info("\n🎉 Demonstration completed successfully!")
        logger.info("The enhanced PackagingSystem provides:")
        logger.info("✅ Automated containerization with multi-stage builds")
        logger.info("✅ Security scanning integration")
        logger.info("✅ Comprehensive deployment manifests")
        logger.info("✅ Environment-specific optimizations")
        logger.info("✅ Health checks and monitoring")
        logger.info("✅ Multi-target deployment support")

    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
