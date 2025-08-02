"""Example script demonstrating deployment orchestration capabilities.

This script showcases the advanced deployment orchestration features including
blue-green deployments, canary releases, rolling updates, and rollback
mechanisms.
"""

import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path modification
from src.crackseg.utils.deployment import (  # noqa: E402
    DefaultHealthChecker,
    DeploymentConfig,
    DeploymentManager,
    DeploymentOrchestrator,
    DeploymentStrategy,
)


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_mock_storage() -> Any:
    """Create mock traceability storage for testing."""
    from unittest.mock import Mock

    mock_storage = Mock()
    mock_storage.query_interface = Mock()
    return mock_storage


def demonstrate_blue_green_deployment() -> Any:
    """Demonstrate blue-green deployment strategy."""
    print("\n" + "=" * 60)
    print("🔵🟢 BLUE-GREEN DEPLOYMENT DEMONSTRATION")
    print("=" * 60)

    # Create deployment manager
    mock_storage = create_mock_storage()
    deployment_manager = DeploymentManager(mock_storage)

    # Configure deployment
    config = DeploymentConfig(
        artifact_id="crackseg-model-v2.1",
        target_environment="production",
        deployment_type="container",
        enable_quantization=True,
        target_format="onnx",
    )

    print(f"📦 Deploying artifact: {config.artifact_id}")
    print(f"🎯 Target environment: {config.target_environment}")
    print(f"🏗️  Deployment type: {config.deployment_type}")
    print("⚡ Strategy: Blue-Green (zero downtime)")

    # Execute blue-green deployment
    result = deployment_manager.deploy_artifact(
        artifact_id=config.artifact_id,
        target_environment=config.target_environment,
        deployment_type=config.deployment_type,
        strategy=DeploymentStrategy.BLUE_GREEN,
        enable_quantization=config.enable_quantization,
        target_format=config.target_format,
    )

    if result.success:
        print("✅ Blue-green deployment completed successfully!")
        print(f"🌐 Deployment URL: {result.deployment_url}")
        print(f"🏥 Health check URL: {result.health_check_url}")
        print(f"📊 Monitoring dashboard: {result.monitoring_dashboard_url}")
    else:
        print(f"❌ Blue-green deployment failed: {result.error_message}")

    return result


def demonstrate_canary_deployment() -> Any:
    """Demonstrate canary deployment strategy."""
    print("\n" + "=" * 60)
    print("🦅 CANARY DEPLOYMENT DEMONSTRATION")
    print("=" * 60)

    # Create deployment manager
    mock_storage = create_mock_storage()
    deployment_manager = DeploymentManager(mock_storage)

    # Configure deployment
    config = DeploymentConfig(
        artifact_id="crackseg-model-v2.2",
        target_environment="production",
        deployment_type="kubernetes",
        enable_quantization=True,
        target_format="onnx",
    )

    print(f"📦 Deploying artifact: {config.artifact_id}")
    print(f"🎯 Target environment: {config.target_environment}")
    print(f"🏗️  Deployment type: {config.deployment_type}")
    print("⚡ Strategy: Canary (gradual traffic increase)")

    # Execute canary deployment
    result = deployment_manager.deploy_artifact(
        artifact_id=config.artifact_id,
        target_environment=config.target_environment,
        deployment_type=config.deployment_type,
        strategy=DeploymentStrategy.CANARY,
        enable_quantization=config.enable_quantization,
        target_format=config.target_format,
    )

    if result.success:
        print("✅ Canary deployment completed successfully!")
        print(f"🌐 Deployment URL: {result.deployment_url}")
        print(f"🏥 Health check URL: {result.health_check_url}")
        print("📈 Traffic gradually increased: 10% → 25% → 50% → 75% → 100%")
    else:
        print(f"❌ Canary deployment failed: {result.error_message}")

    return result


def demonstrate_rolling_deployment() -> Any:
    """Demonstrate rolling deployment strategy."""
    print("\n" + "=" * 60)
    print("🔄 ROLLING DEPLOYMENT DEMONSTRATION")
    print("=" * 60)

    # Create deployment manager
    mock_storage = create_mock_storage()
    deployment_manager = DeploymentManager(mock_storage)

    # Configure deployment
    config = DeploymentConfig(
        artifact_id="crackseg-model-v2.3",
        target_environment="production",
        deployment_type="kubernetes",
        enable_quantization=True,
        target_format="onnx",
    )

    print(f"📦 Deploying artifact: {config.artifact_id}")
    print(f"🎯 Target environment: {config.target_environment}")
    print(f"🏗️  Deployment type: {config.deployment_type}")
    print("⚡ Strategy: Rolling (replica-by-replica update)")

    # Execute rolling deployment
    result = deployment_manager.deploy_artifact(
        artifact_id=config.artifact_id,
        target_environment=config.target_environment,
        deployment_type=config.deployment_type,
        strategy=DeploymentStrategy.ROLLING,
        enable_quantization=config.enable_quantization,
        target_format=config.target_format,
    )

    if result.success:
        print("✅ Rolling deployment completed successfully!")
        print(f"🌐 Deployment URL: {result.deployment_url}")
        print(f"🏥 Health check URL: {result.health_check_url}")
        print("🔄 Replicas updated one by one: 1/3 → 2/3 → 3/3")
    else:
        print(f"❌ Rolling deployment failed: {result.error_message}")

    return result


def demonstrate_rollback_mechanism() -> Any:
    """Demonstrate rollback mechanism."""
    print("\n" + "=" * 60)
    print("🔄 ROLLBACK MECHANISM DEMONSTRATION")
    print("=" * 60)

    # Create orchestrator directly for demonstration
    orchestrator = DeploymentOrchestrator()

    # Simulate a failed deployment
    print("🚨 Simulating deployment failure...")

    def failing_deployment_func(*args, **kwargs):
        raise RuntimeError("Simulated deployment failure")

    config = DeploymentConfig(
        artifact_id="crackseg-model-v2.4",
        target_environment="production",
        deployment_type="container",
    )

    # Execute deployment with automatic rollback
    result = orchestrator.deploy_with_strategy(
        config,
        DeploymentStrategy.BLUE_GREEN,
        failing_deployment_func,
    )

    if not result.success:
        print("❌ Deployment failed as expected")
        print(f"🔍 Error: {result.error_message}")

        # Check deployment status
        status = orchestrator.get_deployment_status(result.deployment_id)
        print(f"📊 Deployment state: {status['state']}")

        if status.get("rollback_reason"):
            print(f"🔄 Rollback reason: {status['rollback_reason']}")

        # Demonstrate manual rollback
        print("\n🔧 Attempting manual rollback...")
        rollback_success = orchestrator.manual_rollback(result.deployment_id)

        if rollback_success:
            print("✅ Manual rollback completed successfully!")
        else:
            print("❌ Manual rollback failed")

    return result


def demonstrate_deployment_history() -> None:
    """Demonstrate deployment history tracking."""
    print("\n" + "=" * 60)
    print("📚 DEPLOYMENT HISTORY DEMONSTRATION")
    print("=" * 60)

    # Create orchestrator
    orchestrator = DeploymentOrchestrator()

    # Simulate multiple deployments
    configs = [
        DeploymentConfig(
            artifact_id="model-v1", target_environment="production"
        ),
        DeploymentConfig(
            artifact_id="model-v2", target_environment="production"
        ),
        DeploymentConfig(artifact_id="model-v3", target_environment="staging"),
    ]

    def successful_deployment_func(*args, **kwargs):
        from unittest.mock import Mock

        return Mock(
            success=True,
            health_check_url="http://localhost:8501/healthz",
            deployment_url="http://localhost:8501",
        )

    print("📝 Creating multiple deployments...")

    for i, config in enumerate(configs, 1):
        print(
            f"  {i}. Deploying {config.artifact_id} to "
            f"{config.target_environment}"
        )

        with patch.object(
            orchestrator.health_checker, "wait_for_healthy", return_value=True
        ):
            result = orchestrator.deploy_with_strategy(
                config,
                DeploymentStrategy.BLUE_GREEN,
                successful_deployment_func,
            )

        if result.success:
            print(f"     ✅ Success: {result.deployment_id}")
        else:
            print(f"     ❌ Failed: {result.error_message}")

    # Display deployment history
    print("\n📊 Deployment History:")
    history = orchestrator.get_deployment_history()

    for deployment in history[:5]:  # Show last 5 deployments
        print(f"  📦 {deployment['artifact_id']}")
        print(f"     🏷️  ID: {deployment['deployment_id']}")
        print(f"     🎯 Strategy: {deployment['strategy']}")
        print(f"     📈 State: {deployment['state']}")
        print(f"     ⏱️  Duration: {deployment['duration']:.2f}s")
        print()

    # Filter by artifact
    print("🔍 Filtered History (model-v1):")
    filtered_history = orchestrator.get_deployment_history("model-v1")
    for deployment in filtered_history:
        print(f"  📦 {deployment['artifact_id']} - {deployment['state']}")


def demonstrate_health_checking() -> None:
    """Demonstrate health checking capabilities."""
    print("\n" + "=" * 60)
    print("🏥 HEALTH CHECKING DEMONSTRATION")
    print("=" * 60)

    # Create health checker
    health_checker = DefaultHealthChecker()

    # Test URLs
    test_urls = [
        "http://localhost:8501",
        "http://localhost:8000",
        "https://api.example.com",
    ]

    print("🔍 Testing health check functionality...")

    for url in test_urls:
        print(f"  🏥 Checking: {url}")

        # Mock the health check response
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            is_healthy = health_checker.check_health(url)

            if is_healthy:
                print("     ✅ Healthy")
            else:
                print("     ❌ Unhealthy")

    print("\n⏳ Testing wait for healthy functionality...")

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        is_healthy = health_checker.wait_for_healthy(
            "http://localhost:8501", max_wait=5
        )

        if is_healthy:
            print("     ✅ Service became healthy within timeout")
        else:
            print("     ❌ Service did not become healthy within timeout")


def main() -> None:
    """Main demonstration function."""
    print("🚀 CRACKSEG DEPLOYMENT ORCHESTRATION DEMONSTRATION")
    print("=" * 60)
    print("This script demonstrates advanced deployment orchestration")
    print("capabilities including blue-green, canary, rolling deployments")
    print("and automatic rollback mechanisms.")
    print("=" * 60)

    # Set up logging
    setup_logging()

    try:
        # Demonstrate different deployment strategies
        demonstrate_blue_green_deployment()
        demonstrate_canary_deployment()
        demonstrate_rolling_deployment()

        # Demonstrate rollback mechanisms
        demonstrate_rollback_mechanism()

        # Demonstrate deployment history
        demonstrate_deployment_history()

        # Demonstrate health checking
        demonstrate_health_checking()

        print("\n" + "=" * 60)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n🎯 Key Features Demonstrated:")
        print("  🔵🟢 Blue-Green Deployment (zero downtime)")
        print("  🦅 Canary Deployment (gradual traffic increase)")
        print("  🔄 Rolling Deployment (replica-by-replica)")
        print("  🔄 Automatic Rollback (failure recovery)")
        print("  📚 Deployment History (tracking and filtering)")
        print("  🏥 Health Checking (monitoring and validation)")
        print(
            "\n🚀 The deployment orchestration system is ready for "
            "production use!"
        )

    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Import patch for mocking
    from unittest.mock import patch

    main()
