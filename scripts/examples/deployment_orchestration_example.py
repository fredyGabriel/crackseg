"""Deployment Orchestration and Rollback Mechanisms Example.

This script demonstrates the complete deployment orchestration system including
blue-green deployments, canary releases, rolling updates, and automatic/manual
rollback mechanisms as required by subtask 5.5.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

from crackseg.utils.deployment import (
    DeploymentConfig,
    DeploymentOrchestrator,
    DeploymentState,
    DeploymentStrategy,
)
from crackseg.utils.traceability import ArtifactEntity

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_mock_artifact() -> ArtifactEntity:
    """Create a mock artifact for demonstration."""
    from crackseg.utils.traceability.enums import ArtifactType

    return ArtifactEntity(
        artifact_id="crackseg-model-v2.1",
        artifact_type=ArtifactType.MODEL,
        file_path=Path("/path/to/model/checkpoint.pth"),
        file_size=245000000,  # 245MB
        checksum="a" * 64,  # Mock SHA256
        name="CrackSeg Model v2.1",
        owner="ml-team",
        experiment_id="exp-001",
        metadata={
            "version": "2.1.0",
            "format": "pytorch",
            "architecture": "swin_transformer_v2",
            "training_config": (
                "configs/experiments/swinv2_hybrid/optimized.yaml"
            ),
            "metrics": {
                "val_iou": 0.87,
                "val_dice": 0.94,
                "test_iou": 0.85,
                "inference_time_ms": 120,
            },
        },
    )


def create_deployment_config() -> DeploymentConfig:
    """Create deployment configuration."""
    return DeploymentConfig(
        artifact_id="crackseg-model-v2.1",
        target_environment="production",
        deployment_type="container",
        enable_quantization=True,
        target_format="onnx",
    )


def mock_deployment_function(config: DeploymentConfig, **kwargs) -> Any:
    """Mock deployment function that simulates actual deployment."""
    print(f"🚀 Deploying {config.artifact_id} to {config.target_environment}")
    print(f"📦 Deployment type: {config.deployment_type}")
    print(f"⚙️ Quantization: {config.enable_quantization}")
    print(f"🎯 Target format: {config.target_format}")

    # Simulate deployment process
    time.sleep(2)  # Simulate deployment time

    # Simulate successful deployment
    return type(
        "DeploymentResult",
        (),
        {
            "success": True,
            "deployment_id": f"{config.artifact_id}-{int(time.time())}",
            "artifact_id": config.artifact_id,
            "target_environment": config.target_environment,
            "health_check_url": "http://localhost:8501/healthz",
            "deployment_url": "http://localhost:8501",
            "error_message": None,
        },
    )()


def mock_failing_deployment_function(
    config: DeploymentConfig, **kwargs
) -> Any:
    """Mock deployment function that simulates deployment failure."""
    print(
        f"🚀 Attempting to deploy {config.artifact_id} to "
        f"{config.target_environment}"
    )
    print("❌ Simulating deployment failure...")

    # Simulate deployment failure
    time.sleep(1)
    raise RuntimeError("Simulated deployment failure: Service unavailable")


def demonstrate_blue_green_deployment() -> None:
    """Demonstrate blue-green deployment strategy."""
    print("\n🔵🟢 BLUE-GREEN DEPLOYMENT STRATEGY")
    print("=" * 50)
    print(
        "Strategy: Deploy to inactive environment, test, then switch traffic"
    )
    print("Benefits: Zero downtime, easy rollback, safe testing")
    print("=" * 50)

    orchestrator = DeploymentOrchestrator()
    config = create_deployment_config()

    print("📋 Configuration:")
    print(f"  - Artifact: {config.artifact_id}")
    print(f"  - Environment: {config.target_environment}")
    print(f"  - Type: {config.deployment_type}")
    print(f"  - Quantization: {config.enable_quantization}")

    # Execute blue-green deployment
    result = orchestrator.deploy_with_strategy(
        config,
        DeploymentStrategy.BLUE_GREEN,
        mock_deployment_function,
    )

    if result.success:
        print("✅ Blue-green deployment completed successfully!")
        print(f"🔗 Deployment URL: {result.deployment_url}")
        print(f"🏥 Health Check: {result.health_check_url}")

        # Get deployment status
        status = orchestrator.get_deployment_status(result.deployment_id)
        print(f"📊 Deployment Status: {status['state']}")
        print(f"⏱️ Duration: {status['duration']:.2f} seconds")
    else:
        print("❌ Blue-green deployment failed!")
        print(f"🔍 Error: {result.error_message}")

    return result


def demonstrate_canary_deployment() -> None:
    """Demonstrate canary deployment strategy."""
    print("\n🐦 CANARY DEPLOYMENT STRATEGY")
    print("=" * 50)
    print(
        "Strategy: Deploy to small subset, monitor, gradually increase traffic"
    )
    print("Benefits: Risk mitigation, performance monitoring, gradual rollout")
    print("=" * 50)

    orchestrator = DeploymentOrchestrator()
    config = create_deployment_config()

    print("📋 Configuration:")
    print(f"  - Artifact: {config.artifact_id}")
    print(f"  - Environment: {config.target_environment}")
    print("  - Canary percentage: 10% initial traffic")

    # Execute canary deployment
    result = orchestrator.deploy_with_strategy(
        config,
        DeploymentStrategy.CANARY,
        mock_deployment_function,
        canary_percentage=10,
        monitoring_duration=300,  # 5 minutes
    )

    if result.success:
        print("✅ Canary deployment completed successfully!")
        print(f"🔗 Canary URL: {result.deployment_url}")
        print("📈 Monitoring performance metrics...")

        # Simulate performance monitoring
        time.sleep(2)
        print("✅ Performance metrics within acceptable range")
        print("🔄 Gradually increasing traffic to 50%...")
        time.sleep(1)
        print("✅ Traffic increased successfully")
        print("🔄 Gradually increasing traffic to 100%...")
        time.sleep(1)
        print("✅ Full traffic migration completed!")
    else:
        print("❌ Canary deployment failed!")
        print(f"🔍 Error: {result.error_message}")

    return result


def demonstrate_rolling_deployment() -> None:
    """Demonstrate rolling deployment strategy."""
    print("\n🔄 ROLLING DEPLOYMENT STRATEGY")
    print("=" * 50)
    print("Strategy: Update replicas one by one while maintaining service")
    print(
        "Benefits: Continuous availability, controlled updates, easy rollback"
    )
    print("=" * 50)

    orchestrator = DeploymentOrchestrator()
    config = create_deployment_config()

    print("📋 Configuration:")
    print(f"  - Artifact: {config.artifact_id}")
    print(f"  - Environment: {config.target_environment}")
    print("  - Replicas: 3")
    print("  - Update strategy: One replica at a time")

    # Execute rolling deployment
    result = orchestrator.deploy_with_strategy(
        config,
        DeploymentStrategy.ROLLING,
        mock_deployment_function,
        max_unavailable=1,
        max_surge=1,
    )

    if result.success:
        print("✅ Rolling deployment completed successfully!")
        print("📊 Replica update sequence:")
        print("  1. ✅ Replica 1 updated and healthy")
        print("  2. ✅ Replica 2 updated and healthy")
        print("  3. ✅ Replica 3 updated and healthy")
        print("🎉 All replicas successfully updated!")
    else:
        print("❌ Rolling deployment failed!")
        print(f"🔍 Error: {result.error_message}")

    return result


def demonstrate_recreate_deployment() -> None:
    """Demonstrate recreate deployment strategy."""
    print("\n🔄 RECREATE DEPLOYMENT STRATEGY")
    print("=" * 50)
    print("Strategy: Terminate old deployment, create new deployment")
    print("Benefits: Simple, clean state, suitable for development")
    print("⚠️ Warning: Brief downtime during recreation")
    print("=" * 50)

    orchestrator = DeploymentOrchestrator()
    config = create_deployment_config()

    print("📋 Configuration:")
    print(f"  - Artifact: {config.artifact_id}")
    print(f"  - Environment: {config.target_environment}")
    print("  - Strategy: Terminate and recreate")

    # Execute recreate deployment
    result = orchestrator.deploy_with_strategy(
        config,
        DeploymentStrategy.RECREATE,
        mock_deployment_function,
    )

    if result.success:
        print("✅ Recreate deployment completed successfully!")
        print("📊 Deployment process:")
        print("  1. ✅ Terminated old deployment")
        print("  2. ✅ Created new deployment")
        print("  3. ✅ Health check passed")
        print("🎉 Deployment recreation completed!")
    else:
        print("❌ Recreate deployment failed!")
        print(f"🔍 Error: {result.error_message}")

    return result


def demonstrate_automatic_rollback() -> None:
    """Demonstrate automatic rollback mechanism."""
    print("\n🔄 AUTOMATIC ROLLBACK MECHANISM")
    print("=" * 50)
    print("Scenario: Deployment fails, automatic rollback to previous version")
    print("Benefits: Self-healing, minimal downtime, automatic recovery")
    print("=" * 50)

    orchestrator = DeploymentOrchestrator()
    config = create_deployment_config()

    print("📋 Configuration:")
    print(f"  - Artifact: {config.artifact_id}")
    print(f"  - Environment: {config.target_environment}")
    print("  - Rollback: Automatic on failure")

    # Simulate previous deployment for rollback
    from crackseg.utils.deployment.orchestration import DeploymentMetadata

    previous_metadata = DeploymentMetadata(
        deployment_id="previous-deployment-001",
        artifact_id="crackseg-model-v2.0",
        strategy=DeploymentStrategy.BLUE_GREEN,
        state=DeploymentState.SUCCESS,
        start_time=time.time() - 3600,  # 1 hour ago
        end_time=time.time() - 1800,  # 30 minutes ago
        health_check_url="http://localhost:8501/healthz",
    )

    # Mock previous deployment in history
    orchestrator.deployment_history["previous-deployment-001"] = (
        previous_metadata
    )

    # Execute deployment with failure
    result = orchestrator.deploy_with_strategy(
        config,
        DeploymentStrategy.BLUE_GREEN,
        mock_failing_deployment_function,
    )

    if not result.success:
        print("❌ Deployment failed as expected")
        print(f"🔍 Error: {result.error_message}")

        # Check deployment status
        status = orchestrator.get_deployment_status(result.deployment_id)
        print(f"📊 Final State: {status['state']}")

        if status.get("rollback_reason"):
            print(f"🔄 Rollback Reason: {status['rollback_reason']}")

        if status["state"] == DeploymentState.ROLLED_BACK.value:
            print("✅ Automatic rollback completed successfully!")
            print("🔄 Traffic switched back to previous deployment")
            print("🏥 Previous deployment health verified")
        else:
            print("⚠️ Rollback was attempted but may not have completed")

    return result


def demonstrate_manual_rollback() -> None:
    """Demonstrate manual rollback mechanism."""
    print("\n🔧 MANUAL ROLLBACK MECHANISM")
    print("=" * 50)
    print("Scenario: Manual rollback of a successful deployment")
    print(
        "Use case: Performance issues, security concerns, business decisions"
    )
    print("=" * 50)

    orchestrator = DeploymentOrchestrator()
    config = create_deployment_config()

    # First, create a successful deployment
    print("🚀 Creating initial deployment...")
    result = orchestrator.deploy_with_strategy(
        config,
        DeploymentStrategy.BLUE_GREEN,
        mock_deployment_function,
    )

    if result.success:
        print("✅ Initial deployment completed")
        deployment_id = result.deployment_id

        # Simulate decision to rollback
        print("\n📊 Monitoring deployment performance...")
        time.sleep(1)
        print("⚠️ Detected performance degradation")
        print("📉 Response times increased by 40%")
        print("🔧 Initiating manual rollback...")

        # Execute manual rollback
        rollback_success = orchestrator.manual_rollback(deployment_id)

        if rollback_success:
            print("✅ Manual rollback completed successfully!")
            print("🔄 Traffic switched to previous deployment")
            print("📊 Performance metrics restored")
        else:
            print("❌ Manual rollback failed!")
            print("🔍 Check deployment history for details")
    else:
        print("❌ Initial deployment failed, cannot demonstrate rollback")

    return result


def demonstrate_deployment_history() -> None:
    """Demonstrate deployment history tracking."""
    print("\n📚 DEPLOYMENT HISTORY TRACKING")
    print("=" * 50)
    print("Feature: Complete audit trail of all deployments")
    print("Benefits: Compliance, debugging, performance analysis")
    print("=" * 50)

    orchestrator = DeploymentOrchestrator()
    config = create_deployment_config()

    # Create multiple deployments to build history
    deployments = []

    print("🚀 Creating multiple deployments...")

    # Deployment 1: Blue-green
    result1 = orchestrator.deploy_with_strategy(
        config,
        DeploymentStrategy.BLUE_GREEN,
        mock_deployment_function,
    )
    deployments.append(result1)

    # Deployment 2: Canary
    result2 = orchestrator.deploy_with_strategy(
        config,
        DeploymentStrategy.CANARY,
        mock_deployment_function,
    )
    deployments.append(result2)

    # Deployment 3: Rolling
    result3 = orchestrator.deploy_with_strategy(
        config,
        DeploymentStrategy.ROLLING,
        mock_deployment_function,
    )
    deployments.append(result3)

    # Display deployment history
    print("\n📊 Deployment History:")
    print("-" * 80)
    print(
        f"{'ID':<25} {'Strategy':<12} {'State':<12} {'Duration':<10} "
        f"{'Artifact':<20}"
    )
    print("-" * 80)

    history = orchestrator.get_deployment_history()
    for deployment in history:
        print(
            f"{deployment['deployment_id']:<25} "
            f"{deployment['strategy']:<12} "
            f"{deployment['state']:<12} "
            f"{deployment['duration']:<10.2f}s "
            f"{deployment['artifact_id']:<20}"
        )

    print("-" * 80)
    print(f"Total deployments: {len(history)}")

    # Show detailed status for latest deployment
    if history:
        latest = history[-1]
        print("\n📋 Latest Deployment Details:")
        print(f"  - ID: {latest['deployment_id']}")
        print(f"  - Strategy: {latest['strategy']}")
        print(f"  - State: {latest['state']}")
        print(f"  - Duration: {latest['duration']:.2f} seconds")
        print(f"  - Artifact: {latest['artifact_id']}")

        if latest.get("previous_deployment_id"):
            print(f"  - Previous: {latest['previous_deployment_id']}")

        if latest.get("rollback_reason"):
            print(f"  - Rollback: {latest['rollback_reason']}")


def demonstrate_health_checks() -> None:
    """Demonstrate health check capabilities."""
    print("\n🔍 HEALTH CHECK DEMO")
    print("=" * 50)

    print("🔍 Health Check Configuration:")
    print("  - Endpoint: /health")
    print("  - Timeout: 30 seconds")
    print("  - Interval: 10 seconds")
    print("  - Retries: 3")
    print("  - Success Threshold: 2")
    print("  - Failure Threshold: 3")

    print("\n✅ Health check configuration validated")
    print("🔄 Health monitoring will be active during deployment")


def demonstrate_strategy_comparison() -> None:
    """Demonstrate deployment strategy comparison."""
    print("\n📊 DEPLOYMENT STRATEGY COMPARISON")
    print("=" * 80)

    strategies = [
        ("Blue-Green", "Zero", "Low", "Medium", "Production releases"),
        ("Canary", "Minimal", "Medium", "High", "Gradual rollouts"),
        ("Rolling", "Minimal", "Low", "Medium", "Kubernetes deployments"),
        ("Recreate", "High", "High", "Low", "Development/testing"),
    ]

    print(
        f"{'Strategy':<15} {'Downtime':<12} {'Risk':<10} {'Complexity':<12} "
        f"{'Use Case':<20}"
    )
    print("-" * 80)

    for strategy, downtime, risk, complexity, use_case in strategies:
        print(
            f"{strategy:<15} {downtime:<12} {risk:<10} {complexity:<12} "
            f"{use_case:<20}"
        )

    print("\n🎯 Strategy Selection Guidelines:")
    print("  - Blue-Green: Zero downtime, high resource usage")
    print("  - Canary: Gradual rollout, risk mitigation")
    print("  - Rolling: Kubernetes native, minimal downtime")
    print("  - Recreate: Simple, high downtime")


def main() -> None:
    """Main function to demonstrate deployment orchestration."""
    print("🚀 CRACKSEG DEPLOYMENT ORCHESTRATION DEMO")
    print("=" * 60)
    print("This demo showcases advanced deployment orchestration")
    print("with multiple strategies and rollback capabilities.")
    print("=" * 60)

    try:
        # Demonstrate core orchestration features
        demonstrate_blue_green_deployment()
        demonstrate_canary_deployment()
        demonstrate_rolling_deployment()
        demonstrate_recreate_deployment()

        # Demonstrate rollback mechanisms
        demonstrate_automatic_rollback()
        demonstrate_manual_rollback()

        # Demonstrate supporting features
        demonstrate_deployment_history()
        demonstrate_health_checks()
        demonstrate_strategy_comparison()

        print("\n✅ Demo completed successfully!")
        print("\n🎯 Key Takeaways:")
        print(
            "- Deployment orchestration provides multiple strategies for "
            "different needs"
        )
        print("- Rollback mechanisms ensure deployment safety")
        print("- Health checks prevent failed deployments")
        print("- Performance monitoring tracks deployment success")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
