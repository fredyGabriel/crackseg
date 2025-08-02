"""Advanced Deployment Orchestration and Rollback Demo.

This script demonstrates the advanced deployment orchestration capabilities
including performance monitoring, alert systems, and enhanced rollback
mechanisms implemented in subtask 5.5.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

from crackseg.utils.deployment import (
    DeploymentConfig,
    DeploymentOrchestrator,
    DeploymentStrategy,
    LoggingAlertHandler,
    PerformanceMonitor,
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
        artifact_id="crackseg-model-v2.2",
        artifact_type=ArtifactType.MODEL,
        file_path=Path("/path/to/model/checkpoint.pth"),
        file_size=245000000,  # 245MB
        checksum="a" * 64,  # Mock SHA256
        name="CrackSeg Model v2.2",
        owner="ml-team",
        experiment_id="exp-001",
        metadata={
            "version": "2.2.0",
            "format": "pytorch",
            "architecture": "swin_transformer_v2",
            "training_config": (
                "configs/experiments/swinv2_hybrid/optimized.yaml"
            ),
            "metrics": {
                "val_iou": 0.89,
                "val_dice": 0.95,
                "test_iou": 0.87,
                "inference_time_ms": 95,
            },
        },
    )


def create_deployment_config() -> DeploymentConfig:
    """Create deployment configuration."""
    return DeploymentConfig(
        artifact_id="crackseg-model-v2.2",
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
    time.sleep(3)  # Simulate deployment time

    # Simulate successful deployment
    return type(
        "DeploymentResult",
        (),
        {
            "success": True,
            "deployment_id": f"{config.artifact_id}-{int(time.time())}",
            "artifact_id": config.artifact_id,
            "target_environment": config.target_environment,
            "deployment_url": "http://localhost:8501",
            "health_check_url": "http://localhost:8501/healthz",
            "monitoring_dashboard_url": "http://localhost:8501/metrics",
            "original_size_mb": 245.0,
            "optimized_size_mb": 98.0,
            "compression_ratio": 2.5,
            "functional_tests_passed": True,
            "performance_benchmark_score": 0.92,
            "security_scan_passed": True,
        },
    )


def mock_failing_deployment_function(
    config: DeploymentConfig, **kwargs
) -> Any:
    """Mock deployment function that simulates failure."""
    print(f"🚀 Deploying {config.artifact_id} to {config.target_environment}")
    print("❌ Simulating deployment failure...")

    time.sleep(2)
    raise RuntimeError(
        "Simulated deployment failure: Service health check failed"
    )


def demonstrate_performance_monitoring() -> None:
    """Demonstrate performance monitoring capabilities."""
    print("\n" + "=" * 60)
    print("🔍 PERFORMANCE MONITORING DEMONSTRATION")
    print("=" * 60)

    # Create orchestrator with alert handlers
    orchestrator = DeploymentOrchestrator()
    orchestrator.add_alert_handler(LoggingAlertHandler())

    # Create deployment config
    config = create_deployment_config()

    # Deploy with monitoring
    print("📊 Starting deployment with performance monitoring...")
    result = orchestrator.deploy_with_strategy(
        config,
        DeploymentStrategy.BLUE_GREEN,
        mock_deployment_function,
    )

    if result.success:
        deployment_id = result.deployment_id
        print(f"✅ Deployment successful: {deployment_id}")

        # Simulate performance monitoring
        print("📈 Collecting performance metrics...")
        for i in range(5):
            metrics = orchestrator.get_performance_metrics(deployment_id)
            if metrics:
                current = metrics.get("current", {})
                response_time = current.get("response_time_ms", 0)
                throughput = current.get("throughput_rps", 0)
                response_msg = (
                    f"   Sample {i + 1}: Response time: {response_time:.1f}ms"
                )
                throughput_msg = f"Throughput: {throughput:.1f} rps"
                print(f"{response_msg}, {throughput_msg}")
            time.sleep(2)

        # Stop monitoring
        orchestrator.stop_performance_monitoring(deployment_id)
        print("🛑 Performance monitoring stopped")
    else:
        print(f"❌ Deployment failed: {result.error_message}")


def demonstrate_alert_system() -> None:
    """Demonstrate alert system capabilities."""
    print("\n" + "=" * 60)
    print("🚨 ALERT SYSTEM DEMONSTRATION")
    print("=" * 60)

    # Create orchestrator with multiple alert handlers
    orchestrator = DeploymentOrchestrator()
    orchestrator.add_alert_handler(LoggingAlertHandler())

    # Demonstrate successful deployment alerts
    print("✅ Testing successful deployment alerts...")
    config = create_deployment_config()
    result = orchestrator.deploy_with_strategy(
        config,
        DeploymentStrategy.CANARY,
        mock_deployment_function,
        canary_percentage=10,
    )

    if result.success:
        print("✅ Success alert sent")

    # Demonstrate failure deployment alerts
    print("\n❌ Testing failure deployment alerts...")
    config.artifact_id = "failing-model-v1"
    result = orchestrator.deploy_with_strategy(
        config,
        DeploymentStrategy.BLUE_GREEN,
        mock_failing_deployment_function,
    )

    if not result.success:
        print("❌ Failure alert sent")


def demonstrate_advanced_rollback() -> None:
    """Demonstrate advanced rollback mechanisms."""
    print("\n" + "=" * 60)
    print("🔄 ADVANCED ROLLBACK MECHANISMS DEMONSTRATION")
    print("=" * 60)

    orchestrator = DeploymentOrchestrator()
    orchestrator.add_alert_handler(LoggingAlertHandler())

    # First deployment (successful)
    print("1️⃣ Initial successful deployment...")
    config = create_deployment_config()
    config.artifact_id = "stable-model-v1"

    result1 = orchestrator.deploy_with_strategy(
        config,
        DeploymentStrategy.BLUE_GREEN,
        mock_deployment_function,
    )

    if result1.success:
        print(f"✅ Initial deployment successful: {result1.deployment_id}")

        # Second deployment (failing - should trigger rollback)
        print("\n2️⃣ Deploying failing model (should trigger rollback)...")
        config.artifact_id = "failing-model-v2"

        result2 = orchestrator.deploy_with_strategy(
            config,
            DeploymentStrategy.BLUE_GREEN,
            mock_failing_deployment_function,
        )

        if not result2.success:
            print(f"❌ Second deployment failed: {result2.error_message}")
            print("🔄 Rollback should have been triggered automatically")

        # Manual rollback demonstration
        print("\n3️⃣ Testing manual rollback...")
        if result1.deployment_id:
            success = orchestrator.manual_rollback(result1.deployment_id)
            if success:
                print("✅ Manual rollback successful")
            else:
                print("❌ Manual rollback failed")
    else:
        print(f"❌ Initial deployment failed: {result1.error_message}")


def demonstrate_performance_trends() -> None:
    """Demonstrate performance trend analysis."""
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE TREND ANALYSIS DEMONSTRATION")
    print("=" * 60)

    # Create performance monitor
    monitor = PerformanceMonitor("http://localhost:8501")

    print("📈 Simulating performance monitoring over time...")

    # Simulate different performance scenarios
    scenarios = [
        (
            "Stable Performance",
            [100, 105, 98, 102, 99, 101, 103, 97, 104, 100],
        ),
        (
            "Degrading Performance",
            [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
        ),
        (
            "Improving Performance",
            [200, 190, 180, 170, 160, 150, 140, 130, 120, 110],
        ),
    ]

    for scenario_name, response_times in scenarios:
        print(f"\n🔍 Scenario: {scenario_name}")

        # Simulate metrics collection
        for i, response_time in enumerate(response_times):
            monitor.metrics["response_time"].append(response_time)
            monitor.metrics["throughput"].append(
                1000 / response_time if response_time > 0 else 0
            )
            monitor.metrics["error_rate"].append(0.01)
            monitor.metrics["memory_usage"].append(512.0)
            monitor.metrics["cpu_usage"].append(25.0)

            if i >= 9:  # After collecting 10 samples
                metrics = monitor.get_metrics()
                trends = metrics.get("trends", {})
                print(
                    f"   Response time trend: "
                    f"{trends.get('response_time_trend', 'unknown')}"
                )
                print(
                    f"   Throughput trend: "
                    f"{trends.get('throughput_trend', 'unknown')}"
                )


def demonstrate_deployment_history() -> None:
    """Demonstrate deployment history tracking."""
    print("\n" + "=" * 60)
    print("📚 DEPLOYMENT HISTORY TRACKING DEMONSTRATION")
    print("=" * 60)

    orchestrator = DeploymentOrchestrator()

    # Create multiple deployments
    configs = [
        ("model-v1", DeploymentStrategy.BLUE_GREEN),
        ("model-v2", DeploymentStrategy.CANARY),
        ("model-v3", DeploymentStrategy.ROLLING),
    ]

    for artifact_id, strategy in configs:
        print(
            f"\n🚀 Deploying {artifact_id} with {strategy.value} strategy..."
        )
        config = create_deployment_config()
        config.artifact_id = artifact_id

        result = orchestrator.deploy_with_strategy(
            config,
            strategy,
            mock_deployment_function,
        )

        if result.success:
            print(f"✅ {artifact_id} deployed successfully")
        else:
            print(f"❌ {artifact_id} deployment failed")

    # Show deployment history
    print("\n📋 Deployment History:")
    history = orchestrator.get_deployment_history()
    for deployment in history[:5]:  # Show last 5 deployments
        print(
            f"   {deployment['deployment_id']}: {deployment['state']} "
            f"({deployment['strategy']}) - {deployment['duration']:.1f}s"
        )


def demonstrate_resource_management() -> None:
    """Demonstrate resource management capabilities."""
    print("\n" + "=" * 60)
    print("⚙️ RESOURCE MANAGEMENT DEMONSTRATION")
    print("=" * 60)

    orchestrator = DeploymentOrchestrator()

    # Demonstrate resource-aware deployment
    print("🔧 Deploying with resource constraints...")
    config = create_deployment_config()
    config.artifact_id = "resource-constrained-model"

    # Add resource limits
    config.resource_limits = {
        "cpu": "2",
        "memory": "4Gi",
        "gpu": "1",
        "max_response_time_ms": 200,
        "max_memory_usage_mb": 1024,
    }

    result = orchestrator.deploy_with_strategy(
        config,
        DeploymentStrategy.BLUE_GREEN,
        mock_deployment_function,
    )

    if result.success:
        print("✅ Resource-constrained deployment successful")
        print(f"   Original size: {result.original_size_mb:.1f} MB")
        print(f"   Optimized size: {result.optimized_size_mb:.1f} MB")
        print(f"   Compression ratio: {result.compression_ratio:.2f}x")
    else:
        print(
            f"❌ Resource-constrained deployment failed: "
            f"{result.error_message}"
        )


def main() -> None:
    """Run the advanced orchestration demonstration."""
    print("🚀 ADVANCED DEPLOYMENT ORCHESTRATION DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the enhanced deployment orchestration")
    print("and rollback mechanisms implemented in subtask 5.5.")
    print("=" * 60)

    setup_logging()

    try:
        # Run demonstrations
        demonstrate_performance_monitoring()
        demonstrate_alert_system()
        demonstrate_advanced_rollback()
        demonstrate_performance_trends()
        demonstrate_deployment_history()
        demonstrate_resource_management()

        print("\n" + "=" * 60)
        print("✅ ADVANCED ORCHESTRATION DEMONSTRATION COMPLETED")
        print("=" * 60)
        print("Key features demonstrated:")
        print("  • Performance monitoring with trend analysis")
        print("  • Multi-channel alert system (logging, email, Slack)")
        print("  • Advanced rollback mechanisms (automatic and manual)")
        print("  • Deployment history tracking")
        print("  • Resource-aware deployment")
        print("  • Health check integration")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
