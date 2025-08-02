#!/usr/bin/env python3
"""Health monitoring demonstration script.

This script demonstrates the health monitoring capabilities integrated
with the deployment orchestration system.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from src.crackseg.utils.deployment.config import DeploymentResult  # noqa: E402
from src.crackseg.utils.deployment.health_monitoring import (  # noqa: E402
    DefaultHealthChecker,
    DefaultResourceMonitor,
    DeploymentHealthMonitor,
)
from src.crackseg.utils.deployment.orchestration import (  # noqa: E402
    DeploymentConfig,
    DeploymentOrchestrator,
    LoggingAlertHandler,
)


def create_mock_deployment_result() -> DeploymentResult:
    """Create a mock deployment result for demonstration."""
    return DeploymentResult(
        success=True,
        deployment_id="demo-deployment-123",
        artifact_id="crackseg-model-v1",
        target_environment="production",
        deployment_url="http://localhost:8080",
        health_check_url="http://localhost:8080/health",
    )


def create_mock_deployment_config() -> DeploymentConfig:
    """Create a mock deployment configuration."""
    from src.crackseg.utils.deployment.config import DeploymentConfig

    return DeploymentConfig(
        artifact_id="crackseg-model-v1",
        target_environment="production",
        enable_health_checks=True,
        enable_metrics_collection=True,
    )


async def demo_health_monitoring() -> None:
    """Demonstrate health monitoring capabilities."""
    print("🏥 Health Monitoring Demonstration")
    print("=" * 50)

    # Initialize orchestrator with health monitoring
    orchestrator = DeploymentOrchestrator()

    # Add alert handler
    alert_handler = LoggingAlertHandler()
    orchestrator.add_alert_handler(alert_handler)

    # Create mock deployment result
    result = create_mock_deployment_result()
    config = create_mock_deployment_config()

    print(f"✅ Created deployment result: {result.deployment_id}")
    print(f"🔗 Health check URL: {result.health_check_url}")
    print(f"📊 Metrics URL: {result.metrics_url}")

    # Add deployment to monitoring
    if result.health_check_url:
        orchestrator.add_deployment_monitoring(
            deployment_id=result.deployment_id,
            health_check_url=result.health_check_url,
            process_name=config.artifact_id,
            check_interval=10,  # Check every 10 seconds for demo
        )

    print(f"📡 Added deployment {result.deployment_id} to monitoring")

    # Start health monitoring
    orchestrator.start_health_monitoring()
    print("🚀 Started health monitoring")

    # Simulate monitoring for 30 seconds
    print("⏳ Monitoring for 30 seconds...")
    for i in range(6):  # 6 iterations * 5 seconds = 30 seconds
        await asyncio.sleep(5)

        # Get health status
        health_status = orchestrator.get_deployment_health_status(
            result.deployment_id
        )
        print(
            f"📊 Health status (iteration {i + 1}): "
            f"{health_status.get('health_status', 'unknown')}"
        )

        if health_status.get("resource_metrics"):
            metrics = health_status["resource_metrics"]
            print(f"   CPU: {metrics.get('cpu_usage_percent', 0):.1f}%")
            print(f"   Memory: {metrics.get('memory_usage_mb', 0):.1f}MB")

    # Get all health statuses
    all_statuses = orchestrator.get_all_health_statuses()
    print(f"\n📋 All monitored deployments: {len(all_statuses)}")

    # Stop monitoring
    orchestrator.stop_health_monitoring()
    print("🛑 Stopped health monitoring")

    # Remove deployment from monitoring
    orchestrator.remove_deployment_monitoring(result.deployment_id)
    print(f"🗑️  Removed deployment {result.deployment_id} from monitoring")


async def demo_standalone_health_monitor() -> None:
    """Demonstrate standalone health monitor capabilities."""
    print("\n🔧 Standalone Health Monitor Demonstration")
    print("=" * 50)

    # Create standalone health monitor
    health_monitor = DeploymentHealthMonitor()

    # Add multiple deployments to monitoring
    deployments = [
        {
            "id": "web-service-1",
            "url": "http://localhost:8080/health",
            "process": "web-service",
        },
        {
            "id": "api-service-1",
            "url": "http://localhost:8081/health",
            "process": "api-service",
        },
        {
            "id": "model-service-1",
            "url": "http://localhost:8082/health",
            "process": "model-service",
        },
    ]

    for deployment in deployments:
        health_monitor.add_deployment_monitoring(
            deployment_id=deployment["id"],
            health_check_url=deployment["url"],
            process_name=deployment["process"],
            check_interval=15,
        )
        print(f"➕ Added {deployment['id']} to monitoring")

    # Start monitoring
    health_monitor.start_monitoring()
    print("🚀 Started standalone health monitoring")

    # Monitor for 20 seconds
    print("⏳ Monitoring for 20 seconds...")
    for i in range(4):  # 4 iterations * 5 seconds = 20 seconds
        await asyncio.sleep(5)

        # Get all statuses
        all_statuses = health_monitor.get_all_deployment_statuses()
        print(
            f"📊 Status update {i + 1}: {len(all_statuses)} deployments "
            "monitored"
        )

        for deployment_id, status in all_statuses.items():
            health_status = status.get("health_status", "unknown")
            print(f"   {deployment_id}: {health_status}")

    # Stop monitoring
    health_monitor.stop_monitoring()
    print("🛑 Stopped standalone health monitoring")

    # Export monitoring data
    export_path = Path("outputs/monitoring_data.json")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    health_monitor.export_monitoring_data(export_path)
    print(f"💾 Exported monitoring data to {export_path}")


async def demo_resource_monitoring() -> None:
    """Demonstrate resource monitoring capabilities."""
    print("\n💻 Resource Monitoring Demonstration")
    print("=" * 50)

    # Create resource monitor
    resource_monitor = DefaultResourceMonitor()

    # Get system metrics
    system_metrics = resource_monitor.get_system_metrics()
    print("📊 System Resource Metrics:")
    print(f"   CPU Usage: {system_metrics.cpu_usage_percent:.1f}%")
    print(f"   Memory Usage: {system_metrics.memory_usage_mb:.1f}MB")
    print(f"   Disk Usage: {system_metrics.disk_usage_percent:.1f}%")
    print(f"   Network IO: {system_metrics.network_io_mbps:.2f}Mbps")

    # Get process metrics (if available)
    process_metrics = resource_monitor.get_process_metrics("python")
    print("\n🐍 Python Process Metrics:")
    print(f"   CPU Usage: {process_metrics.cpu_usage_percent:.1f}%")
    print(f"   Memory Usage: {process_metrics.memory_usage_mb:.1f}MB")


async def demo_health_checker() -> None:
    """Demonstrate health checker capabilities."""
    print("\n🔍 Health Checker Demonstration")
    print("=" * 50)

    # Create health checker
    health_checker = DefaultHealthChecker()

    # Test URLs (these will fail in demo, but show the interface)
    test_urls = [
        "http://localhost:8080/health",
        "http://localhost:8081/health",
        "http://localhost:8082/health",
    ]

    for url in test_urls:
        print(f"🔍 Checking health of {url}")
        result = health_checker.check_health(url, timeout=5)

        if result.success:
            print(
                "   ✅ Healthy - Response time: "
                f"{result.response_time_ms:.1f}ms"
            )
        else:
            print(f"   ❌ Unhealthy - Error: {result.error_message}")
            print(f"   ⏱️  Response time: {result.response_time_ms:.1f}ms")


async def main() -> None:
    """Run all health monitoring demonstrations."""
    print("🚀 CrackSeg Health Monitoring System Demo")
    print("=" * 60)

    try:
        # Demo 1: Integrated health monitoring with orchestrator
        await demo_health_monitoring()

        # Demo 2: Standalone health monitor
        await demo_standalone_health_monitor()

        # Demo 3: Resource monitoring
        await demo_resource_monitoring()

        # Demo 4: Health checker
        await demo_health_checker()

        print("\n✅ All demonstrations completed successfully!")
        print("\n📚 Key Features Demonstrated:")
        print(
            "   • Integrated health monitoring with deployment orchestration"
        )
        print("   • Standalone health monitoring system")
        print("   • Resource monitoring (CPU, memory, disk, network)")
        print("   • HTTP-based health checking")
        print("   • Alert system integration")
        print("   • Monitoring data export")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        logging.exception("Demonstration failed")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run demonstrations
    asyncio.run(main())
