#!/usr/bin/env python3
"""Multi-target deployment demonstration script.

This script demonstrates the multi-target deployment capabilities,
showing how to deploy to different environments with specific
configurations and validations.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from src.crackseg.utils.deployment.config import (  # noqa: E402
    DeploymentConfig,
    DeploymentResult,
)
from src.crackseg.utils.deployment.multi_target import (  # noqa: E402
    MultiTargetDeploymentManager,
    TargetEnvironment,
)


def create_mock_deployment_config() -> DeploymentConfig:
    """Create a mock deployment configuration."""
    return DeploymentConfig(
        artifact_id="crackseg-model-v1",
        target_environment="production",
        enable_health_checks=True,
        enable_metrics_collection=True,
    )


def mock_deployment_func(
    config: DeploymentConfig, **kwargs
) -> DeploymentResult:
    """Mock deployment function for demonstration."""
    # Simulate deployment process
    time.sleep(1)  # Simulate deployment time

    # Simulate different success rates based on environment
    target_env = config.target_environment
    if target_env == "production":
        success_rate = 0.95  # 95% success rate for production
    elif target_env == "staging":
        success_rate = 0.90  # 90% success rate for staging
    else:
        success_rate = 0.85  # 85% success rate for other environments

    import random

    success = random.random() < success_rate

    if success:
        return DeploymentResult(
            success=True,
            deployment_id=f"deploy-{target_env}-{int(time.time())}",
            artifact_id=config.artifact_id,
            target_environment=target_env,
            deployment_url=f"http://{target_env}.crackseg.com",
            health_check_url=f"http://{target_env}.crackseg.com/health",
        )
    else:
        return DeploymentResult(
            success=False,
            deployment_id=f"failed-{target_env}-{int(time.time())}",
            artifact_id=config.artifact_id,
            target_environment=target_env,
            error_message=f"Mock deployment failed for {target_env}",
        )


async def demo_environment_validation() -> None:
    """Demonstrate environment validation capabilities."""
    print("🔍 Environment Validation Demonstration")
    print("=" * 50)

    manager = MultiTargetDeploymentManager()

    # Validate each environment
    for environment in TargetEnvironment:
        print(f"\n📋 Validating {environment.value} environment...")
        validation = manager.validate_environment_readiness(environment)

        status_text = "✅ READY" if validation["ready"] else "❌ NOT READY"
        print(f"   Status: {status_text}")

        if validation["issues"]:
            print("   Issues:")
            for issue in validation["issues"]:
                print(f"     ❌ {issue}")

        if validation["warnings"]:
            print("   Warnings:")
            for warning in validation["warnings"]:
                print(f"     ⚠️  {warning}")


async def demo_single_environment_deployment() -> None:
    """Demonstrate deployment to a single environment."""
    print("\n🚀 Single Environment Deployment Demonstration")
    print("=" * 50)

    manager = MultiTargetDeploymentManager()
    config = create_mock_deployment_config()

    # Deploy to staging environment
    environment = TargetEnvironment.STAGING
    print(f"📦 Deploying to {environment.value}...")

    result = manager.deploy_to_environment(
        config=config,
        environment=environment,
        deployment_func=mock_deployment_func,
    )

    if result.success:
        print("   ✅ Deployment successful!")
        print(f"   🆔 Deployment ID: {result.deployment_id}")
        print(f"   🔗 URL: {result.deployment_url}")
        print(f"   🏥 Health Check: {result.health_check_url}")
    else:
        print(f"   ❌ Deployment failed: {result.error_message}")


async def demo_multi_environment_deployment() -> None:
    """Demonstrate deployment to multiple environments."""
    print("\n🌍 Multi-Environment Deployment Demonstration")
    print("=" * 50)

    manager = MultiTargetDeploymentManager()
    config = create_mock_deployment_config()

    # Deploy to multiple environments
    environments = [
        TargetEnvironment.DEVELOPMENT,
        TargetEnvironment.STAGING,
        TargetEnvironment.PRODUCTION,
    ]

    print(f"📦 Deploying to {len(environments)} environments...")
    print("   Environments:", ", ".join([env.value for env in environments]))

    results = manager.deploy_to_multiple_environments(
        config=config,
        environments=environments,
        deployment_func=mock_deployment_func,
    )

    # Display results
    print("\n📊 Deployment Results:")
    for environment, result in results.items():
        if result.success:
            print(f"   ✅ {environment.value}: SUCCESS")
            print(f"      🆔 {result.deployment_id}")
            print(f"      🔗 {result.deployment_url}")
        else:
            print(f"   ❌ {environment.value}: FAILED")
            print(f"      💥 {result.error_message}")


async def demo_environment_configurations() -> None:
    """Demonstrate environment-specific configurations."""
    print("\n⚙️  Environment Configurations Demonstration")
    print("=" * 50)

    manager = MultiTargetDeploymentManager()

    for environment in TargetEnvironment:
        config = manager.get_environment_config(environment)
        print(f"\n📋 {environment.value.upper()} Configuration:")
        print(f"   Strategy: {config.deployment_strategy.value}")
        print(f"   Health Check Timeout: {config.health_check_timeout}s")
        print(f"   Max Retries: {config.max_retries}")
        print(f"   Auto Rollback: {'✅' if config.auto_rollback else '❌'}")

        if config.performance_thresholds:
            print("   Performance Thresholds:")
            for metric, threshold in config.performance_thresholds.items():
                print(f"     {metric}: {threshold}")

        if config.resource_limits:
            print("   Resource Limits:")
            for resource, limit in config.resource_limits.items():
                print(f"     {resource}: {limit}")


async def demo_deployment_status_tracking() -> None:
    """Demonstrate deployment status tracking across environments."""
    print("\n📊 Deployment Status Tracking Demonstration")
    print("=" * 50)

    manager = MultiTargetDeploymentManager()
    config = create_mock_deployment_config()

    # Deploy to multiple environments first
    environments = [TargetEnvironment.DEVELOPMENT, TargetEnvironment.STAGING]
    results = manager.deploy_to_multiple_environments(
        config=config,
        environments=environments,
        deployment_func=mock_deployment_func,
    )

    # Track status across environments
    print("📈 Tracking deployment status across environments...")

    for environment, result in results.items():
        if result.success:
            print(f"\n🔍 Status for {environment.value}:")
            status = manager.get_deployment_status_across_environments(
                result.deployment_id
            )

            for env, env_status in status.items():
                if "error" in env_status:
                    print(f"   {env.value}: ❌ Error - {env_status['error']}")
                else:
                    print(f"   {env.value}: ✅ Available")


async def demo_rollback_capabilities() -> None:
    """Demonstrate rollback capabilities across environments."""
    print("\n🔄 Rollback Capabilities Demonstration")
    print("=" * 50)

    manager = MultiTargetDeploymentManager()
    config = create_mock_deployment_config()

    # Deploy to multiple environments
    environments = [TargetEnvironment.DEVELOPMENT, TargetEnvironment.STAGING]
    results = manager.deploy_to_multiple_environments(
        config=config,
        environments=environments,
        deployment_func=mock_deployment_func,
    )

    # Simulate rollback
    print("🔄 Performing rollback across environments...")

    for environment, result in results.items():
        if result.success:
            print(f"\n🔄 Rolling back {environment.value} deployment...")
            rollback_results = manager.rollback_across_environments(
                result.deployment_id, [environment]
            )

            for env, success in rollback_results.items():
                if success:
                    print(f"   ✅ Rollback successful for {env.value}")
                else:
                    print(f"   ❌ Rollback failed for {env.value}")


async def demo_configuration_export() -> None:
    """Demonstrate configuration export capabilities."""
    print("\n💾 Configuration Export Demonstration")
    print("=" * 50)

    manager = MultiTargetDeploymentManager()

    # Export configurations
    export_path = Path("outputs/environment_configs.json")
    manager.export_environment_configs(export_path)

    print(f"📁 Exported environment configurations to: {export_path}")
    print("   This file contains all environment-specific settings")
    print(
        "   including deployment strategies, resource limits, and security "
        "requirements"
    )


async def main() -> None:
    """Run all multi-target deployment demonstrations."""
    print("🚀 CrackSeg Multi-Target Deployment System Demo")
    print("=" * 60)

    try:
        # Demo 1: Environment validation
        await demo_environment_validation()

        # Demo 2: Single environment deployment
        await demo_single_environment_deployment()

        # Demo 3: Multi-environment deployment
        await demo_multi_environment_deployment()

        # Demo 4: Environment configurations
        await demo_environment_configurations()

        # Demo 5: Deployment status tracking
        await demo_deployment_status_tracking()

        # Demo 6: Rollback capabilities
        await demo_rollback_capabilities()

        # Demo 7: Configuration export
        await demo_configuration_export()

        print("\n✅ All demonstrations completed successfully!")
        print("\n📚 Key Features Demonstrated:")
        print("   • Environment-specific configurations")
        print("   • Multi-target deployment orchestration")
        print("   • Environment validation and readiness checks")
        print("   • Deployment status tracking across environments")
        print("   • Rollback capabilities across environments")
        print("   • Configuration export and management")

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
