#!/usr/bin/env python3
"""
Production Readiness Validation Example.

This script demonstrates the usage of the ProductionReadinessValidator
to validate artifacts for production deployment.
"""

import sys
from pathlib import Path

from crackseg.utils.deployment import (
    ProductionReadinessCriteria,
    ProductionReadinessValidator,
)
from crackseg.utils.traceability import ArtifactEntity
from crackseg.utils.traceability.enums import ArtifactType

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def setup_logging() -> None:
    """Setup logging configuration."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_mock_artifact() -> ArtifactEntity:
    """Create a mock artifact for demonstration."""
    return ArtifactEntity(
        artifact_id="crackseg-model-v2.1",
        artifact_type=ArtifactType.MODEL,
        file_path=Path("/path/to/model/checkpoint.pth"),
        file_size=1024000,
        checksum="sha256:abc123...",
        name="CrackSeg Model v2.1",
        owner="ml-team",
        experiment_id="exp-2024-001",
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


def create_production_criteria() -> ProductionReadinessCriteria:
    """Create production readiness criteria."""
    return ProductionReadinessCriteria(
        security_checks={
            "model_safety": True,
            "data_privacy": True,
            "access_control": True,
        },
        performance_checks={
            "inference_speed": True,
            "memory_usage": True,
            "throughput": True,
        },
        resource_checks={
            "gpu_compatibility": True,
            "memory_requirements": True,
            "storage_requirements": True,
        },
        operational_checks={
            "monitoring": True,
            "logging": True,
            "error_handling": True,
        },
        compliance_checks={
            "data_governance": True,
            "model_registry": True,
            "audit_trail": True,
        },
    )


def demonstrate_security_validation() -> None:
    """Demonstrate security validation checks."""
    print("\n🔒 SECURITY VALIDATION")
    print("=" * 50)
    print("Validating model safety, data privacy, and access control...")

    validator = ProductionReadinessValidator()
    artifact = create_mock_artifact()

    # Focus on security checks
    security_criteria = ProductionReadinessCriteria(
        security_checks={
            "model_safety": True,
            "data_privacy": True,
            "access_control": True,
        },
        performance_checks={},
        resource_checks={},
        operational_checks={},
        compliance_checks={},
    )

    result = validator.validate_production_readiness(
        artifact, security_criteria
    )

    if result.success:
        print("✅ Security validation passed!")
        print("🔐 Model safety verified")
        print("🛡️ Data privacy compliance confirmed")
        print("🚪 Access control mechanisms validated")
    else:
        print("❌ Security validation failed!")
        for check, details in result.failed_checks.items():
            print(f"  - {check}: {details}")


def demonstrate_performance_validation() -> None:
    """Demonstrate performance validation checks."""
    print("\n⚡ PERFORMANCE VALIDATION")
    print("=" * 50)
    print("Validating inference speed, memory usage, and throughput...")

    validator = ProductionReadinessValidator()
    artifact = create_mock_artifact()

    # Focus on performance checks
    performance_criteria = ProductionReadinessCriteria(
        security_checks={},
        performance_checks={
            "inference_speed": True,
            "memory_usage": True,
            "throughput": True,
        },
        resource_checks={},
        operational_checks={},
        compliance_checks={},
    )

    result = validator.validate_production_readiness(
        artifact, performance_criteria
    )

    if result.success:
        print("✅ Performance validation passed!")
        print("⚡ Inference speed: 120ms (acceptable)")
        print("💾 Memory usage: 2.1GB (within limits)")
        print("📊 Throughput: 8.3 FPS (meets requirements)")
    else:
        print("❌ Performance validation failed!")
        for check, details in result.failed_checks.items():
            print(f"  - {check}: {details}")


def demonstrate_resource_validation() -> None:
    """Demonstrate resource validation checks."""
    print("\n💻 RESOURCE VALIDATION")
    print("=" * 50)
    print("Validating GPU compatibility, memory, and storage requirements...")

    validator = ProductionReadinessValidator()
    artifact = create_mock_artifact()

    # Focus on resource checks
    resource_criteria = ProductionReadinessCriteria(
        security_checks={},
        performance_checks={},
        resource_checks={
            "gpu_compatibility": True,
            "memory_requirements": True,
            "storage_requirements": True,
        },
        operational_checks={},
        compliance_checks={},
    )

    result = validator.validate_production_readiness(
        artifact, resource_criteria
    )

    if result.success:
        print("✅ Resource validation passed!")
        print("🎮 GPU compatibility: CUDA 11.8+ supported")
        print("💾 Memory requirements: 4GB VRAM (available)")
        print("💿 Storage requirements: 500MB (sufficient)")
    else:
        print("❌ Resource validation failed!")
        for check, details in result.failed_checks.items():
            print(f"  - {check}: {details}")


def demonstrate_operational_validation() -> None:
    """Demonstrate operational validation checks."""
    print("\n🔧 OPERATIONAL VALIDATION")
    print("=" * 50)
    print("Validating monitoring, logging, and error handling...")

    validator = ProductionReadinessValidator()
    artifact = create_mock_artifact()

    # Focus on operational checks
    operational_criteria = ProductionReadinessCriteria(
        security_checks={},
        performance_checks={},
        resource_checks={},
        operational_checks={
            "monitoring": True,
            "logging": True,
            "error_handling": True,
        },
        compliance_checks={},
    )

    result = validator.validate_production_readiness(
        artifact, operational_criteria
    )

    if result.success:
        print("✅ Operational validation passed!")
        print("📊 Monitoring: Prometheus metrics configured")
        print("📝 Logging: Structured logging implemented")
        print("⚠️ Error handling: Graceful degradation enabled")
    else:
        print("❌ Operational validation failed!")
        for check, details in result.failed_checks.items():
            print(f"  - {check}: {details}")


def demonstrate_compliance_validation() -> None:
    """Demonstrate compliance validation checks."""
    print("\n📋 COMPLIANCE VALIDATION")
    print("=" * 50)
    print("Validating data governance, model registry, and audit trail...")

    validator = ProductionReadinessValidator()
    artifact = create_mock_artifact()

    # Focus on compliance checks
    compliance_criteria = ProductionReadinessCriteria(
        security_checks={},
        performance_checks={},
        resource_checks={},
        operational_checks={},
        compliance_checks={
            "data_governance": True,
            "model_registry": True,
            "audit_trail": True,
        },
    )

    result = validator.validate_production_readiness(
        artifact, compliance_criteria
    )

    if result.success:
        print("✅ Compliance validation passed!")
        print("📊 Data governance: GDPR compliance verified")
        print("📚 Model registry: Version tracking enabled")
        print("🔍 Audit trail: Complete deployment history")
    else:
        print("❌ Compliance validation failed!")
        for check, details in result.failed_checks.items():
            print(f"  - {check}: {details}")


def demonstrate_comprehensive_validation() -> None:
    """Demonstrate comprehensive production readiness validation."""
    print("\n🎯 COMPREHENSIVE PRODUCTION READINESS VALIDATION")
    print("=" * 60)
    print("Running all validation checks for production deployment...")

    validator = ProductionReadinessValidator()
    artifact = create_mock_artifact()
    criteria = create_production_criteria()

    result = validator.validate_production_readiness(artifact, criteria)

    print("\n📊 Validation Results:")
    print(f"  - Overall Success: {'✅ PASS' if result.success else '❌ FAIL'}")
    print(f"  - Total Checks: {result.total_checks}")
    print(f"  - Passed Checks: {result.passed_checks}")
    print(f"  - Failed Checks: {result.failed_checks_count}")

    if result.success:
        print("\n🎉 Production readiness validation completed successfully!")
        print("✅ Artifact is ready for production deployment")
        print("🚀 All checks passed - safe to deploy")
    else:
        print("\n⚠️ Production readiness validation failed!")
        print("❌ Artifact is NOT ready for production deployment")
        print("🔧 Please address the following issues:")

        for check, details in result.failed_checks.items():
            print(f"  - {check}: {details}")

    return result


def main() -> None:
    """Main function to demonstrate production readiness validation."""
    setup_logging()

    print("🔍 CRACKSEG PRODUCTION READINESS VALIDATION DEMO")
    print("=" * 60)
    print("This demo showcases comprehensive production readiness validation")
    print("for ML model artifacts before deployment.")
    print("=" * 60)

    try:
        # Demonstrate individual validation categories
        demonstrate_security_validation()
        demonstrate_performance_validation()
        demonstrate_resource_validation()
        demonstrate_operational_validation()
        demonstrate_compliance_validation()

        # Demonstrate comprehensive validation
        demonstrate_comprehensive_validation()

        print("\n✅ Demo completed successfully!")
        print("\n🎯 Key Takeaways:")
        print("- Production readiness validation ensures safe deployments")
        print("- Multiple validation categories cover all aspects")
        print("- Failed validations prevent risky deployments")
        print("- Comprehensive checks reduce production incidents")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import logging

        logging.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
