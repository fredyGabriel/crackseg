"""Configuration classes for packaging system."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PackagingResult:
    """Result of packaging operation."""

    success: bool
    package_path: str | None = None
    package_size_mb: float = 0.0
    container_image_name: str | None = None
    dockerfile_path: str | None = None

    # Dependencies
    requirements_path: str | None = None
    dependencies_count: int = 0

    # Containerization details
    image_size_mb: float = 0.0
    build_time_seconds: float = 0.0
    layer_count: int = 0
    security_scan_results: dict[str, Any] = field(default_factory=dict)

    # Deployment manifests
    kubernetes_manifests: list[str] = field(default_factory=list)
    docker_compose_path: str | None = None
    helm_chart_path: str | None = None

    # Error information
    error_message: str | None = None


@dataclass
class ContainerizationConfig:
    """Configuration for containerization process."""

    # Base image selection
    base_image: str = "python:3.12-slim"
    gpu_support: bool = False
    multi_stage: bool = True

    # Security settings
    non_root_user: bool = True
    security_scan: bool = True
    vulnerability_check: bool = True

    # Optimization settings
    layer_caching: bool = True
    compression: bool = True
    multi_arch: bool = False

    # Registry settings
    registry_url: str | None = None
    image_tag: str | None = None
    push_to_registry: bool = False
