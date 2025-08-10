"""Constants and factory helpers for deployment packaging system."""

from __future__ import annotations

from typing import Final

from .config import ContainerizationConfig

SUPPORTED_TARGETS: Final[list[str]] = [
    "docker",
    "kubernetes",
    "serverless",
    "edge",
]

BASE_IMAGES: Final[dict[str, str]] = {
    "python": "python:3.12-slim",
    "gpu": "nvidia/cuda:12.1-devel-ubuntu20.04",
    "minimal": "alpine:3.18",
    "ml": "python:3.12-slim",
}


def default_container_configs() -> dict[str, ContainerizationConfig]:
    """Build default containerization configs for environments."""
    return {
        "production": ContainerizationConfig(
            base_image="python:3.12-slim",
            non_root_user=True,
            security_scan=True,
            layer_caching=True,
            compression=True,
        ),
        "staging": ContainerizationConfig(
            base_image="python:3.12-slim",
            non_root_user=True,
            security_scan=False,
            layer_caching=True,
            compression=False,
        ),
        "development": ContainerizationConfig(
            base_image="python:3.12-slim",
            non_root_user=False,
            security_scan=False,
            layer_caching=False,
            compression=False,
        ),
    }


PACKAGE_DIRECTORIES: Final[list[str]] = [
    "app",
    "config",
    "scripts",
    "tests",
    "docs",
    "k8s",
    "helm",
]
