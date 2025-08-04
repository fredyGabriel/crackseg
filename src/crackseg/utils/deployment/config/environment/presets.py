"""Predefined environment configurations."""

from typing import Any

from .config import ResourceRequirements


def load_predefined_configs() -> dict[str, dict[str, dict[str, Any]]]:
    """Load predefined environment configurations.

    Returns:
        Dictionary of predefined configurations
    """
    return {
        "production": {
            "container": {
                "resources": ResourceRequirements(
                    cpu_cores=4.0,
                    memory_mb=8192,
                    gpu_memory_mb=8192,
                    storage_gb=10.0,
                    network_bandwidth_mbps=1000,
                ),
                "python_version": "3.9",
                "base_image": "python:3.9-slim",
                "required_packages": [
                    "torch>=1.9.0",
                    "torchvision>=0.10.0",
                    "streamlit>=1.0.0",
                    "opencv-python>=4.5.0",
                    "pillow>=8.0.0",
                    "numpy>=1.21.0",
                ],
                "system_dependencies": [
                    "libgl1-mesa-glx",
                    "libglib2.0-0",
                    "libsm6",
                    "libxext6",
                    "libxrender-dev",
                ],
                "replicas": 3,
                "autoscaling": True,
                "max_replicas": 10,
                "log_level": "WARNING",
            },
            "serverless": {
                "resources": ResourceRequirements(
                    cpu_cores=2.0,
                    memory_mb=4096,
                    gpu_memory_mb=0,
                    storage_gb=1.0,
                    network_bandwidth_mbps=100,
                ),
                "python_version": "3.9",
                "required_packages": [
                    "torch>=1.9.0",
                    "streamlit>=1.0.0",
                    "opencv-python>=4.5.0",
                    "pillow>=8.0.0",
                    "numpy>=1.21.0",
                ],
                "replicas": 1,
                "autoscaling": False,
                "log_level": "INFO",
            },
            "edge": {
                "resources": ResourceRequirements(
                    cpu_cores=1.0,
                    memory_mb=2048,
                    gpu_memory_mb=0,
                    storage_gb=2.0,
                    network_bandwidth_mbps=50,
                ),
                "python_version": "3.8",
                "required_packages": [
                    "torch>=1.8.0",
                    "opencv-python>=4.5.0",
                    "pillow>=8.0.0",
                    "numpy>=1.19.0",
                ],
                "replicas": 1,
                "autoscaling": False,
                "log_level": "ERROR",
            },
        },
        "staging": {
            "container": {
                "resources": ResourceRequirements(
                    cpu_cores=2.0,
                    memory_mb=4096,
                    gpu_memory_mb=4096,
                    storage_gb=5.0,
                    network_bandwidth_mbps=500,
                ),
                "python_version": "3.9",
                "base_image": "python:3.9-slim",
                "required_packages": [
                    "torch>=1.9.0",
                    "torchvision>=0.10.0",
                    "streamlit>=1.0.0",
                    "opencv-python>=4.5.0",
                    "pillow>=8.0.0",
                    "numpy>=1.21.0",
                ],
                "system_dependencies": [
                    "libgl1-mesa-glx",
                    "libglib2.0-0",
                    "libsm6",
                    "libxext6",
                    "libxrender-dev",
                ],
                "replicas": 2,
                "autoscaling": True,
                "max_replicas": 5,
                "log_level": "INFO",
            },
            "serverless": {
                "resources": ResourceRequirements(
                    cpu_cores=1.0,
                    memory_mb=2048,
                    gpu_memory_mb=0,
                    storage_gb=1.0,
                    network_bandwidth_mbps=100,
                ),
                "python_version": "3.9",
                "required_packages": [
                    "torch>=1.9.0",
                    "streamlit>=1.0.0",
                    "opencv-python>=4.5.0",
                    "pillow>=8.0.0",
                    "numpy>=1.21.0",
                ],
                "replicas": 1,
                "autoscaling": False,
                "log_level": "INFO",
            },
            "edge": {
                "resources": ResourceRequirements(
                    cpu_cores=1.0,
                    memory_mb=1024,
                    gpu_memory_mb=0,
                    storage_gb=1.0,
                    network_bandwidth_mbps=25,
                ),
                "python_version": "3.8",
                "required_packages": [
                    "torch>=1.8.0",
                    "opencv-python>=4.5.0",
                    "pillow>=8.0.0",
                    "numpy>=1.19.0",
                ],
                "replicas": 1,
                "autoscaling": False,
                "log_level": "WARNING",
            },
        },
        "development": {
            "container": {
                "resources": ResourceRequirements(
                    cpu_cores=1.0,
                    memory_mb=2048,
                    gpu_memory_mb=2048,
                    storage_gb=2.0,
                    network_bandwidth_mbps=100,
                ),
                "python_version": "3.9",
                "base_image": "python:3.9-slim",
                "required_packages": [
                    "torch>=1.9.0",
                    "torchvision>=0.10.0",
                    "streamlit>=1.0.0",
                    "opencv-python>=4.5.0",
                    "pillow>=8.0.0",
                    "numpy>=1.21.0",
                    "pytest>=6.0.0",
                    "black>=21.0.0",
                    "ruff>=0.1.0",
                ],
                "system_dependencies": [
                    "libgl1-mesa-glx",
                    "libglib2.0-0",
                    "libsm6",
                    "libxext6",
                    "libxrender-dev",
                ],
                "replicas": 1,
                "autoscaling": False,
                "log_level": "DEBUG",
            },
            "serverless": {
                "resources": ResourceRequirements(
                    cpu_cores=1.0,
                    memory_mb=1024,
                    gpu_memory_mb=0,
                    storage_gb=1.0,
                    network_bandwidth_mbps=50,
                ),
                "python_version": "3.9",
                "required_packages": [
                    "torch>=1.9.0",
                    "streamlit>=1.0.0",
                    "opencv-python>=4.5.0",
                    "pillow>=8.0.0",
                    "numpy>=1.21.0",
                ],
                "replicas": 1,
                "autoscaling": False,
                "log_level": "DEBUG",
            },
            "edge": {
                "resources": ResourceRequirements(
                    cpu_cores=1.0,
                    memory_mb=512,
                    gpu_memory_mb=0,
                    storage_gb=0.5,
                    network_bandwidth_mbps=10,
                ),
                "python_version": "3.8",
                "required_packages": [
                    "torch>=1.8.0",
                    "opencv-python>=4.5.0",
                    "pillow>=8.0.0",
                    "numpy>=1.19.0",
                ],
                "replicas": 1,
                "autoscaling": False,
                "log_level": "INFO",
            },
        },
    }
