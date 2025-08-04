"""
Data generation helpers for E2E testing. This module provides
utilities for generating test data, sample configurations, and test
scenarios for the CrackSeg Streamlit application testing.
"""

import random
import string
from pathlib import Path
from typing import Any


def generate_random_string(
    length: int = 10,
    include_uppercase: bool = True,
    include_numbers: bool = True,
    include_special: bool = False,
) -> str:
    """
    Generate random string for testing purposes. Args: length: Length of
    the generated string include_uppercase: Include uppercase letters
    include_numbers: Include numeric digits include_special: Include
    special characters Returns: Random string with specified
    characteristics Example: >>> test_string = generate_random_string( ...
    length=8, include_numbers=True ... ) >>> len(test_string) 8
    """
    chars = string.ascii_lowercase

    if include_uppercase:
        chars += string.ascii_uppercase
    if include_numbers:
        chars += string.digits
    if include_special:
        chars += "!@#$%^&*"

    return "".join(random.choices(chars, k=length))


def generate_random_email(domain: str = "test.com") -> str:
    """Generate random email address for testing.

    Args:
        domain: Email domain to use

    Returns:
        Random email address

    Example:
        >>> email = generate_random_email("example.com")
        >>> "@example.com" in email
        True
    """
    username = generate_random_string(
        length=8, include_uppercase=False, include_numbers=True
    )
    return f"{username}@{domain}"


def generate_random_filename(
    extension: str = "jpg",
    prefix: str = "test_image",
    length: int = 6,
) -> str:
    """Generate random filename for testing file uploads.

    Args:
        extension: File extension
        prefix: Filename prefix
        length: Length of random part

    Returns:
        Random filename

    Example:
        >>> filename = generate_random_filename("png", "crack", 8)
        >>> filename.startswith("crack")
        True
        >>> filename.endswith(".png")
        True
    """
    random_part = generate_random_string(
        length=length, include_uppercase=False, include_numbers=True
    )
    return f"{prefix}_{random_part}.{extension}"


def generate_test_image_path(base_dir: str = "data/unified") -> Path:
    """Generate path to a test image file.

    Args:
        base_dir: Base directory for test images

    Returns:
        Path to test image file

    Example:
        >>> image_path = generate_test_image_path("test_data")
        >>> isinstance(image_path, Path)
        True
    """
    # Use existing test images from the project
    test_images = [
        "101.jpg",
        "102.jpg",
        "104.jpg",
        "105.jpg",
        "106.jpg",
    ]

    selected_image = random.choice(test_images)
    return Path(base_dir) / "images" / selected_image


def get_sample_file_paths() -> dict[str, Path]:
    """Get sample file paths for testing file operations.

    Returns:
        Dictionary mapping file types to sample paths

    Example:
        >>> paths = get_sample_file_paths()
        >>> "test_image" in paths
        True
    """
    return {
        "test_image": generate_test_image_path(),
        "config_file": Path("configs/base.yaml"),
        "checkpoint_file": Path("checkpoints/test_model.pth"),
        "output_dir": Path("outputs/test_run"),
        "artifacts_dir": Path("test-artifacts"),
    }


def get_sample_config_data() -> dict[str, Any]:
    """Get sample configuration data for testing.

    Returns:
        Dictionary with sample configuration values

    Example:
        >>> config = get_sample_config_data()
        >>> "model" in config
        True
    """
    return {
        "model": {
            "architecture": "unet",
            "encoder": "resnet50",
            "num_classes": 2,
            "input_channels": 3,
        },
        "training": {
            "batch_size": 8,
            "epochs": 10,
            "learning_rate": 0.001,
            "optimizer": "adam",
        },
        "data": {
            "data_root": "data/unified",
            "root_dir": "data/unified",
            "image_size": 512,
        },
        "evaluation": {
            "metrics": ["iou", "dice", "precision", "recall"],
            "save_predictions": True,
            "threshold": 0.5,
        },
    }


def get_test_data_combinations() -> list[dict[str, Any]]:
    """
    Get various test data combinations for parameterized testing. Returns:
    List of dictionaries with different test scenarios Example: >>>
    combinations = get_test_data_combinations() >>> len(combinations) > 0
    True
    """
    return [
        # Basic training configuration
        {
            "scenario": "basic_training",
            "model_type": "unet",
            "encoder": "resnet34",
            "batch_size": 4,
            "epochs": 5,
            "expected_pages": ["config", "train", "results"],
        },
        # Advanced configuration
        {
            "scenario": "advanced_config",
            "model_type": "deeplabv3",
            "encoder": "efficientnet_b0",
            "batch_size": 8,
            "epochs": 10,
            "use_attention": True,
            "expected_pages": ["config", "advanced_config", "architecture"],
        },
        # Large model scenario
        {
            "scenario": "large_model",
            "model_type": "swin_unet",
            "encoder": "swin_base",
            "batch_size": 2,
            "epochs": 3,
            "memory_optimized": True,
            "expected_pages": ["architecture", "train"],
        },
        # Evaluation only
        {
            "scenario": "evaluation_only",
            "model_type": "unet",
            "load_checkpoint": True,
            "evaluate_only": True,
            "expected_pages": ["results"],
        },
        # Error scenarios
        {
            "scenario": "invalid_config",
            "model_type": "invalid_model",
            "batch_size": -1,
            "epochs": 0,
            "should_fail": True,
            "expected_error": "Invalid configuration",
        },
    ]


def generate_streamlit_test_scenarios() -> list[dict[str, Any]]:
    """Generate test scenarios specific to Streamlit UI interactions.

    Returns:
        List of Streamlit-specific test scenarios

    Example:
        >>> scenarios = generate_streamlit_test_scenarios()
        >>> all("page" in scenario for scenario in scenarios)
        True
    """
    return [
        {
            "name": "config_page_basic_interaction",
            "page": "Configuration",
            "actions": [
                {
                    "type": "select",
                    "element": "model_architecture",
                    "value": "U-Net",
                },
                {"type": "select", "element": "encoder", "value": "ResNet50"},
                {"type": "number_input", "element": "batch_size", "value": 8},
                {"type": "button", "element": "save_config"},
            ],
            "expected_outcomes": ["config_saved", "success_message"],
        },
        {
            "name": "file_upload_workflow",
            "page": "Data Upload",
            "actions": [
                {
                    "type": "file_upload",
                    "element": "image_upload",
                    "file": "test_image.jpg",
                },
                {
                    "type": "file_upload",
                    "element": "mask_upload",
                    "file": "test_mask.png",
                },
                {"type": "button", "element": "process_files"},
            ],
            "expected_outcomes": ["files_uploaded", "preview_displayed"],
        },
        {
            "name": "training_workflow",
            "page": "Training",
            "actions": [
                {"type": "button", "element": "start_training"},
                {
                    "type": "wait",
                    "condition": "training_progress",
                    "timeout": 60,
                },
                {"type": "button", "element": "stop_training"},
            ],
            "expected_outcomes": [
                "training_started",
                "progress_displayed",
                "training_stopped",
            ],
        },
        {
            "name": "results_visualization",
            "page": "Results",
            "actions": [
                {
                    "type": "select",
                    "element": "result_type",
                    "value": "segmentation_overlay",
                },
                {"type": "slider", "element": "threshold", "value": 0.7},
                {"type": "button", "element": "download_results"},
            ],
            "expected_outcomes": [
                "visualization_updated",
                "download_triggered",
            ],
        },
    ]


def generate_browser_test_matrix() -> list[dict[str, Any]]:
    """Generate test matrix for cross-browser testing.

    Returns:
        List of browser configurations for testing

    Example:
        >>> matrix = generate_browser_test_matrix()
        >>> all("browser" in config for config in matrix)
        True
    """
    return [
        {
            "browser": "chrome",
            "version": "latest",
            "headless": True,
            "window_size": (1920, 1080),
            "mobile": False,
        },
        {
            "browser": "firefox",
            "version": "latest",
            "headless": True,
            "window_size": (1920, 1080),
            "mobile": False,
        },
        {
            "browser": "chrome",
            "version": "latest",
            "headless": True,
            "window_size": (375, 667),  # iPhone 6/7/8 size
            "mobile": True,
        },
        {
            "browser": "chrome",
            "version": "latest",
            "headless": False,
            "window_size": (1366, 768),  # Common laptop size
            "mobile": False,
            "debug": True,
        },
    ]


def get_performance_test_data() -> dict[str, Any]:
    """Get data for performance testing scenarios.

    Returns:
        Dictionary with performance test configuration

    Example:
        >>> perf_data = get_performance_test_data()
        >>> "load_test" in perf_data
        True
    """
    return {
        "load_test": {
            "concurrent_users": [1, 5, 10],
            "page_load_threshold": 5.0,  # seconds
            "response_time_threshold": 2.0,  # seconds
        },
        "stress_test": {
            "file_sizes": ["1MB", "10MB", "50MB"],
            "batch_sizes": [1, 8, 16, 32],
            "timeout_threshold": 30.0,  # seconds
        },
        "memory_test": {
            "large_datasets": True,
            "memory_threshold_mb": 2048,
            "gc_monitoring": True,
        },
    }
