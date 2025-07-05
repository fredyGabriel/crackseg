"""Utility functions for E2E testing with Selenium and Streamlit.

This module provides a comprehensive utility library with helper functions
for common Selenium operations, data manipulation, file handling, and
Streamlit-specific interactions.

The utilities are organized into focused modules:
- element: Element interaction utilities
- data: Data generation helpers
- file: File I/O utilities
- string: String manipulation functions
- time: Date/time helpers
- config: Configuration readers
- streamlit: Streamlit-specific utilities
"""

# Element interaction utilities
# Configuration readers
from .config import (
    create_test_config_file,
    get_browser_config,
    get_streamlit_test_config,
    get_test_environment_config,
    load_test_config,
    merge_configs,
    validate_crackseg_config,
)

# Data generation helpers
from .data import (
    generate_random_email,
    generate_random_filename,
    generate_random_string,
    generate_test_image_path,
    get_sample_config_data,
    get_sample_file_paths,
    get_test_data_combinations,
)
from .element import (
    click_element_safely,
    find_element_with_retry,
    get_element_attribute_safely,
    get_element_text_safely,
    is_element_present,
    is_element_visible,
    scroll_to_element,
    wait_for_element_to_be_clickable,
    wait_for_element_to_disappear,
    wait_for_element_visible,
    wait_for_elements_count,
)

# File I/O utilities
from .file import (
    cleanup_test_files,
    create_temp_config_file,
    ensure_artifacts_dir,
    get_test_config_path,
    read_test_config,
    save_test_artifacts,
    validate_file_exists,
)

# Streamlit-specific utilities
from .streamlit import (
    click_streamlit_button,
    get_crackseg_app_status,
    get_streamlit_session_state,
    navigate_to_page,
    start_streamlit_app,
    stop_streamlit_app,
    upload_file,
    wait_for_streamlit_ready,
    wait_for_streamlit_rerun,
)

# String manipulation functions
from .string import (
    clean_element_text,
    extract_numbers_from_text,
    format_test_name,
    normalize_whitespace,
    sanitize_filename,
    validate_text_contains,
)

# Date/time helpers
from .time import (
    format_duration,
    get_current_timestamp,
    get_test_timeout,
    is_timestamp_recent,
    parse_duration_string,
    wait_with_timeout,
)

__all__ = [
    # Element utilities
    "click_element_safely",
    "find_element_with_retry",
    "get_element_attribute_safely",
    "get_element_text_safely",
    "is_element_present",
    "is_element_visible",
    "scroll_to_element",
    "wait_for_element_to_be_clickable",
    "wait_for_element_to_disappear",
    "wait_for_element_visible",
    "wait_for_elements_count",
    # Data utilities
    "generate_random_email",
    "generate_random_filename",
    "generate_random_string",
    "generate_test_image_path",
    "get_sample_config_data",
    "get_sample_file_paths",
    "get_test_data_combinations",
    # File utilities
    "cleanup_test_files",
    "create_temp_config_file",
    "ensure_artifacts_dir",
    "get_test_config_path",
    "read_test_config",
    "save_test_artifacts",
    "validate_file_exists",
    # String utilities
    "clean_element_text",
    "extract_numbers_from_text",
    "format_test_name",
    "normalize_whitespace",
    "sanitize_filename",
    "validate_text_contains",
    # Time utilities
    "format_duration",
    "get_current_timestamp",
    "get_test_timeout",
    "is_timestamp_recent",
    "parse_duration_string",
    "wait_with_timeout",
    # Config utilities
    "create_test_config_file",
    "get_browser_config",
    "get_streamlit_test_config",
    "get_test_environment_config",
    "load_test_config",
    "merge_configs",
    "validate_crackseg_config",
    # Streamlit utilities
    "click_streamlit_button",
    "get_crackseg_app_status",
    "get_streamlit_session_state",
    "navigate_to_page",
    "start_streamlit_app",
    "stop_streamlit_app",
    "upload_file",
    "wait_for_streamlit_ready",
    "wait_for_streamlit_rerun",
]
