"""
Test helpers for E2E testing with comprehensive setup/teardown and
coordination utilities. This module provides specialized helper
functions for advanced E2E test scenarios, including test
setup/teardown procedures, API integration utilities, test
coordination, and performance monitoring. These helpers complement the
existing utils library with higher-level orchestration capabilities.
The helpers are organized into focused modules: - setup_teardown:
Advanced test environment management - api_integration: HTTP/API
testing utilities - test_coordination: Multi-test and parallel
execution helpers - performance_monitoring: Performance metrics and
monitoring
"""

# Setup and teardown helpers
# API integration helpers
from .api_integration import (
    APITestHelper,
    simulate_api_load,
    test_crackseg_endpoints,
    validate_api_responses,
    verify_streamlit_health,
)

# Performance monitoring helpers
from .performance_monitoring import (
    PerformanceMonitor,
    generate_performance_report,
    measure_page_load_time,
    monitor_memory_usage,
    track_user_interaction_latency,
)
from .setup_teardown import (
    TestEnvironmentManager,
    cleanup_test_environment,
    create_clean_test_environment,
    manage_test_artifacts,
    restore_default_state,
    setup_crackseg_test_state,
)

# Test coordination helpers
from .test_coordination import (
    TestCoordinator,
    coordinate_parallel_tests,
    manage_test_dependencies,
    orchestrate_test_sequence,
    synchronize_test_states,
)

__all__ = [
    # Setup/teardown helpers
    "TestEnvironmentManager",
    "create_clean_test_environment",
    "cleanup_test_environment",
    "setup_crackseg_test_state",
    "restore_default_state",
    "manage_test_artifacts",
    # API integration helpers
    "APITestHelper",
    "verify_streamlit_health",
    "test_crackseg_endpoints",
    "simulate_api_load",
    "validate_api_responses",
    # Test coordination helpers
    "TestCoordinator",
    "coordinate_parallel_tests",
    "manage_test_dependencies",
    "synchronize_test_states",
    "orchestrate_test_sequence",
    # Performance monitoring helpers
    "PerformanceMonitor",
    "measure_page_load_time",
    "monitor_memory_usage",
    "track_user_interaction_latency",
    "generate_performance_report",
]
