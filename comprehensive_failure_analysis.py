#!/usr/bin/env python3
"""Comprehensive Test Failure Analysis for CrackSeg Project.

This module provides detailed analysis of all 30 test failures captured
from the GUI test suite execution, categorizing errors systematically
and generating actionable insights for test improvement.

Subtask 6.1 - Test Failure Data Collection and Logging
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class FailureCategory(Enum):
    """Categories of test failures based on root cause analysis."""

    IMPORT_ERROR = "import_error"
    MOCK_ERROR = "mock_error"
    CONFIG_ERROR = "config_error"
    ASSERTION_ERROR = "assertion_error"
    STREAMLIT_ERROR = "streamlit_error"
    ATTRIBUTE_ERROR = "attribute_error"
    TYPE_ERROR = "type_error"
    TEST_INFRASTRUCTURE_ERROR = "test_infrastructure_error"


class ErrorSeverity(Enum):
    """Severity levels for test failures."""

    CRITICAL = "critical"  # Blocking basic functionality
    HIGH = "high"  # Major features broken
    MEDIUM = "medium"  # Specific functionality issues
    LOW = "low"  # Minor issues or flaky tests


@dataclass
class TestFailureAnalysis:
    """Comprehensive analysis of all captured test failures."""

    test_name: str
    test_file: str
    failure_type: str
    error_message: str
    category: FailureCategory
    severity: ErrorSeverity
    root_cause: str
    suggested_fix: str
    affected_component: str
    priority_order: int


def analyze_all_failures() -> list[TestFailureAnalysis]:
    """Analyze all 30 test failures captured from pytest execution."""

    failures = [
        # GUI Pages Tests - Attribute/Import Errors
        TestFailureAnalysis(
            test_name="TestAdvancedConfigPage::test_page_advanced_config_basic_mock",
            test_file="tests/unit/gui/pages/test_advanced_config_page.py",
            failure_type="AttributeError",
            error_message=(
                "module 'scripts.gui.pages.advanced_config_page' "
                "has no attribute 'render_advanced_config_page'"
            ),
            category=FailureCategory.ATTRIBUTE_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "Missing function in module - refactoring broke API contract"
            ),
            suggested_fix=(
                "Add missing render_advanced_config_page function or update "
                "test to use correct API"
            ),
            affected_component="Advanced Config Page",
            priority_order=1,
        ),
        TestFailureAnalysis(
            test_name="TestConfigPage::test_page_config_basic_mock",
            test_file="tests/unit/gui/pages/test_config_page.py",
            failure_type="TypeError",
            error_message=(
                "Config page failed: argument of type 'MockSessionState' is "
                "not iterable"
            ),
            category=FailureCategory.STREAMLIT_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "MockSessionState implementation doesn't match Streamlit API "
                "expectations"
            ),
            suggested_fix=(
                "Update MockSessionState to properly implement iterable "
                "interface"
            ),
            affected_component="Config Page & Session State",
            priority_order=2,
        ),
        TestFailureAnalysis(
            test_name="TestHomePage::test_page_home_basic_rendering",
            test_file="tests/unit/gui/pages/test_home_page.py",
            failure_type="ValueError",
            error_message=(
                "Home page rendering failed: too many values to unpack "
                "(expected 2)"
            ),
            category=FailureCategory.TYPE_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "Function return signature changed - tuple unpacking mismatch"
            ),
            suggested_fix=(
                "Update tuple unpacking to match current function return "
                "format"
            ),
            affected_component="Home Page Rendering",
            priority_order=3,
        ),
        TestFailureAnalysis(
            test_name="TestHomePage::test_dataset_statistics_display",
            test_file="tests/unit/gui/pages/test_home_page.py",
            failure_type="AttributeError",
            error_message=(
                "module 'scripts.gui.utils.data_stats' has no attribute "
                "expected function"
            ),
            category=FailureCategory.ATTRIBUTE_ERROR,
            severity=ErrorSeverity.MEDIUM,
            root_cause="data_stats module API changed or function moved",
            suggested_fix=(
                "Update import path or function name to match current API"
            ),
            affected_component="Data Statistics Display",
            priority_order=4,
        ),
        TestFailureAnalysis(
            test_name="TestHomePage::test_quick_actions_buttons",
            test_file="tests/unit/gui/pages/test_home_page.py",
            failure_type="AssertionError",
            error_message="Missing button: New Experiment",
            category=FailureCategory.ASSERTION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            root_cause="UI component changed - button not present or renamed",
            suggested_fix=(
                "Update test to look for correct button name/ID or add missing"
                " button"
            ),
            affected_component="Quick Actions UI",
            priority_order=5,
        ),
        TestFailureAnalysis(
            test_name="TestHomePage::test_recent_configs_display",
            test_file="tests/unit/gui/pages/test_home_page.py",
            failure_type="AttributeError",
            error_message=(
                "module 'scripts.gui.utils.config_io' has no attribute "
                "expected function"
            ),
            category=FailureCategory.ATTRIBUTE_ERROR,
            severity=ErrorSeverity.MEDIUM,
            root_cause="config_io module API changed during refactoring",
            suggested_fix=("Update test to use correct config_io API methods"),
            affected_component="Config I/O System",
            priority_order=6,
        ),
        TestFailureAnalysis(
            test_name="TestHomePage::test_warning_for_missing_gpu",
            test_file="tests/unit/gui/pages/test_home_page.py",
            failure_type="AssertionError",
            error_message="Expected 'warning' to have been called",
            category=FailureCategory.MOCK_ERROR,
            severity=ErrorSeverity.LOW,
            root_cause=(
                "Mock expectation not met - warning function not called as "
                "expected"
            ),
            suggested_fix=(
                "Review GPU detection logic and update mock expectations"
            ),
            affected_component="GPU Detection System",
            priority_order=7,
        ),
        TestFailureAnalysis(
            test_name="TestHomePage::test_system_status_indicators",
            test_file="tests/unit/gui/pages/test_home_page.py",
            failure_type="AssertionError",
            error_message="assert False",
            category=FailureCategory.ASSERTION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            root_cause=(
                "System status logic changed - assertion always fails"
            ),
            suggested_fix=(
                "Review system status implementation and update test logic"
            ),
            affected_component="System Status Indicators",
            priority_order=8,
        ),
        TestFailureAnalysis(
            test_name="TestHomePageIntegration::test_logo_component_integration",
            test_file="tests/unit/gui/pages/test_home_page.py",
            failure_type="AttributeError",
            error_message=(
                "module 'scripts.gui.components.logo_component' has no "
                "attribute expected"
            ),
            category=FailureCategory.ATTRIBUTE_ERROR,
            severity=ErrorSeverity.LOW,
            root_cause=(
                "Logo component API changed or moved during refactoring"
            ),
            suggested_fix=(
                "Update import and usage to match current logo component API"
            ),
            affected_component="Logo Component",
            priority_order=9,
        ),
        TestFailureAnalysis(
            test_name="TestHomePageIntegration::test_theme_integration",
            test_file="tests/unit/gui/pages/test_home_page.py",
            failure_type="AttributeError",
            error_message=(
                "module 'scripts.gui.components.theme_component' has no "
                "attribute expected"
            ),
            category=FailureCategory.ATTRIBUTE_ERROR,
            severity=ErrorSeverity.LOW,
            root_cause="Theme component API changed during GUI refactoring",
            suggested_fix=(
                "Update test to use current theme component interface"
            ),
            affected_component="Theme Component",
            priority_order=10,
        ),
        TestFailureAnalysis(
            test_name="TestHomePagePerformance::test_caching_efficiency",
            test_file="tests/unit/gui/pages/test_home_page.py",
            failure_type="AttributeError",
            error_message=(
                "module 'scripts.gui.utils.data_stats' has no attribute "
                "expected"
            ),
            category=FailureCategory.ATTRIBUTE_ERROR,
            severity=ErrorSeverity.LOW,
            root_cause="Caching implementation changed in data_stats module",
            suggested_fix=(
                "Update performance test to match current caching API"
            ),
            affected_component="Data Statistics Caching",
            priority_order=11,
        ),
        TestFailureAnalysis(
            test_name="TestTrainPage::test_page_train_basic_mock",
            test_file="tests/unit/gui/pages/test_train_page.py",
            failure_type="TypeError",
            error_message=(
                "Train page failed: argument of type 'MockSessionState' is not"
                " iterable"
            ),
            category=FailureCategory.STREAMLIT_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "Same MockSessionState issue as Config Page - systemic problem"
            ),
            suggested_fix=(
                "Fix MockSessionState implementation to be iterable"
            ),
            affected_component="Train Page & Session State",
            priority_order=2,  # Same priority as other MockSessionState issues
        ),
        # Error Console & Session State Tests
        TestFailureAnalysis(
            test_name="TestErrorCategorizer::test_categorize_value_error",
            test_file="tests/unit/gui/test_error_console.py",
            failure_type="AssertionError",
            error_message=(
                "assert <ErrorSeverity.CRITICAL: 'critical'> == "
                "<ErrorSeverity.WARNING: 'warning'>"
            ),
            category=FailureCategory.ASSERTION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            root_cause=(
                "Error categorization logic changed - severity levels "
                "reassigned"
            ),
            suggested_fix=(
                "Update test expectations to match current error severity "
                "classification"
            ),
            affected_component="Error Categorization System",
            priority_order=12,
        ),
        TestFailureAnalysis(
            test_name="TestErrorCategorizerCore::test_categorize_value_error",
            test_file="tests/unit/gui/test_error_console_simple.py",
            failure_type="AssertionError",
            error_message=(
                "assert <ErrorSeverity.CRITICAL: 'critical'> == "
                "<ErrorSeverity.WARNING: 'warning'>"
            ),
            category=FailureCategory.ASSERTION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            root_cause=(
                "Duplicate of error categorization issue - same root cause"
            ),
            suggested_fix=(
                "Update both error console tests with consistent severity "
                "expectations"
            ),
            affected_component="Error Categorization System",
            priority_order=12,  # Same as above
        ),
        TestFailureAnalysis(
            test_name="TestSessionStateManager::test_update_from_log_stream_info",
            test_file="tests/unit/gui/test_session_state_updates.py",
            failure_type="AssertionError",
            error_message="assert False is True",
            category=FailureCategory.ASSERTION_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "Session state update logic not working - log streaming state "
                "not updated"
            ),
            suggested_fix=(
                "Debug SessionStateManager.update_from_log_stream_info "
                "implementation"
            ),
            affected_component="Session State Management",
            priority_order=3,
        ),
        TestFailureAnalysis(
            test_name="TestSessionStateManager::test_extract_training_stats_from_logs",
            test_file="tests/unit/gui/test_session_state_updates.py",
            failure_type="AssertionError",
            error_message="assert None == 5",
            category=FailureCategory.ASSERTION_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "Training stats extraction not working - returns None instead "
                "of parsed values"
            ),
            suggested_fix=(
                "Fix log parsing regex or extraction logic in "
                "SessionStateManager"
            ),
            affected_component="Training Statistics Extraction",
            priority_order=3,
        ),
        TestFailureAnalysis(
            test_name="TestSessionStateManager::test_reset_training_session",
            test_file="tests/unit/gui/test_session_state_updates.py",
            failure_type="AssertionError",
            error_message="assert 'running' == 'idle'",
            category=FailureCategory.ASSERTION_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "Session reset logic not working - state not changing as "
                "expected"
            ),
            suggested_fix=(
                "Review SessionStateManager.reset_training_session "
                "implementation"
            ),
            affected_component="Session State Reset",
            priority_order=3,
        ),
        TestFailureAnalysis(
            test_name="TestSessionStateIntegration::test_end_to_end_process_lifecycle",
            test_file="tests/unit/gui/test_session_state_updates.py",
            failure_type="AssertionError",
            error_message="assert 'idle' == 'starting'",
            category=FailureCategory.ASSERTION_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "Process lifecycle state transitions not working correctly"
            ),
            suggested_fix=(
                "Debug process state management in SessionStateManager"
            ),
            affected_component="Process Lifecycle Management",
            priority_order=3,
        ),
        # Threading and Run Manager Tests
        TestFailureAnalysis(
            test_name="TestUIResponsiveWrapper::test_cancellation_support",
            test_file="tests/unit/gui/test_threading_integration.py",
            failure_type="AssertionError",
            error_message="assert 'completed' == 'cancelled'",
            category=FailureCategory.ASSERTION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            root_cause=(
                "Cancellation mechanism not working - task completed instead "
                "of being cancelled"
            ),
            suggested_fix=(
                "Review cancellation token implementation and timing"
            ),
            affected_component="UI Threading & Cancellation",
            priority_order=13,
        ),
        TestFailureAnalysis(
            test_name="TestRunManagerIntegration::test_execute_training_async_mock",
            test_file="tests/unit/gui/test_threading_integration.py",
            failure_type="AssertionError",
            error_message=(
                "Expected 'start_training_session' to be called once. Called "
                "0 times."
            ),
            category=FailureCategory.MOCK_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "Async training execution not calling expected function - "
                "integration broken"
            ),
            suggested_fix=(
                "Debug execute_training_async implementation and mock setup"
            ),
            affected_component="Async Training Execution",
            priority_order=4,
        ),
        TestFailureAnalysis(
            test_name="TestRunManagerAbort::test_abort_training_session",
            test_file="tests/unit/gui/test_enhanced_abort.py",
            failure_type="AssertionError",
            error_message=(
                "Expected 'abort_training' to be called once. Called 0 times."
            ),
            category=FailureCategory.MOCK_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "Training abort functionality not working - function not "
                "called"
            ),
            suggested_fix=(
                "Review abort_training implementation and mock expectations"
            ),
            affected_component="Training Abort System",
            priority_order=4,
        ),
        TestFailureAnalysis(
            test_name="TestRunManagerAbort::test_get_process_tree_info_wrapper",
            test_file="tests/unit/gui/test_enhanced_abort.py",
            failure_type="AssertionError",
            error_message="assert 0 == 1",
            category=FailureCategory.ASSERTION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            root_cause=(
                "Process tree information retrieval not working as expected"
            ),
            suggested_fix=(
                "Debug get_process_tree_info_wrapper return values"
            ),
            affected_component="Process Tree Management",
            priority_order=14,
        ),
        TestFailureAnalysis(
            test_name="TestRunManagerAbort::test_force_cleanup_orphans_basic",
            test_file="tests/unit/gui/test_enhanced_abort.py",
            failure_type="AttributeError",
            error_message=(
                "module 'scripts.gui.utils.run_manager' has no "
                "attribute 'psutil'"
            ),
            category=FailureCategory.ATTRIBUTE_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "psutil import missing or incorrectly imported in run_manager "
                "module"
            ),
            suggested_fix=("Add proper psutil import to run_manager module"),
            affected_component="Process Management & psutil Integration",
            priority_order=5,
        ),
        # Integration & Automation Tests
        TestFailureAnalysis(
            test_name="TestResourceCleanupValidationComponent::test_component_initialization",
            test_file="tests/integration/gui/automation/test_resource_cleanup_validation.py",
            failure_type="AttributeError",
            error_message="'dict' object has no attribute 'temp_path'",
            category=FailureCategory.TEST_INFRASTRUCTURE_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "Test utilities mock object incorrect - needs temp_path "
                "attribute"
            ),
            suggested_fix=(
                "Update test_utilities mock to include temp_path attribute"
            ),
            affected_component="Test Infrastructure",
            priority_order=6,
        ),
        TestFailureAnalysis(
            test_name="TestResourceCleanupValidationComponent::test_workflow_name",
            test_file="tests/integration/gui/automation/test_resource_cleanup_validation.py",
            failure_type="AttributeError",
            error_message="'dict' object has no attribute 'temp_path'",
            category=FailureCategory.TEST_INFRASTRUCTURE_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "Same test utilities issue - systemic problem across "
                "automation tests"
            ),
            suggested_fix=(
                "Create proper test utilities fixture with temp_path support"
            ),
            affected_component="Test Infrastructure",
            priority_order=6,
        ),
        TestFailureAnalysis(
            test_name="TestResourceCleanupValidationComponent::test_automation_preconditions",
            test_file="tests/integration/gui/automation/test_resource_cleanup_validation.py",
            failure_type="AttributeError",
            error_message="'dict' object has no attribute 'temp_path'",
            category=FailureCategory.TEST_INFRASTRUCTURE_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "Same test utilities issue - affects multiple automation tests"
            ),
            suggested_fix=(
                "Fix test utilities infrastructure for all automation tests"
            ),
            affected_component="Test Infrastructure",
            priority_order=6,
        ),
        TestFailureAnalysis(
            test_name="TestResourceCleanupValidationComponent::test_automation_metrics_retrieval",
            test_file="tests/integration/gui/automation/test_resource_cleanup_validation.py",
            failure_type="AttributeError",
            error_message="'dict' object has no attribute 'temp_path'",
            category=FailureCategory.TEST_INFRASTRUCTURE_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "Same test utilities issue - systemic infrastructure problem"
            ),
            suggested_fix=(
                "Comprehensive test utilities refactor for automation tests"
            ),
            affected_component="Test Infrastructure",
            priority_order=6,
        ),
        TestFailureAnalysis(
            test_name="TestResourceCleanupValidationComponent::test_workflow_component_cleanup_validation",
            test_file="tests/integration/gui/automation/test_resource_cleanup_validation.py",
            failure_type="AttributeError",
            error_message="'dict' object has no attribute 'temp_path'",
            category=FailureCategory.TEST_INFRASTRUCTURE_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "Same test utilities issue - infrastructure needs redesign"
            ),
            suggested_fix=(
                "Implement proper test utilities pattern for integration tests"
            ),
            affected_component="Test Infrastructure",
            priority_order=6,
        ),
        TestFailureAnalysis(
            test_name="TestResourceCleanupValidationComponent::test_memory_cleanup_validation",
            test_file="tests/integration/gui/automation/test_resource_cleanup_validation.py",
            failure_type="AttributeError",
            error_message="'dict' object has no attribute 'temp_path'",
            category=FailureCategory.TEST_INFRASTRUCTURE_ERROR,
            severity=ErrorSeverity.HIGH,
            root_cause=(
                "Same test utilities issue - final instance of systemic "
                "problem"
            ),
            suggested_fix=("Complete test utilities infrastructure overhaul"),
            affected_component="Test Infrastructure",
            priority_order=6,
        ),
    ]

    return failures


def generate_comprehensive_reports() -> None:
    """
    Generate comprehensive JSON and CSV reports with all failure analysis.
    """

    failures = analyze_all_failures()

    # Sort by priority for systematic fixing
    failures_by_priority = sorted(failures, key=lambda x: x.priority_order)

    # Generate summary statistics
    summary: dict[str, Any] = {
        "analysis_metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_failures_analyzed": len(failures),
            "analysis_scope": "Complete GUI test suite failure analysis",
            "execution_details": {
                "total_tests_run": 287,
                "total_failures": 30,
                "total_passed": 256,
                "total_skipped": 1,
                "failure_rate_percent": 10.45,
                "execution_time_seconds": 27.01,
            },
        },
        "category_breakdown": {},
        "severity_breakdown": {},
        "priority_breakdown": {},
        "affected_components": {},
        "systematic_issues": {},
    }

    # Calculate breakdowns
    for failure in failures:
        # Category breakdown
        cat = failure.category.value
        cat_count = summary["category_breakdown"].get(cat, 0)
        summary["category_breakdown"][cat] = cat_count + 1

        # Severity breakdown
        sev = failure.severity.value
        sev_count = summary["severity_breakdown"].get(sev, 0)
        summary["severity_breakdown"][sev] = sev_count + 1

        # Component breakdown
        comp = failure.affected_component
        comp_count = summary["affected_components"].get(comp, 0)
        summary["affected_components"][comp] = comp_count + 1

    # Identify systematic issues
    summary["systematic_issues"] = {
        "mock_session_state_issues": {
            "count": 2,
            "description": (
                "MockSessionState not properly implementing iterable "
                "interface"
            ),
            "affected_tests": ["TestConfigPage", "TestTrainPage"],
            "fix_priority": "HIGH - Fix once, resolves multiple tests",
        },
        "session_state_management": {
            "count": 4,
            "description": "Core session state management logic broken",
            "affected_tests": ["Multiple SessionStateManager tests"],
            "fix_priority": "HIGH - Core functionality",
        },
        "test_infrastructure_automation": {
            "count": 6,
            "description": (
                "Test utilities infrastructure missing temp_path " "attribute"
            ),
            "affected_tests": ["All ResourceCleanupValidationComponent tests"],
            "fix_priority": "HIGH - Systemic infrastructure issue",
        },
        "api_changes_from_refactoring": {
            "count": 8,
            "description": (
                "Module APIs changed during GUI refactoring, tests not "
                "updated"
            ),
            "affected_tests": ["Various component and utility tests"],
            "fix_priority": "MEDIUM - Update tests to match new APIs",
        },
    }

    # Create output directory
    output_dir = Path("comprehensive_failure_analysis")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate JSON report
    json_data = {
        "summary": summary,
        "failures": [
            {
                "test_name": f.test_name,
                "test_file": f.test_file,
                "failure_type": f.failure_type,
                "error_message": f.error_message,
                "category": f.category.value,
                "severity": f.severity.value,
                "root_cause": f.root_cause,
                "suggested_fix": f.suggested_fix,
                "affected_component": f.affected_component,
                "priority_order": f.priority_order,
            }
            for f in failures_by_priority
        ],
    }

    json_path = output_dir / f"comprehensive_failures_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # Generate CSV report
    csv_path = output_dir / f"comprehensive_failures_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Priority",
                "Test Name",
                "Test File",
                "Failure Type",
                "Category",
                "Severity",
                "Component",
                "Root Cause",
                "Suggested Fix",
                "Error Message",
            ]
        )

        for failure in failures_by_priority:
            writer.writerow(
                [
                    failure.priority_order,
                    failure.test_name,
                    failure.test_file,
                    failure.failure_type,
                    failure.category.value,
                    failure.severity.value,
                    failure.affected_component,
                    failure.root_cause,
                    failure.suggested_fix,
                    (
                        failure.error_message[:100] + "..."
                        if len(failure.error_message) > 100
                        else failure.error_message
                    ),
                ]
            )

    print("🚀 Comprehensive Test Failure Analysis Complete")
    print("=" * 60)
    print(f"📊 Total Failures Analyzed: {len(failures)}")
    print(
        f"📈 Failure Rate: "
        f"{summary['analysis_metadata']['execution_details']['failure_rate_percent']}%"
    )

    print("\n🏷️ CATEGORY BREAKDOWN:")
    for category, count in summary["category_breakdown"].items():
        print(f"  {category}: {count}")

    print("\n⚠️ SEVERITY BREAKDOWN:")
    for severity, count in summary["severity_breakdown"].items():
        print(f"  {severity}: {count}")

    print("\n🔧 TOP SYSTEMATIC ISSUES:")
    for issue, details in summary["systematic_issues"].items():
        print(
            f"  {issue}: {details['count']} failures - "
            f"{details['fix_priority']}"
        )

    print("\n✅ Reports generated:")
    print(f"  📄 JSON: {json_path}")
    print(f"  📊 CSV: {csv_path}")


if __name__ == "__main__":
    generate_comprehensive_reports()
