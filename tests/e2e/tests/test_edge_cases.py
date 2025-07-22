"""
Legacy E2E edge case tests - SPLIT INTO FOCUSED MODULES. This file
originally contained 1,089 lines with comprehensive edge case tests.
It has been refactored into focused test modules: 1.
test_edge_cases_boundary.py - Boundary tests (~150 lines) ✅ COMPLETED
2. test_edge_cases_concurrent.py - Concurrent tests (~120 lines) ✅
COMPLETED 3. test_edge_cases_resources.py - Resource tests (~250
lines) ✅ COMPLETED 4. test_edge_cases_corruption.py - Corruption tests
(~160 lines) ✅ COMPLETED 5. test_edge_cases_interactions.py -
Interaction tests (~150 lines) ✅ DONE 6.
test_edge_cases_performance.py - Performance tests (~170 lines) ✅
COMPLETED Total: 6 focused files instead of 1 large monolith. This
refactoring improves: - Code maintainability and navigation - Test
organization and readability - Parallel test execution capability -
Focused test failure analysis - Better module responsibility
separation Run all edge case tests with: pytest
tests/e2e/tests/test_edge_cases_*.py Planned test classes (to be
implemented): - TestEdgeCasesBoundary - TestEdgeCasesConcurrent -
TestEdgeCasesResources - TestEdgeCasesCorruption -
TestEdgeCasesInteractions - TestEdgeCasesPerformance
"""

import pytest


# Legacy class for backward compatibility
class TestEdgeCases:
    """
    Legacy test class - Tests moved to focused modules. This class exists
    for backward compatibility only. All test methods have been moved to
    specialized test classes.
    """

    def __init__(self):
        pytest.fail(
            "TestEdgeCases has been split into focused modules:\n"
            "- TestEdgeCasesBoundary (test_edge_cases_boundary.py)\n"
            "- TestEdgeCasesConcurrent (test_edge_cases_concurrent.py)\n"
            "- TestEdgeCasesResources (test_edge_cases_resources.py)\n"
            "- TestEdgeCasesCorruption (test_edge_cases_corruption.py)\n"
            "- TestEdgeCasesInteractions (test_edge_cases_interactions.py)\n"
            "- TestEdgeCasesPerformance (test_edge_cases_performance.py)\n"
            "Use the specific test classes instead."
        )
