# Test Maintenance Procedures and Regression Prevention

**Document Version**: 1.0
**Last Updated**: July 13, 2025
**Scope**: GUI Test Infrastructure Management

## Overview

This document provides detailed procedures for maintaining test health and preventing regressions in
the CrackSeg project's GUI testing infrastructure. These procedures are designed to sustain the
improvements achieved through the systematic test failure analysis and correction process.

## Daily Test Health Monitoring

### Morning Health Check Routine

```bash
#!/bin/bash
# daily_test_health_check.sh
set -e

echo "🔍 CrackSeg Daily Test Health Check - $(date)"
echo "=================================================="

# Activate environment
conda activate crackseg

# 1. Quick GUI test suite execution
echo "📋 Executing core GUI test suite..."
pytest tests/unit/gui/pages/ tests/integration/gui/ -v --tb=short --maxfail=5

# 2. Coverage monitoring
echo "📊 Checking current coverage..."
pytest tests/unit/gui/ --cov=scripts/gui --cov-report=term-missing --cov-fail-under=34

# 3. Quality gates verification
echo "🛡️ Verifying quality gates..."
black --check .
ruff check .
basedpyright .

echo "✅ Daily health check completed successfully"
```

### Health Check Metrics Tracking

Create a daily metrics file to track progress:

```bash
# Store daily metrics
echo "$(date),$(pytest tests/unit/gui/ --cov=scripts/gui --cov-report=term | grep TOTAL | awk '{print $4}')" >> test_coverage_daily.csv
```

### Failure Response Protocol

When daily health checks fail:

1. **Immediate Response** (< 15 minutes):
   - Identify failing test categories
   - Check if failures are new or recurring
   - Verify environment setup (conda activation, dependencies)

2. **Investigation** (< 1 hour):
   - Run failing tests individually with `-vvv` for detailed output
   - Check recent commits for changes affecting test infrastructure
   - Verify mock paths and component dependencies

3. **Resolution** (< 4 hours):
   - Apply fixes following established patterns from subtask 6.5
   - Update test documentation if new patterns emerge
   - Commit fixes with clear categorization

## Weekly Comprehensive Validation

### Full Test Suite Analysis

```bash
#!/bin/bash
# weekly_comprehensive_check.sh

echo "🔬 Weekly Comprehensive Test Analysis - $(date)"
echo "================================================"

conda activate crackseg

# 1. Full test execution with detailed reporting
pytest tests/ --tb=long --maxfail=20 > weekly_test_report_$(date +%Y%m%d).txt 2>&1

# 2. Coverage analysis with HTML report
pytest tests/ --cov=src --cov=scripts/gui --cov-report=html --cov-report=term > weekly_coverage_$(date +%Y%m%d).txt

# 3. Test categorization update
python test_failure_categorization.py --output weekly_categorization_$(date +%Y%m%d).json

# 4. Performance regression check
pytest tests/unit/gui/ --benchmark-only --benchmark-json=weekly_performance_$(date +%Y%m%d).json

echo "📊 Generating weekly summary report..."
python scripts/utils/generate_weekly_test_summary.py
```

### Weekly Review Process

1. **Test Health Assessment**:
   - Compare current vs. previous week's failure counts
   - Identify new failure patterns or recurring issues
   - Assess coverage progression toward 80% target

2. **Infrastructure Evolution**:
   - Review new test patterns introduced during the week
   - Update centralized mock libraries if needed
   - Consolidate repeated mock patterns

3. **Documentation Updates**:
   - Update maintenance procedures based on new learnings
   - Document any new failure categories discovered
   - Enhance testing guidelines with recent best practices

## Regression Prevention Protocols

### Pre-Commit Hook Configuration

Create `.githooks/pre-commit` to enforce quality standards:

```bash
#!/bin/bash
# Pre-commit hook for test regression prevention

echo "🔍 Pre-commit test validation..."

conda activate crackseg

# 1. Check if test files are modified
MODIFIED_TESTS=$(git diff --cached --name-only | grep "test_.*\.py$" || true)

if [ ! -z "$MODIFIED_TESTS" ]; then
    echo "📋 Testing modified test files..."

    # Run only modified tests
    for test_file in $MODIFIED_TESTS; do
        echo "Testing: $test_file"
        pytest "$test_file" -v
        if [ $? -ne 0 ]; then
            echo "❌ Test failed: $test_file"
            exit 1
        fi
    done
fi

# 2. Quality gates for all modified files
MODIFIED_FILES=$(git diff --cached --name-only | grep "\.py$" || true)

if [ ! -z "$MODIFIED_FILES" ]; then
    echo "🛡️ Running quality gates on modified files..."

    # Format check
    black --check $MODIFIED_FILES
    if [ $? -ne 0 ]; then
        echo "❌ Black formatting failed. Run: black $MODIFIED_FILES"
        exit 1
    fi

    # Linting
    ruff check $MODIFIED_FILES
    if [ $? -ne 0 ]; then
        echo "❌ Ruff linting failed. Run: ruff check $MODIFIED_FILES --fix"
        exit 1
    fi

    # Type checking
    basedpyright $MODIFIED_FILES
    if [ $? -ne 0 ]; then
        echo "❌ Type checking failed"
        exit 1
    fi
fi

echo "✅ Pre-commit validation passed"
```

### Mock Path Validation System

Create automated validation for mock paths:

```python
# scripts/utils/validate_mock_paths.py
import ast
import importlib
import sys
from pathlib import Path
from typing import List, Tuple

def validate_mock_paths_in_file(test_file: Path) -> List[Tuple[str, str, bool]]:
    """Validate that all mock paths in a test file point to actual functions."""
    violations = []

    with open(test_file, 'r') as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Name) and node.func.id == 'patch') or \
               (isinstance(node.func, ast.Attribute) and node.func.attr == 'patch'):
                if node.args:
                    mock_path = ast.literal_eval(node.args[0])
                    is_valid = validate_mock_path(mock_path)
                    violations.append((test_file.name, mock_path, is_valid))

    return violations

def validate_mock_path(mock_path: str) -> bool:
    """Check if a mock path points to an actual function/class."""
    try:
        module_path, attr_name = mock_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return hasattr(module, attr_name)
    except (ImportError, ValueError, AttributeError):
        return False

if __name__ == "__main__":
    test_dir = Path("tests")
    all_violations = []

    for test_file in test_dir.rglob("test_*.py"):
        violations = validate_mock_paths_in_file(test_file)
        all_violations.extend(violations)

    invalid_mocks = [(f, p, v) for f, p, v in all_violations if not v]

    if invalid_mocks:
        print("❌ Invalid mock paths detected:")
        for file, path, _ in invalid_mocks:
            print(f"  {file}: {path}")
        sys.exit(1)
    else:
        print("✅ All mock paths validated successfully")
```

### Coverage Regression Prevention

Implement coverage tracking with ratcheting:

```python
# scripts/utils/coverage_ratchet.py
import json
import sys
from pathlib import Path

def check_coverage_regression(current_coverage: float, baseline_file: str = "coverage_baseline.json") -> bool:
    """Ensure coverage doesn't regress below baseline."""
    baseline_path = Path(baseline_file)

    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)

        baseline_coverage = baseline_data.get('coverage', 0.0)

        if current_coverage < baseline_coverage - 1.0:  # Allow 1% tolerance
            print(f"❌ Coverage regression detected: {current_coverage:.1f}% < {baseline_coverage:.1f}%")
            return False

    # Update baseline if coverage improved
    if not baseline_path.exists() or current_coverage > baseline_data.get('coverage', 0.0):
        with open(baseline_path, 'w') as f:
            json.dump({
                'coverage': current_coverage,
                'date': str(datetime.now().isoformat())
            }, f, indent=2)
        print(f"✅ Coverage baseline updated to {current_coverage:.1f}%")

    return True
```

## Test Infrastructure Evolution Guidelines

### Mock Pattern Standardization

#### Centralized Mock Library

Create `tests/fixtures/gui_mocks.py` for reusable patterns:

```python
from unittest.mock import MagicMock, Mock
from typing import Any, List

class StreamlitMockFactory:
    """Factory for creating standardized Streamlit component mocks."""

    @staticmethod
    def create_columns_mock(default_count: int = 3):
        """Create a dynamic columns mock that adapts to usage."""
        def mock_columns_side_effect(num_cols: Any, **kwargs: Any) -> List[MagicMock]:
            if hasattr(num_cols, "__len__") and not isinstance(num_cols, str):
                actual_num = len(num_cols)
            elif isinstance(num_cols, int):
                actual_num = num_cols
            else:
                actual_num = default_count

            cols = []
            for i in range(actual_num):
                col = MagicMock()
                col.__enter__ = Mock(return_value=col)
                col.__exit__ = Mock(return_value=None)
                cols.append(col)
            return cols

        mock = MagicMock()
        mock.side_effect = mock_columns_side_effect
        return mock

    @staticmethod
    def create_session_state_mock():
        """Create a comprehensive session state mock."""
        class MockSessionState:
            def __init__(self):
                self._data = {}

            def __contains__(self, key: str) -> bool:
                return key in self._data

            def __iter__(self):
                return iter(self._data)

            def __getitem__(self, key: str):
                return self._data[key]

            def __setitem__(self, key: str, value: Any) -> None:
                self._data[key] = value

            def get(self, key: str, default: Any = None):
                return self._data.get(key, default)

            def keys(self):
                return self._data.keys()

            def values(self):
                return self._data.values()

            def items(self):
                return self._data.items()

        return MockSessionState()
```

#### Test Template Generation

Create templates for new test files:

```python
# scripts/utils/generate_test_template.py
def generate_gui_test_template(component_name: str, component_path: str) -> str:
    """Generate a standardized test file template for GUI components."""
    return f'''"""Unit tests for {component_name}."""

import pytest
from unittest.mock import MagicMock, Mock, patch
from typing import Any

from tests.fixtures.gui_mocks import StreamlitMockFactory
from {component_path} import {component_name}


class Test{component_name}:
    """Test suite for {component_name} functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_session = StreamlitMockFactory.create_session_state_mock()
        self.mock_columns = StreamlitMockFactory.create_columns_mock()

    def test_component_initialization(self) -> None:
        """Test component can be initialized."""
        # Basic smoke test
        component = {component_name}()
        assert component is not None

    @patch("streamlit.session_state")
    @patch("streamlit.columns")
    def test_basic_rendering(
        self, mock_columns: MagicMock, mock_session_state: MagicMock
    ) -> None:
        """Test basic component rendering."""
        mock_session_state.return_value = self.mock_session
        mock_columns.return_value = self.mock_columns.side_effect(3)

        # Test that component renders without errors
        try:
            component = {component_name}()
            component.render()
        except Exception as e:
            pytest.fail(f"Component rendering failed: {{e}}")
'''
```

### Performance Monitoring Integration

#### Test Execution Time Tracking

```python
# conftest.py addition for performance monitoring
import time
import json
from pathlib import Path

@pytest.fixture(autouse=True)
def track_test_performance(request):
    """Track test execution times for performance regression detection."""
    start_time = time.time()

    yield

    execution_time = time.time() - start_time

    # Log slow tests (>5 seconds)
    if execution_time > 5.0:
        slow_tests_file = Path("test_performance_log.json")

        if slow_tests_file.exists():
            with open(slow_tests_file, 'r') as f:
                data = json.load(f)
        else:
            data = {"slow_tests": []}

        data["slow_tests"].append({
            "test_name": request.node.name,
            "execution_time": execution_time,
            "timestamp": time.time()
        })

        with open(slow_tests_file, 'w') as f:
            json.dump(data, f, indent=2)
```

## Conclusion

These maintenance procedures and regression prevention protocols provide a comprehensive framework
for sustaining test health in the CrackSeg project. The systematic approach ensures that
improvements achieved through the test failure analysis and correction process are maintained and
continuously enhanced.

Key benefits of this framework:

- **Proactive monitoring** prevents test degradation
- **Automated validation** catches regressions early
- **Standardized patterns** reduce maintenance overhead
- **Continuous improvement** through systematic tracking

Regular application of these procedures will support the project's goal of achieving 80% test
coverage while maintaining high code quality and development velocity.

---

**Implementation Priority**: High
**Review Frequency**: Monthly
**Owner**: Development Team
**Stakeholders**: QA, DevOps, Project Management
