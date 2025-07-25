# GUI Testing Best Practices for CrackSeg

**Version**: 1.0
**Last Updated**: December 19, 2024
**Scope**: Comprehensive testing guidelines for CrackSeg GUI components

## Table of Contents

1. [Overview](#overview)
2. [Testing Philosophy](#testing-philosophy)
3. [Test Organization](#test-organization)
4. [Component Testing Patterns](#component-testing-patterns)
5. [Streamlit-Specific Testing](#streamlit-specific-testing)
6. [Coverage Requirements](#coverage-requirements)
7. [CI/CD Integration](#cicd-integration)
8. [Common Patterns and Examples](#common-patterns-and-examples)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance Guidelines](#maintenance-guidelines)

## Overview

This guide establishes testing standards for the CrackSeg GUI components built with Streamlit. It
provides practical patterns, examples, and guidelines to achieve and maintain >80% test coverage
while ensuring code quality and reliability.

### Key Principles

- **Test Real Behavior**: Focus on actual user interactions and business logic
- **Maintainable Tests**: Write tests that adapt to implementation changes
- **Comprehensive Coverage**: Target >80% coverage on critical GUI components
- **CI/CD Integration**: Automated testing with quality gates

## Testing Philosophy

### Production-First Approach

Following the project's testing standards, GUI tests should validate real behavior rather than
implementation details:

```python
# ✅ Good: Test actual user behavior
def test_device_selector_updates_session_state():
    """Test that device selection updates session state correctly."""
    with patch('streamlit.session_state', {}) as mock_state:
        device_selector = DeviceSelector()
        device_selector.render()

        # Simulate user selection
        mock_state['selected_device'] = 'cuda:0'

        assert mock_state['selected_device'] == 'cuda:0'
        assert device_selector.get_selected_device() == 'cuda:0'

# ❌ Bad: Test implementation details
def test_device_selector_has_specific_widget_count():
    """Don't test internal widget structure."""
    device_selector = DeviceSelector()
    assert len(device_selector._widgets) == 3  # Brittle internal detail
```

### Test Adaptation Strategy

When tests fail due to code changes:

1. **Analyze the failure**: Is it a bug or improved behavior?
2. **Adapt test expectations**: Update tests to match correct behavior
3. **Preserve intent**: Maintain the original test purpose
4. **Never break production**: Don't modify production code to satisfy tests

## Test Organization

### Directory Structure

```txt
tests/
├── unit/
│   └── gui/
│       ├── components/
│       │   ├── test_device_selector.py
│       │   ├── test_confirmation_dialog.py
│       │   └── test_theme_component.py
│       ├── pages/
│       │   ├── test_home_page.py
│       │   ├── test_config_page.py
│       │   └── test_train_page.py
│       └── utils/
│           ├── test_gui_config.py
│           └── test_session_state.py
├── integration/
│   └── gui/
│       ├── test_workflows.py
│       └── test_page_navigation.py
└── conftest.py
```

### File Naming Conventions

- **Unit Tests**: `test_{component_name}.py`
- **Integration Tests**: `test_{workflow_name}.py`
- **Test Classes**: `Test{ComponentName}`
- **Test Methods**: `test_{specific_behavior}`

## Component Testing Patterns

### Streamlit Component Testing

```python
import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from scripts.gui.components.device_selector import DeviceSelector

class TestDeviceSelector:
    """Test suite for DeviceSelector component."""

    @pytest.fixture
    def mock_session_state(self):
        """Provide clean session state for each test."""
        with patch('streamlit.session_state', {}) as mock_state:
            yield mock_state

    @pytest.fixture
    def device_selector(self):
        """Create DeviceSelector instance for testing."""
        return DeviceSelector()

    def test_initialization(self, device_selector):
        """Test component can be created with default values."""
        assert device_selector is not None
        assert hasattr(device_selector, 'render')

    def test_render_without_errors(self, device_selector, mock_session_state):
        """Test component renders without raising exceptions."""
        with patch('streamlit.selectbox') as mock_selectbox:
            mock_selectbox.return_value = 'cpu'

            # Should not raise any exceptions
            device_selector.render()

            # Verify selectbox was called
            mock_selectbox.assert_called_once()

    def test_device_selection_updates_state(self, device_selector, mock_session_state):
        """Test device selection updates session state correctly."""
        with patch('streamlit.selectbox') as mock_selectbox:
            mock_selectbox.return_value = 'cuda:0'

            device_selector.render()

            # Verify state was updated
            assert mock_session_state.get('selected_device') == 'cuda:0'
```

### Page Testing Pattern

```python
import pytest
from unittest.mock import patch, MagicMock
from scripts.gui.pages.home_page import HomePage

class TestHomePage:
    """Test suite for HomePage functionality."""

    @pytest.fixture
    def home_page(self):
        """Create HomePage instance for testing."""
        return HomePage()

    @pytest.fixture
    def mock_streamlit_components(self):
        """Mock all Streamlit components used by HomePage."""
        with patch.multiple(
            'streamlit',
            title=MagicMock(),
            markdown=MagicMock(),
            columns=MagicMock(return_value=[MagicMock(), MagicMock()]),
            button=MagicMock(return_value=False),
            session_state={}
        ) as mocks:
            yield mocks

    def test_render_displays_title(self, home_page, mock_streamlit_components):
        """Test that render displays the correct page title."""
        home_page.render()

        # Verify title was set
        mock_streamlit_components['title'].assert_called_once()

        # Check title content
        call_args = mock_streamlit_components['title'].call_args[0]
        assert 'CrackSeg' in call_args[0]

    def test_navigation_buttons_present(self, home_page, mock_streamlit_components):
        """Test that navigation buttons are rendered."""
        home_page.render()

        # Verify buttons were created
        assert mock_streamlit_components['button'].call_count >= 2
```

## Streamlit-Specific Testing

### Session State Testing

```python
def test_session_state_persistence():
    """Test session state persists across component renders."""
    with patch('streamlit.session_state', {}) as mock_state:
        component = MyComponent()

        # First render
        component.render()
        mock_state['test_value'] = 'initial'

        # Second render should preserve state
        component.render()
        assert mock_state['test_value'] == 'initial'
```

### Widget Interaction Testing

```python
def test_widget_interaction_flow():
    """Test complete widget interaction workflow."""
    with patch('streamlit.session_state', {}) as mock_state:
        with patch('streamlit.selectbox') as mock_selectbox:
            with patch('streamlit.button') as mock_button:

                # Setup widget returns
                mock_selectbox.return_value = 'option1'
                mock_button.return_value = True

                component = MyComponent()
                component.render()

                # Verify interaction sequence
                mock_selectbox.assert_called_once()
                mock_button.assert_called_once()

                # Verify state changes
                assert mock_state['selected_option'] == 'option1'
                assert mock_state['button_clicked'] is True
```

## Coverage Requirements

### Target Coverage Levels

- **Critical Components**: >90% coverage
- **Standard Components**: >80% coverage
- **Utility Functions**: >85% coverage
- **Integration Tests**: >70% coverage

### Coverage Measurement

```bash
# Run coverage analysis
conda activate crackseg && pytest tests/unit/gui/ tests/integration/gui/ \
                --cov=gui \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-report=xml

# View detailed coverage report
open htmlcov/index.html
```

### Coverage Exclusions

```python
# Use pragma comments for legitimate exclusions
def debug_function():  # pragma: no cover
    """Debug function not included in coverage."""
    pass

# Exclude error handling that's hard to test
try:
    risky_operation()
except SpecificError:  # pragma: no cover
    handle_error()
```

## CI/CD Integration

### Automated Testing Workflow

The project includes automated testing through GitHub Actions:

```yaml
# .github/workflows/test-reporting.yml
name: Test Coverage Reporting
on: [push, pull_request]

jobs:
  test-coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest-cov
      - name: Run tests with coverage
        run: |
          pytest tests/unit/gui/ tests/integration/gui/ \
            --cov=gui \
            --cov-report=xml \
            --cov-fail-under=80
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
```

### Quality Gates

All GUI code must pass these quality gates:

```bash
# Format code
conda activate crackseg && black gui/

# Lint code
conda activate crackseg && python -m ruff check gui/ --fix

# Type checking
conda activate crackseg && basedpyright gui/

# Run tests
conda activate crackseg && pytest tests/unit/gui/ -v
```

## Common Patterns and Examples

### Mock Configuration

```python
@pytest.fixture
def mock_config():
    """Provide mock configuration for testing."""
    return {
        'model': {
            'architecture': 'unet',
            'num_classes': 2
        },
        'training': {
            'batch_size': 16,
            'epochs': 100
        }
    }
```

### Error Handling Tests

```python
def test_handles_missing_config_gracefully():
    """Test component handles missing configuration gracefully."""
    with patch('streamlit.error') as mock_error:
        component = ConfigComponent()
        component.load_config('nonexistent.yaml')

        # Should display error message
        mock_error.assert_called_once()
        assert 'Configuration file not found' in mock_error.call_args[0][0]
```

### File Upload Testing

```python
def test_file_upload_processing():
    """Test file upload functionality."""
    mock_file = MagicMock()
    mock_file.name = 'test_config.yaml'
    mock_file.read.return_value = b'model:\n  architecture: unet'

    with patch('streamlit.file_uploader', return_value=mock_file):
        component = FileUploadComponent()
        result = component.process_upload()

        assert result is not None
        assert result['model']['architecture'] == 'unet'
```

## Troubleshooting

### Common Issues

#### Import Errors

```python
# Problem: Module not found during testing
# Solution: Add proper path setup in conftest.py

# conftest.py
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
```

#### Mock Issues

```python
# Problem: Mocks not working correctly
# Solution: Use patch.object for specific methods

# Instead of:
@patch('streamlit.selectbox')
def test_component(mock_selectbox):
    pass

# Use:
@patch.object(streamlit, 'selectbox')
def test_component(mock_selectbox):
    pass
```

#### Session State Issues

```python
# Problem: Session state not persisting
# Solution: Use proper session state mocking

@pytest.fixture
def mock_session_state():
    """Provide persistent session state mock."""
    state = {}
    with patch('streamlit.session_state', state):
        yield state
```

### Debugging Test Failures

1. **Run single test**: `pytest tests/unit/gui/test_component.py::TestComponent::test_method -v`
2. **Add debugging**: Use `pytest.set_trace()` for breakpoints
3. **Check coverage**: `pytest --cov=gui --cov-report=html`
4. **Verbose output**: Add `-v` and `-s` flags for detailed output

## Maintenance Guidelines

### Regular Maintenance Tasks

1. **Weekly**: Run full test suite and check coverage
2. **Monthly**: Review and update test patterns
3. **Quarterly**: Audit test quality and remove obsolete tests
4. **Release**: Ensure all tests pass and coverage >80%

### Test Refactoring

When refactoring tests:

1. **Preserve intent**: Keep the original test purpose
2. **Improve clarity**: Make tests more readable
3. **Reduce duplication**: Extract common patterns to fixtures
4. **Update documentation**: Keep examples current

### Performance Optimization

```python
# Use fixtures for expensive setup
@pytest.fixture(scope="session")
def expensive_setup():
    """Setup that runs once per test session."""
    return create_expensive_resource()

# Use parametrize for multiple test cases
@pytest.mark.parametrize("input_value,expected", [
    ("cpu", "CPU"),
    ("cuda:0", "CUDA:0"),
    ("cuda:1", "CUDA:1"),
])
def test_device_formatting(input_value, expected):
    assert format_device(input_value) == expected
```

## Conclusion

This guide provides the foundation for maintaining high-quality GUI tests in the CrackSeg project.
By following these patterns and practices, we ensure reliable, maintainable, and comprehensive test
coverage that supports the project's quality objectives.

### Key Takeaways

1. **Test real behavior**, not implementation details
2. **Adapt tests** to code changes, don't break production
3. **Maintain >80% coverage** on critical components
4. **Integrate with CI/CD** for automated quality assurance
5. **Follow consistent patterns** for maintainability

### Resources

- [Testing Standards](/.cursor/rules/testing-standards.mdc)
- [Development Workflow](/.cursor/rules/development-workflow.mdc)
- [GUI Test Coverage Analysis](../reports/gui_test_coverage_analysis.md)
- [CI/CD Integration Guide](ci_cd_integration_guide.md)
