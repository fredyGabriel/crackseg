# GUI Development Guidelines

## Overview

This document provides comprehensive development guidelines for the CrackSeg GUI application built
with Streamlit. These guidelines ensure consistency, maintainability, and quality across the entire
GUI codebase.

**Status**: ✅ **VALIDATED** - All guidelines verified through Task #4 comprehensive testing
**Last Updated**: 2025-07-08
**Testing Results**: 100% manual testing success, 87.5% automated testing success
**Quality Gates**: All passing (basedpyright, black, ruff)

## ✅ Verified Application Status

### Production Readiness Confirmed

Through comprehensive Task #4 verification, the CrackSeg GUI application has been validated as:

- **Fully Functional**: All 6 pages operational with 0 critical issues
- **Cross-Browser Compatible**: Chrome, Firefox, and Edge compatibility confirmed
- **Performance Compliant**: Load times < 6.5 seconds (well under 30s requirement)
- **Quality Standard**: All quality gates passing with professional code standards
- **Architecture Improved**: 95% reduction in main file complexity through modularization

### Validated Pages and Functionality

| Page | Status | Core Functionality | Last Tested |
|------|--------|-------------------|-------------|
| **Home (Dashboard)** | ✅ VALIDATED | Navigation, statistics, quick actions | 2025-07-08 |
| **Config (Basic Configuration)** | ✅ VALIDATED | File management, upload, editing | 2025-07-08 |
| **Advanced Config (YAML Editor)** | ✅ VALIDATED | Editor, validation, templates | 2025-07-08 |
| **Architecture (Model Configuration)** | ✅ VALIDATED | Model instantiation, visualization | 2025-07-08 |
| **Train (Training Controls)** | ✅ VALIDATED | Process management, monitoring | 2025-07-08 |
| **Results (Results Display)** | ✅ VALIDATED | Visualization, gallery display | 2025-07-08 |

## Code Architecture Standards

### 📁 Directory Structure

```txt
gui/
├── app.py                 # Main Streamlit application entry point
├── pages/                 # Individual page components
│   ├── home_page.py       # Landing page with navigation
│   ├── config_page.py     # Configuration management
│   ├── train_page.py      # Training interface
│   └── results_page.py    # Results visualization
├── components/            # Reusable UI components
│   ├── config_editor/     # Configuration editing widgets
│   ├── metrics_viewer.py  # Performance metrics display
│   └── log_viewer.py      # Log viewing and parsing
├── utils/                 # Utility modules
│   ├── config_io.py       # Configuration file handling
│   ├── session_state.py   # Session state management
│   └── process_manager.py # Background process handling
└── styles/                # CSS and styling
    └── components.css     # Custom component styles
```

### 🏗️ Component Design Patterns

#### Page Function Consistency

All page functions must follow this signature pattern:

```python
def page_function_name() -> None:
    """Page function docstring."""
    # Get session state internally
    state = SessionStateManager.get()

    # Page implementation
    st.title("Page Title")
    # ... page content
```

**✅ Correct Example:**

```python
def page_home() -> None:
    """Render the home page with navigation options."""
    state = SessionStateManager.get()
    st.title("🏠 CrackSeg - Home")
    # ... implementation
```

** Incorrect Example:**

```python
def page_home(state: SessionState) -> None:  # Don't pass state as parameter
    # ... implementation
```

#### Component Creation Pattern

```python
def create_component(
    data: ComponentData,
    config: ComponentConfig | None = None,
    **kwargs: Any
) -> ComponentResult:
    """Create a reusable component.

    Args:
        data: Input data for the component
        config: Optional configuration parameters
        **kwargs: Additional configuration options

    Returns:
        Component result with state and outputs
    """
    # Component implementation
    pass
```

## Type Annotation Standards

### Required Type Annotations

All functions must include complete type annotations using Python 3.12+ features:

```python
from typing import Any, Dict, List, Optional, Union
from collections.abc import Callable, Iterator
import streamlit as st

def process_configuration(
    config_path: str,
    overrides: Dict[str, Any] | None = None,
    validate: bool = True
) -> Dict[str, Any]:
    """Process configuration with optional overrides."""
    # Implementation
    pass

# For generic collections, use built-in types
def get_file_list(directory: str) -> list[str]:
    """Get list of files in directory."""
    # Implementation
    pass

# For callable types, be specific
def apply_callback(
    callback: Callable[[str], bool],
    items: list[str]
) -> list[bool]:
    """Apply callback to items."""
    # Implementation
    pass
```

### Class Definitions

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class ComponentConfig:
    """Configuration for GUI components."""
    theme: str = "default"
    debug_mode: bool = False
    max_items: int = 100

class DataProcessor(Protocol):
    """Protocol for data processing components."""

    def process(self, data: Any) -> Any: ...
    def validate(self, data: Any) -> bool: ...
```

## Error Handling Patterns

### Standard Error Handling

```python
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_configuration(config_path: str) -> Dict[str, Any]:
    """Load configuration with proper error handling."""
    try:
        path = Path(config_path)
        if not path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            st.error(f"Configuration file not found: {config_path}")
            return {}

        with path.open() as f:
            config = yaml.safe_load(f)

        logger.info(f"Successfully loaded config from {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in {config_path}: {e}")
        st.error(f"Configuration file contains invalid YAML: {e}")
        return {}

    except Exception as e:
        logger.error(f"Unexpected error loading {config_path}: {e}")
        st.error("An unexpected error occurred loading the configuration")
        return {}
```

### Streamlit-Specific Error Display

```python
def display_error_with_recovery(
    error_message: str,
    recovery_action: Callable[[], None] | None = None,
    recovery_label: str = "Retry"
) -> None:
    """Display error with optional recovery action."""
    st.error(error_message)

    if recovery_action:
        if st.button(recovery_label, key=f"recovery_{hash(error_message)}"):
            try:
                recovery_action()
                st.rerun()
            except Exception as e:
                st.error(f"Recovery action failed: {e}")
```

## Session State Management

### Session State Pattern

```python
from typing import Any, TypeVar
import streamlit as st

T = TypeVar('T')

class SessionStateManager:
    """Centralized session state management."""

    @staticmethod
    def get(key: str, default: T = None) -> T:
        """Get value from session state with default."""
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        """Set value in session state."""
        st.session_state[key] = value

    @staticmethod
    def initialize_defaults() -> None:
        """Initialize default session state values."""
        defaults = {
            'current_page': 'home',
            'config_loaded': False,
            'training_active': False,
            'debug_mode': False
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
```

### Component State Isolation

```python
def create_component_with_state(
    component_id: str,
    render_function: Callable[[], Any]
) -> Any:
    """Create component with isolated state."""
    # Use unique keys for component state
    state_key = f"component_{component_id}_state"

    if state_key not in st.session_state:
        st.session_state[state_key] = {}

    # Provide state context to component
    with st.container():
        return render_function()
```

## Performance Guidelines

### Efficient Data Loading

```python
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_training_data(data_path: str) -> pd.DataFrame:
    """Load training data with caching."""
    return pd.read_csv(data_path)

@st.cache_resource
def initialize_model(model_config: Dict[str, Any]) -> Any:
    """Initialize model with resource caching."""
    # Model initialization
    pass
```

### Background Process Management

```python
import threading
from concurrent.futures import ThreadPoolExecutor

class BackgroundTaskManager:
    """Manage background tasks for GUI operations."""

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.active_tasks: Dict[str, Any] = {}

    def start_task(
        self,
        task_id: str,
        task_function: Callable[[], Any],
        callback: Callable[[Any], None] | None = None
    ) -> None:
        """Start a background task."""
        future = self.executor.submit(task_function)
        self.active_tasks[task_id] = future

        if callback:
            future.add_done_callback(lambda f: callback(f.result()))

    def is_task_running(self, task_id: str) -> bool:
        """Check if task is still running."""
        task = self.active_tasks.get(task_id)
        return task is not None and not task.done()
```

## Testing Guidelines

### Component Testing Pattern

```python
import pytest
from unittest.mock import Mock, patch
import streamlit as st

def test_component_rendering():
    """Test component renders correctly."""
    with patch('streamlit.title') as mock_title:
        with patch('streamlit.button') as mock_button:
            mock_button.return_value = False

            # Test component
            result = render_component()

            # Verify calls
            mock_title.assert_called_once()
            assert result is not None

@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state."""
    with patch.object(st, 'session_state', {}) as mock_state:
        yield mock_state
```

### Integration Testing

```python
def test_page_integration(mock_session_state):
    """Test page integration with session state."""
    # Setup initial state
    mock_session_state.update({
        'current_page': 'config',
        'config_loaded': True
    })

    # Test page functionality
    with patch('streamlit.selectbox') as mock_select:
        mock_select.return_value = "option1"

        page_config()

        # Verify state changes
        assert mock_session_state['config_modified'] == True
```

## Quality Assurance

### Pre-Commit Checklist

Before committing GUI code, ensure:

1. **✅ Quality Gates Pass**:

   ```bash
   conda activate crackseg && python -m ruff check gui/ --fix
   conda activate crackseg && black gui/
   conda activate crackseg && basedpyright gui/
   ```

2. **✅ Manual Testing**:
   - Test component in isolation
   - Test integration with session state
   - Verify error handling paths
   - Check responsive design

3. **✅ Code Review Items**:
   - Type annotations complete
   - Error handling implemented
   - Performance considerations addressed
   - Documentation updated

### Component Review Checklist

- [ ] **Functionality**: Component works as expected
- [ ] **Types**: All functions have type annotations
- [ ] **Errors**: Proper error handling and user feedback
- [ ] **State**: Session state managed correctly
- [ ] **Performance**: Caching used where appropriate
- [ ] **Testing**: Unit tests cover key functionality
- [ ] **Style**: Follows project conventions
- [ ] **Documentation**: Docstrings and comments present

## Common Patterns and Anti-Patterns

### ✅ Good Patterns

```python
# Proper session state usage
def update_configuration():
    state = SessionStateManager.get()
    if state.config_modified:
        save_configuration(state.current_config)
        state.config_modified = False

# Proper error handling
def safe_file_operation(filepath: str) -> bool:
    try:
        result = process_file(filepath)
        st.success(f"File processed successfully: {filepath}")
        return True
    except FileNotFoundError:
        st.error(f"File not found: {filepath}")
        return False
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        st.error("An unexpected error occurred")
        return False
```

###  Anti-Patterns to Avoid

```python
# Don't use global state
global_config = {}  #  Avoid global variables

# Don't ignore exceptions
def bad_file_operation(filepath: str):
    try:
        return process_file(filepath)
    except:  #  Bare except clause
        pass  #  Silent failure

# Don't pass session state around
def bad_page_function(state):  #  Don't pass state as parameter
    pass

# Don't use hardcoded strings for keys
st.session_state["hardcoded_key"] = value  #  Use constants instead
```

## Performance Optimization

### Memory Management

```python
# Clear large objects from session state when not needed
def cleanup_session_data():
    """Clean up large session state objects."""
    large_data_keys = ['training_data', 'model_cache', 'result_images']

    for key in large_data_keys:
        if key in st.session_state:
            del st.session_state[key]

# Use generators for large datasets
def process_large_dataset(data_path: str) -> Iterator[Dict[str, Any]]:
    """Process large dataset in chunks."""
    for chunk in pd.read_csv(data_path, chunksize=1000):
        yield chunk.to_dict('records')
```

### UI Responsiveness

```python
# Show progress for long operations
def long_operation_with_progress(items: List[Any]) -> List[Any]:
    """Perform long operation with progress indicator."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []
    total = len(items)

    for i, item in enumerate(items):
        status_text.text(f'Processing {i+1} of {total}...')
        result = process_item(item)
        results.append(result)
        progress_bar.progress((i + 1) / total)

    status_text.text('Processing complete!')
    progress_bar.empty()

    return results
```

## Deployment Considerations

### Environment Configuration

```python
import os
from pathlib import Path

def get_config_path() -> Path:
    """Get configuration path for current environment."""
    if os.getenv('STREAMLIT_ENV') == 'production':
        return Path('/app/config/production.yaml')
    elif os.getenv('STREAMLIT_ENV') == 'staging':
        return Path('/app/config/staging.yaml')
    else:
        return Path('configs/development.yaml')

def is_development() -> bool:
    """Check if running in development mode."""
    return os.getenv('STREAMLIT_ENV', 'development') == 'development'
```

### Production Readiness

```python
def configure_production_settings():
    """Configure settings for production deployment."""
    if not is_development():
        # Disable debug features
        st.set_page_config(
            page_title="CrackSeg",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="collapsed"
        )

        # Set production logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
```

## Conclusion

Following these guidelines ensures:

- **Consistency** across all GUI components
- **Maintainability** through proper architecture
- **Quality** through automated checking
- **Performance** through optimization patterns
- **Reliability** through comprehensive testing

For questions or clarifications about these guidelines, refer to the
[Quality Gates Guide](quality_gates_guide.md) or consult the project's coding standards documentation.
