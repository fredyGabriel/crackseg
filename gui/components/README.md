# GUI Components - CrackSeg Project

## Overview

This directory contains all GUI components for the CrackSeg application, organized
following ML project best practices with modular architecture and clear separation
of concerns.

## Directory Structure

```bash
gui/components/
├── core/                    # Core UI components
│   ├── loading/            # Loading indicators
│   ├── progress/           # Progress tracking
│   └── navigation/         # Navigation components
├── data/                   # Data-related components
│   ├── file_browser/       # File browsing and selection
│   ├── upload/             # File upload components
│   └── gallery/            # Results gallery
├── ml/                     # ML-specific components
│   ├── device/             # Device selection
│   ├── config/             # Configuration editing
│   └── tensorboard/        # TensorBoard integration
├── ui/                     # UI utilities and helpers
│   ├── theme/              # Theming and styling
│   ├── dialogs/            # Dialog components
│   └── error/              # Error handling
└── deprecated/             # Obsolete components (for removal)
```

## Component Categories

### Core Components (`core/`)

- **Loading**: Spinners and loading indicators
- **Progress**: Progress bars and step tracking
- **Navigation**: Page routing and navigation

### Data Components (`data/`)

- **File Browser**: File selection and browsing
- **Upload**: File upload with validation
- **Gallery**: Results display and management

### ML Components (`ml/`)

- **Device**: GPU/CPU selection and detection
- **Config**: Configuration editing and validation
- **TensorBoard**: Training monitoring integration

### UI Utilities (`ui/`)

- **Theme**: Branding and styling
- **Dialogs**: Confirmation and error dialogs
- **Error**: Error handling and display

## Migration Guide

### From Old Structure

- `progress_bar.py` → `core/progress/standard.py`
- `progress_bar_optimized.py` → `core/progress/optimized.py`
- `loading_spinner.py` → `core/loading/standard.py`
- `loading_spinner_optimized.py` → `core/loading/optimized.py`
- `file_browser_component.py` → `data/file_browser/main.py`
- `device_selector.py` → `ml/device/selector.py`

### Deprecated Files

- `file_browser.py` (obsolete)
- `tensorboard_component.py` (replaced by tensorboard/ subdirectory)

## Best Practices

1. **Single Responsibility**: Each component has one clear purpose
2. **Modular Design**: Components are self-contained and reusable
3. **Type Safety**: Full type annotations using Python 3.12+ features
4. **Error Handling**: Comprehensive error states and recovery
5. **Performance**: Optimized versions available for heavy operations
6. **Testing**: Each component has corresponding unit tests

## Usage Examples

```python
# Core components
from gui.components.core.loading import LoadingSpinner
from gui.components.core.progress import ProgressBar

# Data components
from gui.components.data.file_browser import FileBrowser
from gui.components.data.upload import FileUpload

# ML components
from gui.components.ml.device import DeviceSelector
from gui.components.ml.config import ConfigEditor

# UI utilities
from gui.components.ui.theme import ThemeComponent
from gui.components.ui.dialogs import ConfirmationDialog
```

## Quality Standards

- All components must pass quality gates: `black`, `ruff`, `basedpyright`
- Maximum file size: 300 lines (preferred), 400 lines (absolute max)
- Complete type annotations and docstrings
- Comprehensive error handling
- Unit tests with >80% coverage
