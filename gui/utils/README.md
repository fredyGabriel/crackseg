# GUI Utils - CrackSeg Project

## Overview

This directory contains all GUI utilities for the CrackSeg application, organized
following ML project best practices with modular architecture and clear separation
of concerns.

## Directory Structure

```bash
gui/utils/
├── core/                    # Core utility functions
│   ├── session/            # Session state management
│   ├── config/             # Configuration utilities
│   └── validation/         # Data validation utilities
├── ui/                     # UI-related utilities
│   ├── theme/              # Theme and styling
│   ├── styling/            # CSS and styling utilities
│   └── dialogs/            # Dialog and form utilities
├── ml/                     # ML-specific utilities
│   ├── training/           # Training state and monitoring
│   ├── tensorboard/        # TensorBoard integration
│   └── architecture/       # Model architecture utilities
├── data/                   # Data-related utilities
│   ├── parsing/            # Data parsing and processing
│   ├── export/             # Export and import utilities
│   └── reports/            # Report generation utilities
├── process/                # Process management utilities
│   ├── manager/            # Process lifecycle management
│   ├── streaming/          # Stream processing utilities
│   └── threading/          # Threading utilities
└── deprecated/             # Obsolete utilities (for removal)
```

## Utility Categories

### Core Utilities (`core/`)

- **Session**: Session state management and persistence
- **Config**: Configuration loading and validation
- **Validation**: Data validation and error handling

### UI Utilities (`ui/`)

- **Theme**: Theme management and color schemes
- **Styling**: CSS generation and styling utilities
- **Dialogs**: Dialog components and form utilities

### ML Utilities (`ml/`)

- **Training**: Training state monitoring and management
- **TensorBoard**: TensorBoard integration and visualization
- **Architecture**: Model architecture visualization

### Data Utilities (`data/`)

- **Parsing**: Data parsing and log processing
- **Export**: Export/import functionality
- **Reports**: Report generation and analysis

### Process Utilities (`process/`)

- **Manager**: Process lifecycle management
- **Streaming**: Stream processing and monitoring
- **Threading**: Threading utilities and synchronization

## Migration Guide

### From Old Structure

- `session_state.py` → `core/session/state.py`
- `theme.py` → `ui/theme/manager.py`
- `styling.py` → `ui/styling/css.py`
- `auto_save.py` → `core/session/auto_save.py`
- `performance_optimizer.py` → `ui/theme/optimizer.py`
- `session_sync.py` → `core/session/sync.py`
- `architecture_viewer.py` → `ml/architecture/viewer.py`
- `tb_manager.py` → `ml/tensorboard/manager.py`
- `training_state.py` → `ml/training/state.py`
- `log_parser.py` → `data/parsing/logs.py`
- `export_manager.py` → `data/export/manager.py`
- `process_manager.py` → `process/manager/main.py`

### Deprecated Files

- `streaming_examples.py` (obsoleto, ejemplos de desarrollo)
- `override_examples.py` (obsoleto, ejemplos de desarrollo)
- `demo_tensorboard.py` (obsoleto, demo de desarrollo)

## Best Practices

1. **Single Responsibility**: Each utility has one clear purpose
2. **Modular Design**: Utilities are self-contained and reusable
3. **Type Safety**: Full type annotations using Python 3.12+ features
4. **Error Handling**: Comprehensive error states and recovery
5. **Performance**: Optimized for efficiency
6. **Testing**: Each utility has corresponding unit tests

## Usage Examples

```python
# Core utilities
from gui.utils.core.session import SessionStateManager
from gui.utils.core.config import ConfigManager

# UI utilities
from gui.utils.ui.theme import ThemeManager
from gui.utils.ui.styling import CSSGenerator

# ML utilities
from gui.utils.ml.training import TrainingStateManager
from gui.utils.ml.tensorboard import TensorBoardManager

# Data utilities
from gui.utils.data.parsing import LogParser
from gui.utils.data.export import ExportManager

# Process utilities
from gui.utils.process.manager import ProcessManager
from gui.utils.process.streaming import StreamProcessor
```

## Quality Standards

- All utilities must pass quality gates: `black`, `ruff`, `basedpyright`
- Maximum file size: 300 lines (preferred), 400 lines (absolute max)
- Complete type annotations and docstrings
- Comprehensive error handling
- Unit tests with >80% coverage
