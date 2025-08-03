"""
GUI Utils for CrackSeg Application

This package contains all GUI utilities organized in a modular structure:

- core/: Core utilities (session, config, validation)
- ui/: UI utilities (theme, styling, dialogs)
- ml/: ML utilities (training, tensorboard, architecture)
- data/: Data utilities (parsing, export, reports)
- process/: Process utilities (manager, streaming, threading)
- deprecated/: Obsolete utilities (for removal)

All utilities follow ML project best practices with type safety,
error handling, and performance optimization.
"""

# Core utilities
from .core import (
    AutoSaveManager,
    ConfigIO,
    ErrorMessageFactory,
    ErrorState,
    GUIConfig,
    SessionState,
    SessionStateManager,
    SessionSyncManager,
)

# Data utilities
from .data import (
    DataStats,
    ExportManager,
    LogParser,
)

# ML utilities
from .ml import (
    ArchitectureViewer,
    TensorBoardManager,
    TrainingStateManager,
)

# Process utilities
from .process import (
    ProcessManager,
    StreamProcessor,
    ThreadingManager,
)

# UI utilities
from .ui import (
    ColorScheme,
    CSSGenerator,
    PerformanceOptimizer,
    SaveDialog,
    ThemeConfig,
    ThemeManager,
)

__all__ = [
    # Core
    "SessionState",
    "SessionStateManager",
    "AutoSaveManager",
    "SessionSyncManager",
    "GUIConfig",
    "ConfigIO",
    "ErrorState",
    "ErrorMessageFactory",
    # UI
    "ThemeManager",
    "ThemeConfig",
    "ColorScheme",
    "PerformanceOptimizer",
    "CSSGenerator",
    "SaveDialog",
    # ML
    "TrainingStateManager",
    "TensorBoardManager",
    "ArchitectureViewer",
    # Data
    "LogParser",
    "ExportManager",
    "DataStats",
    # Process
    "ProcessManager",
    "StreamProcessor",
    "ThreadingManager",
]
