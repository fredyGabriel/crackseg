"""
GUI Components for CrackSeg Application

This package contains all GUI components organized in a modular structure:

- core/: Core UI components (loading, progress, navigation)
- data/: Data-related components (file browser, upload, gallery)
- ml/: ML-specific components (device, config, tensorboard)
- ui/: UI utilities (theme, dialogs, error handling)
- deprecated/: Obsolete components (for removal)

All components follow ML project best practices with type safety,
error handling, and performance optimization.
"""

# Core components
from .core import (
    LoadingSpinner,
    OptimizedLoadingSpinner,
    OptimizedProgressBar,
    OptimizedStepBasedProgress,
    PageRouter,
    ProgressBar,
    SidebarComponent,
    StepBasedProgress,
)

# Data components
from .data import (
    FileBrowserComponent,
    FileUploadComponent,
    MetricsViewer,
    ResultsDisplay,
    ResultsGalleryComponent,
)

# ML components
from .ml import (
    ConfigEditorComponent,
    DeviceCardRenderer,
    DeviceDetector,
    DeviceInfo,
    DeviceSelectorRenderer,
    OptimizedDeviceSelector,
    TensorBoardComponent,
)

# UI utilities
from .ui import (
    AutoSaveManager,
    ConfirmationDialog,
    ConfirmationRenderer,
    ConfirmationUtils,
    ErrorConsole,
    HeaderComponent,
    LogoComponent,
    LogViewer,
    ThemeComponent,
)

__all__ = [
    # Core
    "LoadingSpinner",
    "OptimizedLoadingSpinner",
    "ProgressBar",
    "StepBasedProgress",
    "OptimizedProgressBar",
    "OptimizedStepBasedProgress",
    "PageRouter",
    "SidebarComponent",
    # Data
    "FileBrowserComponent",
    "FileUploadComponent",
    "ResultsGalleryComponent",
    "ResultsDisplay",
    "MetricsViewer",
    # ML
    "OptimizedDeviceSelector",
    "DeviceSelectorRenderer",
    "DeviceCardRenderer",
    "DeviceDetector",
    "DeviceInfo",
    "ConfigEditorComponent",
    "TensorBoardComponent",
    # UI
    "ThemeComponent",
    "LogoComponent",
    "HeaderComponent",
    "ConfirmationDialog",
    "ConfirmationRenderer",
    "ConfirmationUtils",
    "ErrorConsole",
    "AutoSaveManager",
    "LogViewer",
]
