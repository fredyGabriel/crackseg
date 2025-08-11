"""GUI configuration contracts and page registry for CrackSeg.

This module exposes a TypedDict schema (`PageConfig`) and a compatible
alias (`GUIConfig`) expected by external imports/tests. It also provides
the `PAGE_CONFIG` registry used across the GUI.
"""

from typing import TypedDict


class PageConfig(TypedDict):
    title: str
    icon: str
    description: str
    requires: list[str]


# Backward-compatibility alias expected by tests/imports
GUIConfig = PageConfig

# Page configuration registry
PAGE_CONFIG: dict[str, PageConfig] = {
    "Home": {
        "title": "CrackSeg Dashboard",
        "icon": "🏠",
        "description": "Main dashboard with project overview and stats.",
        "requires": [],
    },
    "Config": {
        "title": "Experiment Configuration",
        "icon": "📄",
        "description": "Configure model and training parameters.",
        "requires": [],
    },
    "Advanced Config": {
        "title": "Advanced Configuration Editor",
        "icon": "⚙️",
        "description": "Advanced YAML editor for configurations.",
        "requires": [],
    },
    "Architecture": {
        "title": "Model Architecture Viewer",
        "icon": "🏗️",
        "description": "Visualize the model architecture.",
        "requires": ["config_loaded"],
    },
    "Train": {
        "title": "Training & Monitoring",
        "icon": "🚀",
        "description": "Start and monitor training sessions.",
        "requires": ["config_loaded", "run_directory"],
    },
    "Results": {
        "title": "Results & Analysis",
        "icon": "📊",
        "description": "View training results and visualizations.",
        "requires": ["run_directory"],
    },
}

__all__ = ["PageConfig", "GUIConfig", "PAGE_CONFIG"]
