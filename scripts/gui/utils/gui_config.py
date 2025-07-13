"""
Configuration constants and shared settings for the CrackSeg GUI application.

This module contains page configurations, constants, and shared settings
used throughout the application.
"""

# Importar TypedDict
from typing import TypedDict


# Definir el tipo
class PageConfig(TypedDict):
    title: str
    icon: str
    description: str
    requires: list[str]


# Anotar PAGE_CONFIG
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
