"""
Configuration constants and shared settings for the CrackSeg GUI application.

This module contains page configurations, constants, and shared settings
used throughout the application.
"""

# Shared page configuration
PAGE_CONFIG = {
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
        "requires": ["config_loaded"],
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
