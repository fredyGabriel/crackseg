"""
Configuration constants and shared settings for the CrackSeg GUI application.

This module contains page configurations, constants, and shared settings
used throughout the application.
"""

# Shared page configuration
PAGE_CONFIG = {
    "Config": {
        "icon": "🔧",
        "description": "Configure model and training parameters",
        "requires": [],
    },
    "Advanced Config": {
        "icon": "⚙️",
        "description": "Advanced YAML editor with live validation",
        "requires": [],
    },
    "Architecture": {
        "icon": "🏗️",
        "description": "Visualize model architecture",
        "requires": ["config_loaded"],
    },
    "Train": {
        "icon": "🚀",
        "description": "Launch and monitor training",
        "requires": ["config_loaded", "run_directory"],
    },
    "Results": {
        "icon": "📊",
        "description": "View results and export reports",
        "requires": ["run_directory"],
    },
}
