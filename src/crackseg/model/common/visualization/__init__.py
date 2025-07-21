"""Architecture visualization module for U-Net models.

This module provides functionality to render U-Net architecture diagrams
using matplotlib (preferred) or graphviz (fallback).

Main functions:
- render_unet_architecture_diagram: Main entry point for visualization
- render_unet_architecture_matplotlib: Matplotlib-based rendering
- render_unet_architecture_graphviz: Graphviz-based rendering (legacy)

See ADR-001 for migration from graphviz to matplotlib.
"""

from .graphviz_renderer import render_unet_architecture_graphviz
from .main import render_unet_architecture_diagram
from .matplotlib_renderer import render_unet_architecture_matplotlib

__all__ = [
    "render_unet_architecture_diagram",
    "render_unet_architecture_matplotlib",
    "render_unet_architecture_graphviz",
]
