"""
Diagnostic components for TensorBoard status display. This module
provides detailed diagnostic panels and action controls for
TensorBoard management.
"""

from .action_controls import ActionControls
from .diagnostic_panel import DiagnosticPanel

__all__ = [
    "DiagnosticPanel",
    "ActionControls",
]
