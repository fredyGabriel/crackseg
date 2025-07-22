"""
TensorBoard component for CrackSeg GUI - Compatibility Layer. This
file provides backward compatibility by import ing from the new
modular tensorboard component structure. All functionality has been
refactored into specialized modules for better maintainability. For
new code, import directly from : from
scripts.gui.components.tensorboard import TensorBoardComponent Legacy
import s will continue to work: from
scripts.gui.components.tensorboard_component import
TensorBoardComponent
"""

# Import from the new modular structure
from gui.components.tensorboard import TensorBoardComponent

# Re-export for backward compatibility
__all__ = ["TensorBoardComponent"]

# Note: The original 783-line implementation has been refactored into a modular
# architecture located in scripts/gui/components/tensorboard/ directory.
# See docs/reports/tensorboard_component_refactoring_summary.md for details.
