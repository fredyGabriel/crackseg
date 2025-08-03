"""TensorBoard component module for CrackSeg GUI.

This module provides a refactored, modular architecture for TensorBoard
integration within Streamlit applications. The component is split into
specialized sub-modules for better maintainability and testing.

Example:
    >>> from  scripts.gui.components.tensorboard  import  TensorBoardComponent
    >>> tb_component = TensorBoardComponent()
    >>> tb_component.render(log_dir=Path("outputs/logs"))
"""

from .component import TensorBoardComponent

__all__ = ["TensorBoardComponent"]
