"""
Results page for the CrackSeg application.

This module has been refactored into a modular structure for better
maintainability and adherence to code quality standards.

The functionality is now distributed across:
- setup_section: Setup guide for incomplete configurations
- config_section: Configuration panel for results display
- gallery_section: Results gallery with triplet visualization
- tensorboard_section: TensorBoard integration and visualization
- metrics_section: Training metrics analysis and display
- comparison_section: Model comparison tools and registry
- utils: Helper functions and utilities

For backward compatibility, the main function is imported from the
refactored module structure.
"""

# Import the main function from the refactored module
from scripts.gui.pages.results import page_results

# Export for backward compatibility
__all__ = ["page_results"]
