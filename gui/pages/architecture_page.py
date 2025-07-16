"""
Architecture page for the CrackSeg application.

This module has been refactored into a modular structure for better
maintainability and adherence to code quality standards.

The functionality is now distributed across:
- config_section: Configuration file selection
- model_section: Model instantiation and device management
- visualization_section: Architecture diagram generation
- info_section: Model information and statistics display
- utils: Common utilities and helper functions
"""

# Import the main function from the refactored module
from gui.pages.architecture.page_architecture import page_architecture

# Export for backward compatibility
__all__ = ["page_architecture"]
