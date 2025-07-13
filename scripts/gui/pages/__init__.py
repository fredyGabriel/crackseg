"""
Pages package for the CrackSeg GUI application.

This package contains all the individual page modules for the application.
"""

from scripts.gui.pages.advanced_config_page import page_advanced_config
from scripts.gui.pages.architecture import page_architecture
from scripts.gui.pages.config_page import page_config
from scripts.gui.pages.home_page import page_home
from scripts.gui.pages.results import page_results
from scripts.gui.pages.train_page import page_train

__all__ = [
    "page_advanced_config",
    "page_config",
    "page_architecture",
    "page_train",
    "page_results",
    "page_home",
]
