"""
Pages package for the CrackSeg GUI application.

This package contains all the individual page modules for the application.
"""

from gui.pages.advanced_config_page import page_advanced_config
from gui.pages.architecture import page_architecture
from gui.pages.config_page import page_config
from gui.pages.home_page import page_home
from gui.pages.page_train import page_train
from gui.pages.results import page_results

__all__ = [
    "page_advanced_config",
    "page_config",
    "page_architecture",
    "page_train",
    "page_results",
    "page_home",
]
