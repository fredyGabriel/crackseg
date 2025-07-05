"""Page Object Model package for E2E testing.

This package provides a complete Page Object Model implementation for
the CrackSeg Streamlit application, including base classes, locators,
and page-specific objects for all application pages.

Example:
    >>> from tests.e2e.pages import ConfigPage, ArchitecturePage
    >>>
    >>> config_page = ConfigPage(driver)
    >>> config_page.navigate_to_page().load_configuration_file("basic.yaml")
    >>>
    >>> arch_page = ArchitecturePage(driver)
    >>> arch_page.navigate_to_page().instantiate_model()
"""

from .advanced_config_page import AdvancedConfigPage
from .architecture_page import ArchitecturePage
from .base_page import BasePage
from .config_page import ConfigPage
from .locators import (
    AdvancedConfigPageLocators,
    ArchitecturePageLocators,
    BaseLocators,
    ConfigPageLocators,
    ResultsPageLocators,
    SidebarLocators,
    TrainPageLocators,
)
from .results_page import ResultsPage
from .train_page import TrainPage

# Export all page objects
__all__ = [
    # Base classes
    "BasePage",
    # Page objects
    "ConfigPage",
    "AdvancedConfigPage",
    "ArchitecturePage",
    "TrainPage",
    "ResultsPage",
    # Locator classes
    "BaseLocators",
    "SidebarLocators",
    "ConfigPageLocators",
    "AdvancedConfigPageLocators",
    "ArchitecturePageLocators",
    "TrainPageLocators",
    "ResultsPageLocators",
]
