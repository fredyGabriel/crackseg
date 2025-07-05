"""Centralized element locators for Streamlit application pages.

This module provides a centralized location for all element selectors used
across different page objects, improving maintainability and reducing
duplication.
"""

from selenium.webdriver.common.by import By


class BaseLocators:
    """Common locators shared across all pages."""

    # Streamlit app structure
    STREAMLIT_APP = (By.CSS_SELECTOR, "[data-testid='stApp']")
    MAIN_CONTENT = (By.CSS_SELECTOR, ".main .block-container")
    SIDEBAR = (By.CSS_SELECTOR, "[data-testid='stSidebar']")

    # Navigation elements
    SIDEBAR_NAV_SECTION = (By.CSS_SELECTOR, "[data-testid='stSidebar'] h3")
    PAGE_TITLE = (By.CSS_SELECTOR, "h1")
    BREADCRUMBS = (By.XPATH, "//*[contains(text(), 'Navigation:')]")

    # Common Streamlit widgets
    BUTTON = (By.CSS_SELECTOR, "[data-testid='stButton'] button")
    SELECTBOX = (By.CSS_SELECTOR, "[data-testid='stSelectbox']")
    TEXT_INPUT = (By.CSS_SELECTOR, "[data-testid='stTextInput'] input")
    FILE_UPLOADER = (By.CSS_SELECTOR, "[data-testid='stFileUploader']")
    EXPANDER = (By.CSS_SELECTOR, "[data-testid='stExpander']")

    # Status indicators
    SUCCESS_MESSAGE = (
        By.CSS_SELECTOR,
        "[data-testid='stAlert'][data-baseweb='notification']",
    )
    ERROR_MESSAGE = (
        By.CSS_SELECTOR,
        "[data-testid='stAlert'][data-baseweb='notification']",
    )
    WARNING_MESSAGE = (
        By.CSS_SELECTOR,
        "[data-testid='stAlert'][data-baseweb='notification']",
    )

    # Loading indicators
    SPINNER = (By.CSS_SELECTOR, "[data-testid='stSpinner']")
    PROGRESS_BAR = (By.CSS_SELECTOR, "[data-testid='stProgress']")


class SidebarLocators:
    """Locators specific to sidebar navigation."""

    # Logo and branding
    LOGO = (By.CSS_SELECTOR, "[data-testid='stSidebar'] img")

    @staticmethod
    def nav_button(page_name: str) -> tuple[str, str]:
        """Generate navigation button locator using multiple strategies.

        This method implements a multi-strategy approach to locate navigation
        buttons in the Streamlit sidebar, providing robust fallback mechanisms
        for different rendering scenarios.

        Strategies (in order of preference):
        1. Key-based selector using Streamlit's button key attribute
        2. Data-testid based selector for Streamlit buttons
        3. Text content matching within sidebar context
        4. Aria-label based matching for accessibility

        Args:
            page_name: The visible name of the page to navigate to.

        Returns:
            A tuple containing the By strategy and a complex CSS selector
            with multiple fallback strategies.

        Example:
            >>> locator = SidebarLocators.nav_button("Config")
            >>> # Returns CSS selector with multiple strategies
        """
        # Convert page name to key format (spaces to underscores, lowercase)
        page_key = page_name.replace(" ", "_").lower()

        # Multi-strategy CSS selector with fallbacks
        selector = (
            f"[data-testid='stSidebar'] button[key*='nav_btn_{page_key}'], "
            f"[data-testid='stSidebar'] [data-testid='stButton'] "
            f"button[key*='nav_btn_{page_key}'], "
            f"[data-testid='stSidebar'] button[aria-label*='{page_name}'], "
            f"[data-testid='stSidebar'] button:has-text('{page_name}'), "
            f"[data-testid='stSidebar'] button"
        )

        return (By.CSS_SELECTOR, selector)

    @staticmethod
    def nav_button_xpath_fallback(page_name: str) -> tuple[str, str]:
        """XPath-based fallback locator for navigation buttons.

        This method provides XPath-based locating as a fallback when
        CSS selectors fail. It uses multiple XPath strategies to handle
        different DOM structures.

        Args:
            page_name: The visible name of the page to navigate to.

        Returns:
            A tuple containing By.XPATH and the XPath selector string.
        """
        return (
            By.XPATH,
            f"//div[@data-testid='stSidebar']//button"
            f"[normalize-space()='{page_name}'] | "
            f"//div[@data-testid='stSidebar']//button"
            f"[contains(normalize-space(), '{page_name}')] | "
            f"//div[@data-testid='stSidebar']//button"
            f"[.//p[normalize-space()='{page_name}']] | "
            f"//div[@data-testid='stSidebar']//button"
            f"[.//span[normalize-space()='{page_name}']] | "
            f"//div[@data-testid='stSidebar']//button"
            f"[@aria-label='{page_name}']",
        )

    @staticmethod
    def nav_button_by_key(page_name: str) -> tuple[str, str]:
        """Direct key-based locator for navigation buttons.

        This method targets buttons using their Streamlit key attribute
        directly, which is the most reliable method when available.

        Args:
            page_name: The visible name of the page to navigate to.

        Returns:
            A tuple containing the By strategy and key-based selector.
        """
        page_key = page_name.replace(" ", "_").lower()
        return (
            By.CSS_SELECTOR,
            f"button[key='nav_btn_{page_key}'], "
            f"[data-testid='stButton'] button[key='nav_btn_{page_key}']",
        )

    # Status indicators
    CONFIG_STATUS = (By.XPATH, "//*[contains(text(), 'Configuration:')]")
    RUN_DIR_STATUS = (By.XPATH, "//*[contains(text(), 'Run Directory:')]")


class ConfigPageLocators:
    """Locators specific to the Configuration page."""

    # Page elements
    PAGE_TITLE = (By.XPATH, "//h1[contains(text(), 'Configuration Manager')]")

    # Configuration selection
    CONFIG_FILE_SELECTOR = (By.CSS_SELECTOR, "[data-testid='stSelectbox']")
    CONFIG_UPLOAD = (By.CSS_SELECTOR, "[data-testid='stFileUploader']")
    LOAD_CONFIG_BUTTON = (
        By.XPATH,
        "//button[contains(text(), 'Load Configuration')]",
    )

    # Configuration display
    CONFIG_VIEWER = (By.CSS_SELECTOR, "[data-testid='stExpander']")
    CONFIG_CONTENT = (By.CSS_SELECTOR, "code, pre")

    # Actions
    SAVE_CONFIG_BUTTON = (By.XPATH, "//button[contains(text(), 'Save')]")
    VALIDATE_CONFIG_BUTTON = (
        By.XPATH,
        "//button[contains(text(), 'Validate')]",
    )

    @staticmethod
    def selectbox_option(option_text: str) -> tuple[str, str]:
        """Locator for a specific selectbox option by its text."""
        return (By.XPATH, f"//li[text()='{option_text}']")


class ArchitecturePageLocators:
    """Locators specific to the Architecture page."""

    # Page elements
    PAGE_TITLE = (By.XPATH, "//h1[contains(text(), 'Model Architecture')]")

    # Model instantiation
    INSTANTIATE_MODEL_BUTTON = (
        By.XPATH,
        "//button[contains(text(), 'Instantiate Model')]",
    )
    MODEL_SUMMARY = (By.CSS_SELECTOR, "[data-testid='stText']")

    # Architecture visualization
    ARCHITECTURE_DIAGRAM = (By.CSS_SELECTOR, "svg, img")
    DOWNLOAD_DIAGRAM_BUTTON = (
        By.XPATH,
        "//button[contains(text(), 'Download')]",
    )

    # Model information
    MODEL_INFO_EXPANDER = (
        By.XPATH,
        "//div[contains(text(), 'Model Information')]",
    )
    PARAMETER_COUNT = (By.XPATH, "//*[contains(text(), 'parameters')]")


class TrainPageLocators:
    """Locators specific to the Training page."""

    # Page elements
    PAGE_TITLE = (By.XPATH, "//h1[contains(text(), 'Training')]")

    # Training controls
    START_TRAINING_BUTTON = (
        By.XPATH,
        "//button[contains(text(), 'Start Training')]",
    )
    STOP_TRAINING_BUTTON = (
        By.XPATH,
        "//button[contains(text(), 'Stop Training')]",
    )
    RESUME_TRAINING_BUTTON = (
        By.XPATH,
        "//button[contains(text(), 'Resume Training')]",
    )

    # Training status
    TRAINING_STATUS = (By.CSS_SELECTOR, "[data-testid='stAlert']")
    PROGRESS_BAR = (By.CSS_SELECTOR, "[data-testid='stProgress']")
    EPOCH_INFO = (By.XPATH, "//*[contains(text(), 'Epoch')]")

    # Metrics display
    METRICS_SECTION = (By.CSS_SELECTOR, "[data-testid='stMetric']")
    LOSS_METRIC = (By.XPATH, "//*[contains(text(), 'Loss')]")
    ACCURACY_METRIC = (By.XPATH, "//*[contains(text(), 'Accuracy')]")


class ResultsPageLocators:
    """Locators specific to the Results page."""

    # Page elements
    PAGE_TITLE = (By.XPATH, "//h1[contains(text(), 'Results')]")

    # Tabs
    GALLERY_TAB = (By.XPATH, "//div[contains(text(), 'Results Gallery')]")
    TENSORBOARD_TAB = (By.XPATH, "//div[contains(text(), 'TensorBoard')]")
    METRICS_TAB = (By.XPATH, "//div[contains(text(), 'Metrics Analysis')]")
    COMPARISON_TAB = (By.XPATH, "//div[contains(text(), 'Model Comparison')]")

    # Gallery elements
    RESULTS_GALLERY = (By.CSS_SELECTOR, "[data-testid='stImage']")
    IMAGE_TRIPLET = (By.CSS_SELECTOR, ".image-triplet")

    # Export controls
    EXPORT_BUTTON = (By.XPATH, "//button[contains(text(), 'Export')]")
    DOWNLOAD_RESULTS_BUTTON = (
        By.XPATH,
        "//button[contains(text(), 'Download Results')]",
    )


class AdvancedConfigPageLocators:
    """Locators specific to the Advanced Configuration page."""

    # Page elements
    PAGE_TITLE = (By.XPATH, "//h1[contains(text(), 'Advanced')]")

    # YAML editor
    YAML_EDITOR = (By.CSS_SELECTOR, "textarea, .ace_editor")
    SAVE_YAML_BUTTON = (By.XPATH, "//button[contains(text(), 'Save YAML')]")
    VALIDATE_YAML_BUTTON = (
        By.XPATH,
        "//button[contains(text(), 'Validate YAML')]",
    )

    # Validation feedback
    VALIDATION_SUCCESS = (By.CSS_SELECTOR, "[data-testid='stSuccess']")
    VALIDATION_ERROR = (By.CSS_SELECTOR, "[data-testid='stError']")

    # Template controls
    TEMPLATE_SELECTOR = (By.CSS_SELECTOR, "[data-testid='stSelectbox']")
    LOAD_TEMPLATE_BUTTON = (
        By.XPATH,
        "//button[contains(text(), 'Load Template')]",
    )
