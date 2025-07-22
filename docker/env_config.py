"""Environment configuration data models."""

from dataclasses import dataclass, field


@dataclass
class EnvironmentConfig:
    """
    Configuration container for environment-specific settings. This class
    encapsulates all environment variables with type safety, validation,
    and default values appropriate for crack segmentation testing.
    """

    # Environment Identification
    node_env: str = "local"
    crackseg_env: str = "local"
    project_name: str = "crackseg"

    # Application Configuration
    streamlit_server_headless: bool = False
    streamlit_server_port: int = 8501
    streamlit_server_address: str = "localhost"
    streamlit_browser_stats: bool = False

    # Development Features
    debug: bool = True
    log_level: str = "INFO"
    development_mode: bool = True
    hot_reload_enabled: bool = True

    # Testing Configuration
    test_browser: str = "chrome"
    test_parallel_workers: str | int = "auto"
    test_timeout: int = 300
    test_headless: bool = True
    test_debug: bool = False
    coverage_enabled: bool = True
    html_report_enabled: bool = True

    # Service Endpoints
    selenium_hub_host: str = "localhost"
    selenium_hub_port: int = 4444
    streamlit_host: str = "localhost"
    streamlit_port: int = 8501

    # Paths Configuration
    project_root: str = "/app"
    test_results_path: str = "./test-results"
    test_data_path: str = "./test-data"
    test_artifacts_path: str = "./test-artifacts"
    selenium_videos_path: str = "./selenium-videos"

    # ML/Training Configuration
    pytorch_cuda_alloc_conf: str = "max_split_size_mb:512"
    cuda_visible_devices: str = "0"
    model_cache_dir: str = "./cache/models"
    dataset_cache_dir: str = "./cache/datasets"

    # Performance Tuning
    pytest_opts: str = "--verbose --tb=short --strict-markers"
    max_browser_instances: int = 2
    browser_window_size: str = "1920,1080"
    selenium_implicit_wait: int = 10
    selenium_page_load_timeout: int = 30

    # Security (sensitive values handled separately)
    secrets: dict[str, str] = field(default_factory=dict)

    # Feature Flags
    feature_flags: dict[str, bool] = field(
        default_factory=lambda: {
            "FEATURE_ADVANCED_METRICS": True,
            "FEATURE_TENSORBOARD": True,
            "FEATURE_MODEL_COMPARISON": True,
            "FEATURE_EXPERIMENT_TRACKING": True,
        }
    )

    # Resource Constraints
    memory_limit: str = "4g"
    cpu_limit: str = "2"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration values and ensure consistency."""
        # Validate ports
        if not (1024 <= self.streamlit_server_port <= 65535):
            raise ValueError(
                f"Invalid Streamlit port: {self.streamlit_server_port}"
            )
        if not (1024 <= self.selenium_hub_port <= 65535):
            raise ValueError(
                f"Invalid Selenium hub port: {self.selenium_hub_port}"
            )

        # Validate timeout values
        if self.test_timeout <= 0:
            raise ValueError(
                f"Test timeout must be positive: {self.test_timeout}"
            )
        if self.selenium_implicit_wait < 0:
            raise ValueError(
                f"Selenium wait must be non-negative: "
                f"{self.selenium_implicit_wait}"
            )

        # Validate browser configuration
        valid_browsers = {"chrome", "firefox", "edge", "safari"}
        browsers = {b.strip().lower() for b in self.test_browser.split(",")}
        if not browsers.issubset(valid_browsers):
            invalid = browsers - valid_browsers
            raise ValueError(f"Invalid browsers: {invalid}")

        # Validate log level (case insensitive)
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log level: {self.log_level}")

        # Validate paths (basic checks)
        if not self.project_root:
            raise ValueError("Project root cannot be empty")
