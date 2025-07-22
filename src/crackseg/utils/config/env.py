import os

try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
except ImportError:
    # Graceful fallback when python-dotenv is not available
    def load_dotenv(dotenv_path: str | None = None, **kwargs) -> bool:
        """Fallback when dotenv is not available."""
        return False


def load_env(dotenv_path: str | None = None) -> bool:
    """Load environment variables from a .env file."""
    return load_dotenv(dotenv_path)  # type: ignore[misc]


def get_env_var(key: str, default: str | None = None) -> str | None:
    """Get an environment variable or return default if not set."""
    return os.getenv(key, default)
