import os

from dotenv import load_dotenv


def load_env(dotenv_path: str | None = None) -> None:
    """Load environment variables from a .env file."""
    load_dotenv(dotenv_path)


def get_env_var(key: str, default: str | None = None) -> str | None:
    """Get an environment variable or return default if not set."""
    return os.getenv(key, default)
