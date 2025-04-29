import os
from dotenv import load_dotenv
from typing import Optional


def load_env(dotenv_path: Optional[str] = None) -> None:
    """Load environment variables from a .env file."""
    load_dotenv(dotenv_path)


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get an environment variable or return default if not set."""
    return os.getenv(key, default)
