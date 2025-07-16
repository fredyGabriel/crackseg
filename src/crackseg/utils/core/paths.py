import os


def get_abs_path(path: str) -> str:
    """Return the absolute path for a given relative path."""
    return os.path.abspath(path)


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
