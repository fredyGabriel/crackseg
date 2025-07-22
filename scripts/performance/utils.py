"""Shared utilities for performance maintenance modules."""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Type definitions
type MaintenanceLog = list[str]


def safe_print(text: str) -> None:
    """Print text with Unicode-safe emoji handling for Windows.

    Args:
        text: Text to print, may contain emojis or Unicode characters
    """
    # Define emoji replacements for Windows compatibility
    emoji_replacements = {
        "🔍": "[CHECK]",
        "❌": "[ERROR]",
        "✅": "[SUCCESS]",
        "⚠️": "[WARNING]",
        "📊": "[METRICS]",
        "🚀": "[LAUNCH]",
        "🔧": "[CONFIG]",
        "📈": "[PROGRESS]",
        "💾": "[SAVE]",
        "🎯": "[TARGET]",
    }

    # Replace emojis for Windows compatibility
    for emoji, replacement in emoji_replacements.items():
        text = text.replace(emoji, replacement)

    # Safe print with error handling
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback to ASCII if Unicode fails
        print(text.encode("ascii", errors="replace").decode("ascii"))


def setup_logging() -> logging.Logger:
    """Configure logging for performance maintenance."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("performance_maintenance.log"),
        ],
    )
    return logging.getLogger("performance_maintenance")


def get_project_paths() -> dict[str, Path]:
    """Get standardized project paths for maintenance operations.

    Returns:
        Dictionary mapping path names to Path objects
    """
    project_root = Path(__file__).parent.parent.parent

    return {
        "project_root": project_root,
        "src_dir": project_root / "src",
        "tests_dir": project_root / "tests",
        "scripts_dir": project_root / "scripts",
        "docs_dir": project_root / "docs",
        "artifacts": project_root / "artifacts",
        "performance_dir": project_root / "tests" / "performance",
        "cleanup_dir": project_root / "tests" / "e2e" / "cleanup",
    }


def log_action(action: str, status: str = "INFO") -> None:
    """Log a maintenance action with timestamp.

    Args:
        action: Description of the action performed
        status: Log level (INFO, WARNING, ERROR)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{timestamp}] {status}: {action}"

    if status == "ERROR":
        logging.error(message)
    elif status == "WARNING":
        logging.warning(message)
    else:
        logging.info(message)
