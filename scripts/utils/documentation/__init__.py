"""Documentation utilities for CrackSeg project.

This package contains utilities for managing project documentation:
- Project tree generation
- Documentation cataloging
- Report organization
"""

from .catalog_documentation import main as catalog_docs
from .generate_project_tree import main as generate_tree
from .organize_reports import main as organize_reports

__all__ = [
    "catalog_docs",
    "generate_tree",
    "organize_reports",
]
