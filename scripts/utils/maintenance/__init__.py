"""Maintenance utilities for CrackSeg project.

This package contains utilities for maintaining the project:
- Workspace cleanup
- Dependency updates
- Rules auditing
- Setup verification
"""

from .audit_rules_checklist import main as audit_rules
from .check_updates import main as check_updates
from .clean_workspace import main as clean_workspace
from .validate_rule_references import main as validate_rules
from .verify_setup import main as verify_setup

__all__ = [
    "audit_rules",
    "check_updates",
    "clean_workspace",
    "validate_rules",
    "verify_setup",
]
