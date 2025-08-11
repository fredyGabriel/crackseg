"""Shim to maintain backward compatibility. Use `crackseg.training.environment_setup`."""

from crackseg.training.environment_setup import setup_environment  # re-export

__all__ = ["setup_environment"]
