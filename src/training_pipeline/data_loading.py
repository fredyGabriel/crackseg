"""Shim to maintain backward compatibility. Use `crackseg.training.data_loading`."""

from crackseg.training.data_loading import load_data  # re-export

__all__ = ["load_data"]
