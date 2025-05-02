"""Logging utilities for the crack segmentation project."""

# Import classes and functions from submodules
from .base import (
    BaseLogger,
    get_logger,
    NoOpLogger,
    # setup_logging, # Does not exist in base.py
    # log_system_info, # Does not exist
    # log_memory_usage, # Does not exist
    # log_gpu_memory_usage, # Does not exist
    # log_cpu_usage, # Does not exist
    # log_disk_usage, # Does not exist
    # flatten_dict, # Removed if not used/defined
    # log_metrics_dict # Removed
)
from .experiment import ExperimentLogger

# Define what gets imported with 'from src.utils.logging import *'
__all__ = [
    # Base classes and functions
    'BaseLogger',
    'get_logger',
    'NoOpLogger',
    # 'setup_logging', # Removed
    # 'log_system_info', # Removed
    # 'log_memory_usage', # Removed
    # 'log_gpu_memory_usage', # Removed
    # 'log_cpu_usage', # Removed
    # 'log_disk_usage', # Removed
    # 'flatten_dict', # Removed
    # 'log_metrics_dict', # Removed
    # Experiment specific logger
    'ExperimentLogger',
]
