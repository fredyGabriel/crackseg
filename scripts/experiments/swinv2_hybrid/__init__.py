"""
SwinV2 Hybrid Architecture Experiment Package

This package contains the complete experiment setup for the SwinV2 + ASPP + CNN
hybrid architecture with Focal Dice Loss for crack segmentation.

Modules:
    - run_swinv2_hybrid_experiment: Main experiment runner
    - test_swinv2_hybrid_setup: Setup validation script
    - swinv2_hybrid_analysis: Analysis and visualization wrapper
"""

__version__ = "1.0.0"
__author__ = "CrackSeg Team"
__description__ = (
    "SwinV2 Hybrid Architecture Experiment for Crack Segmentation"
)

from .run_swinv2_hybrid_experiment import main as run_experiment
from .swinv2_hybrid_analysis import main as run_analysis
from .test_swinv2_hybrid_setup import main as test_setup

__all__ = ["run_experiment", "test_setup", "run_analysis"]
