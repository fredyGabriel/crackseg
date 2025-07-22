"""
End-to-End (E2E) Testing Package for CrackSeg Pipeline.

This package contains comprehensive end-to-end testing utilities for the
CrackSeg training pipeline, including configuration, training, evaluation,
and checkpointing components.

Main Components:
- test_pipeline_e2e.py: Main E2E test script
- modules/: Supporting modules for different pipeline stages

Usage:
    from scripts.experiments.e2e import run_e2e_test
    from scripts.experiments.e2e.modules import config, training
"""

from .test_pipeline_e2e import run_e2e_test

__all__ = ["run_e2e_test"]
