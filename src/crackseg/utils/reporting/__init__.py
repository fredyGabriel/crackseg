"""Public reporting shim aggregating result/report APIs.

Stable import surface:

from crackseg.utils.reporting import (
    save_evaluation_results,
    ExperimentDataSaver,
    ValidationReporter,
)
"""

from __future__ import annotations

from crackseg.evaluation.utils.results import (
    save_evaluation_results,  # re-export
)
from crackseg.utils.deployment.validation.reporting.core import (
    ValidationReporter,  # re-export
)
from crackseg.utils.experiment_saver import ExperimentDataSaver  # re-export

__all__ = [
    "save_evaluation_results",
    "ExperimentDataSaver",
    "ValidationReporter",
]
