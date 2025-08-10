"""Utilities for loading experiment data from directories.

Extracted from reporting.data_loader to reduce module size and improve modularity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config_from_dir(experiment_dir: Path) -> DictConfig:
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        return OmegaConf.create({})
    with open(config_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}
    return OmegaConf.create(config_dict)


def load_metrics_from_dir(experiment_dir: Path) -> dict[str, Any]:
    metrics_data: dict[str, Any] = {}
    metrics_dir = experiment_dir / "metrics"
    if not metrics_dir.exists():
        return metrics_data

    summary_path = metrics_dir / "complete_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                metrics_data["complete_summary"] = json.load(f)
        except Exception:
            pass

    metrics_file = metrics_dir / "metrics.jsonl"
    if metrics_file.exists():
        try:
            epoch_metrics: list[dict[str, Any]] = []
            with open(metrics_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        epoch_metrics.append(json.loads(line))
            metrics_data["epoch_metrics"] = epoch_metrics
        except Exception:
            pass

    val_metrics_file = metrics_dir / "validation_metrics.json"
    if val_metrics_file.exists():
        try:
            with open(val_metrics_file, encoding="utf-8") as f:
                metrics_data["validation_metrics"] = json.load(f)
        except Exception:
            pass

    test_metrics_file = metrics_dir / "test_metrics.json"
    if test_metrics_file.exists():
        try:
            with open(test_metrics_file, encoding="utf-8") as f:
                metrics_data["test_metrics"] = json.load(f)
        except Exception:
            pass

    return metrics_data


def load_artifacts_from_dir(experiment_dir: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}

    checkpoints_dir = experiment_dir / "checkpoints"
    if checkpoints_dir.exists():
        checkpoint_files = [
            "model_best.pth",
            "model_latest.pth",
            "optimizer_best.pth",
            "optimizer_latest.pth",
            "scheduler_best.pth",
            "scheduler_latest.pth",
        ]
        for checkpoint_file in checkpoint_files:
            checkpoint_path = checkpoints_dir / checkpoint_file
            if checkpoint_path.exists():
                artifacts[checkpoint_file] = checkpoint_path

    logs_dir = experiment_dir / "logs"
    if logs_dir.exists():
        for log_file in ["training.log", "validation.log", "test.log"]:
            log_path = logs_dir / log_file
            if log_path.exists():
                artifacts[f"log_{log_file}"] = log_path

    viz_dir = experiment_dir / "visualizations"
    if viz_dir.exists():
        viz_files = list(viz_dir.glob("*.png")) + list(viz_dir.glob("*.jpg"))
        for viz_file in viz_files:
            artifacts[f"viz_{viz_file.name}"] = viz_file

    predictions_dir = experiment_dir / "predictions"
    if predictions_dir.exists():
        pred_files = list(predictions_dir.glob("*.png")) + list(
            predictions_dir.glob("*.jpg")
        )
        for pred_file in pred_files:
            artifacts[f"pred_{pred_file.name}"] = pred_file

    return artifacts


def load_metadata_from_dir(experiment_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "experiment_dir": str(experiment_dir),
        "experiment_name": experiment_dir.name,
    }

    info_file = experiment_dir / "experiment_info.json"
    if info_file.exists():
        try:
            with open(info_file, encoding="utf-8") as f:
                metadata.update(json.load(f))
        except Exception:
            pass

    git_file = experiment_dir / "git_info.json"
    if git_file.exists():
        try:
            with open(git_file, encoding="utf-8") as f:
                metadata["git_info"] = json.load(f)
        except Exception:
            pass

    sys_file = experiment_dir / "system_info.json"
    if sys_file.exists():
        try:
            with open(sys_file, encoding="utf-8") as f:
                metadata["system_info"] = json.load(f)
        except Exception:
            pass

    config_path = experiment_dir / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
                metadata["config_summary"] = {
                    "model": config_dict.get("model", {}).get(
                        "name", "unknown"
                    ),
                    "dataset": config_dict.get("data", {}).get(
                        "dataset", "unknown"
                    ),
                    "optimizer": config_dict.get("training", {}).get(
                        "optimizer", "unknown"
                    ),
                    "epochs": config_dict.get("training", {}).get("epochs", 0),
                }
        except Exception:
            pass

    return metadata


def validate_experiment_structure(experiment_dir: Path) -> bool:
    if not (experiment_dir / "config.yaml").exists():
        return False
    for optional in [
        "metrics",
        "checkpoints",
        "logs",
        "visualizations",
        "predictions",
    ]:
        if (experiment_dir / optional).exists():
            return True
    return False


def build_experiment_summary(experiment_data: Any) -> dict[str, Any]:
    summary = {
        "experiment_id": experiment_data.experiment_id,
        "experiment_name": experiment_data.metadata.get(
            "experiment_name", "unknown"
        ),
        "config_summary": experiment_data.metadata.get("config_summary", {}),
        "metrics_summary": {},
        "artifacts_count": len(experiment_data.artifacts),
    }
    if "complete_summary" in experiment_data.metrics:
        cs = experiment_data.metrics["complete_summary"]
        summary["metrics_summary"] = {
            "best_epoch": cs.get("best_epoch", 0),
            "best_iou": cs.get("best_iou", 0.0),
            "best_f1": cs.get("best_f1", 0.0),
            "best_precision": cs.get("best_precision", 0.0),
            "best_recall": cs.get("best_recall", 0.0),
            "final_loss": cs.get("final_loss", 0.0),
            "training_time": cs.get("training_time", 0.0),
        }
    return summary
