"""Checkpointing functions for end-to-end pipeline testing."""

from typing import Any

import torch

from .dataclasses import TrainingRunArgs


def _save_checkpoint_with_config(
    args: TrainingRunArgs,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    is_best: bool = False,
) -> None:
    """Saves a checkpoint with training state and metrics."""
    from crackseg.utils.logging import get_logger

    logger = get_logger("E2ETestCheckpointing") if get_logger else None

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": args.model.state_dict(),
        "optimizer_state_dict": args.optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    # Save regular checkpoint
    checkpoint_path = (
        args.checkpoints_dir / f"checkpoint_epoch_{epoch}.pth.tar"
    )
    torch.save(checkpoint, checkpoint_path)
    if logger:
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Save best model if specified
    if is_best:
        best_path = args.checkpoints_dir / "model_best.pth.tar"
        torch.save(checkpoint, best_path)
        if logger:
            logger.info(f"Best model saved: {best_path}")

    # Save last checkpoint
    last_path = args.checkpoints_dir / "model_last.pth.tar"
    torch.save(checkpoint, last_path)
    if logger:
        logger.info(f"Last model saved: {last_path}")


def _load_best_checkpoint(args: TrainingRunArgs) -> str:
    """Loads the best checkpoint and returns the path."""
    from crackseg.utils.logging import get_logger

    logger = get_logger("E2ETestCheckpointing") if get_logger else None

    best_checkpoint_path = args.checkpoints_dir / "model_best.pth.tar"

    if not best_checkpoint_path.exists():
        if logger:
            logger.warning("Best checkpoint not found, using last checkpoint")
        best_checkpoint_path = args.checkpoints_dir / "model_last.pth.tar"

    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=args.device)
        args.model.load_state_dict(checkpoint["model_state_dict"])
        if logger:
            logger.info(f"Loaded checkpoint from: {best_checkpoint_path}")
        return str(best_checkpoint_path)
    else:
        if logger:
            logger.error("No checkpoint found!")
        return ""


def _finalize_and_save_results(args: Any) -> dict[str, Any]:
    """Compiles and saves final training and evaluation results."""
    import yaml

    from crackseg.utils.logging import get_logger

    logger = get_logger("E2ETestResults") if get_logger else None

    final_results_data = {
        "train_summary": {
            "loss": args.final_train_loss,
            "metrics": args.final_train_metrics,
        },
        "val_summary": {
            "loss": args.final_val_loss,
            "metrics": args.final_val_metrics,
        },
        "test_summary": {"loss": args.test_loss, "metrics": args.test_metrics},
        "training_epochs": args.epochs,
        "best_validation_metric_value": args.best_metric_val,
        "loaded_checkpoint_for_test": args.loaded_checkpoint_path,
        "experiment_directory": str(args.exp_dir),
    }

    results_path = args.metrics_dir / "final_e2e_results.yaml"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        yaml.dump(final_results_data, f, default_flow_style=False)

    if logger:
        logger.info(f"Final results saved to {results_path}")

    return final_results_data
