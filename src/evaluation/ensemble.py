from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.evaluation.loading import load_model_from_checkpoint
from src.utils.logging import get_logger
from src.utils.visualization import visualize_predictions

log = get_logger("evaluation.ensemble")


@dataclass
class TensorShapeConfig:
    num_dims_image: int
    num_channels_rgb: int
    num_dims_mask: int


def _process_ensemble_batch(
    batch: Any,
    models: list[torch.nn.Module],
    device: torch.device,
    metrics_fn_dict: dict[
        str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ],
    shape_config: TensorShapeConfig,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Processes a single batch for ensemble evaluation, returning metrics and
    tensors."""
    if isinstance(batch, dict):
        inputs, targets = batch["image"], batch["mask"]
    else:
        inputs, targets = batch

    # Shape adjustments for inputs and targets
    if (
        len(inputs.shape) == shape_config.num_dims_image
        and inputs.shape[-1] == shape_config.num_channels_rgb
    ):
        inputs = inputs.permute(0, 3, 1, 2)
    if len(targets.shape) == shape_config.num_dims_mask:
        targets = targets.unsqueeze(1)

    # Type conversion for targets
    if hasattr(targets, "long"):
        targets = targets.long() if targets.dtype != torch.float32 else targets
    else:
        targets = torch.tensor(targets, dtype=torch.float32)

    inputs = inputs.to(device)
    targets = targets.to(device)

    # Ensemble prediction (assumes models is not empty)
    # Relies on the outer torch.no_grad() context from ensemble_evaluate
    template_output = models[0](inputs)  # For shape and dtype
    ensemble_sum = torch.zeros_like(template_output)
    for model in models:
        ensemble_sum += model(inputs)

    # len(models) will be > 0 due to check in ensemble_evaluate
    ensemble_output_tensor = ensemble_sum / len(models)

    # Calculate metrics
    current_batch_metrics = {}
    for name, metric_fn in metrics_fn_dict.items():
        metric_value = metric_fn(ensemble_output_tensor, targets).item()
        current_batch_metrics[f"ensemble_{name}"] = metric_value

    return (
        current_batch_metrics,
        inputs.cpu(),
        targets.cpu(),
        ensemble_output_tensor.cpu(),
    )


def _load_ensemble_models(
    checkpoint_paths: list[str], device: torch.device
) -> list[torch.nn.Module]:
    """Loads models from checkpoint paths."""
    models = []
    log.info(f"Loading {len(checkpoint_paths)} models for ensemble.")
    for checkpoint_path in checkpoint_paths:
        try:
            log.info(f"Loading model from checkpoint: {checkpoint_path}")
            model, _ = load_model_from_checkpoint(
                checkpoint_path=checkpoint_path, device=device
            )
            models.append(model)
        except Exception as e:
            log.error(
                f"Error loading model {checkpoint_path}: {e}", exc_info=True
            )
            # Consider a strategy: re-raise, skip, or collect failures
            # For now, if a model fails, it might compromise the ensemble
            raise  # Re-raise the exception to halt if a model fails
    if not models:
        raise ValueError(
            "No models were successfully loaded for the ensemble."
        )
    return models


def _finalize_ensemble_evaluation(
    output_dir_path: Path,
    run_config_to_save: DictConfig,
    results_data: dict[str, Any],
    viz_data: dict[str, list[torch.Tensor]],
) -> None:
    """Averages metrics, visualizes predictions, and saves all results."""
    # Unpack results_data
    summed_ensemble_results = results_data["summed_metrics"]
    num_batches = results_data["num_batches"]
    checkpoint_paths = results_data["checkpoint_paths"]
    num_models = results_data["num_models"]

    # Unpack viz_data
    all_inputs_list = viz_data.get("inputs", [])
    all_targets_list = viz_data.get("targets", [])
    all_ensemble_outputs_list = viz_data.get("outputs", [])

    # Average metrics
    averaged_ensemble_results = {**summed_ensemble_results}
    if num_batches > 0:
        for key in averaged_ensemble_results:
            averaged_ensemble_results[key] /= num_batches
    else:
        for key in averaged_ensemble_results:
            averaged_ensemble_results[key] = (
                0.0  # Handle empty dataloader case
            )

    # Convert stored tensors for visualization
    if all_inputs_list:
        all_inputs_cpu = torch.cat(all_inputs_list, dim=0)
        all_targets_cpu = torch.cat(all_targets_list, dim=0)
        all_ensemble_outputs_cpu = torch.cat(all_ensemble_outputs_list, dim=0)

        ensemble_vis_dir = output_dir_path / "visualizations" / "ensemble"
        ensemble_vis_dir.mkdir(parents=True, exist_ok=True)

        visualize_predictions(
            all_inputs_cpu,
            all_targets_cpu,
            all_ensemble_outputs_cpu,
            str(ensemble_vis_dir),
            num_samples=min(5, len(all_inputs_cpu)),
        )

    # Save ensemble results (metrics)
    ensemble_metrics_dir = output_dir_path / "metrics" / "ensemble"
    ensemble_metrics_dir.mkdir(parents=True, exist_ok=True)

    ensemble_eval_payload = {
        "metrics": averaged_ensemble_results,
        "checkpoints": checkpoint_paths,
        "num_models": num_models,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    yaml_path = ensemble_metrics_dir / "ensemble_results.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(ensemble_eval_payload, f, default_flow_style=False)
    log.info(f"Ensemble evaluation metrics saved to {yaml_path}")

    # Save the ensemble configuration for reproducibility
    output_config_path = output_dir_path / "ensemble_config.yaml"
    with open(output_config_path, "w", encoding="utf-8") as f:
        yaml.dump(OmegaConf.to_container(run_config_to_save, resolve=True), f)
    log.info(f"Ensemble configuration saved to {output_config_path}")


def ensemble_evaluate(
    checkpoint_paths: list[str],
    config: DictConfig,  # Assuming device and output_dir are in config
    dataloader: DataLoader[Any],
    metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
) -> dict[str, float]:
    """
    Perform ensemble evaluation with multiple model checkpoints.

    Args:
        checkpoint_paths: List of paths to model checkpoints.
        config: Configuration (Hydra DictConfig). Expected to contain
                `device` and `output_dir` (e.g., config.device,
                config.output_dir).
        dataloader: DataLoader with evaluation data.
        metrics: Dictionary of metric functions.

    Returns:
        Dictionary of averaged ensemble evaluation results (metrics).
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoints provided for ensemble evaluation.")

    # Extract device and output_dir from config - adapt path as needed
    device = torch.device(
        config.device_str
    )  # e.g., config.setup.device or config.train.device
    output_dir = Path(config.output_dir_str)  # e.g., config.paths.output_dir

    log.info(
        f"Performing ensemble evaluation with {len(checkpoint_paths)} models."
    )
    log.info(f"Using device: {device}")
    log.info(f"Output directory: {output_dir}")

    # Load base configuration from the first model (for reference and merging)
    # This base_cfg might be different from the input `config` if `config`
    # is a specific ensemble config rather than the base model's config.
    first_model_path = Path(checkpoint_paths[0])
    # Assuming checkpoint_path is a dir.
    # If it's a file: first_model_path.parent
    base_config_path = first_model_path / ".hydra" / "config.yaml"
    if not base_config_path.exists():
        log.error(f"Base config from model not found: {base_config_path}")
        raise FileNotFoundError(
            f"Base config from model not found: {base_config_path}"
        )

    # This loaded_base_cfg is from the first model's training run.
    _ = OmegaConf.load(
        base_config_path
    )  # Loaded for val. and logging, but `config` param is used for the run
    log.info(f"Loaded base config from first model: {base_config_path}")

    # The `config` argument to this function can be an ensemble-specific
    # config.
    # If we want to log the *effective* config for the ensemble run,
    # it might be a merge of `loaded_base_cfg` and the passed `config`.
    # For now, _finalize_ensemble_evaluation will save the `config` passed to
    # ensemble_evaluate.
    # If `config` is meant to override parts of `loaded_base_cfg` for the
    # ensemble process:
    # effective_run_config = OmegaConf.merge(loaded_base_cfg, config)
    # For simplicity, let's assume `config` is the main config for this run.

    models = _load_ensemble_models(checkpoint_paths, device)

    # Initialize summed results (metrics are summed over batches first)
    # Metric names are already prefixed with "ensemble_"
    # by _process_ensemble_batch
    summed_ensemble_results = {
        f"ensemble_{name}": 0.0 for name in metrics.keys()
    }

    all_inputs_list: list[torch.Tensor] = []
    all_targets_list: list[torch.Tensor] = []
    all_ensemble_outputs_list: list[torch.Tensor] = []

    # Extract shape constants from config for _process_ensemble_batch
    # Assuming config.data is populated from configs/data/default.yaml
    tensor_shape_cfg = TensorShapeConfig(
        num_dims_image=config.data.num_dims_image,
        num_channels_rgb=config.data.num_channels_rgb,
        num_dims_mask=config.data.num_dims_mask,
    )

    # Evaluate
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            (
                current_batch_metrics,
                inputs_cpu,
                targets_cpu,
                ensemble_output_cpu,
            ) = _process_ensemble_batch(
                batch_data,
                models,
                device,
                metrics,
                tensor_shape_cfg,
            )

            for metric_key, metric_value in current_batch_metrics.items():
                # Ensure key exists, though it should from initialization
                if metric_key in summed_ensemble_results:
                    summed_ensemble_results[metric_key] += metric_value
                else:
                    log.warning(
                        f"Unexpected metric key {metric_key} from batch "
                        "processing."
                    )

            # Use num_batches_visualize from config.evaluation
            if batch_idx < config.evaluation.num_batches_visualize:
                all_inputs_list.append(inputs_cpu)
                all_targets_list.append(targets_cpu)
                all_ensemble_outputs_list.append(ensemble_output_cpu)

            if (batch_idx + 1) % 10 == 0:
                log.info(
                    f"Ensemble evaluated {batch_idx + 1}/{len(dataloader)} "
                    "batches"
                )

    # Prepare data for _finalize_ensemble_evaluation
    results_payload = {
        "summed_metrics": summed_ensemble_results,
        "num_batches": len(dataloader),
        "checkpoint_paths": checkpoint_paths,
        "num_models": len(models),
    }
    visualization_payload = {
        "inputs": all_inputs_list,
        "targets": all_targets_list,
        "outputs": all_ensemble_outputs_list,
    }

    _finalize_ensemble_evaluation(
        output_dir_path=output_dir,
        run_config_to_save=config,
        results_data=results_payload,
        viz_data=visualization_payload,
    )

    # For the return value, we need the averaged metrics
    # _finalize_ensemble_evaluation calculates them, but doesn't return them
    # directly.
    # We can recalculate here for the return, or modify _finalize to return
    # them.
    # For now, let's keep _finalize as a void function and recalculate for
    # clarity.
    final_averaged_metrics = {**summed_ensemble_results}
    if len(dataloader) > 0:
        for key in final_averaged_metrics:
            final_averaged_metrics[key] /= len(dataloader)
    else:
        for key in final_averaged_metrics:
            final_averaged_metrics[key] = 0.0

    log.info("Ensemble evaluation complete.")
    return final_averaged_metrics
