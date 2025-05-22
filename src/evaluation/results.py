import os
from datetime import datetime
from typing import Any

import yaml
from omegaconf import OmegaConf


def save_evaluation_results(
    results: dict[str, float],
    config: Any,
    checkpoint_path: str,
    output_dir: str,
) -> None:
    """
    Save evaluation results to file.

    Args:
        results: Dictionary of evaluation results
        config: Configuration used for evaluation
        checkpoint_path: Path to the model checkpoint
        output_dir: Directory to save results
    """
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Ensure config is OmegaConf
    if not OmegaConf.is_config(config):
        config = OmegaConf.create(config)

    # Save results as YAML
    yaml_path = os.path.join(metrics_dir, "evaluation_results.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "results": results,
                "config": OmegaConf.to_container(config),
                "checkpoint": checkpoint_path,
                "timestamp": datetime.now().isoformat(),
            },
            f,
        )

    # Save results as TXT
    txt_path = os.path.join(metrics_dir, "evaluation_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("Results:\n")
        for k, v in results.items():
            f.write(f"  {k}: {v}\n")
