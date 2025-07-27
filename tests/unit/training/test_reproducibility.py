# ruff: noqa: PLR2004
"""Tests for training reproducibility.

This module tests reproducibility of training with:
1. The same seed (to verify exact reproducibility)
2. Different seeds (to verify statistical similarity)

This focuses only on random number generation, not the full training process.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from crackseg.utils.core.seeds import set_random_seeds

# --- Fixtures ---


@pytest.fixture(autouse=True)
def hydra_cleanup():
    """Clean up Hydra's global state before each test."""
    GlobalHydra.instance().clear()
    yield


@pytest.fixture
def base_config():
    """Base configuration for reproducibility tests."""
    # Robust way to find project root based on this file's location
    # Assuming project root contains 'src' and 'configs'
    current_file_dir = Path(__file__).parent
    project_root = current_file_dir.parent.parent.parent  # Go up three levels
    config_dir_abs = project_root / "configs"

    # Use initialize_config_dir
    initialize_config_dir(
        config_dir=str(config_dir_abs),
        version_base=None,
        job_name="repro_test",
    )

    # Load the config using hydra.compose
    cfg = compose(config_name="base.yaml")

    # Ensure random_seed exists and set it
    if "random_seed" not in cfg:
        # Add default if missing, or raise error if required
        cfg.random_seed = 42
    else:
        cfg.random_seed = 42  # Base seed

    return cfg


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """Temporary directory for test results."""
    return tmp_path / "reproducibility_test"


def generate_random_numbers(seed: int, n_samples: int = 10) -> dict[str, Any]:
    """Generate random numbers using given seed.

    Args:
        seed: Random seed to use
        n_samples: Number of random samples to generate

    Returns:
        Dict with random values from numpy and PyTorch
    """
    # Set seeds
    set_random_seeds(seed)

    # Generate numpy random numbers
    np_rand = np.random.rand(n_samples)

    # Generate PyTorch random numbers
    torch_rand = torch.rand(n_samples)

    return {
        "seed": seed,
        "np_sum": float(np_rand.sum()),
        "torch_sum": float(torch_rand.sum().item()),
    }


def compare_runs(
    results_same_seed: list[dict[str, Any]],
    results_diff_seeds: list[dict[str, Any]],
) -> dict[str, float]:
    """Compare results from runs with same and different seeds.

    Args:
        results_same_seed: List of results from runs with the same seed
        results_diff_seeds: List of results from runs with different seeds

    Returns:
        Dict with differences between runs
    """
    # Compare runs with same seed - they should be identical
    if results_same_seed:
        same_seed_diffs = {}

        # Get first run as reference
        reference = results_same_seed[0]

        # Check if all runs with the same seed have identical results
        for i, result in enumerate(results_same_seed[1:], 1):
            np_diff = abs(reference["np_sum"] - result["np_sum"])
            torch_diff = abs(reference["torch_sum"] - result["torch_sum"])
            same_seed_diffs[f"run_{i}_np_diff"] = float(np_diff)
            same_seed_diffs[f"run_{i}_torch_diff"] = float(torch_diff)
    else:
        same_seed_diffs = {}

    # Compare runs with different seeds - they should be different
    if results_diff_seeds:
        diff_seeds_stats = {}
        # Get random number sums
        np_sums = [r["np_sum"] for r in results_diff_seeds]
        torch_sums = [r["torch_sum"] for r in results_diff_seeds]

        # Compute statistics
        diff_seeds_stats["np_sum_mean"] = float(np.mean(np_sums))
        diff_seeds_stats["np_sum_std"] = float(np.std(np_sums))
        diff_seeds_stats["torch_sum_mean"] = float(np.mean(torch_sums))
        diff_seeds_stats["torch_sum_std"] = float(np.std(torch_sums))
    else:
        diff_seeds_stats = {}

    return {**same_seed_diffs, **diff_seeds_stats}


# --- Test Cases ---


def test_exact_reproducibility(base_config: Any, results_dir: Path) -> None:
    """Test that multiple runs with the same seed produce identical results."""
    N_RUNS = 3
    results = []

    for i in range(N_RUNS):
        run_name = f"same_seed_{base_config.random_seed}_run_{i}"
        # Create a copy to avoid modifying the original
        cfg = OmegaConf.create(OmegaConf.to_container(base_config))
        seed = int(cfg.random_seed)

        # Generate random numbers with this seed
        result = generate_random_numbers(seed)
        result["run_name"] = run_name
        results.append(result)

        # Save results
        run_dir = results_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "results.json", "w") as f:
            json.dump(result, f, indent=4)

    # Compare results
    stats = compare_runs(results, [])  # Empty list for diff_seeds

    # Check that differences are very small for all metrics with same seed
    for metric_name, diff in stats.items():
        if metric_name.endswith("_diff"):
            msg = f"High difference ({diff}) detected for {metric_name}"
            assert diff < 1e-6, msg


def test_statistical_similarity(base_config: Any, results_dir: Path) -> None:
    """Test that runs with different seeds produce statistically similar
    but not identical results."""
    N_RUNS = 5
    results = []

    # Generate different seeds
    seeds = np.random.randint(1000, 10000, size=N_RUNS)

    for i, seed in enumerate(seeds):
        run_name = f"diff_seed_{seed}_run_{i}"

        # Generate random numbers with this seed
        result = generate_random_numbers(int(seed))
        result["run_name"] = run_name
        results.append(result)

        # Save results
        run_dir = results_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "results.json", "w") as f:
            json.dump(result, f, indent=4)

    # Compare results
    stats = compare_runs([], results)  # Empty list for same_seed

    # For different seeds, we expect non-zero std
    # but it shouldn't be extremely high relative to the mean
    if "np_sum_std" in stats and "np_sum_mean" in stats:
        rel_std = stats["np_sum_std"] / abs(stats["np_sum_mean"])
        assert (
            0 < rel_std < 0.3
        ), f"NumPy random variation ({rel_std}) outside expected range"

    if "torch_sum_std" in stats and "torch_sum_mean" in stats:
        # Avoid division by zero
        if stats["torch_sum_mean"] != 0:
            rel_std = stats["torch_sum_std"] / abs(stats["torch_sum_mean"])
            assert (
                0 < rel_std < 0.3
            ), f"PyTorch random variation ({rel_std}) outside expected range"
