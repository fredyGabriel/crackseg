# ruff: noqa: PLR2004
"""Unit tests for the data splitting functionality."""

import math
import os
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pytest import MonkeyPatch

from crackseg.data.splitting import DatasetCreationConfig

# Add src/ to sys.path to ensure correct import
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from crackseg.data.dataset import CrackSegmentationDataset  # noqa: E402
from crackseg.data.splitting import (  # noqa: E402
    create_split_datasets,
    get_all_samples,
    split_indices,
)

# --- Fixture for temporary data directory ---


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> str:
    """Creates a temporary directory structure for testing get_all_samples."""
    data_dir = tmp_path / "test_data"
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir()

    # Create sample files
    (images_dir / "img1.png").touch()
    (masks_dir / "img1.png").touch()
    (images_dir / "img2.jpg").touch()
    (masks_dir / "img2.png").touch()
    (images_dir / "img3.tif").touch()
    # No mask for img3
    (images_dir / "img4_no_mask.jpeg").touch()
    # Mask without corresponding image
    (masks_dir / "mask_only.png").touch()
    # Image with unsupported extension
    (images_dir / "img5.txt").touch()
    (masks_dir / "img5.png").touch()  # Mask exists but image ext not supported

    return str(data_dir)


# --- Tests for split_indices ---


def test_split_indices_basic():
    """Test basic splitting with explicit ratios."""
    num_samples = 100
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    splits = split_indices(num_samples, ratios, seed=42, shuffle=True)

    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 15
    assert len(splits["test"]) == 15
    assert sum(len(v) for v in splits.values()) == num_samples
    # Check for unique indices across splits
    all_indices = splits["train"] + splits["val"] + splits["test"]
    assert len(set(all_indices)) == num_samples


def test_split_indices_infer_test():
    """Test splitting when test ratio is inferred."""
    num_samples = 100
    ratios = {"train": 0.8, "val": 0.1}  # Test should be 0.1
    splits = split_indices(num_samples, ratios, seed=42, shuffle=True)

    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10
    assert sum(len(v) for v in splits.values()) == num_samples


def test_split_indices_reproducible():
    """Test that splitting is reproducible with the same seed."""
    num_samples = 50
    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    splits1 = split_indices(num_samples, ratios, seed=123, shuffle=True)
    splits2 = split_indices(num_samples, ratios, seed=123, shuffle=True)

    assert splits1["train"] == splits2["train"]
    assert splits1["val"] == splits2["val"]
    assert splits1["test"] == splits2["test"]


def test_split_indices_different_without_seed():
    """Test that splits are different without a seed (if shuffled)."""
    num_samples = 50
    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    # Note: Relies on shuffle=True default
    splits1 = split_indices(num_samples, ratios, seed=None)
    splits2 = split_indices(num_samples, ratios, seed=None)

    # High probability they will be different, though technically could be same
    assert (
        splits1["train"] != splits2["train"]
        or splits1["val"] != splits2["val"]
        or splits1["test"] != splits2["test"]
    )


def test_split_indices_no_shuffle():
    """Test splitting without shuffling."""
    num_samples = 20
    ratios = {"train": 0.5, "val": 0.3, "test": 0.2}
    splits = split_indices(num_samples, ratios, shuffle=False)

    assert splits["train"] == list(range(10))
    assert splits["val"] == list(range(10, 16))
    assert splits["test"] == list(range(16, 20))
    assert sum(len(v) for v in splits.values()) == num_samples


def test_split_indices_invalid_ratio_sum():
    """Test error handling for ratios summing > 1.0."""
    num_samples = 100
    ratios = {"train": 0.7, "val": 0.2, "test": 0.2}  # Sums to 1.1
    with pytest.raises(ValueError, match="Sum of ratios must be close to 1.0"):
        split_indices(num_samples, ratios)


def test_split_indices_invalid_ratio_sum_infer():
    """Test error handling for ratios summing > 1.0 when inferring test."""
    num_samples = 100
    ratios = {"train": 0.8, "val": 0.3}  # Sums > 1.0
    with pytest.raises(
        ValueError, match="Sum of provided ratios cannot exceed 1.0"
    ):
        split_indices(num_samples, ratios)


def test_split_indices_invalid_keys():
    """Test error handling for invalid keys in ratios."""
    num_samples = 100
    ratios = {"training": 0.8, "validation": 0.2}
    with pytest.raises(ValueError, match="Ratio keys must be"):
        split_indices(num_samples, ratios)


def test_split_indices_invalid_values():
    """Test error handling for ratio values outside [0, 1]."""
    num_samples = 100
    ratios_neg = {"train": -0.1, "val": 0.9, "test": 0.2}
    ratios_high = {"train": 1.1, "val": -0.1}
    with pytest.raises(ValueError, match="Ratio values must be between"):
        split_indices(num_samples, ratios_neg)
    with pytest.raises(ValueError, match="Ratio values must be between"):
        split_indices(num_samples, ratios_high)


def test_split_indices_edge_case_few_samples():
    """Test splitting with very few samples."""
    num_samples = 5
    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    splits = split_indices(num_samples, ratios, seed=42)

    train_expected = math.floor(5 * 0.6)  # 3
    val_expected = math.floor(5 * 0.2)  # 1
    test_expected = 5 - train_expected - val_expected  # 1

    assert len(splits["train"]) == train_expected
    assert len(splits["val"]) == val_expected
    assert len(splits["test"]) == test_expected
    assert sum(len(v) for v in splits.values()) == num_samples


def test_split_indices_edge_case_zero_samples():
    """Test splitting with zero samples."""
    num_samples = 0
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    splits = split_indices(num_samples, ratios, seed=42)

    assert len(splits["train"]) == 0
    assert len(splits["val"]) == 0
    assert len(splits["test"]) == 0
    assert sum(len(v) for v in splits.values()) == num_samples


def test_split_indices_rounding_adjustment():
    """Test that index counts are adjusted correctly due to rounding."""
    # Example where simple floor rounding might mismatch total
    num_samples = 10
    ratios = {"train": 0.33, "val": 0.33, "test": 0.34}
    splits = split_indices(num_samples, ratios, seed=42)

    # train = floor(3.3) = 3
    # val = floor(3.3) = 3
    # test = 10 - 3 - 3 = 4
    assert len(splits["train"]) == 3
    assert len(splits["val"]) == 3
    assert len(splits["test"]) == 4
    assert sum(len(v) for v in splits.values()) == num_samples

    ratios_infer = {"train": 0.33, "val": 0.33}  # test inferred as ~0.34
    splits_infer = split_indices(num_samples, ratios_infer, seed=42)
    assert len(splits_infer["train"]) == 3
    assert len(splits_infer["val"]) == 3
    assert len(splits_infer["test"]) == 4  # Check inference & remainder logic
    assert sum(len(v) for v in splits_infer.values()) == num_samples


# --- Tests for get_all_samples ---


def test_get_all_samples_finds_pairs(temp_data_dir: str) -> None:
    """Test that get_all_samples finds correct image/mask pairs."""
    all_samples = get_all_samples(temp_data_dir)

    # Expected pairs (img1.png, img2.jpg)
    assert len(all_samples) == 2
    expected_pairs = [
        (
            str(Path(temp_data_dir) / "images" / "img1.png"),
            str(Path(temp_data_dir) / "masks" / "img1.png"),
        ),
        (
            str(Path(temp_data_dir) / "images" / "img2.jpg"),
            str(Path(temp_data_dir) / "masks" / "img2.png"),
        ),
    ]
    # Convert to sets for order-independent comparison
    assert set(all_samples) == set(expected_pairs)


def test_get_all_samples_ignores_missing_masks(temp_data_dir: str) -> None:
    """
    Test that images without corresponding masks are ignored (with
    warning).
    """
    with pytest.warns(UserWarning, match="Mask not found for image: img3.tif"):
        all_samples = get_all_samples(temp_data_dir)
    # Should only find img1 and img2 pairs
    assert len(all_samples) == 2
    assert not any("img3.tif" in pair[0] for pair in all_samples)
    assert not any("img4_no_mask.jpeg" in pair[0] for pair in all_samples)


def test_get_all_samples_ignores_missing_images(temp_data_dir: str) -> None:
    """Test that masks without corresponding images are ignored."""
    all_samples = get_all_samples(temp_data_dir)
    # Should only find img1 and img2 pairs
    assert len(all_samples) == 2
    assert not any("mask_only.png" in pair[1] for pair in all_samples)


def test_get_all_samples_ignores_unsupported_extensions(
    temp_data_dir: str,
) -> None:
    """Test that images with unsupported extensions are ignored."""
    # img5.txt exists, but is not in image_extensions
    # img5.png mask exists
    all_samples = get_all_samples(temp_data_dir)
    assert len(all_samples) == 2  # Only img1 and img2
    # Verify img5.txt was ignored
    assert not any("img5.txt" in pair[0] for pair in all_samples)


def test_get_all_samples_missing_images_dir(tmp_path: Path) -> None:
    """Test error handling when the 'images' directory is missing."""
    data_dir = tmp_path / "missing_images"
    masks_dir = data_dir / "masks"
    masks_dir.mkdir(parents=True)
    (masks_dir / "some_mask.png").touch()

    with pytest.raises(FileNotFoundError, match="Images directory not found"):
        get_all_samples(str(data_dir))


def test_get_all_samples_missing_masks_dir(tmp_path: Path) -> None:
    """Test error handling when the 'masks' directory is missing."""
    data_dir = tmp_path / "missing_masks"
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "some_image.jpg").touch()

    with pytest.raises(FileNotFoundError, match="Masks directory not found"):
        get_all_samples(str(data_dir))


def test_get_all_samples_empty_dirs(tmp_path: Path) -> None:
    """Test handling of empty 'images' and 'masks' directories."""
    data_dir = tmp_path / "empty_dirs"
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir()

    with pytest.warns(UserWarning, match="No matching image/mask pairs found"):
        all_samples = get_all_samples(str(data_dir))
    assert len(all_samples) == 0


# --- Tests for create_split_datasets ---


def test_create_split_datasets_basic(
    temp_data_dir: str, monkeypatch: MonkeyPatch
) -> None:
    """Test basic creation of split datasets."""
    # Setup temp directories for train/val/test structure
    import os

    # Create directories expected by the new function signature
    for split in ["train", "val", "test"]:
        os.makedirs(
            os.path.join(temp_data_dir, split, "images"), exist_ok=True
        )
        os.makedirs(os.path.join(temp_data_dir, split, "masks"), exist_ok=True)

    # Mock transform config for all splits
    transform_cfg_dict = {
        "train": {"Resize": {"height": 64, "width": 64}},
        "val": {"Resize": {"height": 64, "width": 64}},
        "test": {"Resize": {"height": 64, "width": 64}},
    }
    transform_cfg = OmegaConf.create(transform_cfg_dict)

    # Create actual sample files for testing
    # Minimal valid PNG file content
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00"
        b"\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0c"
        b"IDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    for split in ["train", "val"]:
        img_path = os.path.join(
            temp_data_dir, split, "images", f"img{split}.png"
        )
        mask_path = os.path.join(
            temp_data_dir, split, "masks", f"img{split}.png"
        )
        # Create empty files
        with open(img_path, "wb") as f:
            f.write(png_data)
        with open(mask_path, "wb") as f:
            f.write(png_data)

    # Call the function with real files instead of mocking
    config = DatasetCreationConfig(
        data_root=temp_data_dir,
        transform_cfg=transform_cfg,
        dataset_cls=CrackSegmentationDataset,
        seed=99,
        cache_flag=False,
    )
    datasets = create_split_datasets(config=config)

    assert isinstance(datasets, dict)
    assert "train" in datasets
    assert "val" in datasets

    assert isinstance(datasets["train"], CrackSegmentationDataset)
    assert isinstance(datasets["val"], CrackSegmentationDataset)

    # Datasets should now contain the actual files we created
    assert len(datasets["train"]) == 1
    assert len(datasets["val"]) == 1

    # Check dataset properties
    assert datasets["train"].mode == "train"
    assert datasets["val"].mode == "val"
    assert datasets["train"].seed == 99
    assert datasets["val"].in_memory_cache is False


def test_create_split_datasets_missing_cls():
    """Test error when dataset_cls is not provided."""
    transform_cfg_dict = {
        "train": {"resize": {"height": 64, "width": 64}},
        "val": {"resize": {"height": 64, "width": 64}},
        "test": {"resize": {"height": 64, "width": 64}},
    }
    transform_cfg = OmegaConf.create(transform_cfg_dict)

    with pytest.raises(ValueError, match="dataset_cls must be provided"):
        config = DatasetCreationConfig(
            data_root="dummy",
            transform_cfg=transform_cfg,
            dataset_cls=None,  # type: ignore[arg-type]
        )
        create_split_datasets(config=config)


def test_create_split_datasets_missing_transform(temp_data_dir: str) -> None:
    """Test error when transform config is missing for a split."""
    import os

    for split in ["train", "val", "test"]:
        os.makedirs(
            os.path.join(temp_data_dir, split, "images"), exist_ok=True
        )
        os.makedirs(os.path.join(temp_data_dir, split, "masks"), exist_ok=True)

    # Missing 'test' config
    transform_cfg_dict = {
        "train": {"resize": {"height": 64, "width": 64}},
        "val": {"resize": {"height": 64, "width": 64}},
    }
    transform_cfg = OmegaConf.create(transform_cfg_dict)

    with pytest.raises(ValueError, match="Transform config missing for split"):
        config = DatasetCreationConfig(
            data_root=temp_data_dir,
            transform_cfg=transform_cfg,
            dataset_cls=CrackSegmentationDataset,
        )
        create_split_datasets(config=config)


# Test warning for zero samples in a split?
# Handled inside the function itself with warnings.warn
