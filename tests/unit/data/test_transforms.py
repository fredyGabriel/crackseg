import pytest
import numpy as np
import torch
from src.data import transforms as tr


def test_get_basic_transforms_modes():
    """Test get_basic_transforms returns a pipeline for each mode."""
    for mode in ["train", "val", "test"]:
        pipeline = tr.get_basic_transforms(mode, image_size=(32, 32))
        assert hasattr(pipeline, "__call__")
    with pytest.raises(ValueError):
        tr.get_basic_transforms("invalid", image_size=(32, 32))


def test_apply_transforms_numpy():
    """Test apply_transforms with numpy arrays as input."""
    img = np.ones((32, 32, 3), dtype=np.uint8) * 127
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    pipeline = tr.get_basic_transforms("val", image_size=(32, 32))
    result = tr.apply_transforms(img, mask, pipeline)
    assert isinstance(result, dict)
    assert "image" in result
    assert "mask" in result
    assert isinstance(result["image"], torch.Tensor)
    assert isinstance(result["mask"], torch.Tensor)
    assert result["image"].shape[1:] == result["mask"].shape


def test_apply_transforms_path(tmp_path):
    """Test apply_transforms with image/mask file paths."""
    img = np.ones((16, 16, 3), dtype=np.uint8) * 200
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"
    import cv2
    cv2.imwrite(str(img_path), img)
    cv2.imwrite(str(mask_path), mask)
    pipeline = tr.get_basic_transforms("test", image_size=(16, 16))
    result = tr.apply_transforms(str(img_path), str(mask_path), pipeline)
    assert isinstance(result["image"], torch.Tensor)
    assert isinstance(result["mask"], torch.Tensor)
    assert result["image"].shape[1:] == result["mask"].shape


def test_apply_transforms_no_transform():
    """Test apply_transforms with no pipeline returns tensors."""
    img = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (8, 8), dtype=np.uint8) * 255
    result = tr.apply_transforms(img, mask, None)
    assert isinstance(result["image"], torch.Tensor)
    assert isinstance(result["mask"], torch.Tensor)
    assert result["image"].shape[1:] == result["mask"].shape


def test_get_transforms_from_config_list():
    """Test get_transforms_from_config with a list of transforms."""
    config = [
        {"name": "Resize", "params": {"height": 8, "width": 8}},
        {
            "name": "Normalize",
            "params": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
        },
        {"name": "ToTensorV2", "params": {}}
    ]
    pipeline = tr.get_transforms_from_config(config, mode="train")
    assert hasattr(pipeline, "__call__")


def test_get_transforms_from_config_dict():
    """Test get_transforms_from_config with a dict of transforms."""
    config = {
        "Resize": {"height": 8, "width": 8},
        "Normalize": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5]
        },
        "ToTensorV2": {}
    }
    pipeline = tr.get_transforms_from_config(config, mode="val")
    assert hasattr(pipeline, "__call__")


def test_get_transforms_from_config_invalid():
    """Test get_transforms_from_config raises on invalid input."""
    with pytest.raises(ValueError):
        tr.get_transforms_from_config([{"params": {}}], mode="train")
    with pytest.raises(ValueError):
        tr.get_transforms_from_config(
            [{"name": "NonExistent", "params": {}}], mode="train"
        )
