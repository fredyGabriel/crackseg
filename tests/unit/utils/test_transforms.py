"""Unit tests for the transforms module."""

import albumentations as A
import numpy as np
import pytest
import torch
from PIL import Image

from src.data.transforms import get_basic_transforms, apply_transforms


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Create a sample binary mask for testing."""
    return np.random.randint(0, 2, (64, 64), dtype=np.uint8)


@pytest.fixture
def temp_image_file(tmp_path, sample_image):
    """Create a temporary image file for testing."""
    img_path = tmp_path / "test_image.png"
    Image.fromarray(sample_image).save(img_path)
    return str(img_path)


@pytest.fixture
def temp_mask_file(tmp_path, sample_mask):
    """Create a temporary mask file for testing."""
    mask_path = tmp_path / "test_mask.png"
    Image.fromarray(sample_mask).save(mask_path)
    return str(mask_path)


def test_get_basic_transforms_invalid_mode():
    """Test that get_basic_transforms raises ValueError for invalid mode."""
    with pytest.raises(ValueError, match="Invalid mode"):
        get_basic_transforms(mode="invalid")


def test_get_basic_transforms_default_params():
    """Test get_basic_transforms with default parameters."""
    transform = get_basic_transforms(mode="val")
    assert isinstance(transform, A.Compose)

    transform_names = [type(t).__name__ for t in transform.transforms]
    assert "Resize" in transform_names
    assert "Normalize" in transform_names
    assert "ToTensorV2" in transform_names


@pytest.mark.parametrize("mode", ["train", "val", "test"])
def test_get_basic_transforms_all_modes(mode):
    """Test get_basic_transforms for all valid modes."""
    transform = get_basic_transforms(mode=mode)
    transform_names = [type(t).__name__ for t in transform.transforms]

    # Common transforms (excluding Resize for train mode)
    if mode != "train":
        assert "Resize" in transform_names
    assert "Normalize" in transform_names
    assert "ToTensorV2" in transform_names

    # Training mode should have additional augmentations
    if mode == "train":
        assert "Resize" not in transform_names  # Replaced by RandomSizedCrop
        assert "RandomSizedCrop" in transform_names
        assert "HueSaturationValue" in transform_names
        assert "HorizontalFlip" in transform_names
        assert "VerticalFlip" in transform_names
        assert "RandomRotate90" in transform_names
        assert "RandomBrightnessContrast" in transform_names
        assert "GaussNoise" in transform_names
    else:
        assert "RandomSizedCrop" not in transform_names
        assert "HueSaturationValue" not in transform_names
        assert "HorizontalFlip" not in transform_names
        assert "VerticalFlip" not in transform_names


def test_get_basic_transforms_custom_size():
    """Test get_basic_transforms with custom image size."""
    custom_size = (256, 256)
    transform = get_basic_transforms(mode="val", image_size=custom_size)

    resize_transform = next(
        t for t in transform.transforms if isinstance(t, A.Resize)
    )
    assert resize_transform.height == custom_size[0]
    assert resize_transform.width == custom_size[1]


def test_get_basic_transforms_custom_normalization():
    """Test get_basic_transforms with custom normalization parameters."""
    custom_mean = (0.485, 0.456, 0.406)  # ImageNet mean values
    custom_std = (0.229, 0.224, 0.225)   # ImageNet std values
    transform = get_basic_transforms(
        mode="val",
        mean=custom_mean,
        std=custom_std
    )

    normalize_transform = next(
        t for t in transform.transforms if isinstance(t, A.Normalize)
    )
    assert normalize_transform.mean == custom_mean
    assert normalize_transform.std == custom_std


def test_apply_transforms_with_arrays(sample_image, sample_mask):
    """Test apply_transforms with numpy array inputs."""
    transforms = get_basic_transforms(mode="train")
    result = apply_transforms(sample_image, mask=sample_mask,
                              transforms=transforms)

    assert "image" in result
    assert "mask" in result
    assert isinstance(result["image"], torch.Tensor)
    assert isinstance(result["mask"], torch.Tensor)
    assert result["image"].shape[-2:] == (512, 512)
    assert result["mask"].shape[-2:] == (512, 512)


def test_apply_transforms_with_files(temp_image_file, temp_mask_file):
    """Test apply_transforms with file path inputs."""
    transforms = get_basic_transforms(mode="val")
    result = apply_transforms(
        temp_image_file,
        mask=temp_mask_file,
        transforms=transforms
    )

    assert "image" in result
    assert "mask" in result
    assert isinstance(result["image"], torch.Tensor)
    assert isinstance(result["mask"], torch.Tensor)


def test_apply_transforms_without_mask(sample_image):
    """Test apply_transforms with image only (no mask)."""
    transforms = get_basic_transforms(mode="test")
    result = apply_transforms(sample_image, transforms=transforms)

    assert "image" in result
    assert isinstance(result["image"], torch.Tensor)
    assert result["image"].shape[-2:] == (512, 512)


def test_apply_transforms_deterministic():
    """Test that transformations are deterministic in val/test modes."""
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    transforms = get_basic_transforms(mode="val")

    result1 = apply_transforms(image, transforms=transforms)
    result2 = apply_transforms(image, transforms=transforms)

    torch.testing.assert_close(result1["image"], result2["image"])


def test_apply_transforms_augmentation_variability():
    """Test that training mode transformations produce variable results."""
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    transforms = get_basic_transforms(mode="train")

    results = [
        apply_transforms(image, transforms=transforms)["image"]
        for _ in range(5)
    ]

    assert any(
        not torch.allclose(results[0], result)
        for result in results[1:]
    )


def test_transforms_tensor_output(sample_image, sample_mask):
    """Test that transformations output tensors with correct shape and type."""
    transforms = get_basic_transforms(mode="val")
    result = apply_transforms(sample_image, mask=sample_mask,
                              transforms=transforms)

    assert result["image"].dtype == torch.float32
    assert result["mask"].dtype == torch.float32
    assert len(result["image"].shape) == 3  # C,H,W
    assert len(result["mask"].shape) == 2  # H,W


def test_transforms_normalization_range(sample_image):
    """Test that normalization produces values in expected range."""
    transforms = get_basic_transforms(mode="val")
    result = apply_transforms(sample_image, transforms=transforms)

    min_val = result["image"].min().item()
    max_val = result["image"].max().item()
    # Rango ligeramente más amplio para tolerancia
    assert -2.15 <= min_val <= 2.65
    assert -2.15 <= max_val <= 2.65


def test_transforms_mask_values(sample_image, sample_mask):
    """Test that mask values remain binary after transformations."""
    transforms = get_basic_transforms(mode="train")
    result = apply_transforms(sample_image, mask=sample_mask,
                              transforms=transforms)

    # Check that mask values remain binary (or very close to it)
    unique_values = torch.unique(result["mask"])
    # Allow for small floating point tolerances
    is_close_to_0 = torch.isclose(unique_values, torch.tensor(0.0))
    is_close_to_1 = torch.isclose(unique_values, torch.tensor(1.0))
    assert torch.all(is_close_to_0 | is_close_to_1)


def test_transforms_channel_order(sample_image):
    """Test that image tensor has correct channel order (C,H,W)."""
    transforms = get_basic_transforms(mode="val")
    result = apply_transforms(sample_image, transforms=transforms)

    # Check channel order and dimensions
    assert result["image"].shape[0] == 3  # RGB channels first
    assert len(result["image"].shape) == 3  # C,H,W format


def test_transforms_mask_shape_consistency(sample_image, sample_mask):
    """Test that mask shape matches image spatial dimensions."""
    transforms = get_basic_transforms(mode="val")
    result = apply_transforms(
        sample_image,
        mask=sample_mask,
        transforms=transforms
    )

    # Verificar que las dimensiones espaciales coinciden
    assert result["image"].shape[-2:] == result["mask"].shape
    assert len(result["mask"].shape) == 2  # H,W format
