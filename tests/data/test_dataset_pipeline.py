"""Test for the dataset pipeline and transformations."""

import numpy as np
import cv2
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf

from src.data import create_crackseg_dataset


@pytest.fixture
def test_data_dir():
    """Create and return a temporary directory with test images."""
    # Create test directory in tests/data/
    test_dir = Path("tests/data/test_images")
    test_dir.mkdir(exist_ok=True)

    # Create dummy image and mask
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48] = 255  # White square in center
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 255  # Smaller white square in center

    # Save files
    img_path = test_dir / "test_img.png"
    mask_path = test_dir / "test_mask.png"
    cv2.imwrite(str(img_path), img)
    cv2.imwrite(str(mask_path), mask)

    yield test_dir

    # Cleanup
    for f in test_dir.glob("*.png"):
        f.unlink()
    test_dir.rmdir()


def test_dataset_pipeline(test_data_dir):
    """Test the dataset creation and sample loading."""
    # Use test images
    samples_list = [
        (str(test_data_dir / "test_img.png"),
         str(test_data_dir / "test_mask.png")),
    ]

    # Cargar y modificar configuraciones para pruebas
    data_cfg = OmegaConf.create({
        "data_root": str(test_data_dir),
        "image_size": [64, 64],  # Mismo tamaño que las imágenes de prueba
        "batch_size": 1,
        "num_workers": 0,
        "in_memory_cache": False,
        "train_split": 0.7,
        "val_split": 0.15,
        "test_split": 0.15
    })

    transform_cfg = OmegaConf.create({
        "resize": {
            "enabled": True,
            "height": 64,
            "width": 64
        },
        "normalize": {
            "enabled": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "train": {
            "random_crop": {"enabled": False},
            "horizontal_flip": {"enabled": False},
            "vertical_flip": {"enabled": False},
            "rotate": {"enabled": False},
            "color_jitter": {"enabled": False}
        }
    })

    # Crear dataset para modo 'train'
    dataset = create_crackseg_dataset(
        data_cfg=data_cfg,
        transform_cfg=transform_cfg,
        mode="train",
        samples_list=samples_list
    )

    # Verificaciones básicas
    assert len(dataset) == len(samples_list), \
        "Dataset length should match samples list length"

    # Verificar una muestra
    sample = dataset[0]
    assert isinstance(sample, dict), "Sample should be a dictionary"
    assert "image" in sample, "Sample should contain 'image'"
    assert "mask" in sample, "Sample should contain 'mask'"
    assert isinstance(sample["image"], torch.Tensor), \
        "Image should be a torch.Tensor"
    assert isinstance(sample["mask"], torch.Tensor), \
        "Mask should be a torch.Tensor"

    # Verificar dimensiones
    assert len(sample["image"].shape) == 3, \
        "Image should have 3 dimensions (C, H, W)"
    assert sample["image"].shape[0] == 3, \
        "Image should have 3 channels"
    assert len(sample["mask"].shape) == 2, \
        "Mask should have 2 dimensions (H, W)"
    assert sample["image"].shape[1:] == sample["mask"].shape, \
        "Image and mask spatial dimensions should match" 