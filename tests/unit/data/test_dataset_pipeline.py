"""Test for the dataset pipeline and transformations."""

import cv2
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.data import create_crackseg_dataset
from src.data import transforms as tr


@pytest.fixture
def test_data_dir(tmp_path):
    """Create and return a temporary directory with test images.
    Uses tmp_path for isolation.
    """
    test_dir = tmp_path / "test_images"
    test_dir.mkdir(parents=True, exist_ok=True)

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
        (
            str(test_data_dir / "test_img.png"),
            str(test_data_dir / "test_mask.png"),
        ),
    ]

    # Cargar y modificar configuraciones para pruebas
    data_cfg = OmegaConf.create(
        {
            "data_root": str(test_data_dir),
            "image_size": [64, 64],  # Mismo tamaño que las imágenes de prueba
            "batch_size": 1,
            "num_workers": 0,
            "in_memory_cache": False,
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
        }
    )

    # La configuración de transformación debe ser un DictConfig
    transform_cfg = OmegaConf.create(
        [
            {"name": "Resize", "params": {"height": 64, "width": 64}},
            {
                "name": "Normalize",
                "params": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            },
            {"name": "ToTensorV2", "params": {}},
        ]
    )

    # Crear dataset para modo 'train'
    dataset = create_crackseg_dataset(
        data_cfg=data_cfg,
        transform_cfg=transform_cfg,  # type: ignore
        mode="train",
        samples_list=samples_list,
    )

    # Verificaciones básicas
    assert len(dataset) == len(
        samples_list
    ), "Dataset length should match samples list length"

    # Verificar una muestra
    sample = dataset[0]
    assert isinstance(sample, dict), "Sample should be a dictionary"
    assert "image" in sample, "Sample should contain 'image'"
    assert "mask" in sample, "Sample should contain 'mask'"
    assert isinstance(
        sample["image"], torch.Tensor
    ), "Image should be a torch.Tensor"
    assert isinstance(
        sample["mask"], torch.Tensor
    ), "Mask should be a torch.Tensor"

    # Verificar dimensiones
    assert (
        len(sample["image"].shape) == 3  # noqa: PLR2004
    ), "Image should have 3 dimensions (C, H, W)"  # noqa: PLR2004
    assert (
        sample["image"].shape[0] == 3  # noqa: PLR2004
    ), "Image should have 3 channels"  # noqa: PLR2004
    assert (
        len(sample["mask"].shape) == 3  # noqa: PLR2004
        and sample["mask"].shape[0] == 1  # noqa: PLR2004
    ), "Mask should have shape (1, H, W)"  # noqa: PLR2004
    assert (
        sample["image"].shape[1:] == sample["mask"].shape[1:]
    ), "Image and mask spatial dimensions should match"


class DummySegmentationDataset(Dataset):
    """Dummy dataset for integration test."""

    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            result = (
                self.transform(img, mask)
                if self.transform is not None
                else None if self.transform is not None else (None, None)
            )
            img = result["image"]  # type: ignore
            mask = result["mask"]  # type: ignore
        return img, mask


def test_data_pipeline_end_to_end(tmp_path):
    """Integration test: full data pipeline from images to dataloader batch."""
    # Create dummy data
    n = 8
    images = [
        np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        for _ in range(n)
    ]
    masks = [
        np.random.randint(0, 2, (32, 32), dtype=np.uint8) * 255
        for _ in range(n)
    ]
    # Compose transforms
    pipeline = tr.get_basic_transforms("train", image_size=(32, 32))

    def transform_fn(img, mask):
        return tr.apply_transforms(img, mask, pipeline)

    # Use a temp directory for any file operations if needed
    # Create dataset and dataloader
    dataset = DummySegmentationDataset(images, masks, transform=transform_fn)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    # Iterate and check batch shapes
    for imgs, masks in loader:
        assert isinstance(imgs, torch.Tensor)
        assert isinstance(masks, torch.Tensor)
        assert imgs.shape[1:] == (3, 32, 32)
        assert masks.shape[1:] == (32, 32)
        assert imgs.shape[0] == masks.shape[0] <= 4  # noqa: PLR2004
