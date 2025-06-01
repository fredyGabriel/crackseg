# ruff: noqa: PLR2004
import os
import sys
from pathlib import Path

import pytest
import torch  # Import torch
from PIL import Image

# Add src/ to sys.path to ensure correct import
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from src.data.dataset import CrackSegmentationDataset  # noqa: E402


def create_image(
    path: str,
    size: tuple[int, int] = (16, 16),
    color: tuple[int, int, int] = (255, 0, 0),
    exif_orientation: int | None = None,
) -> None:
    img = Image.new("RGB", size, color)
    if exif_orientation is not None:
        exif = img.getexif()
        # 274 is the EXIF tag for orientation
        exif[274] = exif_orientation
        img.save(path, exif=exif)
    else:
        img.save(path)


def create_mask(
    path: str, size: tuple[int, int] = (16, 16), value: int = 128
) -> None:
    mask = Image.new("L", size, value)
    mask.save(path)


def test_dataset_basic(tmp_path: Path) -> None:
    # Structure: data_root/mode/images, data_root/mode/masks
    data_root = tmp_path / "data"
    images_dir = data_root / "train" / "images"
    masks_dir = data_root / "train" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    # Create 3 valid pairs
    for i in range(3):
        create_image(str(images_dir / f"img{i}.png"))
        create_mask(str(masks_dir / f"img{i}.png"))

    # Crear lista de muestras manualmente ya que el dataset ahora requiere
    # samples_list
    samples_list = [
        (str(images_dir / f"img{i}.png"), str(masks_dir / f"img{i}.png"))
        for i in range(3)
    ]

    # Actualizar la forma de crear el dataset
    ds = CrackSegmentationDataset(mode="train", samples_list=samples_list)

    assert len(ds) == 3
    sample = ds[0]
    assert "image" in sample and "mask" in sample
    # Check tensor properties
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32
    # Default image_size es variable, verificar solo las dimensiones
    assert sample["image"].ndim == 3  # C, H, W
    assert sample["mask"].ndim == 3  # 1, H, W
    assert sample["mask"].shape[0] == 1
    assert sample["image"].shape[0] == 3  # Canales RGB


def test_dataset_missing_mask(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "train" / "images"
    masks_dir = data_root / "train" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    create_image(str(images_dir / "img0.png"))
    # Don't create the corresponding mask

    # Lista vacía, entonces lanzará ValueError
    with pytest.raises(ValueError, match="samples_list must be provided"):
        CrackSegmentationDataset(mode="train", samples_list=None)


def test_dataset_missing_dirs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    # Don't create subdirectories
    # Ya que la clase ahora requiere samples_list, debemos probar
    # con una lista de archivos que no existen
    non_existent_img = str(data_root / "non_existent.jpg")
    non_existent_mask = str(data_root / "non_existent_mask.png")

    ds = CrackSegmentationDataset(
        mode="train", samples_list=[(non_existent_img, non_existent_mask)]
    )

    # Ahora el error ocurrirá al intentar cargar la imagen
    with pytest.raises(
        RuntimeError, match="No valid image/mask pairs could be loaded"
    ):
        _ = ds[0]


def test_dataset_corrupt_image(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "train" / "images"
    masks_dir = data_root / "train" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    # Create valid image/mask pair
    create_image(str(images_dir / "img0.png"))
    create_mask(str(masks_dir / "img0.png"))
    # Create corrupt image and valid mask
    with open(str(images_dir / "img1.png"), "wb") as f:
        f.write(b"not an image")
    create_mask(str(masks_dir / "img1.png"))

    # Crear lista de muestras manualmente
    samples_list = [
        (str(images_dir / f"img{i}.png"), str(masks_dir / f"img{i}.png"))
        for i in range(2)
    ]

    ds = CrackSegmentationDataset(mode="train", samples_list=samples_list)

    # Dataset should find 2 pairs, but only process 1 valid one
    assert len(ds) == 2
    # __getitem__ should skip the corrupt one and return the valid one
    # Requesting index 1 (corrupt) should trigger error handling
    # and return index 0 (valid) after skipping.
    sample = ds[1]  # This will internally try idx=1, fail, try idx=0
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)
    assert sample["image"].ndim == 3
    assert sample["mask"].ndim == 3
    assert sample["mask"].shape[0] == 1
    assert sample["image"].shape[0] == 3  # Canales RGB


def test_dataset_exif_orientation(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "train" / "images"
    masks_dir = data_root / "train" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    # Create image with EXIF orientation
    create_image(str(images_dir / "img0.png"), exif_orientation=3)
    create_mask(str(masks_dir / "img0.png"))

    # Crear lista de muestras manualmente
    samples_list = [
        (str(images_dir / "img0.png"), str(masks_dir / "img0.png"))
    ]

    ds = CrackSegmentationDataset(mode="train", samples_list=samples_list)

    sample = ds[0]
    # Just check if a valid tensor is returned. EXIF handled by PIL/cv2.
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)


def test_dataset_with_transform(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "train" / "images"
    masks_dir = data_root / "train" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    create_image(str(images_dir / "img0.png"), size=(16, 16))
    create_mask(str(masks_dir / "img0.png"), size=(16, 16))

    samples_list = [
        (str(images_dir / "img0.png"), str(masks_dir / "img0.png"))
    ]

    # Usar image_size para forzar un resize a (8, 8)
    ds = CrackSegmentationDataset(
        mode="train", samples_list=samples_list, image_size=(8, 8)
    )
    sample = ds[0]
    # Verificar que la imagen y la máscara han sido redimensionadas
    assert sample["image"].shape == (3, 8, 8)
    assert sample["mask"].shape == (1, 8, 8)


def test_dataset_all_samples_corrupt(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_dir = data_root / "train" / "images"
    masks_dir = data_root / "train" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    # Create only corrupt files
    with open(str(images_dir / "img0.png"), "wb") as f:
        f.write(b"not an image")
    # Corresponding mask is needed for scan
    create_mask(str(masks_dir / "img0.png"))

    # Crear lista de muestras manualmente
    samples_list = [
        (str(images_dir / "img0.png"), str(masks_dir / "img0.png"))
    ]

    ds = CrackSegmentationDataset(mode="train", samples_list=samples_list)

    # __getitem__ should raise RuntimeError after failing all samples
    with pytest.raises(
        RuntimeError, match="No valid image/mask pairs could be loaded"
    ):
        _ = ds[0]
