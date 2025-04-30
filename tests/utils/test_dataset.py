import sys
import os
import pytest
from PIL import Image
import torch  # Import torch

# Add src/ to sys.path to ensure correct import
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../..')
    )
)
from src.data.dataset import CrackSegmentationDataset  # noqa: E402


def create_image(path, size=(16, 16), color=(255, 0, 0),
                 exif_orientation=None):
    img = Image.new("RGB", size, color)
    if exif_orientation is not None:
        exif = img.getexif()
        # 274 is the EXIF tag for orientation
        exif[274] = exif_orientation
        img.save(path, exif=exif)
    else:
        img.save(path)


def create_mask(path, size=(16, 16), value=128):
    mask = Image.new("L", size, value)
    mask.save(path)


def test_dataset_basic(tmp_path):
    # Structure: data_root/mode/images, data_root/mode/masks
    data_root = tmp_path / "data"
    images_dir = data_root / "train" / "images"
    masks_dir = data_root / "train" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    # Create 3 valid pairs
    for i in range(3):
        create_image(images_dir / f"img{i}.png")
        create_mask(masks_dir / f"img{i}.png")
    # Add mode parameter
    ds = CrackSegmentationDataset(str(data_root), mode="train")
    assert len(ds) == 3
    sample = ds[0]
    assert "image" in sample and "mask" in sample
    # Check tensor properties
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32
    # Default image_size is (512, 512)
    assert sample["image"].shape == (3, 512, 512)  # C, H, W
    assert sample["mask"].shape == (512, 512)    # H, W


def test_dataset_missing_mask(tmp_path):
    data_root = tmp_path / "data"
    images_dir = data_root / "train" / "images"
    masks_dir = data_root / "train" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    create_image(images_dir / "img0.png")
    # Don't create the corresponding mask
    # Expect RuntimeError because no samples will be found
    with pytest.raises(RuntimeError, match="No image/mask pairs found"):
        CrackSegmentationDataset(str(data_root), mode="train")


def test_dataset_missing_dirs(tmp_path):
    data_root = tmp_path / "data"
    # Don't create subdirectories
    with pytest.raises(FileNotFoundError):
        # Add mode parameter
        CrackSegmentationDataset(str(data_root), mode="train")


def test_dataset_corrupt_image(tmp_path):
    data_root = tmp_path / "data"
    images_dir = data_root / "train" / "images"
    masks_dir = data_root / "train" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    # Create valid image/mask pair
    create_image(images_dir / "img0.png")
    create_mask(masks_dir / "img0.png")
    # Create corrupt image and valid mask
    with open(images_dir / "img1.png", "wb") as f:
        f.write(b"not an image")
    create_mask(masks_dir / "img1.png")
    # Add mode parameter
    ds = CrackSegmentationDataset(str(data_root), mode="train")
    # Dataset should find 2 pairs, but only process 1 valid one
    assert len(ds) == 2
    # __getitem__ should skip the corrupt one and return the valid one
    # Requesting index 1 (corrupt) should trigger error handling
    # and return index 0 (valid) after skipping.
    sample = ds[1]  # This will internally try idx=1, fail, try idx=0
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)
    assert sample["image"].shape == (3, 512, 512)
    assert sample["mask"].shape == (512, 512)


def test_dataset_exif_orientation(tmp_path):
    data_root = tmp_path / "data"
    images_dir = data_root / "train" / "images"
    masks_dir = data_root / "train" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    # Create image with EXIF orientation
    create_image(images_dir / "img0.png", exif_orientation=3)
    create_mask(masks_dir / "img0.png")
    # Add mode parameter
    ds = CrackSegmentationDataset(str(data_root), mode="train")
    sample = ds[0]
    # Just check if a valid tensor is returned. EXIF handled by PIL/cv2.
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)


@pytest.mark.skip(reason="Transform parameter removed, test needs redesign or \
removal")
def test_dataset_with_transform(tmp_path):
    # ... (original test content, now skipped) ...
    data_root = tmp_path / "data"
    images_dir = data_root / "train" / "images"
    masks_dir = data_root / "train" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    create_image(images_dir / "img0.png")
    create_mask(masks_dir / "img0.png")

    def dummy_transform(sample):
        # Cambia el color de la imagen a azul
        img = sample["image"].copy()
        img.paste((0, 0, 255), [0, 0, img.size[0], img.size[1]])
        return {"image": img, "mask": sample["mask"]}

    ds = CrackSegmentationDataset(
        # This call will fail
        str(data_root), transform=dummy_transform, mode="train"
    )
    sample = ds[0]
    assert sample["image"].getpixel((0, 0)) == (0, 0, 255)


def test_dataset_all_samples_corrupt(tmp_path):
    data_root = tmp_path / "data"
    images_dir = data_root / "train" / "images"
    masks_dir = data_root / "train" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    # Create only corrupt files
    with open(images_dir / "img0.png", "wb") as f:
        f.write(b"not an image")
    # Corresponding mask is needed for scan
    create_mask(masks_dir / "img0.png")

    # Add mode parameter
    ds = CrackSegmentationDataset(str(data_root), mode="train")
    # __getitem__ should raise RuntimeError after failing all samples
    with pytest.raises(RuntimeError,
                       match="No valid image/mask pairs could be loaded"):
        _ = ds[0]
