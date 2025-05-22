import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import transforms as tr


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
            result = self.transform(img, mask)
            img = result["image"]
            mask = result["mask"]
        return img, mask


def test_data_pipeline_end_to_end():
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
