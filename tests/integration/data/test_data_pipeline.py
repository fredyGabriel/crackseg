from collections.abc import Callable, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from src.data import transforms as tr


class DummySegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dummy dataset for integration test."""

    def __init__(
        self,
        images: Sequence[NDArray[np.uint8]],
        masks: Sequence[NDArray[np.uint8]],
        transform: (
            Callable[
                [NDArray[np.uint8], NDArray[np.uint8]], dict[str, torch.Tensor]
            ]
            | None
        ) = None,
    ):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            result = self.transform(img, mask)
            img = result["image"]
            mask = result["mask"]
        # Asegura que ambos sean torch.Tensor
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.asarray(img))
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.asarray(mask))
        return img, mask


def test_data_pipeline_end_to_end() -> None:
    """Integration test: full data pipeline from images to dataloader batch."""
    # Create dummy data
    n = 8
    images: list[NDArray[np.uint8]] = [
        np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        for _ in range(n)
    ]
    masks: list[NDArray[np.uint8]] = [
        np.random.randint(0, 2, (32, 32), dtype=np.uint8) * 255
        for _ in range(n)
    ]
    # Compose transforms
    pipeline = tr.get_basic_transforms("train", image_size=(32, 32))

    def transform_fn(
        img: NDArray[np.uint8], mask: NDArray[np.uint8]
    ) -> dict[str, torch.Tensor]:
        return tr.apply_transforms(img, mask, pipeline)

    # Create dataset and dataloader
    dataset = DummySegmentationDataset(images, masks, transform=transform_fn)
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset, batch_size=4, shuffle=False
    )
    # Iterate and check batch shapes
    for imgs, masks in loader:
        assert isinstance(imgs, torch.Tensor)
        assert isinstance(masks, torch.Tensor)
        assert imgs.shape[1:] == (3, 32, 32)
        assert masks.shape[1:] == (32, 32)
        assert imgs.shape[0] == masks.shape[0] <= 4  # noqa: PLR2004
