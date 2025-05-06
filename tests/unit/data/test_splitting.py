import pytest
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from src.data.splitting import (
    split_indices, get_all_samples, create_split_datasets
)


class DummyDataset(Dataset):

    def __init__(self, mode, samples_list, **kwargs):
        self.mode = mode
        self.samples = samples_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def test_split_indices_basic():
    """Test correct splitting of indices with valid ratios."""
    indices = split_indices(
        100, {'train': 0.7, 'val': 0.2, 'test': 0.1}, seed=42
    )
    assert set(indices.keys()) == {'train', 'val', 'test'}
    assert sum(len(v) for v in indices.values()) == 100
    assert abs(len(indices['train']) - 70) <= 1
    assert abs(len(indices['val']) - 20) <= 1
    assert abs(len(indices['test']) - 10) <= 1


def test_split_indices_missing_test():
    """Test that 'test' ratio is inferred if missing."""
    indices = split_indices(
        50, {'train': 0.6, 'val': 0.2}, seed=1
    )
    assert set(indices.keys()) == {'train', 'val', 'test'}
    assert sum(len(v) for v in indices.values()) == 50


def test_split_indices_invalid_keys():
    with pytest.raises(ValueError):
        split_indices(10, {'foo': 0.5, 'bar': 0.5})


def test_split_indices_invalid_sum():
    with pytest.raises(ValueError):
        split_indices(10, {'train': 0.7, 'val': 0.4})


def test_split_indices_invalid_values():
    with pytest.raises(ValueError):
        split_indices(10, {'train': -0.1, 'val': 1.1, 'test': 0.0})


def test_get_all_samples(tmp_path):
    """Test that get_all_samples finds image/mask pairs."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    (images_dir / "img1.png").write_bytes(b"fakeimg1")
    (masks_dir / "img1.png").write_bytes(b"fakemask1")
    (images_dir / "img2.jpg").write_bytes(b"fakeimg2")
    (masks_dir / "img2.png").write_bytes(b"fakemask2")
    (images_dir / "img3.png").write_bytes(b"fakeimg3")
    pairs = get_all_samples(str(tmp_path))
    assert len(pairs) == 2
    assert all(
        Path(img).exists() and Path(mask).exists() for img, mask in pairs
    )


def test_get_all_samples_missing_dirs(tmp_path):
    """Test get_all_samples raises if images or masks dir missing."""
    with pytest.raises(FileNotFoundError):
        get_all_samples(str(tmp_path))
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError):
        get_all_samples(str(tmp_path))


def test_create_split_datasets(tmp_path):
    """Test create_split_datasets creates datasets for each split."""
    for split in ["train", "val", "test"]:
        images_dir = tmp_path / split / "images"
        masks_dir = tmp_path / split / "masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        (images_dir / f"{split}1.png").write_bytes(b"img")
        (masks_dir / f"{split}1.png").write_bytes(b"mask")
    transform_cfg = OmegaConf.create({
        'train': {}, 'val': {}, 'test': {}
    })
    datasets = create_split_datasets(
        data_root=str(tmp_path),
        transform_cfg=transform_cfg,
        dataset_cls=DummyDataset
    )
    assert set(datasets.keys()) == {'train', 'val', 'test'}
    for split in ['train', 'val', 'test']:
        assert isinstance(datasets[split], DummyDataset)
        assert len(datasets[split]) == 1


def test_create_split_datasets_missing_split(tmp_path):
    """Test create_split_datasets skips missing split dirs."""
    images_dir = tmp_path / "train" / "images"
    masks_dir = tmp_path / "train" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    (images_dir / "img1.png").write_bytes(b"img")
    (masks_dir / "img1.png").write_bytes(b"mask")
    transform_cfg = OmegaConf.create({'train': {}, 'val': {}, 'test': {}})
    datasets = create_split_datasets(
        data_root=str(tmp_path),
        transform_cfg=transform_cfg,
        dataset_cls=DummyDataset
    )
    assert 'train' in datasets
    assert 'val' not in datasets or len(datasets['val']) == 0
    assert 'test' not in datasets or len(datasets['test']) == 0


def test_create_split_datasets_no_dataset_cls(tmp_path):
    """Test create_split_datasets raises if dataset_cls is not provided."""
    transform_cfg = OmegaConf.create({'train': {}, 'val': {}, 'test': {}})
    with pytest.raises(ValueError):
        create_split_datasets(
            data_root=str(tmp_path),
            transform_cfg=transform_cfg
        )
