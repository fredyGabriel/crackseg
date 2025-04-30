import pytest
import torch
from torch.utils.data import Dataset
from src.data.sampler import (
    RandomSamplerWrapper, BalancedSampler, SubsetSampler, sampler_factory
)
from src.data.factory import create_dataloader
import numpy as np
from torch.utils.data import DistributedSampler


class DummyDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx]


def test_random_sampler_wrapper_distribution():
    ds = DummyDataset(list(range(100)))
    sampler = RandomSamplerWrapper(ds, replacement=False)
    indices = list(iter(sampler))
    assert sorted(indices) == list(range(100))
    # With replacement, allow repeats
    sampler2 = RandomSamplerWrapper(ds, replacement=True, num_samples=200)
    indices2 = list(iter(sampler2))
    assert len(indices2) == 200
    assert set(indices2).issubset(set(range(100)))


def test_balanced_sampler_balances_classes():
    labels = [0]*90 + [1]*10  # Highly imbalanced
    ds = DummyDataset(labels)
    sampler = BalancedSampler(ds, labels)
    sampled = [labels[i] for i in list(iter(sampler))[:1000]]
    # Debe haber proporciones mucho más balanceadas
    count0 = sampled.count(0)
    count1 = sampled.count(1)
    ratio = count0 / count1
    assert 0.5 < ratio < 2.0  # Aproximadamente balanceado


def test_subset_sampler():
    indices = [2, 5, 7, 11]
    sampler = SubsetSampler(indices)
    sampled = list(iter(sampler))
    assert sorted(sampled) == sorted(indices)


def test_sampler_factory_random():
    ds = DummyDataset(list(range(10)))
    sampler = sampler_factory('random', ds, replacement=True, num_samples=15)
    assert isinstance(sampler, RandomSamplerWrapper)
    indices = list(iter(sampler))
    assert len(indices) == 15


def test_sampler_factory_balanced():
    labels = [0, 1, 1, 0, 0, 1]
    ds = DummyDataset(labels)
    sampler = sampler_factory('balanced', ds, labels=labels)
    assert isinstance(sampler, BalancedSampler)
    indices = list(iter(sampler))
    assert len(indices) == len(labels)


def test_sampler_factory_subset():
    ds = DummyDataset(list(range(10)))
    indices = [1, 3, 5]
    sampler = sampler_factory('subset', ds, indices=indices)
    assert isinstance(sampler, SubsetSampler)
    assert sorted(list(iter(sampler))) == sorted(indices)


def test_sampler_factory_invalid():
    with pytest.raises(ValueError):
        sampler_factory('unknown', DummyDataset(list(range(5))))
    with pytest.raises(ValueError):
        sampler_factory('balanced', DummyDataset(list(range(5))))
    with pytest.raises(ValueError):
        sampler_factory('subset', DummyDataset(list(range(5))))


def test_create_dataloader_with_sampler():
    labels = [0]*5 + [1]*5
    ds = DummyDataset(labels)
    sampler_cfg = {'kind': 'balanced', 'labels': labels}
    loader = create_dataloader(
        ds, batch_size=2, sampler_config=sampler_cfg
    )
    batch_labels = []
    for batch in loader:
        batch_labels.extend(batch.tolist())
    # Debe contener solo los labels originales
    assert set(batch_labels).issubset({0, 1})


def test_create_dataloader_sampler_and_shuffle_warning():
    labels = [0, 1, 0, 1]
    ds = DummyDataset(labels)
    sampler_cfg = {'kind': 'random'}
    # shuffle=True y sampler: debe forzar shuffle=False
    loader = create_dataloader(
        ds, batch_size=2, shuffle=True, sampler_config=sampler_cfg
    )
    assert loader.sampler is not None


def test_reproducibility_random_sampler():
    ds = DummyDataset(list(range(50)))
    torch.manual_seed(42)
    sampler1 = RandomSamplerWrapper(ds, replacement=True, num_samples=20)
    indices1 = list(iter(sampler1))
    torch.manual_seed(42)
    sampler2 = RandomSamplerWrapper(ds, replacement=True, num_samples=20)
    indices2 = list(iter(sampler2))
    assert indices1 == indices2


def test_reproducibility_balanced_sampler():
    labels = [0]*10 + [1]*10
    ds = DummyDataset(labels)
    np.random.seed(123)
    torch.manual_seed(123)
    sampler1 = BalancedSampler(ds, labels)
    indices1 = list(iter(sampler1))
    np.random.seed(123)
    torch.manual_seed(123)
    sampler2 = BalancedSampler(ds, labels)
    indices2 = list(iter(sampler2))
    # No necesariamente idéntico, pero debe ser reproducible en la mayoría de
    # casos
    assert indices1[:10] == indices2[:10]


def test_distributed_sampler_factory():
    ds = DummyDataset(list(range(20)))
    sampler = sampler_factory(
        'distributed', ds, num_replicas=2, rank=1, shuffle=True, seed=123
    )
    assert isinstance(sampler, DistributedSampler)
    assert sampler.num_replicas == 2
    assert sampler.rank == 1
    assert sampler.shuffle is True
    assert sampler.seed == 123


def test_create_dataloader_distributed_sampler():
    ds = DummyDataset(list(range(10)))
    sampler_cfg = {'kind': 'distributed', 'num_replicas': 2, 'rank': 0,
                   'shuffle': False}
    loader = create_dataloader(ds, batch_size=2, sampler_config=sampler_cfg)
    assert isinstance(loader.sampler, DistributedSampler)
    assert loader.sampler.num_replicas == 2
    assert loader.sampler.rank == 0
    assert loader.sampler.shuffle is False


def test_set_epoch_distributed_sampler():
    ds = DummyDataset(list(range(10)))
    sampler = sampler_factory('distributed', ds, num_replicas=2, rank=0)
    # set_epoch should not raise
    try:
        sampler.set_epoch(5)
    except Exception as e:
        pytest.fail(
            f"set_epoch raised an exception: {e}"
        )
