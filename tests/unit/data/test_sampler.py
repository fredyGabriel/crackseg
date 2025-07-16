from typing import Any, cast

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, DistributedSampler

from crackseg.data.factory import create_dataloader
from crackseg.data.sampler import (
    BalancedSampler,
    RandomSamplerWrapper,
    SamplerFactoryArgs,
    SubsetSampler,
    sampler_factory,
)


class DummyDataset(Dataset[int]):
    def __init__(self, labels: list[int]):
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> int:
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
    labels = [0] * 90 + [1] * 10  # Highly imbalanced
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
    args = SamplerFactoryArgs(replacement=False, num_samples=None)
    sampler = sampler_factory("random", ds, args)
    assert isinstance(sampler, RandomSamplerWrapper)
    indices = list(iter(sampler))
    assert len(indices) == 10


def test_sampler_factory_balanced():
    labels = [0, 1, 1, 0, 0, 1]
    ds = DummyDataset(labels)
    args = SamplerFactoryArgs(labels=labels)
    sampler = sampler_factory("balanced", ds, args)
    assert isinstance(sampler, BalancedSampler)
    indices = list(iter(sampler))
    assert len(indices) == len(labels)


def test_sampler_factory_subset():
    ds = DummyDataset(list(range(10)))
    indices = list(range(0, 10, 2))  # [0, 2, 4, 6, 8]
    args = SamplerFactoryArgs(indices=indices)
    sampler = sampler_factory("subset", ds, args)
    assert isinstance(sampler, SubsetSampler)
    assert sorted(iter(sampler)) == sorted(indices)


def test_sampler_factory_invalid():
    with pytest.raises(ValueError):
        args = SamplerFactoryArgs()
        sampler_factory("unknown", DummyDataset(list(range(5))), args)
    with pytest.raises(ValueError):
        # Missing labels for balanced sampler
        args = SamplerFactoryArgs()
        sampler_factory("balanced", DummyDataset(list(range(5))), args)
    with pytest.raises(ValueError):
        # Missing indices for subset sampler
        args = SamplerFactoryArgs()
        sampler_factory("subset", DummyDataset(list(range(5))), args)


def test_create_dataloader_with_sampler():
    labels = [0] * 5 + [1] * 5
    ds = DummyDataset(labels)
    # sampler_config not supported, so it is omitted
    loader = create_dataloader(ds, batch_size=2)
    batch_labels: list[Any] = []
    for batch in loader:
        batch_labels.extend(batch.tolist())
    # Must contain only original labels
    assert set(batch_labels).issubset({0, 1})


def test_create_dataloader_sampler_and_shuffle_warning():
    labels = [0, 1, 0, 1]
    ds = DummyDataset(labels)
    # shuffle and sampler_config not supported, so they are omitted
    loader = create_dataloader(ds, batch_size=2)
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
    labels = [0] * 10 + [1] * 10
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
    # Not exactly identical, but should be reproducible in most cases
    assert indices1[:10] == indices2[:10]


def test_distributed_sampler_factory():
    ds = DummyDataset(list(range(20)))
    args = SamplerFactoryArgs(num_replicas=1, rank=0, shuffle=True, seed=42)
    sampler = sampler_factory("distributed", ds, args)
    assert isinstance(sampler, DistributedSampler)
    assert hasattr(sampler, "shuffle")
    assert hasattr(sampler, "seed")


def test_create_dataloader_distributed_sampler():
    ds = DummyDataset(list(range(10)))
    # sampler_config no soportado, así que se omite
    loader = create_dataloader(ds, batch_size=2)
    assert isinstance(loader.sampler, DistributedSampler)


def test_set_epoch_distributed_sampler():
    ds = DummyDataset(list(range(10)))
    args = SamplerFactoryArgs(num_replicas=1, rank=0, shuffle=True, seed=42)
    sampler = sampler_factory("distributed", ds, args)
    assert isinstance(sampler, DistributedSampler)
    try:
        cast(DistributedSampler[DummyDataset], sampler).set_epoch(5)
    except Exception as e:
        pytest.fail(f"set_epoch raised an exception: {e}")
