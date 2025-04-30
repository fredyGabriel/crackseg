import torch
from torch.utils.data import (
    Sampler, RandomSampler, SubsetRandomSampler, DistributedSampler
)
from collections import Counter
import numpy as np
from typing import Optional, Sequence, Any


class RandomSamplerWrapper(RandomSampler):
    """Random sampler with configurable replacement and num_samples."""
    def __init__(self, data_source, replacement: bool = False,
                 num_samples: Optional[int] = None):
        super().__init__(data_source, replacement=replacement,
                         num_samples=num_samples)


class BalancedSampler(Sampler):
    """Sampler to balance classes in imbalanced datasets."""
    def __init__(self, data_source, labels: Sequence[Any]):
        self.data_source = data_source
        self.labels = np.array(labels)
        class_counts = Counter(self.labels)
        weights = 1.0 / np.array([
            class_counts[label] for label in self.labels
        ])
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        indices = torch.multinomial(self.weights, len(self.weights),
                                    replacement=True)
        return iter(indices.tolist())

    def __len__(self):
        return len(self.data_source)


class SubsetSampler(SubsetRandomSampler):
    """Sampler for a subset of indices (e.g., validation/test split)."""
    def __init__(self, indices: Sequence[int]):
        super().__init__(indices)


def sampler_factory(
    kind: str,
    data_source,
    labels: Optional[Sequence[Any]] = None,
    indices: Optional[Sequence[int]] = None,
    replacement: bool = False,
    num_samples: Optional[int] = None,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False
):
    """
    Factory to create a sampler by kind: 'random', 'balanced', 'subset',
    'distributed'.
    For 'distributed', uses torch.utils.data.DistributedSampler and accepts
    num_replicas, rank, shuffle, seed, drop_last.
    """
    if kind == 'random':
        return RandomSamplerWrapper(
            data_source, replacement=replacement, num_samples=num_samples
        )
    if kind == 'balanced':
        if labels is None:
            raise ValueError("labels must be provided for BalancedSampler")
        return BalancedSampler(data_source, labels)
    if kind == 'subset':
        if indices is None:
            raise ValueError("indices must be provided for SubsetSampler")
        return SubsetSampler(indices)
    if kind == 'distributed':
        return DistributedSampler(
            data_source,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
    raise ValueError(f"Unknown sampler kind: {kind}")
