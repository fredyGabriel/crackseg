from collections import Counter
from collections.abc import Iterator, Sequence, Sized
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import (
    Dataset,
    DistributedSampler,
    RandomSampler,
    Sampler,
    SubsetRandomSampler,
)


class RandomSamplerWrapper(RandomSampler):
    """Random sampler with configurable replacement and num_samples."""

    def __init__(
        self,
        data_source: Dataset[Any],
        replacement: bool = False,
        num_samples: int | None = None,
    ):
        super().__init__(
            cast(Sized, data_source),
            replacement=replacement,
            num_samples=num_samples,
        )


class BalancedSampler(Sampler[int]):
    """Sampler to balance classes in imbalanced datasets."""

    def __init__(self, data_source: Dataset[Any], labels: Sequence[Any]):
        self.data_source: Dataset[Any] = data_source
        self.labels = np.array(labels)
        class_counts = Counter(self.labels)
        weights = 1.0 / np.array(
            [class_counts[label] for label in self.labels]
        )
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(
            self.weights, len(self.weights), replacement=True
        )
        return iter(cast(list[int], indices.tolist()))

    def __len__(self) -> Any:
        return len(cast(Sized, self.data_source))


class SubsetSampler(SubsetRandomSampler):
    """Sampler for a subset of indices (e.g., validation/test split)."""

    def __init__(self, indices: Sequence[int]):
        super().__init__(indices)


@dataclass
class SamplerFactoryArgs:
    """Arguments for the sampler factory."""

    labels: Sequence[Any] | None = None
    indices: Sequence[int] | None = None
    replacement: bool = False
    num_samples: int | None = None
    num_replicas: int | None = None
    rank: int | None = None
    shuffle: bool = True
    seed: int = 0
    drop_last: bool = False


def sampler_factory(
    kind: str,
    data_source: Dataset[Any],
    args: SamplerFactoryArgs,
) -> Sampler[Any]:
    """
    Factory to create a sampler by kind: 'random', 'balanced', 'subset',
    'distributed'.
    For 'distributed', uses torch.utils.data.DistributedSampler and accepts
    num_replicas, rank, shuffle, seed, drop_last.
    """
    if kind == "random":
        return RandomSamplerWrapper(
            data_source,
            replacement=args.replacement,
            num_samples=args.num_samples,
        )
    if kind == "balanced":
        if args.labels is None:
            raise ValueError("labels must be provided for BalancedSampler")
        return BalancedSampler(data_source, args.labels)
    if kind == "subset":
        if args.indices is None:
            raise ValueError("indices must be provided for SubsetSampler")
        return SubsetSampler(args.indices)
    if kind == "distributed":
        return DistributedSampler(
            data_source,
            num_replicas=args.num_replicas,
            rank=args.rank,
            shuffle=args.shuffle,
            seed=args.seed,
            drop_last=args.drop_last,
        )
    raise ValueError(f"Unknown sampler kind: {kind}")
