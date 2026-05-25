# standard
from abc import ABC, abstractmethod
from typing import Mapping, Sequence, Self

# third-party
import numpy as np

# local
from src.util import sample


class Distribution[T](ABC):
    @abstractmethod
    def sample(self, n: int | None = None, rng: np.random.Generator | None = None) -> T | Sequence[T]:
        pass

class CategoricalDistribution[T](Distribution[T]):
    __slots__ = ("classes", "pmf", "_mode")

    def __init__(
        self,
        distribution: Mapping[T, float] | Sequence[tuple[T, float]],
        normalize: bool = False
    ):
        if isinstance(distribution, Mapping):
            distribution_seq = list(distribution.items())
        elif isinstance(distribution, Sequence):
            distribution_seq = distribution
        else:
            raise TypeError(f"expected Mapping or Sequence but got: {type(distribution)}")
        
        if len(distribution_seq) == 0:
            raise ValueError("distribution cannot be empty")

        classes, pmf = zip(*distribution_seq)
        pmf = np.asarray(pmf, dtype=np.float64)
        total = pmf.sum()

        if np.any(pmf < 0):
            raise ValueError("probabilities must be nonnegative")
        if total <= 0:
            raise ValueError("sum of probabilities must be positive")
        if not normalize and np.abs(total-1) > 1e-6:
            raise ValueError("probabilities must sum to 1")
        
        if normalize:
            pmf = pmf/total
        
        self.classes = tuple(classes)
        self.pmf = pmf
        self._mode = classes[np.argmax(pmf)]  # calculate once


    def sample(self, n: int | None = None, rng: np.random.Generator | None = None) -> T | list[T]:
        return sample(
            elements=self.classes,
            p=self.pmf,
            num_samples=n,
            normalize=False,
            rng = rng if rng is not None else np.random.default_rng()
        )
    
    def mode(self) -> T:
        return self._mode
    
    def condition(self, classes: Sequence[T]) -> Self:
        if len(classes) == 0:
            raise ValueError("class_in cannot be empty")

        class_set = set(classes)

        conditioned = [
            (cls, prob)
            for cls, prob in zip(self.classes, self.pmf)
            if cls in class_set
        ]

        if len(conditioned) == 0:
            raise ValueError("no matching classes found")

        return CategoricalDistribution(conditioned, normalize=True)

class QuantitativeDistribution(Distribution[int | float]):
    _DEFAULT_MONTE_CARLO_SAMPLE_SIZE = 5000

    @abstractmethod
    def _sample_numpy(
        self,
        n: int | None = None,
        rng: np.random.Generator | None = None
    ) -> float | np.ndarray[np.float64]:
        pass

    def sample(self, n: int | None = None, rng: np.random.Generator | None = None) -> float | list[float]:
        np_sample = self._sample_numpy(n, rng)
        return np_sample if n is None else np_sample.tolist()

    def mean(self) -> float:
        return np.mean(self._sample_numpy(self._DEFAULT_MONTE_CARLO_SAMPLE_SIZE)).item()

    def variance(self) -> float:
        return np.var(self._sample_numpy(self._DEFAULT_MONTE_CARLO_SAMPLE_SIZE), ddof=1).item()
