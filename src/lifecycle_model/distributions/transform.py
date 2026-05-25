# standard
from typing import Callable

# third-party
import numpy as np

# local
from src.lifecycle_model.types import QuantitativeDistribution

class TransformDistribution(QuantitativeDistribution):
    __slots__ = ("base", "f")

    def __init__(self, base: QuantitativeDistribution, f: Callable[[float], float]):
        self.base = base
        self.f = f
        self._vectorized_f = np.vectorize(f)

    def _sample_numpy(
        self,
        n: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> float | np.ndarray[np.float64]:
        x = self.base._sample_numpy(n, rng)
        return self.f(x) if n is None else self._vectorized_f(x)

    def mean(self, num_samples: int | None = None) -> float:
        num_samples = num_samples if num_samples is not None else self._DEFAULT_MONTE_CARLO_SAMPLE_SIZE
        return np.mean(self._sample_numpy(n=num_samples)).item()

    def variance(self, num_samples: int | None = None) -> float:
        num_samples = num_samples if num_samples is not None else self._DEFAULT_MONTE_CARLO_SAMPLE_SIZE
        return np.var(self._sample_numpy(n=num_samples), ddof=1).item()
