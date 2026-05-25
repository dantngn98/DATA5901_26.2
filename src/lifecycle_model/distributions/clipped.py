# third-party
import numpy as np

# local
from src.lifecycle_model.types import QuantitativeDistribution

class ClippedDistribution(QuantitativeDistribution):
    __slots__ = ("base", "lower", "upper")

    def __init__(
        self,
        base: QuantitativeDistribution,
        lower: int | float | None = None,
        upper: int | float | None = None,
    ):
        if lower is None and upper is None:
            raise ValueError("must specify lower or upper")

        if (
            lower is not None
            and upper is not None
            and lower > upper
        ):
            raise ValueError("lower must be <= upper")

        self.base = base
        self.lower = lower
        self.upper = upper

    def _sample_numpy(
        self,
        n: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> float | np.ndarray[np.float64]:
        x = self.base._sample_numpy(n, rng)
        x = np.clip(x, self.lower, self.upper)
        return x.item() if n is None else x

    def mean(self, num_samples: int | None = None) -> float:
        num_samples = num_samples if num_samples is not None else self._DEFAULT_MONTE_CARLO_SAMPLE_SIZE
        return np.mean(self._sample_numpy(n=num_samples)).item()

    def variance(self, num_samples: int | None = None) -> float:
        num_samples = num_samples if num_samples is not None else self._DEFAULT_MONTE_CARLO_SAMPLE_SIZE
        return np.var(self._sample_numpy(n=num_samples), ddof=1).item()