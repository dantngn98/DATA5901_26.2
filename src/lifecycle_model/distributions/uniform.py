# third-party
import numpy as np

# local
from src.lifecycle_model.types import QuantitativeDistribution

class UniformDistribution(QuantitativeDistribution):
    __slots__ = ("low", "high")

    def __init__(self, low: int | float, high: int | float):
        if high <= low:
            raise ValueError("high must be greater than low")

        self.low = low
        self.high = high

    def _sample_numpy(
        self,
        n: int | None = None,
        rng: np.random.Generator | None = None
    ) -> float | np.ndarray[np.float64]:
        rng = rng if rng is not None else np.random.default_rng()
        return rng.uniform(self.low, self.high, size=n)

    def mean(self) -> float:
        return (self.low + self.high)/2

    def variance(self) -> float:
        return ((self.high - self.low)**2) / 12
