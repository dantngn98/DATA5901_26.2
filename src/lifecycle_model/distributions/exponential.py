# third-party
import numpy as np

# local
from src.lifecycle_model.types import QuantitativeDistribution

class ExponentialDistribution(QuantitativeDistribution):
    __slots__ = ("rate",)

    def __init__(self, rate: int | float):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate

    def _sample_numpy(
        self,
        n: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> float | np.ndarray[np.float64]:
        rng = rng if rng is not None else np.random.default_rng()
        return rng.exponential(scale=1/self.rate, size=n)

    def mean(self) -> float:
        return 1 / self.rate

    def variance(self) -> float:
        return 1 / (self.rate**2)
