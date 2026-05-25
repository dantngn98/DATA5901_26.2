# third-party
import numpy as np

# local
from src.lifecycle_model.types import QuantitativeDistribution

class TriangularDistribution(QuantitativeDistribution):
    __slots__ = ("low", "mode", "high")

    def __init__(self, low: int | float, mode: int | float, high: int | float):
        if high <= low:
            raise ValueError("high must be greater than low")
        if not (low <= mode <= high):
            raise ValueError("mode must satisfy low <= mode <= high")

        self.low = low
        self.mode = mode
        self.high = high

    def _sample_numpy(
        self,
        n: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> float | np.ndarray[np.float64]:
        rng = rng if rng is not None else np.random.default_rng()
        return rng.triangular(
            left=self.low,
            mode=self.mode,
            right=self.high,
            size=n,
        )

    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3

    def variance(self) -> float:
        a, b, c = self.low, self.mode, self.high
        return (a**2 + b**2 + c**2 - a*b - a*c - b*c) / 18
