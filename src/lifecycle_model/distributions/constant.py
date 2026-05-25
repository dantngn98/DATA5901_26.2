# third-party
import numpy as np

# local
from src.lifecycle_model.types import QuantitativeDistribution


class ConstantDistribution(QuantitativeDistribution):
    __slots__ = ("value")

    def __init__(self, value: int | float):
        self.value = float(value)
    
    def _sample_numpy(
        self,
        n: int | None = None,
        rng: np.random.Generator | None = None
    ) -> float | np.ndarray[np.float64]:
        return (
            self.value if n is None
            else np.full(shape=n, fill_value=self.value)
        )
    
    def mean(self) -> int | float:
        return self.value
    
    def variance(self) -> int | float:
        return 0
