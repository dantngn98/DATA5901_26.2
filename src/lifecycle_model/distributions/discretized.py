# third-party
import numpy as np

# local
from src.lifecycle_model.types import QuantitativeDistribution

class DiscretizedDistribution(QuantitativeDistribution):
    __slots__ = ("base", "bin_width", "method_function")

    def __init__(
        self,
        base: QuantitativeDistribution,
        bin_width: int | float = 1,
        method: str = "middle",
    ):
        if bin_width <= 0:
            raise ValueError("bin width must be positive")
        if method not in {"middle", "floor", "ceil"}:
            raise ValueError("invalid method")

        self.base = base
        self.bin_width = bin_width
        self.method = method
        self._method_function = {
            "middle": lambda x: np.floor(x) + 0.5,
            "floor": np.floor,
            "ceil": np.ceil
        }[method]

    def _sample_numpy(
        self,
        n: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> float | np.ndarray[np.float64]:
        x = self.base._sample_numpy(n, rng)
        x_binned = self._method_function(x/self.bin_width) * self.bin_width
        return x_binned.item() if n is None else x_binned

    def mean(self, num_samples: int | None = None) -> float:
        num_samples = num_samples if num_samples is not None else self._DEFAULT_MONTE_CARLO_SAMPLE_SIZE
        return np.mean(self._sample_numpy(n=num_samples)).item()

    def variance(self, num_samples: int | None = None) -> float:
        num_samples = num_samples if num_samples is not None else self._DEFAULT_MONTE_CARLO_SAMPLE_SIZE
        return np.var(self._sample_numpy(n=num_samples), ddof=1).item()