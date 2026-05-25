# standard
from typing import Sequence

# third-party
import numpy as np

# local
from src.lifecycle_model.types import QuantitativeDistribution

class DiscretizedEmpiricalDistribution(QuantitativeDistribution):
    __slots__ = (
        "num_bins",
        "bin_width",
        "bin_centers",
        "pmf",
        "_mean",
        "_variance"
    )

    def __init__(self, observations: Sequence[float], num_bins: int | None = None):
        if len(observations) == 0:
            raise ValueError("observations cannot be empty")

        x = np.array(observations, dtype=np.float64)

        x_min = np.min(x)
        x_max = np.max(x)

        if x_min == x_max:  # only 1 unique value -> constant distribution
            value = x_max.item()
            self.num_bins = 1
            self.bin_width = 1
            self.bin_centers = [value]
            self.pmf = [1]
            self._mean = value
            self._variance = value
        else:
            num_bins = num_bins if num_bins is not None else max(1, int(np.sqrt(len(x))))
            bin_width = (x_max - x_min)/num_bins
            bin_edges = np.linspace(
                x_min,
                x_max,
                num_bins + 1,
            )
            bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

            counts, _ = np.histogram(
                x,
                bins=bin_edges,
            )
            pmf = counts / counts.sum()

            self.num_bins = num_bins
            self.bin_width = bin_width
            self.bin_centers = bin_centers.tolist()
            self.pmf = pmf.tolist()
            self._mean = np.sum(bin_centers*pmf)
            self._variance = np.sum(((bin_centers-self._mean)**2)*pmf)

    def _sample_numpy(
        self,
        n: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> float | np.ndarray[np.float64]:
        rng = rng if rng is not None else np.random.default_rng()
        sample = rng.choice(self.bin_centers, p=self.pmf, size=n)
        return sample.item() if n is None else sample

    def mean(self) -> float:
        return self._mean

    def variance(self) -> float:
        return self._variance
