# standard
from typing import Any, Sequence

# third-party
import numpy as np

def sample(
    elements: Sequence[Any],
    p: Sequence[float],
    num_samples: int | None = None,
    normalize: bool = False,
    rng: np.random.Generator | None = None
) -> Any | list[Any]:
    # let call to rng.choice handle errors instead of double-checking

    if normalize:
        p = np.asarray(p, dtype=np.float64)
        p /= p.sum()

    rng = rng if rng is not None else np.random.default_rng()
    result = rng.choice(
        np.array(elements, dtype=object),
        size=num_samples,
        p=p
    )
    return result if num_samples is None else result.tolist()
