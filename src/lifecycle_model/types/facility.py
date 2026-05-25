# standard
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class Facility:
    id: str
    storage_cost_rate: float  # average cost per volume per day (units must be consistent with facility)
    profit_rate: float        # average profit per volume per day wrt item sales
