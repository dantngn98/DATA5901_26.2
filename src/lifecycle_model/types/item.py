# standard
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class Item:
    id: str
    volume: float  # ! units must be consistent with facility
    sales_price: float
