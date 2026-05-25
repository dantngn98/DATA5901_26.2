# standard
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

# local
from src.lifecycle_model.types import CostingDimension

class LifecycleState(Enum):
    # "active" inventory
    INITIAL_RECEPTION = "initial_reception"
    SELLABLE_INVENTORY = "sellable_inventory"
    RETURNED = "returned"

    # non-recovery funnel (outcome pending)
    SOLD = "sold"

    # non-recovery funnel outcomes
    UNRETURNED = "unreturned"

    # recovery funnel (outcome pending)
    SEEKING_DISCOUNT = "seeking_discount"
    SEEKING_DONATION = "seeking_donation"
    SEEKING_DISPOSAL = "seeking_disposal"

    # recovery funnel outcomes
    RETURNED_TO_VENDOR = "returned_to_vendor"
    WAREHOUSE_DEALS_AND_GRADE_RESELL = "warehouse_deals_and_grade_resell"
    LIQUIDATED = "liquidated"
    DONATED = "donated"
    DISPOSED = "disposed"

@dataclass(slots=True, frozen=True)
class LifecycleStep:
    start_state: LifecycleState
    end_state: LifecycleState
    realized_wait_time: float
    costs: dict[CostingDimension, float]

Lifecycle = Sequence[LifecycleStep]
