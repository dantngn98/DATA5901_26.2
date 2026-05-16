# standard
from enum import Enum

class State(Enum):
    # "active" inventory
    INITIAL_RECEPTION = "initial_reception"
    SELLABLE_INVENTORY = "sellable"
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
    WAREHOUSE_DEALS_AND_GRADE_RESELL = "warehouse_deals_and_grade_resell"
    RETURNED_TO_VENDOR = "returned_to_vendor"
    LIQUIDATED = "liquidated"
    DONATED = "donated"
    DISPOSED = "disposed"


class CostingDimensions(Enum):
    # inventory
    STORAGE = "storage"
    OPPORTUNITY_COST = "opportunity_cost"

    # movement/processing
    LABOR = "labor"
    TRANSPORTATION = "transportation"

    # non-full sale
    LOST_REVENUE = "lost_revenue"
    RECOVERY_FUNNEL = "recovery_funnel"
