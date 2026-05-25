# standard
from enum import Enum

class CostingDimension(Enum):
    # inventory
    STORAGE = "storage"
    OPPORTUNITY_COST = "opportunity_cost"

    # movement/processing
    LABOR = "labor"
    TRANSPORTATION = "transportation"

    # non-full sale
    LOST_REVENUE = "lost_revenue"
    RECOVERY_FUNNEL = "recovery_funnel"
