# standard
from abc import ABC, abstractmethod
from collections import Counter

# third-party
import numpy as np

# local
from src.lifecycle_model.types import CostingDimension, LifecycleState, Lifecycle


_RECOVERY_FUNNEL_ENTRY_STATES = {
        LifecycleState.RETURNED_TO_VENDOR,
        LifecycleState.SEEKING_DISCOUNT,
        LifecycleState.SEEKING_DONATION,
        LifecycleState.SEEKING_DISPOSAL
    }

class LifecycleModel(ABC):
    AUTOMATED_COSTING_DIMENSIONS = {CostingDimension.STORAGE, CostingDimension.OPPORTUNITY_COST, CostingDimension.LOST_REVENUE}
    INVENTORY_STATES = {
        LifecycleState.INITIAL_RECEPTION, LifecycleState.SELLABLE_INVENTORY, LifecycleState.RETURNED,
        LifecycleState.SEEKING_DISCOUNT, LifecycleState.SEEKING_DONATION, LifecycleState.SEEKING_DISPOSAL
    }
    RECOVERY_FUNNEL_ENTRY_STATES = _RECOVERY_FUNNEL_ENTRY_STATES
    FORWARD_EDGES = {
        LifecycleState.INITIAL_RECEPTION:  {LifecycleState.SELLABLE_INVENTORY, *_RECOVERY_FUNNEL_ENTRY_STATES},
        LifecycleState.SELLABLE_INVENTORY: {LifecycleState.SOLD, *_RECOVERY_FUNNEL_ENTRY_STATES},
        LifecycleState.SOLD:               {LifecycleState.UNRETURNED, LifecycleState.RETURNED},
        LifecycleState.RETURNED:           {LifecycleState.SELLABLE_INVENTORY, *_RECOVERY_FUNNEL_ENTRY_STATES},
        LifecycleState.SEEKING_DISCOUNT:   {LifecycleState.WAREHOUSE_DEALS_AND_GRADE_RESELL, LifecycleState.LIQUIDATED, LifecycleState.SEEKING_DONATION},
        LifecycleState.SEEKING_DONATION:   {LifecycleState.DONATED, LifecycleState.SEEKING_DISPOSAL},
        LifecycleState.SEEKING_DISPOSAL:   {LifecycleState.DISPOSED}
    }
    _WAIT_TIME_DIST_KW = "wait_time_dist"
    _PROBABILITY_FUNC_KW = "probability_func"

    @abstractmethod
    def sample_lifecycle(
        self,
        starting_distribution: dict[LifecycleState, float],
        num_samples: int | None = None
    ) -> Lifecycle | list[Lifecycle]:
        pass

    def aggregate_lifecycle_metrics(
        self,
        starting_distribution: dict[LifecycleState, float],
        num_samples: int
    ) -> dict[str, float]:
        sample_lifecycles = self.sample_lifecycle(starting_distribution, num_samples)
        lifecycle_metrics = [self._lifecycle_metrics(lifecycle) for lifecycle in sample_lifecycles]
        pivoted_metrics = _pivot(lifecycle_metrics)

        return {
            "start_state_distribution": _state_distribution(pivoted_metrics["start_state"]),
            "end_state_distribution": _state_distribution(pivoted_metrics["end_state"]),
            "mean_total_steps": np.mean(pivoted_metrics["total_steps"]),
            "mean_total_time": np.mean(pivoted_metrics["total_time"]),
            **{f"mean_{costing_dimension.value}": np.mean(pivoted_metrics[costing_dimension.value]) for costing_dimension in CostingDimension},
            "mean_total_cost": np.mean(pivoted_metrics["total_cost"]),
        }

    @staticmethod
    def _lifecycle_metrics(lifecycle: Lifecycle) -> dict[str, float]:
        metrics = {
            "start_state": lifecycle[0].start_state,
            "end_state": lifecycle[-1].end_state,
            "total_steps": len(lifecycle),
            "total_time": 0.0,
            **{
                costing_dimension.value: 0.0
                for costing_dimension in CostingDimension
            },
            "total_cost": 0.0,
        }

        for step in lifecycle:
            metrics["total_time"] += step.realized_wait_time
            for costing_dimension in CostingDimension:
                cost = step.costs.get(costing_dimension.value, 0.0)
                metrics[costing_dimension.value] += cost
                metrics["total_cost"] += cost

        return metrics
    
    def _storage_cost(self, state: LifecycleState, t: float) -> float:
        return (
            0 if state not in _INVENTORY_STATES
            else t * self.item.volume * self.facility.storage_cost_rate
        )
    
    def _opportunity_cost(self, state: LifecycleState, t: float) -> float:
        return (
            0 if state not in _INVENTORY_STATES
            else t * self.item.volume * self.facility.profit_rate
        )

    def _lost_revenue_cost(self, outcome: LifecycleState) -> float:
        return self.item.sales_price * self.lost_revenue_rates.get(outcome, 0)


def _pivot(dicts: list[dict]) -> dict:
    res = {}
    for d in dicts:
        for key, value in d.items():
            if key not in res:
                res[key] = []
            res[key].append(value)
    return res

def _state_distribution(observed_states: list[LifecycleState]) -> dict[str, float]:
    counts = Counter(observed_states)
    total = len(observed_states)

    return {
        state: count / total
        for state, count in counts.items()
    }