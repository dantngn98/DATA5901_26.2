# standard
from abc import ABC, abstractmethod

# local
from src.lifecycle_model import WeightedDigraph
from src.lifecycle_model.types import (
    Item, Facility,
    CostingDimension,
    LifecycleState, LifecycleStep, Lifecycle,
    QuantitativeDistribution, 
)
from src.lifecycle_model.distributions import ConstantDistribution
from src.util import sample


# ============================================================
#  (IMPLEMENTATION-DETAIL) CONSTANTS
# ============================================================

# for default attributes; instantiate once
_ZERO_DIST = ConstantDistribution(0)
_ZERO_FUNC = lambda t: 0

_RECOVERY_FUNNEL_ENTRY_STATES = (
    LifecycleState.RETURNED_TO_VENDOR,
    LifecycleState.SEEKING_DISCOUNT,
    LifecycleState.SEEKING_DONATION,
    LifecycleState.SEEKING_DISPOSAL
)


# ============================================================
# LIFECYCLE MODEL(S)
# ============================================================

"""
total steps
total time
mean costs
"""

class LifecycleModel(ABC):
    @abstractmethod
    def sample_lifecycle(
        self,
        starting_distribution: dict[LifecycleState, float],
        num_samples: int | None = None
    ) -> Lifecycle | list[Lifecycle]:
        pass

    @abstractmethod
    def lifecycle_metrics(
        self,
        starting_distribution: dict[LifecycleState, float],
        num_samples: int
    ) -> dict[str, float]:
        pass
    

class _ForwardLifecycleModel(LifecycleModel):
    ...

class _BackwardLifecycleModel:
    ...

class CostModel:
    AUTOMATED_COSTING_DIMENSIONS = {CostingDimensions.STORAGE, CostingDimensions.OPPORTUNITY_COST, CostingDimensions.LOST_REVENUE}
    

    def __init__(
        self,
        facility: Facility,
        item: Item,
        state_wait_time_distributions: dict[State, QuantitativeDistribution],
        edge_cost_functions: dict[str, dict[str, dict[str, Callable[[float], float]]]],  # start -> end -> costing dimension -> C(t)
        edge_probability_functions: dict[str, dict[str, Callable[[float], float]]],      # start -> end -> P_t
        lost_revenue_rates: dict[State, float],  # e.g., on average the liquidated item sells for 20% of its normal sales price;
                                                 # should only be provided for the terminal outcomes (forward direction) to prevent double counting
        reversed: bool = False  # ! directionality of edge_cost_functions and edge_probability_functions needs to be consistent with reversed
    ):
        transition_graph = WeightedDigraph.from_dict(  # create edges and default attributes
            edges=self.FORWARD_EDGES,
            default_vertex_attributes={state: _ZERO_DIST for state in State if state},
            default_edge_attributes={
                costing_dimension: _ZERO_FUNC
                for costing_dimension in CostingDimensions
                if costing_dimension not in self.AUTOMATED_COSTING_DIMENSIONS
            }
        )

        if reversed:
            transition_graph = transition_graph.reverse(inplace=True)

        for state, distribution in state_wait_time_distributions.items():  # write wait time distributions
            self.transition_graph.set_vertex_attributes(state, **{self._WAIT_TIME_DIST_KW: distribution})

        for start, end_costdim_costfunc in edge_cost_functions.items():  # write cost and probability functions
            for end, costdim_costfunc in end_costdim_costfunc.items():
                transition_graph.set_edge_attributes(
                    start,
                    end,
                    **costdim_costfunc,
                    **{
                        self._PROBABILITY_FUNC_KW: edge_probability_functions[start][end],
                        CostingDimensions.STORAGE: self._storage_cost,
                        CostingDimensions.OPPORTUNITY_COST: self._opportunity_cost
                    }
                )

        self.facility = facility
        self.item = item
        self.reversed = reversed
        self.transition_graph = transition_graph
        self.lost_revenue_rates = lost_revenue_rates
    
    def sample_lifecycle(self, starting_distribution: dict[State, float]) -> Lifecycle:
        start = sample(zip(*starting_distribution.items()))
        return self._sample_lifecycle_forward(start) if not self.reversed else self._sample_lifecycle_reverse(start)

    def _sample_lifecycle_forward(self, start: State) -> Lifecycle:
        lifecycle = []

        current = self.transition_graph.get_vertex(start)
        while self.transition_graph.out_degree(start) > 0:
            # sample wait time
            realized_wait_time = current.attributes[self._WAIT_TIME_DIST_KW].sample()

            # sample outgoing edge from transition probabilities evaluated at realized wait time
            out_edges = list(self.transition_graph.out_edges(current))
            out_edge = sample(
                out_edges,
                p = [e.attributes[self._PROBABILITY_FUNC_KW](realized_wait_time) for e in out_edges],
                normalize=True
            )
            next_ = out_edge.end.id

            # evaluate costs
            costs = {
                **{
                    costing_dimension: cost_func(realized_wait_time)
                    for costing_dimension, cost_func in out_edge.attributes.items()
                    if costing_dimension in CostingDimensions
                },
                CostingDimensions.LOST_REVENUE: self._lost_revenue_cost(next_)
            }

            # write step
            lifecycle.append(LifecycleStep(current, next_, realized_wait_time, costs))

            # update
            current = next_
        
        return lifecycle

    def _sample_lifecycle_reverse(self, start: State) -> Lifecycle:
        lifecycle = []

        current = self.transition_graph.get_vertex(start)
        while self.transition_graph.out_degree(start) > 0:
            # sample wait time
            realized_wait_time = current.attributes[self._WAIT_TIME_DIST_KW].sample()

            # sample outgoing edge from transition probabilities evaluated at realized wait time
            out_edges = list(self.transition_graph.out_edges(current))
            out_edge = sample(
                out_edges,
                p = [e.attributes[self._PROBABILITY_FUNC_KW](realized_wait_time) for e in out_edges],
                normalize=True
            )
            next_ = out_edge.end.id

            # evaluate costs
            costs = {
                **{
                    costing_dimension: cost_func(realized_wait_time)
                    for costing_dimension, cost_func in out_edge.attributes.items()
                    if costing_dimension in CostingDimensions
                },
                CostingDimensions.LOST_REVENUE: self._lost_revenue_cost(next_)
            }

            # write step
            lifecycle.append(LifecycleStep(current, next_, realized_wait_time, costs))

            # update
            current = next_
        
        return reversed(lifecycle)

    
    def metrics(self):  # sample random walks and calculate aggregate statistics
        ...
    

    def _storage_cost(self, t: float) -> float:
        return t * self.item.volume * self.facility.storage_cost_rate
    
    def _opportunity_cost(self, t: float) -> float:
        return t * self.item.volume * self.facility.profit_rate

    def _lost_revenue_cost(self, outcome: State) -> float:
        return self.item.sales_price * self.lost_revenue_rates.get(outcome, 0)
    
    
    def _validate_transition_probabilities(self):


class ReversedCostModel:
    def __init__()