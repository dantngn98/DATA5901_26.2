# standard
from typing import Callable

# local
from src.lifecycle_model.types import (
    QuantitativeDistribution,
    Item, Facility,
    CostingDimension,
    LifecycleState, LifecycleStep, Lifecycle,
    LifecycleModel
)
from src.lifecycle_model.weighted_digraph import WeightedDigraph
from src.lifecycle_model.distributions import ConstantDistribution
from src.util import sample


# for default attributes; instantiate once
_ZERO_DIST = ConstantDistribution(0)
_ZERO_FUNC = lambda t: 0

class ForwardLifecycleModel(LifecycleModel):
    def __init__(
        self,
        facility: Facility,
        item: Item,
        state_wait_time_distributions: dict[LifecycleState, QuantitativeDistribution],
        edge_cost_functions: dict[str, dict[str, dict[str, Callable[[float], float]]]],  # start -> end -> costing dimension -> cost(t)
        edge_probability_functions: dict[str, dict[str, Callable[[float], float]]],      # start -> end -> probability(t)
        lost_revenue_rates: dict[LifecycleState, float],  # e.g., on average the liquidated item sells for 20% of its normal sales price;
                                                          # should only be provided for the terminal outcomes to prevent double counting
    ):
        transition_graph = WeightedDigraph.from_dict(  # create edges and default attributes
            edges=self.FORWARD_EDGES,
            default_vertex_attributes={
                # use Enum str values internally
                **{state.value: _ZERO_DIST for state in LifecycleState if state},
                self._WAIT_TIME_DIST_KW: _ZERO_DIST
            },
            default_edge_attributes={
                **{
                    costing_dimension.value: _ZERO_FUNC
                    for costing_dimension in CostingDimension
                    if costing_dimension not in self.AUTOMATED_COSTING_DIMENSIONS
                },
                self._PROBABILITY_FUNC_KW: _ZERO_FUNC
            }
        )

        for state, distribution in state_wait_time_distributions.items():  # write wait time distributions
            transition_graph.set_vertex_attributes(state, **{self._WAIT_TIME_DIST_KW: distribution})

        for start, end_costdim_costfunc in edge_cost_functions.items():  # write cost and probability functions
            for end, costdim_costfunc in end_costdim_costfunc.items():
                transition_graph.set_edge_attributes(
                    start,
                    end,
                    **{costdim.value: costfunc for costdim, costfunc in costdim_costfunc.items()},
                    **{self._PROBABILITY_FUNC_KW: edge_probability_functions[start][end]}
                )

        self.facility = facility
        self.item = item
        self.transition_graph = transition_graph.freeze()
        self.lost_revenue_rates = lost_revenue_rates
    
    def sample_lifecycle(
        self,
        starting_distribution: dict[LifecycleState, float],
        num_samples: int | None = None
    ) -> Lifecycle | list[Lifecycle]:
        states, probabilities = zip(*starting_distribution.items())

        if num_samples is None:
            return self._sample_lifecycle(sample(states, probabilities))
        
        starting_states = sample(states, probabilities, num_samples=num_samples)
        return [self._sample_lifecycle(start) for start in starting_states]

    def _sample_lifecycle(self, start: LifecycleState) -> Lifecycle:
        lifecycle = []

        current = self.transition_graph.get_vertex(start)
        while self.transition_graph.out_degree(current.id) > 0:
            # sample wait time
            realized_wait_time = current.attributes[self._WAIT_TIME_DIST_KW].sample()

            # sample outgoing edge from transition probabilities evaluated at realized wait time
            out_edges = list(self.transition_graph.out_edges(current.id))
            out_edge = sample(
                out_edges,
                p = [e.attributes[self._PROBABILITY_FUNC_KW](realized_wait_time) for e in out_edges],
                normalize=True
            )
            next_ = out_edge.end

            # evaluate costs
            costs = {
                **{
                    costing_dimension: cost_func(realized_wait_time)
                    for costing_dimension, cost_func in out_edge.attributes.items()
                    if costing_dimension in CostingDimension
                },
                CostingDimension.STORAGE.value: self._storage_cost(current.id, realized_wait_time),
                CostingDimension.OPPORTUNITY_COST.value: self._opportunity_cost(current.id, realized_wait_time),
                CostingDimension.LOST_REVENUE.value: self._lost_revenue_cost(next_.id)
            }

            # write step
            lifecycle.append(LifecycleStep(current.id, next_.id, realized_wait_time, costs))

            # update
            current = next_
        
        return lifecycle
    