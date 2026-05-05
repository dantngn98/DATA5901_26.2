# local
from src.pipeline.types.bound_condition import BoundCondition
from src.pipeline.types.condition import Condition
from src.pipeline.types.meta_state import MetaState


class _DefaultBoundCondition(BoundCondition):
    def __init__(self, condition: Condition, pre_state: MetaState):
        self.condition = condition
        self.pre_state = pre_state

    def postcondition(self, state: MetaState) -> bool:
        return self.condition.postcondition(state)

    def simulate(self) -> MetaState | None:
        return self.condition.simulate(self.pre_state)


class FlatCondition(Condition):
    """
    A `Condition` where postconditions depend only on the post-state, not pre-state.
    
    Simplifies condition implementation by flattening the validation logic.
    """

    def precondition(self, state: MetaState) -> bool:
        return True

    def postcondition(self, state: MetaState) -> bool:
        return True
    
    def simulate(self, in_state: MetaState) -> MetaState | None:
        return None
    
    def bind(self, state: MetaState) -> BoundCondition | None:
        return None if not self.precondition(state) else _DefaultBoundCondition(self, state)
