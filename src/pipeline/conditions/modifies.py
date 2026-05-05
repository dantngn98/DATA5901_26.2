# local
from src.pipeline.types import FlatCondition, MetaState

class Modifies(FlatCondition):
    def __init__(self): ...

    def precondition(self, state: MetaState) -> bool:
        return state.exists and not state.locked

    # postcondition validation (checking if the value actually changed) infeasible due to possibility of mutable objects

    def simulate(self, in_state: MetaState) -> MetaState | None:
        return None if not self.precondition(in_state) else MetaState(exists=True, locked=False)
