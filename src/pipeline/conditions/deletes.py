# local
from src.pipeline.types import FlatCondition, MetaState

class Deletes(FlatCondition):
    def __init__(self, strict: bool = True):
        self.strict = strict

    def precondition(self, state: MetaState) -> bool:
        if self.strict:
            return state.exists and not state.locked
        return not state.exists or (state.exists and not state.locked)
    
    def postcondition(self, state: MetaState) -> bool:
        return not state.exists
    
    def simulate(self, in_state: MetaState) -> MetaState | None:
        return None if not self.precondition(in_state) else MetaState(exists=False, locked=False)
    
    def __repr__(self) -> str:
        return f"Deletes(strict={self.strict})"
