# local
from src.pipeline.types import FlatCondition, MetaState

class Unlocks(FlatCondition):
    def __init__(self, strict: bool = True):
        self.strict = strict

    def precondition(self, state: MetaState) -> bool:
        return state.exists and not (self.strict and not state.locked)
    
    def postcondition(self, state: MetaState) -> bool:
        return state.exists and not state.locked
    
    def simulate(self, in_state: MetaState) -> MetaState | None:
        return None if not self.precondition(in_state) else MetaState(exists=True, locked=False)
    
    def __repr__(self) -> str:
        return f"Unlocks(strict={self.strict})"
