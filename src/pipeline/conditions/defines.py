# local
from src.pipeline.types import FlatCondition, MetaState

class Defines(FlatCondition):
    def __init__(self, strict: bool = True):
        self.strict = strict

    def precondition(self, state: MetaState) -> bool:
        return not (self.strict and state.exists)
    
    def postcondition(self, state: MetaState) -> bool:
        return state.exists
    
    def simulate(self, in_state: MetaState) -> MetaState | None:
        return None if not self.precondition(in_state) else MetaState(exists=True, locked=False)
    
    def __repr__(self) -> str:
        return f"Defines(strict={self.strict})"
