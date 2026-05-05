# local
from src.pipeline.types import FlatCondition, MetaState

class Requires(FlatCondition):
    def __init__(self): ...

    def precondition(self, state: MetaState) -> bool:
        return state.exists
