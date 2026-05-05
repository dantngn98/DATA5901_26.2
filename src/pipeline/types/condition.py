# standard
from abc import ABC, abstractmethod

# local
from src.pipeline.types.bound_condition import BoundCondition
from src.pipeline.types.meta_state import MetaState


class Condition(ABC):
    """A `Condition` encodes variable-level pre and post `MetaState` validation."""

    def precondition(self, state: MetaState) -> bool:
        return self.bind(state) is not None
    
    def simulate(self, in_state: MetaState) -> MetaState | None:
        bound = self.bind(in_state)
        return bound.simulate() if bound is not None else None

    @abstractmethod
    def bind(self, state: MetaState) -> BoundCondition | None:
        pass
    
    def __repr__(self):
        return self.__class__.__name__
