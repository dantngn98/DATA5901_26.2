# standard
from abc import ABC, abstractmethod

# local
from src.pipeline.types.meta_state import MetaState


class BoundCondition(ABC):
    """
    The result of binding a `Condition` to a specific pre-state.

    Used for conditions that need to reference both pre and post states.
    """

    @abstractmethod
    def postcondition(self, state: MetaState) -> bool:
        pass

    @abstractmethod
    def simulate(self) -> MetaState | None:
        pass
