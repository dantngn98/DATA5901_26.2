# standard
from abc import abstractmethod
from typing import Protocol

class PipelineStep[Context](Protocol):
    """
    A `PipelineStep` is a callable that takes in context, performs computations with and on it, and
    then passes it along.
    """
    
    @abstractmethod
    def __call__(self, context: Context) -> Context:
        pass
