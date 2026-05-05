# standard
from typing import Self

# local
from src.pipeline.types import PipelineStep

class Pipeline[Context]:
    """
    A generic pipeline that executes a sequence of steps on a context object.
    
    Steps can be added dynamically with `extend`, and pipelines can be combined using the + operator.
    """

    __slots__ = ("pipeline")

    def __init__(self, *steps: PipelineStep | None):
        self.pipeline = list(step for step in steps if step is not None)
    
    def __call__(self, context: Context) -> Context:
        c = context
        for step in self.pipeline:
            if step:
                c = step(c)
        return c

    def extend(self, *steps: PipelineStep):
        self.pipeline.extend(steps)

    def __add__(self, other: Self) -> Self:
        if isinstance(other, Pipeline):
            return Pipeline(
                *self.pipeline,
                *other.pipeline
            )
        raise NotImplementedError()
