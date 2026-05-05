# local
from src.config import ContextKeys
from src.pipeline import Context, enforce
from src.pipeline.types import PipelineStep
from src.pipeline.conditions import Defines, Locks, Sequence
from src.util import load


@enforce({
    ContextKeys.DF_RECOVERY_LOADED: Sequence(Defines(strict=True), Locks(strict=True))
})
class Load(PipelineStep):
    def __init__(self, recovery_data_source: str | list[str]):
        self.source = recovery_data_source

    def __call__(self, context: Context) -> Context:
        context[ContextKeys.DF_RECOVERY_LOADED] = load(self.recovery_data_source)
        context.lock(ContextKeys.DF_RECOVERY_LOADED)
        return context
