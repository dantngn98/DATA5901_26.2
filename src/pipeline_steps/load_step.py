# standard
import logging
from typing import Iterable

# local
from src.config import ContextKeys, CSV_DELIMITER
from src.pipeline import Context, enforce
from src.pipeline.types import PipelineStep
from src.pipeline.conditions import Defines, Locks, Sequence
from src.util import load


logger = logging.getLogger(__name__)

@enforce({
    ContextKeys.DF_RECOVERY_LOADED: Sequence(Defines(strict=True), Locks(strict=True))
})
class Load(PipelineStep):
    """
    Loads unprocessed recovery data from provided source file(s) into a Polars DataFrame.
    If multiple files are provided, loads each file individually and then concatenates
    the resultant dataframes together.
    """

    def __init__(self, recovery_data_source: str | Iterable[str]):
        self.recovery_data_source = recovery_data_source

    def __call__(self, context: Context) -> Context:
        logger.info(f"loading data from {self.recovery_data_source}")
        context[ContextKeys.DF_RECOVERY_LOADED] = load(self.recovery_data_source, CSV_DELIMITER)
        logger.info("data loaded")
        context.lock(ContextKeys.DF_RECOVERY_LOADED)
        return context
