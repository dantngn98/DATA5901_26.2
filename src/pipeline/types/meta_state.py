# standard
from dataclasses import dataclass

@dataclass(frozen=True)
class MetaState:
    """
    Represents the metastate of a variable in `Context`.

    Indicates whether a variable exists and whether it is locked.
    """

    exists: bool
    locked: bool
