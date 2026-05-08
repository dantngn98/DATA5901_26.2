# standard
from typing import Iterable

# third-party
import polars as pl

def load(source: str | Iterable[str]) -> pl.DataFrame:
    if not isinstance(source, str) and isinstance(source, Iterable):
        return pl.concat([
            load(fp) for fp in source
        ])
    
    filename, dot, extension = source.rpartition(".")
    assert len(extension) < len(source)  # check existence of .

    dispatch = {
        "csv": lambda fp: pl.read_csv(fp, separator="\t"),  # assume tab-delimited
        "parquet": pl.read_parquet
    }.get(extension)

    if dispatch is None:
        raise ValueError(f"Unknown extension for source file '{source}'")
    return dispatch(source)