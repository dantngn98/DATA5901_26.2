# standard
from typing import Iterable

# third-party
import polars as pl


def load(source: str | Iterable[str], csv_delimiter: str = ",") -> pl.DataFrame:
    # str is Iterable!
    if not isinstance(source, str) and isinstance(source, Iterable):
        return pl.concat([
            _load(fp, csv_delimiter) for fp in source
        ])
    return _load(source, csv_delimiter)

def _load(fp: str, csv_delimiter: str) -> pl.DataFrame:
    if not isinstance(fp, str):
        raise TypeError(f"expected type 'str' but got {type(fp)}")
    
    # "filename" can include path (e.g., dir/example.txt -> ["dir/example", ".", "txt"])
    filename, dot, extension = fp.rpartition(".")

    if len(extension) == len(fp):
        raise ValueError(f"missing file extension: '{fp}'")
    
    dispatch = {
        "csv": lambda fp: pl.read_csv(fp, separator=csv_delimiter),
        "parquet": pl.read_parquet
    }.get(extension)

    if dispatch is None:
        raise ValueError(f"Unknown extension for file '{fp}'")
    return dispatch(fp)
