# standard
from io import BytesIO
from os import PathLike, fspath
from typing import Iterable

# third-party
import boto3
import joblib
import polars as pl

# local
from src.util.s3_path import S3Path


# ============================================================
# DATAFRAME IO UTILS
# ============================================================

def load_dataframe(
    source: (str | PathLike[str]) | Iterable[str | PathLike[str]],
    csv_delimiter: str = ","
) -> pl.DataFrame:
    # need to check this case first because str is Iterable
    if (source_str := _str_or_pathlike_to_string(source)) is not None:
        return _load(source_str, csv_delimiter)
    elif isinstance(source, Iterable):
        dataframes = []
        for fp in source:
            fp_str = _str_or_pathlike_to_string(fp)
            if fp_str is None:
                raise TypeError(f"invalid path element: {type(fp)}")
            dataframes.append(_load(fp_str, csv_delimiter))
        return pl.concat(dataframes)
    raise TypeError(f"expected type 'str' or 'PathLike[str]' (or an 'Iterable' of such) but got {type(source)}")

def _load(fp: str, csv_delimiter: str) -> pl.DataFrame:
    if not isinstance(fp, str):
        raise TypeError(f"expected type 'str' but got {type(fp)}")
    
    # "filename" can include path (e.g., dir/example.txt -> ["dir/example", ".", "txt"])
    filename, dot, extension = fp.rpartition(".")

    if not dot:
        raise ValueError(f"missing file extension: '{fp}'")
    
    dispatch = {
        "csv": lambda fp: pl.read_csv(fp, separator=csv_delimiter),
        "parquet": pl.read_parquet
    }.get(extension)

    if dispatch is None:
        raise ValueError(f"Unknown extension for file '{fp}'")
    return dispatch(fp)


def write_dataframe(
    dataframe: pl.DataFrame,
    destination: str | PathLike[str],
    csv_delimiter: str = ","
):
    if (fp := _str_or_pathlike_to_string(destination)) is None:
        raise TypeError(f"expected type 'str' or 'PathLike[str]' (or an 'Iterable' of such) but got {type(destination)}")
    
    if not isinstance(fp, str):
        raise TypeError(f"expected type 'str' but got {type(fp)}")
    
    # "filename" can include path (e.g., dir/example.txt -> ["dir/example", ".", "txt"])
    filename, dot, extension = fp.rpartition(".")

    if not dot:
        raise ValueError(f"missing file extension: '{fp}'")
    
    if extension == "csv":
        return dataframe.write_csv(fp, separator=csv_delimiter)
    elif extension == "parquet":
        return dataframe.write_parquet()
    raise ValueError(f"Unknown extension for file '{fp}'")


def _str_or_pathlike_to_string(source: object) -> str | None:
    try:
        source_str = fspath(source)  # can be str or bytes
        return source_str if isinstance(source_str, str) else None
    except TypeError:
        return None


# ============================================================
# MODEL IO UTILS
# ============================================================

_s3 = boto3.client("s3")

def load_joblib_from_s3(source: S3Path) -> object:
    buffer = BytesIO()
    _s3.download_fileobj(source.bucket, source.key, buffer)
    buffer.seek(0)
    return joblib.load(buffer)

def write_joblib_to_s3(obj: object, destination: S3Path):
    buffer = BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    _s3.upload_fileobj(buffer, destination.bucket, destination.key)
