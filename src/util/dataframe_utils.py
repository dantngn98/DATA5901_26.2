# standard
from typing import Iterable

# third-party
import pandas as pd

def cast_categoricals(
    df: pd.DataFrame,
    categorical_columns: Iterable[str],
    inplace: bool = True
) -> pd.DataFrame:
    if not inplace:
        df = df.copy()
    for column in categorical_columns:
        if column in df.columns:
            df[column] = df[column].astype("category")
    return df
