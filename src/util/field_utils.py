# used for converting a string to a normalized column name
def normalize_to_column_name(string: str) -> str:
    return "_".join(string.strip().lower().split())