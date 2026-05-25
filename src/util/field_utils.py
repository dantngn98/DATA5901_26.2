# used for converting a string to a normalized column name
def normalize_to_column_name(string: str) -> str:
    if string == "RETAIL" or string == "FBA":
        return "_".join(string.strip().split())
    else:
        return "_".join(string.strip().lower().split())
