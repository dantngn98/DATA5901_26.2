# === paths ===

S3_ROOT_DIR = "s3://"
DATA_DIR = f"{S3_ROOT_DIR}msds-26.2-data"
CLEANED_DATA_DIR = f"{DATA_DIR}/clean"

YEARLY_RECOVERY_DATA_CSV_FPS = [
    f"{DATA_DIR}/sanitized_{year}.csv"
    for year in range(2022, 2026)
]
COMBINED_RECOVERY_DATA_PARQUET_FP = f"{CLEANED_DATA_DIR}/combined_recovery_data.parquet"
# ... etc.


# === field names ===

# TODO: dataframe field names to reduce hard-coded strings?

class ContextKeys:
    DF_RECOVERY_LOADED = "df_recovery_loaded"
    DF_RECOVERY_PREPROCESSED = "df_recovery_preprocessed"
