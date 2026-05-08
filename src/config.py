# === paths ===

S3_ROOT_DIR = "s3://"
DATA_DIR = f"{S3_ROOT_DIR}msds-26.2-data"
CLEANED_DATA_DIR = f"{DATA_DIR}/clean"

YEARLY_RECOVERY_DATA_CSV_FPS = [
    f"{DATA_DIR}/sanitized_{year}.csv"
    for year in range(2022, 2026)
]
COMBINED_RECOVERY_DATA_PARQUET_FP = f"{CLEANED_DATA_DIR}/combined_recovery_data.parquet"
AGGREGATED_DATA_PARQUET_FP = f"{CLEANED_DATA_DIR}/combined_recovery_data_aggregated_with_full_features.parquet"
# ... etc.

# === S3 model storage ===
S3_BUCKET = "msds-26.2-data"       # raw bucket name (no "s3://" prefix) for boto3
MODEL_DIR = "model"                 # S3 key prefix for saved models
CLF_MODEL_S3_KEY = f"{MODEL_DIR}/tuned_xgboost_classification_model.joblib"
REG_MODEL_S3_KEY = f"{MODEL_DIR}/tuned_xgboost_regression_model.joblib"
SHARE_MODELS_S3_PREFIX = f"{MODEL_DIR}/recovery_channel_share_softmax"  # per-channel share regressors live under this prefix


# === field names ===

# TODO: dataframe field names to reduce hard-coded strings?

PREDICTIONS_S3_KEY = f"{MODEL_DIR}/predictions.parquet"


class ContextKeys:
    DF_RECOVERY_LOADED = "df_recovery_loaded"
    DF_RECOVERY_PREPROCESSED = "df_recovery_preprocessed"
    CLF_MODEL = "clf_model"         # trained XGBClassifier (Stage 1)
    REG_MODEL = "reg_model"         # trained XGBRegressor (Stage 2)
    SHARE_MODELS = "share_models"   # dict[channel_name -> XGBRegressor], 9 per-channel share regressors
    PREDICTIONS = "predictions"     # output Polars DataFrame from Predict step
