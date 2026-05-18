# local
from src.util.s3_path import S3Path  # direct import to avoid potential circularity issues


# ============================================================
# PATH CONFIG
# ============================================================
"""
REORGANIZED
s3://msds-26.2-data/
├── data/
│   ├── unprocessed/
│       ├── sanitized_2022.csv, sanitized_2023.csv, ..., sanitized_2025.csv
│       ├── waste_rate_card.csv
│   ├── processed/
│       ├── preprocessed_recovery_data.parquet
├── model/
│   ├── tuned_xgboost_classification_model.joblib
│   ├── tuned_xgboost_regression_model.joblib
│   └── recovery_channel_regression_models/
│       ├── prob_bintool_donations_clf.joblib
│       ├── prob_bintool_donations_reg.joblib
│       ├── prob_bintool_remove_liquidate_clf.joblib
│       ├── prob_bintool_remove_liquidate_reg.joblib
│       └── ...
├── out/
|   ├── predictions.parquet
|   ├── report.html
"""
"""
s3://msds-26.2-data/
├── model/
│   ├── tuned_xgboost_classification_model.joblib
│   ├── tuned_xgboost_regression_model.joblib
|   ├── predictions.parquet
│   └── recovery_channel_regression/
│       ├── prob_bintool_donations_clf.joblib
│       ├── prob_bintool_donations_reg.joblib
│       ├── prob_bintool_remove_liquidate_clf.joblib
│       ├── prob_bintool_remove_liquidate_reg.joblib
│       └── ...
"""
# === S3 data storage ===
S3_BUCKET_URI = "s3://msds-26.2-data"
YEARLY_RECOVERY_DATA_CSV_FPS = [
    str(f"{S3_BUCKET_URI}/data/unprocessed/sanitized_{year}.csv")
    for year in range(2022, 2025)
]
# MODEL_DIR = S3_BUCKET/"model"
# CLF_MODEL_JOBLIB = MODEL_DIR/"tuned_xgboost_classification_model.joblib"
# REG_MODEL_JOBLIB = MODEL_DIR/"tuned_xgboost_regression_model.joblib"
# SHARE_MODELS_DIR = MODEL_DIR/"recovery_channel_regression"
PREDICTIONS_PARQUET = f"{S3_BUCKET_URI}/out/predictions.parquet"


# === S3 model storage ===
S3_BUCKET = "msds-26.2-data"       # raw bucket name (no "s3://" prefix) for boto3
MODEL_DIR = "model"                 # S3 key prefix for saved models
CLF_MODEL_S3_KEY = f"{MODEL_DIR}/tuned_xgboost_classification_model.joblib"
REG_MODEL_S3_KEY = f"{MODEL_DIR}/tuned_xgboost_regression_model.joblib"
SHARE_MODELS_S3_PREFIX = f"{MODEL_DIR}/recovery_channel_share_softmax"  # per-channel share regressors live under this prefix
PREDICTIONS_S3_KEY = f"{MODEL_DIR}/predictions.parquet"


# ============================================================
# DATA LOADING CONFIG
# ============================================================

CSV_DELIMITER = "\t"


# ============================================================
# INPUT DATA CONFIG
# ============================================================

class RecoverySchema:
    # time fields
    YEAR = "year"
    MONTH = "month"
    WEEK = "week"
    WEEK_DT = "week_dt"

    # site fields
    HASHED_FC = "hashed_fc"
    COUNTRY = "country"
    COUNTRY_STATE = "country_state"
    ZIP_CODE = "zip_code"
    SITE_TYPE = "site_type"
    SITE_CATEGORY = "site_category"
    FC_COUNTRY = "fc_country"
    MARKETPLACE_ID = "marketplace_id"

    # product fields
    PRODUCT_TYPE = "product_type"
    PRODUCT_TYPE_GRANULAR = "product_type_granular"
    MACRO_CATEGORY = "macro_category"
    GL_PRODUCT_GROUP = "gl_product_group"
    GL_PRODUCT_GROUP_DESC = "gl_product_group_desc"
    ASIN = "asin"
    VENDOR_CODE = "vendor_code"
    VENDOR_NAME = "vendor_name"
    CATEGORY_CODE = "category_code"
    SUBCATEGORY_CODE = "subcategory_code"

    # recovery/operational fields
    RECOVERY_TYPE = "recovery_type"
    REASON_CODE = "reason_code"
    REASON_CODE_TYPE = "reason_code_type"
    REASON_CODE_STRANDED = "reason_code_stranded"
    STRANDED_POTENTIAL_ISSUE = "stranded_potential_issue"
    STRANDED_SUB_REASON = "stranded_sub_reason"
    APPLICATION_NAME = "application_name"
    ITEM_DISPOSITION_CODE = "item_disposition_code"

    # flag fields
    IS_STRANDED = "is_stranded"
    IS_EXCEPTION_DESTROY = "is_exception_destroy"
    IN_SCOPE_BAN = "in_scope_ban"
    EXCL_FROM_REPORTING = "excl_from_reporting"
    IS_POD = "is_pod"
    IS_EXPIRATION_DATED_PRODUCT = "is_expiration_dated_product"
    IS_HAZMAT = "is_hazmat"
    RAITH_FLAG = "raith_flag"

    # metric fields
    UNITS = "units"
    COGS = "cogs"
    WEIGHT = "weight"


# ============================================================
# SELF-DEFINED NAMES CONFIG
# ============================================================

class ContextKeys:
    DF_RECOVERY_LOADED = "df_recovery_loaded"
    DF_RECOVERY_PREPROCESSED = "df_recovery_preprocessed"
    CLF_MODEL = "clf_model"         # trained XGBClassifier (Stage 1)
    REG_MODEL = "reg_model"         # trained XGBRegressor (Stage 2)
    SHARE_MODELS = "share_models"   # dict[channel_name -> XGBRegressor], 9 per-channel share regressors
    PREDICTIONS = "predictions"     # output Polars DataFrame from Predict step
