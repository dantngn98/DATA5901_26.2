# local
from src.util.s3_path import S3Path  # use direct imports to avoid potential circularity issues
from src.util.field_utils import normalize_to_column_name


# ============================================================
# DATA CONFIG
# ============================================================
# (try to) minimize writing hard-coded strings

class RecoverySchema:  # column names of recovery data
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

# unique values of recovery_type field
# WARNING: if these change, then you'll need to be careful to check if the preprocessing PipelineStep
#          and model configuration need to be updated
class RecoveryTypes:
    SALES = "Sales"
    RETURN_TO_VENDOR = "Return to Vendor"
    WAREHOUSE_DEALS_AND_GR = "Warehouse Deals and G&R"
    DONATIONS = "Donations"
    BINTOOL_DONATIONS = "Bintool Donations"
    LIQUIDATIONS = "Liquidations"
    REMOVE_RETURN = "Remove Return"
    BINTOOL_REMOVE_LIQUIDATE = "Bintool Remove Liquidate"
    REMOVE_LIQUIDATE = "Remove Liquidate"
    BINTOOL_THEFT = "Bintool Theft"
    C_RETURNS = "C-Returns"

RECOVERY_TYPES = {
    getattr(RecoveryTypes, attr)
    for attr in dir(RecoveryTypes)
    if not (attr.startswith("__") and attr.endswith("__"))
}
MACRO_CATEGORIES = {"RETAIL", "FBA"}  # TODO: make these dynamic? (would require eager evaluation)
PRODUCT_TYPES = {"Food", "Non Food", "Pet Food"}


# ============================================================
# LOADING CONFIG
# ============================================================

CSV_DELIMITER = "\t"


# ============================================================
# PREPROCESSING CONFIG
# ============================================================

NORMALIZED_MACRO_CATEGORY_DICT = {
    macro_category: normalize_to_column_name(macro_category)
    for macro_category in MACRO_CATEGORIES
}
NORMALIZED_PRODUCT_TYPES_DICT = {
    product_type: normalize_to_column_name(product_type)
    for product_type in PRODUCT_TYPES
}

RECOVERY_TYPES_TO_DROP = {RecoveryTypes.BINTOOL_THEFT, RecoveryTypes.C_RETURNS}
RECOVERY_TYPES_TO_KEEP = RECOVERY_TYPES - RECOVERY_TYPES_TO_DROP

class ConsolidatedRecoveryTypes:
    SALES = "sales"
    RETURN_TO_VENDOR = "return_to_vendor"
    WAREHOUSE_DEALS_AND_GR = "warehouse_deals_and_gr"
    LIQUIDATIONS = "liquidations"
    DONATIONS = "donations"
    DISPOSAL = "disposal"

CONSOLIDATED_RECOVERY_TYPES = {
    getattr(ConsolidatedRecoveryTypes, attr)
    for attr in dir(ConsolidatedRecoveryTypes)
    if not (attr.startswith("__") and attr.endswith("__"))
}
CONSOLIDATED_RECOVERY_TYPE_DICT = {  # recovery type -> consolidated recovery type
    RecoveryTypes.SALES: ConsolidatedRecoveryTypes.SALES,
    RecoveryTypes.RETURN_TO_VENDOR: ConsolidatedRecoveryTypes.RETURN_TO_VENDOR,
    RecoveryTypes.WAREHOUSE_DEALS_AND_GR: ConsolidatedRecoveryTypes.WAREHOUSE_DEALS_AND_GR,
    RecoveryTypes.DONATIONS: ConsolidatedRecoveryTypes.DONATIONS,
    RecoveryTypes.BINTOOL_DONATIONS: ConsolidatedRecoveryTypes.DONATIONS,
    RecoveryTypes.LIQUIDATIONS: ConsolidatedRecoveryTypes.LIQUIDATIONS,
    RecoveryTypes.REMOVE_RETURN: ConsolidatedRecoveryTypes.DISPOSAL,
    RecoveryTypes.BINTOOL_REMOVE_LIQUIDATE: ConsolidatedRecoveryTypes.DISPOSAL,
    RecoveryTypes.REMOVE_LIQUIDATE: ConsolidatedRecoveryTypes.DISPOSAL
}
assert all(
    recovery_type in CONSOLIDATED_RECOVERY_TYPE_DICT
    for recovery_type in RECOVERY_TYPES
    if recovery_type in RECOVERY_TYPES_TO_KEEP
)

LAG_WEEKS = [1, 4, 12, 13, 52]
ROLLING_WEEKS = [4, 12]
ROLLING_WEEKS_LONG = [26, 52]
EWMA_ALPHAS = [0.5, 0.1]


# ============================================================
# MODEL TRAINING CONFIG
# ============================================================

DEFAULT_TRAIN_YEARS = [2022, 2023, 2024]
DEFAULT_TEST_YEARS = [2025]

CATEGORICAL_COLUMNS = {  # used for casting pandas df columns (if present) 
    RecoverySchema.HASHED_FC,
    RecoverySchema.GL_PRODUCT_GROUP,
    RecoverySchema.COUNTRY,
    RecoverySchema.COUNTRY_STATE,
    RecoverySchema.SITE_TYPE,
    RecoverySchema.SITE_CATEGORY
}

_COMPOSITION_COLUMNS = {
    *NORMALIZED_MACRO_CATEGORY_DICT.values(),
    *NORMALIZED_PRODUCT_TYPES_DICT.values(),
    "hazmat"
}

RECOVERY_RATE_TARGET_COLUMN = "prob_recovered"

# =====================================
# RECOVERY RATE BINARY CLASSIFIER CONFIG
# =====================================
# NOTE: highly coupled w/ preprocessing code, but probably the best we can do without at least
#       a mini declarative feature language that can express and track the different compositions
#       of variable-ordered features (in the sense of a "0-ordered feature" being one already in the
#       data and a "n-ordered feature" being one computed only from (n-1)-ordered features or lower)

_GL_COMPOSITION_COLS = {f"share_{c}" for c in _COMPOSITION_COLUMNS}
_GL_VOLUME_COLS = {
    "units_total", "cogs_total", "weight_total",
    "avg_cogs_per_unit", "avg_weight_per_unit", "cogs_per_unit_weight"
}
_GL_AT_SITE_COLS = {"site_units_share_week", "site_weight_share_week"}
_SITE_CONTEXT_COLS = {
    "site_units_total_week", "site_weight_total_week",
    RecoverySchema.SITE_TYPE, RecoverySchema.SITE_CATEGORY, RecoverySchema.COUNTRY, RecoverySchema.COUNTRY_STATE,
}
_TEMPORAL_SITE_CONTEXT_COLS = {
    *[f"site_units_total_week_lag_{w}w" for w in LAG_WEEKS],
    *[f"site_weight_total_week_lag_{w}w" for w in LAG_WEEKS],
    *[f"site_prob_recovered_week_lag_{w}w" for w in LAG_WEEKS],
    *[f"site_prob_recovered_week_rolling_{w}w" for w in ROLLING_WEEKS + ROLLING_WEEKS_LONG]
}
_CALENDAR_COLS = {"month", "week"}
_TEMPORAL_COMPOSITION_COLS = {
    *[f"share_{c}_lag_{w}w" for c in _COMPOSITION_COLUMNS for w in LAG_WEEKS],
    *[f"share_{c}_rolling_{w}w" for c in _COMPOSITION_COLUMNS for w in ROLLING_WEEKS + ROLLING_WEEKS_LONG],
    *[f"share_{c}_ewma_{a}"for c in _COMPOSITION_COLUMNS for a in EWMA_ALPHAS]
}
_TEMPORAL_VOLUME_COLS = {
    *[f"{v}_lag_{w}w" for v in ["units_total", "cogs_total", "weight_total"] for w in LAG_WEEKS],
    *[f"{v}_rolling_{w}w" for v in ["units_total", "cogs_total", "weight_total"] for w in ROLLING_WEEKS + ROLLING_WEEKS_LONG],
    *[f"{v}_ewma_{a}" for v in ["units_total", "cogs_total", "weight_total"] for a in EWMA_ALPHAS]
}
_TEMPORAL_PROBABILITY_COLS: list[str] = (
    *[f"prob_recovered_lag_{w}w" for w in LAG_WEEKS],
    *[f"prob_recovered_rolling_{w}w" for w in ROLLING_WEEKS + ROLLING_WEEKS_LONG]
    *[f"prob_recovered_ewma_{a}" for a in EWMA_ALPHAS]
)

RECOVERY_RATE_CLF_FEATURE_COLUMNS = (
    _GL_COMPOSITION_COLS        |
    _GL_VOLUME_COLS             |
    _GL_AT_SITE_COLS            |
    _SITE_CONTEXT_COLS          |
    _TEMPORAL_SITE_CONTEXT_COLS |
    _CALENDAR_COLS              |
    _TEMPORAL_COMPOSITION_COLS  |
    _TEMPORAL_VOLUME_COLS       |
    _TEMPORAL_PROBABILITY_COLS
)

RECOVERY_RATE_CLF_DEFAULT_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
    "max_delta_step": 0,
}

# =====================================
# RECOVERY RATE REGRESSOR CONFIG
# =====================================

# RECOVERY_RATE_REG_FEATURE_COLUMNS = RECOVERY_RATE_CLF_FEATURE_COLUMNS + _BASELINE_COLS
RECOVERY_RATE_REG_DEFAULT_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
}

# =====================================
# RECOVERY TYPE SHARE REGRESSORS CONFIG
# =====================================



_TEMPORAL_PER_CHANNEL_GL_COLS: list[str] = (
    [
        f"prob_{ch}_lag_{w}w"
        for ch in CONSOLIDATED_RECOVERY_TYPES
        for w in [1, 4, 12, 13, 52]
        if ch != ConsolidatedRecoveryTypes.SALES
    ]
    + [
        f"prob_{ch}_rolling_{w}w"
        for ch in CONSOLIDATED_RECOVERY_TYPES
        for w in [4, 12, 26, 52]
        if ch != ConsolidatedRecoveryTypes.SALES
    ]
    + [
        f"prob_{ch}_ewma_{a}"
        for ch in CONSOLIDATED_RECOVERY_TYPES
        for a in ["5a", "1a"]
        if ch != ConsolidatedRecoveryTypes.SALES
    ]
)

_TEMPORAL_PER_CHANNEL_SITE_COLS: list[str] = (
    [
        f"site_prob_{ch}_week_lag_{w}w"
        for ch in _CHANNEL_SHORT_NAMES
        for w in [1, 4, 12, 13, 52]
    ]
    + [
        f"site_prob_{ch}_week_rolling_{w}w"
        for ch in _CHANNEL_SHORT_NAMES
        for w in [4, 12, 26, 52]
    ]
    + [
        f"site_prob_{ch}_week_ewma_{a}"
        for ch in _CHANNEL_SHORT_NAMES
        for a in ["5a", "1a"]
    ]
)


# ============================================================
# LIFECYCLE MODEL CONFIG
# ============================================================
# TODO

# ============================================================
# PATH CONFIG
# ============================================================
"""
s3://msds-26.2-data/
├── data/
│   ├── unprocessed/
│       ├── sanitized_2022.csv, sanitized_2023.csv, ..., sanitized_2025.csv
│       ├── inflation_data_north_america_capstone.csv (not used)
│       └── waste_rate_card.csv                       (not used)
│   ├── processed/
│       └── preprocessed_recovery_data.parquet
├── model/
│   ├── tuned_xgboost_classification_model.joblib
│   ├── tuned_xgboost_regression_model.joblib
│   └── recovery_channel_share_softmax/
│       ├── prob_disposal_share.joblib
│       ├── prob_donations_share.joblib
│       ├── prob_liquidations_share.joblib
│       ├── prob_return_to_vendor_share.joblib
│       └── prob_warehouse_deals_and_gr_share.joblib
├── out/
|   ├── predictions.parquet
|   └── report.html
"""

ROOT_DIR = S3Path(bucket="msds-26.2-data")

# === data/ ===
DATA_DIR = ROOT_DIR/"data"

UNPROCESSED_DATA_DIR = DATA_DIR/"unprocessed"
YEARLY_RECOVERY_DATA_CSVS = [
    UNPROCESSED_DATA_DIR/f"sanitized_{year}.csv"
    for year in (2022, 2023, 2024, 2025)
]

PROCESSED_DATA_DIR = DATA_DIR/"processed"
PREPROCESSED_RECOVERY_DATA_PARQUET = PROCESSED_DATA_DIR/"preprocessed_recovery_data.parquet"

# === model/ ===
MODEL_DIR = ROOT_DIR/"model"

# how likely to enter the recovery funnel?
ENTER_RECOVERY_FUNNEL_CLF_MODEL_JOBLIB = MODEL_DIR/"tuned_xgboost_classification_model.joblib"  # classifies sale vs. non-sale outcome
ENTER_RECOVERY_FUNNEL_REG_MODEL_JOBLIB = MODEL_DIR/"tuned_xgboost_regression_model.joblib"      # regressor for propensity for non-sale conditioned on P(Sale) < 1

# conditioned on entering recovery funnel, how likely is each (non-sale) outcome?
RECOVERY_CHANNEL_MODEL_DIR = MODEL_DIR/"recovery_channel_share_softmax"
RECOVERY_CHANNEL_MODEL_JOBLIB_DICT = {
    consolidated_recovery_type: RECOVERY_CHANNEL_MODEL_DIR/f"prob_{consolidated_recovery_type}_share.joblib"
    for consolidated_recovery_type in CONSOLIDATED_RECOVERY_TYPES
}

# === out/ ===
OUT_DIR = ROOT_DIR/"out"
PREDICTIONS_PARQUET = OUT_DIR/"predictions.parquet"
REPORT_HTML = OUT_DIR/"report.html"


# ============================================================
# PIPELINE CONFIG
# ============================================================

class ContextKeys:
    DF_RECOVERY_LOADED = "df_recovery_loaded"
    DF_RECOVERY_PREPROCESSED = "df_recovery_preprocessed"
    CLF_MODEL = "clf_model"         # trained XGBClassifier (Stage 1)
    REG_MODEL = "reg_model"         # trained XGBRegressor (Stage 2)
    SHARE_MODELS = "share_models"   # dict[channel_name -> XGBRegressor], 5 per-channel share regressors
    PREDICTIONS = "predictions"     # output Polars DataFrame from Predict step
    REPORT = "report"               # HTML string produced by Report step
