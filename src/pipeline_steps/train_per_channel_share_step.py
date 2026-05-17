# standard
import logging
import math
import tempfile

# third-party
import boto3
import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# local
from src.config import ContextKeys, S3_BUCKET, SHARE_MODELS_S3_PREFIX
from src.pipeline import Context, enforce
from src.pipeline.conditions import Defines, Locks, Sequence
from src.pipeline.types import PipelineStep
from src.util import load

logger = logging.getLogger(__name__)


# ============================================================
# Channel + feature constants
# ============================================================

# Per-channel target column names (each is a share of total units in the parquet).
RECOVERY_CHANNELS: list[str] = [
    "prob_remove_return",
    "prob_bintool_donations",
    "prob_donations",
    "prob_warehouse_deals_and_gr",
    "prob_liquidations",
    "prob_return_to_vendor",
    "prob_bintool_theft",
    "prob_remove_liquidate",
    "prob_bintool_remove_liquidate",
]

# Short channel names used to construct per-channel temporal feature names.
_CHANNEL_SHORT_NAMES: list[str] = [
    "remove_return", "bintool_donations", "donations",
    "warehouse_deals_and_gr", "liquidations", "return_to_vendor",
    "bintool_theft", "remove_liquidate", "bintool_remove_liquidate",
]

_CAT_COLS: list[str] = [
    "site_type", "site_category", "country", "country_state",
]

_GL_COMPOSITION_COLS: list[str] = [
    "share_food", "share_non_food", "share_pet_food",
    "share_RETAIL", "share_FBA", "share_hazmat",
]

_GL_VOLUME_COLS: list[str] = [
    "units_total", "cogs_total", "weight_total",
    "avg_cogs_per_unit", "avg_weight_per_unit", "cogs_per_unit_weight",
]

_GL_AT_SITE_COLS: list[str] = [
    "site_units_share_week", "site_weight_share_week",
]

_SITE_CONTEXT_COLS: list[str] = [
    "site_units_total_week", "site_weight_total_week",
    "site_type", "site_category", "country", "country_state",
]

_TEMPORAL_SITE_CONTEXT_COLS: list[str] = (
    [f"site_units_total_week_lag_{w}w" for w in [1, 4, 12, 13, 52]]
    + [f"site_weight_total_week_lag_{w}w" for w in [1, 4, 12, 13, 52]]
    + [f"site_prob_recovered_week_lag_{w}w" for w in [1, 4, 12, 13, 52]]
    + [f"site_prob_recovered_week_rolling_{w}w" for w in [4, 12, 26, 52]]
)

_CALENDAR_COLS: list[str] = ["month", "week"]

_TEMPORAL_COMPOSITION_COLS: list[str] = (
    [
        f"share_{c}_lag_{w}w"
        for c in ["RETAIL", "FBA", "hazmat", "food", "non_food", "pet_food"]
        for w in [1, 4, 12, 13, 52]
    ]
    + [
        f"share_{c}_rolling_{w}w"
        for c in ["food", "non_food", "pet_food", "RETAIL", "FBA", "hazmat"]
        for w in [4, 12, 26, 52]
    ]
    + [
        f"share_{c}_ewma_{a}"
        for c in ["RETAIL", "FBA", "hazmat", "food", "non_food", "pet_food"]
        for a in ["5a", "1a"]
    ]
)

_TEMPORAL_VOLUME_COLS: list[str] = (
    [
        f"{v}_lag_{w}w"
        for v in ["units_total", "cogs_total", "weight_total"]
        for w in [1, 4, 12, 13, 52]
    ]
    + [
        f"{v}_rolling_{w}w"
        for v in ["units_total", "cogs_total", "weight_total"]
        for w in [4, 12, 26, 52]
    ]
    + [
        f"{v}_ewma_{a}"
        for v in ["units_total", "cogs_total", "weight_total"]
        for a in ["5a", "1a"]
    ]
)

_TEMPORAL_PROBABILITY_COLS: list[str] = (
    [f"prob_recovered_lag_{w}w" for w in [1, 4, 12, 13, 52]]
    + [f"prob_recovered_rolling_{w}w" for w in [4, 12, 26, 52]]
    + [f"prob_recovered_ewma_{a}" for a in ["5a", "1a"]]
)

_TEMPORAL_PER_CHANNEL_GL_COLS: list[str] = (
    [
        f"prob_{ch}_lag_{w}w"
        for ch in _CHANNEL_SHORT_NAMES
        for w in [1, 4, 12, 13, 52]
    ]
    + [
        f"prob_{ch}_rolling_{w}w"
        for ch in _CHANNEL_SHORT_NAMES
        for w in [4, 12, 26, 52]
    ]
    + [
        f"prob_{ch}_ewma_{a}"
        for ch in _CHANNEL_SHORT_NAMES
        for a in ["5a", "1a"]
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

# Base features. Per-channel temporal columns are filtered against the actual
# frame schema at train time since some can be missing upstream.
_BASE_FEATURE_COLS: list[str] = (
    _GL_COMPOSITION_COLS
    + _GL_VOLUME_COLS
    + _GL_AT_SITE_COLS
    + _SITE_CONTEXT_COLS
    + _TEMPORAL_SITE_CONTEXT_COLS
    + _CALENDAR_COLS
    + _TEMPORAL_COMPOSITION_COLS
    + _TEMPORAL_VOLUME_COLS
    + _TEMPORAL_PROBABILITY_COLS
    + _TEMPORAL_PER_CHANNEL_GL_COLS
    + _TEMPORAL_PER_CHANNEL_SITE_COLS
)

_BASELINE_COLS: list[str] = [
    "baseline_share_mean", "baseline_share_std", "baseline_share_count",
]

_EPS = 1e-7

_DEFAULT_REG_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.15,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "early_stopping_rounds": 30,
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": 42,
    "verbosity": 0,
    "eval_metric": "mae",
    "n_jobs": -1,
}

_DEFAULT_TRAIN_YEARS: list[int] = [2022, 2023, 2024]
_DEFAULT_TEST_YEARS: list[int] = [2025]


# ============================================================
# Transform helpers
# ============================================================

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in _CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _polars_to_pandas_safe(df: pl.DataFrame) -> pd.DataFrame:
    """Convert column-by-column to avoid arrow chunked-array peak memory."""
    data: dict = {}
    for col in df.columns:
        s = df[col]
        if s.dtype == pl.Categorical:
            data[col] = pd.Categorical(s.to_numpy())
        else:
            data[col] = s.to_numpy()
    return pd.DataFrame(data)


# ============================================================
# Per-channel split + baseline
# ============================================================

def _resolve_feature_cols(df: pl.DataFrame) -> list[str]:
    available = set(df.columns)
    return [c for c in _BASE_FEATURE_COLS if c in available]


def _compute_site_gl_share_baseline(df_train_nz: pl.DataFrame) -> pl.DataFrame:
    return (
        df_train_nz
        .group_by(["hashed_fc", "gl_product_group"])
        .agg([
            pl.col("_share").mean().alias("baseline_share_mean"),
            pl.col("_share").std().alias("baseline_share_std"),
            pl.col("_share").count().alias("baseline_share_count"),
        ])
    )


def _build_channel_splits(
    df: pl.DataFrame,
    target_col: str,
    train_years: list[int],
    test_years: list[int],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pl.DataFrame, list[str]]:
    """Filter to recovered subset, build per-channel share + baseline, return matrices."""
    df_train_nz = (
        df
        .filter(pl.col("year").is_in(train_years))
        .filter(pl.col("prob_recovered") > 0)
        .with_columns((pl.col(target_col) / pl.col("prob_recovered")).alias("_share"))
    )
    df_test_nz = (
        df
        .filter(pl.col("year").is_in(test_years))
        .filter(pl.col("prob_recovered") > 0)
        .with_columns((pl.col(target_col) / pl.col("prob_recovered")).alias("_share"))
    )

    site_gl_baseline = _compute_site_gl_share_baseline(df_train_nz)

    df_train_nz = df_train_nz.join(
        site_gl_baseline, on=["hashed_fc", "gl_product_group"], how="left"
    )
    df_test_nz = df_test_nz.join(
        site_gl_baseline, on=["hashed_fc", "gl_product_group"], how="left"
    )

    aug_features = [c for c in feature_cols + _BASELINE_COLS if c in df_train_nz.columns]

    X_train = _cast_categoricals(_polars_to_pandas_safe(df_train_nz.select(aug_features)))
    X_test  = _cast_categoricals(_polars_to_pandas_safe(df_test_nz.select(aug_features)))

    y_train = df_train_nz["_share"].to_numpy()
    y_test  = df_test_nz["_share"].to_numpy()

    logger.info(
        "Channel '%s' splits: train_nz=%d (%s), test_nz=%d (%s), aug_features=%d",
        target_col, len(df_train_nz), train_years, len(df_test_nz), test_years, len(aug_features),
    )
    return X_train, X_test, y_train, y_test, site_gl_baseline, aug_features


# ============================================================
# Per-channel training
# ============================================================

def _train_channel_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    params: dict,
) -> tuple[XGBRegressor, dict]:
    y_train_lg = _logit(np.clip(y_train, _EPS, 1 - _EPS))
    y_test_lg  = _logit(np.clip(y_test,  _EPS, 1 - _EPS))

    model = XGBRegressor(**params)
    model.fit(X_train, y_train_lg, eval_set=[(X_test, y_test_lg)], verbose=False)

    pred = np.clip(_sigmoid(model.predict(X_test)), 0.0, 1.0)
    metrics = {
        "mae":  float(mean_absolute_error(y_test, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, pred))),
        "r2":   float(r2_score(y_test, pred)),
    }
    return model, metrics


def _save_model_to_s3(model: XGBRegressor, bucket: str, key: str) -> None:
    s3_client = boto3.client("s3")
    try:
        with tempfile.TemporaryFile() as fp:
            joblib.dump(model, fp)
            fp.seek(0)
            s3_client.put_object(Body=fp.read(), Bucket=bucket, Key=key)
            logger.info("Channel-share model saved to s3://%s/%s", bucket, key)
    except Exception:
        logger.exception("Failed to save channel-share model to s3://%s/%s", bucket, key)
        raise

# TODO: load models, not preprocessed df

# ============================================================
# Pipeline step
# ============================================================

@enforce({
    # ContextKeys.DF_RECOVERY_PREPROCESSED: Requires(),  # not enforced; read_from makes it optional
    ContextKeys.SHARE_MODELS: Sequence(Defines(strict=True), Locks(strict=True)),
})
class TrainPerChannelShareRegressors(PipelineStep):
    """Pipeline step that trains 9 per-channel share regressors used for softmax allocation.

    For each recovery channel, fits an XGBRegressor on logit(share) where
    share = prob_<channel> / prob_recovered, on rows with prob_recovered > 0.
    Each model carries `site_gl_baseline_`, `aug_features_`, `channel_`, and
    `metrics_` so that downstream inference can recover everything from the
    single artifact. Models are saved to S3 under SHARE_MODELS_S3_PREFIX and
    stored in context under ContextKeys.SHARE_MODELS as a dict keyed by
    channel name.

    Softmax normalisation across channels and combination with the Stage-1
    p_recovered_hat prediction happen at inference, not in this step.
    """

    def __init__(
        self,
        train_years: list[int] | None = None,
        test_years: list[int] | None = None,
        read_from: str | None = None,
        save_to_prefix: str | None = None,
    ):
        self.train_years = train_years if train_years is not None else _DEFAULT_TRAIN_YEARS
        self.test_years = test_years if test_years is not None else _DEFAULT_TEST_YEARS
        self.read_from = read_from
        self.save_to_prefix = save_to_prefix or SHARE_MODELS_S3_PREFIX

    def __call__(self, context: Context) -> Context:
        if self.read_from is not None:
            df = load(self.read_from)
        else:
            df = context[ContextKeys.DF_RECOVERY_PREPROCESSED]

        feature_cols = _resolve_feature_cols(df)
        logger.info(
            "Per-channel share regressor: %d / %d base features resolved against frame schema",
            len(feature_cols), len(_BASE_FEATURE_COLS),
        )

        share_models: dict[str, XGBRegressor] = {}
        prefix = self.save_to_prefix.rstrip("/")
        for i, channel in enumerate(RECOVERY_CHANNELS, start=1):
            logger.info(
                "[%d/%d] Training share regressor for channel '%s'",
                i, len(RECOVERY_CHANNELS), channel,
            )

            X_train, X_test, y_train, y_test, site_gl_baseline, aug_features = (
                _build_channel_splits(
                    df,
                    target_col=channel,
                    train_years=self.train_years,
                    test_years=self.test_years,
                    feature_cols=feature_cols,
                )
            )

            model, metrics = _train_channel_model(
                X_train, X_test, y_train, y_test, _DEFAULT_REG_PARAMS,
            )
            logger.info(
                "  channel=%s  best_iter=%s  MAE=%.4f  R2=%.4f",
                channel, model.best_iteration, metrics["mae"], metrics["r2"],
            )

            # Persist baseline + metadata on the model so a single joblib artifact
            # is sufficient for downstream inference.
            model.site_gl_baseline_ = site_gl_baseline
            model.aug_features_ = aug_features
            model.channel_ = channel
            model.metrics_ = metrics

            _save_model_to_s3(model, S3_BUCKET, f"{prefix}/{channel}_share.joblib")

            share_models[channel] = model

        context[ContextKeys.SHARE_MODELS] = share_models
        context.lock(ContextKeys.SHARE_MODELS)

        return context
