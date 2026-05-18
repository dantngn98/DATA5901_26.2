# standard
import io
import logging
import tempfile

# third-party
import boto3
import joblib
import numpy as np
import pandas as pd
import polars as pl
import shap

# local
from src.config import (
    CLF_MODEL_S3_KEY,
    ContextKeys,
    REG_MODEL_S3_KEY,
    S3_BUCKET,
    SHARE_MODELS_S3_PREFIX,
)
from src.pipeline import Context, enforce
from src.pipeline.conditions import Defines, Locks, Sequence
from src.pipeline.types import PipelineStep
from src.util import load

# Reuse feature lists and helpers — no duplication
from src.pipeline_steps.train_binary_classifier_step import (
    _FEATURE_COLS,
    _cast_categoricals,
)
from src.pipeline_steps.train_regressor_step import (
    _EXTENDED_FEATURE_COLS,
    _fill_site_gl_baseline,
    _prob_mae,
    _sigmoid,
)
from src.pipeline_steps.train_per_channel_share_step import RECOVERY_CHANNELS

logger = logging.getLogger(__name__)

# ============================================================
# SHAP decomposition feature groups
# ============================================================

# "Deviation" features = current-week signals that capture departures from the
# site-GL baseline (GL snapshot cols + all 1-week lags).
_SHAP_DEVIATION_FEATURE_NAMES: frozenset[str] = frozenset([
    "share_food", "share_non_food", "share_pet_food",
    "share_RETAIL", "share_FBA", "share_hazmat",
    "units_total", "cogs_total", "weight_total",
    "avg_cogs_per_unit", "avg_weight_per_unit", "cogs_per_unit_weight",
] + [f for f in _EXTENDED_FEATURE_COLS if f.endswith("_lag_1w")])


# ============================================================
# S3 helpers
# ============================================================

def _register_custom_metrics() -> None:
    """Register custom XGBoost eval metrics in __main__ so joblib can unpickle
    models that were saved from notebooks where these functions lived at top level."""
    import __main__
    if not hasattr(__main__, "prob_mae"):
        __main__.prob_mae = _prob_mae


# TODO: move these helpers to a util so they can be used in other places

def _load_model_from_s3(bucket: str, key: str):
    _register_custom_metrics()
    s3_client = boto3.client("s3")
    with tempfile.TemporaryFile() as fp:
        s3_client.download_fileobj(bucket, key, fp)
        fp.seek(0)
        model = joblib.load(fp)
    logger.info("Loaded model from s3://%s/%s", bucket, key)
    return model


def _load_share_models_from_s3(bucket: str, prefix: str) -> dict:
    prefix = prefix.rstrip("/")
    models = {}
    for channel in RECOVERY_CHANNELS:
        models[channel] = _load_model_from_s3(bucket, f"{prefix}/{channel}_share.joblib")
    return models


def _save_predictions_to_s3(df: pl.DataFrame, s3_path: str) -> None:
    path = s3_path.removeprefix("s3://")
    bucket, _, key = path.partition("/")
    s3_client = boto3.client("s3")
    buf = io.BytesIO()
    df.write_parquet(buf)
    buf.seek(0)
    s3_client.put_object(Body=buf.read(), Bucket=bucket, Key=key)
    logger.info("Predictions saved to s3://%s/%s", bucket, key)


# ============================================================
# Model resolution
# ============================================================

def _resolve_models(
    context: Context,
    clf_key: str,
    reg_key: str,
    share_prefix: str,
    predict_channels: bool,
) -> tuple:
    if ContextKeys.CLF_MODEL in context:
        model_clf = context[ContextKeys.CLF_MODEL]
        logger.info("Classifier loaded from context")
    else:
        model_clf = _load_model_from_s3(S3_BUCKET, clf_key)

    if ContextKeys.REG_MODEL in context:
        model_reg = context[ContextKeys.REG_MODEL]
        logger.info("Regressor loaded from context")
    else:
        model_reg = _load_model_from_s3(S3_BUCKET, reg_key)

    if predict_channels:
        if ContextKeys.SHARE_MODELS in context:
            share_models = context[ContextKeys.SHARE_MODELS]
            logger.info("Share models loaded from context")
        else:
            share_models = _load_share_models_from_s3(S3_BUCKET, share_prefix)
    else:
        share_models = {}

    return model_clf, model_reg, share_models


# ============================================================
# Data filtering
# ============================================================

def _apply_filters(
    df: pl.DataFrame,
    years: list[int] | None,
    sites: list[str] | None,
    gl_groups: list[str] | None,
    weeks: list[int] | None,
) -> pl.DataFrame:
    if years is None:
        max_year = df["year"].max()
        df = df.filter(pl.col("year") == max_year)
        logger.info("Defaulting to last year in data: %d", max_year)
    else:
        df = df.filter(pl.col("year").is_in(years))
    if sites is not None:
        df = df.filter(pl.col("hashed_fc").is_in(sites))
    if gl_groups is not None:
        df = df.filter(pl.col("gl_product_group").is_in(gl_groups))
    if weeks is not None:
        df = df.filter(pl.col("week").is_in(weeks))
    return df


# ============================================================
# Stage predictions
# ============================================================

def _predict_stage1(model_clf, df: pl.DataFrame) -> np.ndarray:
    X = _cast_categoricals(df.select(_FEATURE_COLS).to_pandas())
    return model_clf.predict_proba(X)[:, 1]


def _predict_stage2(
    model_reg, df: pl.DataFrame
) -> tuple[pl.DataFrame, np.ndarray]:
    """Returns (df_ext, e_rate). df_ext carries baseline cols for SHAP."""
    df_ext = df.join(
        model_reg.site_gl_baseline_,
        on=["hashed_fc", "gl_product_group"],
        how="left",
    )
    df_ext = _fill_site_gl_baseline(df_ext, model_reg.baseline_priors_)
    X = _cast_categoricals(df_ext.select(_EXTENDED_FEATURE_COLS).to_pandas())
    e_rate = np.clip(_sigmoid(model_reg.predict(X)), 0.0, 1.0)
    return df_ext, e_rate


def _predict_stage3(
    share_models: dict,
    df: pl.DataFrame,
    combined_rate: np.ndarray,
) -> dict[str, np.ndarray]:
    raw_shares: dict[str, np.ndarray] = {}
    for channel, model in share_models.items():
        df_ch = df.join(
            model.site_gl_baseline_,
            on=["hashed_fc", "gl_product_group"],
            how="left",
        )
        for col in ["baseline_share_mean", "baseline_share_std", "baseline_share_count"]:
            if col in df_ch.columns:
                df_ch = df_ch.with_columns(pl.col(col).fill_null(0.0))
        aug_features = [c for c in model.aug_features_ if c in df_ch.columns]
        X = _cast_categoricals(df_ch.select(aug_features).to_pandas())
        raw_shares[channel] = _sigmoid(model.predict(X))

    raw_matrix = np.stack([raw_shares[ch] for ch in RECOVERY_CHANNELS], axis=1)
    row_sums = raw_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    norm_matrix = raw_matrix / row_sums

    return {
        channel: norm_matrix[:, i] * combined_rate
        for i, channel in enumerate(RECOVERY_CHANNELS)
    }


# ============================================================
# SHAP
# ============================================================

def _sample_idx(n: int, shap_n: int | None, seed: int) -> np.ndarray:
    if shap_n is None or shap_n >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=shap_n, replace=False))


def _run_shap(
    model_clf,
    model_reg,
    X_clf: pd.DataFrame,
    X_reg: pd.DataFrame,
) -> dict:
    logger.info("Running SHAP Stage 1 on %d rows", len(X_clf))
    sv_clf = shap.TreeExplainer(model_clf)(X_clf, check_additivity=False)

    logger.info("Running SHAP Stage 2 on %d rows", len(X_reg))
    sv_reg = shap.TreeExplainer(model_reg)(X_reg, check_additivity=False)

    return {
        "sv_clf_values": sv_clf.values,
        "sv_clf_base": sv_clf.base_values,
        "sv_reg_values": sv_reg.values,
        "sv_reg_base": sv_reg.base_values,
    }


def _compute_shap_decomposition(
    sv_reg_values: np.ndarray,
    sv_reg_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feat_index = {f: i for i, f in enumerate(_EXTENDED_FEATURE_COLS)}
    dev_idx = np.array([
        i for f, i in feat_index.items() if f in _SHAP_DEVIATION_FEATURE_NAMES
    ])
    base_idx = np.array([
        i for f, i in feat_index.items() if f not in _SHAP_DEVIATION_FEATURE_NAMES
    ])

    shap_dev = sv_reg_values[:, dev_idx].sum(axis=1) if len(dev_idx) else np.zeros(len(sv_reg_values))
    shap_base = sv_reg_values[:, base_idx].sum(axis=1) if len(base_idx) else np.zeros(len(sv_reg_values))

    baseline_rate = _sigmoid(sv_reg_base + shap_base)
    deviation_contribution = _sigmoid(sv_reg_base + shap_base + shap_dev) - baseline_rate
    return baseline_rate, deviation_contribution


# ============================================================
# Bucket helpers
# ============================================================

def _rate_bucket(vals: np.ndarray) -> list[str]:
    return list(np.select(
        [vals == 0, vals < 0.1, vals < 0.3, vals < 0.6],
        ["zero", "0-10%", "10-30%", "30-60%"],
        default=">60%",
    ))


def _volume_bucket(vals: np.ndarray) -> list[str]:
    return list(np.select(
        [vals < 10, vals < 100, vals < 1000],
        ["<10", "10-100", "100-1k"],
        default=">1k",
    ))


# ============================================================
# Pipeline step
# ============================================================

@enforce({
    ContextKeys.PREDICTIONS: Sequence(Defines(strict=True), Locks(strict=True)),
})
class Predict(PipelineStep):
    """
    Loads preprocessed data + all three model types (from context or S3),
    applies optional row filters (site / GL / year / week), runs Stage 1
    (classifier → p_nonzero), Stage 2 (regressor → e_rate), Stage 3
    (per-channel share regressors → 4 absolute channel rates), and optionally
    SHAP decomposition.

    Output is a Polars DataFrame stored in ContextKeys.PREDICTIONS with columns:
    identifiers, all stage predictions, SHAP baseline/deviation, diagnostic
    buckets, ground-truth pass-through, and absolute errors.
    """

    def __init__(
        self,
        read_from: str | None = None,
        clf_model_s3_key: str | None = None,
        reg_model_s3_key: str | None = None,
        share_models_s3_prefix: str | None = None,
        predict_channels: bool = True,
        sites: list[str] | None = None,
        gl_groups: list[str] | None = None,
        years: list[int] | None = None,
        weeks: list[int] | None = None,
        run_shap: bool = True,
        shap_n: int | None = None,
        shap_seed: int = 0,
        save_to: str | None = None,
    ):
        self.read_from = read_from
        self.clf_model_s3_key = clf_model_s3_key or CLF_MODEL_S3_KEY
        self.reg_model_s3_key = reg_model_s3_key or REG_MODEL_S3_KEY
        self.share_models_s3_prefix = share_models_s3_prefix or SHARE_MODELS_S3_PREFIX
        self.predict_channels = predict_channels
        self.sites = sites
        self.gl_groups = gl_groups
        self.years = years
        self.weeks = weeks
        self.run_shap = run_shap
        self.shap_n = shap_n
        self.shap_seed = shap_seed
        self.save_to = save_to

    def __call__(self, context: Context) -> Context:
        # 1. Load data
        if self.read_from is not None:
            df = load(self.read_from)
        else:
            df = context[ContextKeys.DF_RECOVERY_PREPROCESSED]

        # 2. Resolve models
        model_clf, model_reg, share_models = _resolve_models(
            context,
            self.clf_model_s3_key,
            self.reg_model_s3_key,
            self.share_models_s3_prefix,
            predict_channels=self.predict_channels,
        )

        # 3. Filter to prediction scope
        df = _apply_filters(df, self.years, self.sites, self.gl_groups, self.weeks)
        logger.info("Predicting on %d rows", len(df))

        # 4. Stage 1: P(recovery > 0)
        p_nonzero = _predict_stage1(model_clf, df)

        # 5. Stage 2: E(rate | recovery > 0); df_ext carries baseline cols for SHAP
        df_ext, e_rate = _predict_stage2(model_reg, df)

        # 6. Combined recovery rate
        combined_rate = p_nonzero * e_rate

        # 7. Stage 3: per-channel absolute recovery rates (skipped when predict_channels=False)
        if self.predict_channels:
            channel_rates = _predict_stage3(share_models, df, combined_rate)
        else:
            channel_rates = {}

        # 8. SHAP decomposition (optionally on a capped sample)
        shap_baseline_rate = np.full(len(df), np.nan)
        shap_deviation_contribution = np.full(len(df), np.nan)

        if self.run_shap:
            idx = _sample_idx(len(df), self.shap_n, self.shap_seed)
            logger.info("Running SHAP on %d / %d rows", len(idx), len(df))
            X_clf_shap = _cast_categoricals(
                df[idx].select(_FEATURE_COLS).to_pandas()
            )
            X_reg_shap = _cast_categoricals(
                df_ext[idx].select(_EXTENDED_FEATURE_COLS).to_pandas()
            )
            shap_raw = _run_shap(model_clf, model_reg, X_clf_shap, X_reg_shap)
            baseline_sample, deviation_sample = _compute_shap_decomposition(
                shap_raw["sv_reg_values"],
                shap_raw["sv_reg_base"],
            )
            shap_baseline_rate[idx] = baseline_sample
            shap_deviation_contribution[idx] = deviation_sample

        # 9. Assemble output DataFrame
        has_gt = "prob_recovered" in df.columns

        out = df.select(["hashed_fc", "gl_product_group", "year", "week"]).with_columns([
            pl.Series("p_nonzero", p_nonzero),
            pl.Series("e_rate", e_rate),
            pl.Series("combined_rate", combined_rate),
        ])

        if self.predict_channels:
            for channel in RECOVERY_CHANNELS:
                out = out.with_columns(pl.Series(f"pred_{channel}", channel_rates[channel]))

        out = out.with_columns([
            pl.Series("shap_baseline_rate", shap_baseline_rate),
            pl.Series("shap_deviation_contribution", shap_deviation_contribution),
        ])

        # Buckets: rate_bucket uses ground truth when available, else combined_rate
        rate_vals = df["prob_recovered"].to_numpy() if has_gt else combined_rate
        out = out.with_columns([
            pl.Series("rate_bucket", _rate_bucket(rate_vals)),
            pl.Series("volume_bucket", _volume_bucket(df["units_total"].to_numpy())),
        ])

        # Ground truth pass-through and absolute errors
        if has_gt:
            y_true = df["prob_recovered"].to_numpy()
            out = out.with_columns([
                pl.Series("prob_recovered", y_true),
                pl.Series("abs_err", np.abs(combined_rate - y_true)),
            ])
            if self.predict_channels:
                for channel in RECOVERY_CHANNELS:
                    if channel in df.columns:
                        y_channel = df[channel].to_numpy()
                        out = out.with_columns([
                            pl.Series(channel, y_channel),
                            pl.Series(
                                f"abs_err_{channel}",
                                np.abs(channel_rates[channel] - y_channel),
                            ),
                        ])

        context[ContextKeys.PREDICTIONS] = out
        context.lock(ContextKeys.PREDICTIONS)

        if self.save_to is not None:
            _save_predictions_to_s3(out, self.save_to)

        return context
