 # standard
import logging

# third-party
import numpy as np
import pandas as pd
import polars as pl
import shap

# local
from src.config import (
    RecoverySchema,
    NORMALIZED_MACRO_CATEGORY_DICT, NORMALIZED_PRODUCT_TYPES_DICT,
    RECOVERY_FUNNEL_CONSOLIDATED_RECOVERY_TYPES,
    RECOVERY_RATE_CLF_FEATURE_COLUMNS, RECOVERY_RATE_REG_FEATURE_COLUMNS, PER_TYPE_REG_FEATURE_COLUMNS,
    RECOVERY_RATE_TARGET_COLUMN, PER_TYPE_TARGET_COLUMN_DICT,
    ContextKeys,
    CATEGORICAL_COLUMNS
)
from src.pipeline import Context, enforce
from src.pipeline.conditions import Defines, Locks, Sequence, Requires
from src.pipeline.types import PipelineStep
from src.util import write_dataframe

# Reuse feature lists and helpers — no duplication
from src.pipeline_steps.train_regressor_step import (
    _fill_site_gl_baseline,
    _prob_mae,
    _sigmoid,
)
from src.util import cast_categoricals


logger = logging.getLogger(__name__)

# ============================================================
# Pipeline step
# ============================================================

@enforce({
    ContextKeys.DF_RECOVERY_PREPROCESSED: Requires(),
    ContextKeys.CLF_MODEL: Requires(),
    ContextKeys.REG_MODEL: Requires(),
    #ContextKeys.SHARE_MODELS: Requires(),  # not required if predict_channels = False
    ContextKeys.PREDICTIONS: Sequence(Defines(strict=True), Locks(strict=True)),
})
class Predict(PipelineStep):
    """
    Loads preprocessed data + all three model types (from context or S3),
    applies optional row filters (site / GL / year / week), runs Stage 1
    (classifier -> p_nonzero), Stage 2 (regressor -> e_rate), Stage 3
    (per-channel share regressors -> 4 absolute channel rates), and optionally
    SHAP decomposition.

    Output is a Polars DataFrame stored in ContextKeys.PREDICTIONS with columns:
    identifiers, all stage predictions, SHAP baseline/deviation, diagnostic
    buckets, ground-truth pass-through, and absolute errors.
    """

    def __init__(
        self,
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
        # 1. Preprocessed data
        df = context[ContextKeys.DF_RECOVERY_PREPROCESSED]

        # 2. Trained models
        model_clf = context[ContextKeys.CLF_MODEL]
        model_reg = context[ContextKeys.REG_MODEL]
        share_models = context[ContextKeys.SHARE_MODELS] if self.predict_channels else {}

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
            X_clf_shap = cast_categoricals(
                df[idx].select(tuple(RECOVERY_RATE_CLF_FEATURE_COLUMNS)).to_pandas(), CATEGORICAL_COLUMNS
            )
            X_reg_shap = cast_categoricals(
                df_ext[idx].select(tuple(RECOVERY_RATE_REG_FEATURE_COLUMNS)).to_pandas(), CATEGORICAL_COLUMNS
            )
            shap_raw = _run_shap(model_clf, model_reg, X_clf_shap, X_reg_shap)
            baseline_sample, deviation_sample = _compute_shap_decomposition(
                shap_raw["sv_reg_values"],
                shap_raw["sv_reg_base"],
            )
            shap_baseline_rate[idx] = baseline_sample
            shap_deviation_contribution[idx] = deviation_sample

        # 9. Assemble output DataFrame
        # Column order: identifiers -> (real -> predicted -> abs_err) per prediction group -> diagnostics
        has_gt = RECOVERY_RATE_TARGET_COLUMN in df.columns

        out = df.select([RecoverySchema.HASHED_FC, RecoverySchema.GL_PRODUCT_GROUP, RecoverySchema.YEAR, RecoverySchema.WEEK])

        # Overall recovery rate group
        if has_gt:
            y_true = df[RECOVERY_RATE_TARGET_COLUMN].to_numpy()
            out = out.with_columns(pl.Series(RECOVERY_RATE_TARGET_COLUMN, y_true))

        out = out.with_columns([
            pl.Series("p_nonzero",      np.round(p_nonzero,      6)),
            pl.Series("e_rate",         np.round(e_rate,         6)),
            pl.Series("combined_rate",  np.round(combined_rate,  6)),
        ])

        if has_gt:
            out = out.with_columns(
                pl.Series("abs_err", np.round(np.abs(combined_rate - y_true), 6))
            )

        # Per-channel group: real -> predicted -> abs_err
        if self.predict_channels:
            for channel, target_column in PER_TYPE_TARGET_COLUMN_DICT.items():
                pred_col = np.round(channel_rates[channel], 6)
                if has_gt and target_column in df.columns:
                    y_channel = df[target_column].to_numpy()
                    out = out.with_columns([
                        pl.Series(target_column,              y_channel),
                        pl.Series(f"pred_{target_column}",    pred_col),
                        pl.Series(f"abs_err_{target_column}", np.round(np.abs(channel_rates[channel] - y_channel), 6)),
                    ])
                else:
                    out = out.with_columns(pl.Series(f"pred_{target_column}", pred_col))

        # Diagnostics
        out = out.with_columns([
            pl.Series("shap_baseline_rate",         np.round(shap_baseline_rate,         6)),
            pl.Series("shap_deviation_contribution", np.round(shap_deviation_contribution, 6)),
        ])

        rate_vals = df[RECOVERY_RATE_TARGET_COLUMN].to_numpy() if has_gt else combined_rate
        out = out.with_columns([
            pl.Series("rate_bucket",   _rate_bucket(rate_vals)),
            pl.Series("volume_bucket", _volume_bucket(df["units_total"].to_numpy())),
        ])

        context[ContextKeys.PREDICTIONS] = out
        context.lock(ContextKeys.PREDICTIONS)

        if self.save_to is not None:
            write_dataframe(out, self.save_to)

        return context


# ============================================================
# SHAP decomposition feature groups
# ============================================================

# "Deviation" features = current-week signals that capture departures from the
# site-GL baseline (GL snapshot cols + all 1-week lags).

_SHAP_DEVIATION_FEATURE_NAMES: frozenset[str] = frozenset(
    [f"share_{c}" for c in list(NORMALIZED_PRODUCT_TYPES_DICT.values()) + list(NORMALIZED_MACRO_CATEGORY_DICT.values())]
    +
    ["share_hazmat","units_total","cogs_total","weight_total","avg_cogs_per_unit","avg_weight_per_unit","cogs_per_unit_weight"]
    +
    [f for f in RECOVERY_RATE_REG_FEATURE_COLUMNS if f.endswith("_lag_1w")]
)


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
        max_year = df[RecoverySchema.YEAR].max()
        df = df.filter(pl.col(RecoverySchema.YEAR) == max_year)
        logger.info("Defaulting to last year in data: %d", max_year)
    else:
        df = df.filter(pl.col(RecoverySchema.YEAR).is_in(years))
    if sites is not None:
        df = df.filter(pl.col(RecoverySchema.HASHED_FC).is_in(sites))
    if gl_groups is not None:
        df = df.filter(pl.col(RecoverySchema.GL_PRODUCT_GROUP).is_in(gl_groups))
    if weeks is not None:
        df = df.filter(pl.col(RecoverySchema.WEEK).is_in(weeks))
    return df


# ============================================================
# Stage predictions
# ============================================================

def _predict_stage1(model_clf, df: pl.DataFrame) -> np.ndarray:
    X = cast_categoricals(df.select(tuple(RECOVERY_RATE_CLF_FEATURE_COLUMNS)).to_pandas(), CATEGORICAL_COLUMNS)
    return model_clf.predict_proba(X)[:, 1]


def _predict_stage2(
    model_reg, df: pl.DataFrame
) -> tuple[pl.DataFrame, np.ndarray]:
    """Returns (df_ext, e_rate). df_ext carries baseline cols for SHAP."""
    df_ext = df.join(
        model_reg.site_gl_baseline_,
        on=[RecoverySchema.HASHED_FC, RecoverySchema.GL_PRODUCT_GROUP],
        how="left",
    )
    df_ext = _fill_site_gl_baseline(df_ext, model_reg.baseline_priors_)
    X = cast_categoricals(df_ext.select(RECOVERY_RATE_REG_FEATURE_COLUMNS).to_pandas(), CATEGORICAL_COLUMNS)
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
            on=[RecoverySchema.HASHED_FC, RecoverySchema.GL_PRODUCT_GROUP],
            how="left",
        )
        features = tuple(PER_TYPE_REG_FEATURE_COLUMNS - {PER_TYPE_TARGET_COLUMN_DICT[channel]})
        X = cast_categoricals(df_ch.select(features).to_pandas(), CATEGORICAL_COLUMNS)
        raw_shares[channel] = _sigmoid(model.predict(X))

    raw_matrix = np.stack([raw_shares[ch] for ch in RECOVERY_FUNNEL_CONSOLIDATED_RECOVERY_TYPES], axis=1)
    row_sums = raw_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    norm_matrix = raw_matrix / row_sums

    return {
        channel: norm_matrix[:, i] * combined_rate
        for i, channel in enumerate(RECOVERY_FUNNEL_CONSOLIDATED_RECOVERY_TYPES)
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
    feat_index = {f: i for i, f in enumerate(RECOVERY_RATE_REG_FEATURE_COLUMNS)}
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
