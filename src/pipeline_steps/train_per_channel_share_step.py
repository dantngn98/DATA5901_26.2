# standard
import logging
import math
import tempfile

# third-party
import boto3
import joblib
import numpy as np
import optuna
import pandas as pd
import polars as pl
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# local
from src.config import ContextKeys, S3_BUCKET, SHARE_MODELS_S3_PREFIX, DEFAULT_TRAIN_YEARS, DEFAULT_TEST_YEARS, CONSOLIDATED_RECOVERY_TYPES, ConsolidatedRecoveryTypes, CATEGORICAL_COLUMNS
from src.pipeline import Context, enforce
from src.pipeline.conditions import Defines, Locks, Sequence
from src.pipeline.types import PipelineStep
from src.util import cast_categoricals, load_joblib_from_s3, write_joblib_to_s3

logger = logging.getLogger(__name__)


# ============================================================
# Channel + feature constants
# ============================================================

RECOVERY_CHANNELS: list[str] = [
    "prob_donations",
    "prob_liquidations",
    "prob_return_to_vendor",
    "prob_warehouse_deals_and_gr",
    "prob_disposal",
]

# Short channel names used to construct per-channel temporal feature names.
_CHANNEL_SHORT_NAMES: list[str] = list(CONSOLIDATED_RECOVERY_TYPES) - set([ConsolidatedRecoveryTypes.SALES])

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

# Per-channel default hyperparameters from the 2026-05-14 Optuna tuning run
# (n_trials=20 per channel, raw 9-channel data, closest raw-channel match per
# consolidated channel).  Used when tune=False.
_DEFAULT_CHANNEL_PARAMS: dict[str, dict] = {
    "prob_donations": {
        "max_depth": 8,
        "learning_rate": 0.010233524192808544,
        "subsample": 0.6618100158148682,
        "colsample_bytree": 0.4103116901613753,
        "min_child_weight": 29,
        "gamma": 2.1717813587820562,
        "reg_alpha": 7.587120682342036,
        "reg_lambda": 1.2536269325652274e-08,
    },
    "prob_liquidations": {
        "max_depth": 8,
        "learning_rate": 0.10154216570970824,
        "subsample": 0.8222127978469346,
        "colsample_bytree": 0.49571853898410817,
        "min_child_weight": 22,
        "gamma": 0.033121217751713956,
        "reg_alpha": 0.00014422243561458065,
        "reg_lambda": 0.03187847480686257,
    },
    "prob_return_to_vendor": {
        "max_depth": 8,
        "learning_rate": 0.1943881912272435,
        "subsample": 0.9182630131548245,
        "colsample_bytree": 0.40022527720500706,
        "min_child_weight": 17,
        "gamma": 2.5072850771118342,
        "reg_alpha": 9.113338418294182e-07,
        "reg_lambda": 0.05392398582308144,
    },
    "prob_warehouse_deals_and_gr": {
        "max_depth": 6,
        "learning_rate": 0.10597071455010725,
        "subsample": 0.7774552662295606,
        "colsample_bytree": 0.40043799848796335,
        "min_child_weight": 41,
        "gamma": 2.9575883549786286,
        "reg_alpha": 0.0008500602063823264,
        "reg_lambda": 0.105063075386338,
    },
    "prob_disposal": {
        "max_depth":        8,
        "learning_rate":    0.10303364894496688,
        "subsample":        0.7949352732615969,
        "colsample_bytree": 0.4266733762096135,
        "min_child_weight": 27,
        "gamma":            1.9174437521560582,
        "reg_alpha":        1.8970289690794687,
        "reg_lambda":       0.047716386888,
    },
}




# ============================================================
# Transform helpers
# ============================================================

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _prob_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Custom XGBoost eval metric: MAE in original probability space."""
    return float(np.mean(np.abs(_sigmoid(y_pred) - _sigmoid(y_true))))

_prob_mae.__name__ = "prob_mae"


def _polars_to_pandas_safe(df: pl.DataFrame) -> pd.DataFrame:
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

    X_train = cast_categoricals(_polars_to_pandas_safe(df_train_nz.select(aug_features)), CATEGORICAL_COLUMNS)
    X_test  = cast_categoricals(_polars_to_pandas_safe(df_test_nz.select(aug_features)), CATEGORICAL_COLUMNS)

    y_train = df_train_nz["_share"].to_numpy()
    y_test  = df_test_nz["_share"].to_numpy()

    logger.info(
        "Channel '%s' splits: train_nz=%d (%s), test_nz=%d (%s), aug_features=%d",
        target_col, len(df_train_nz), train_years, len(df_test_nz), test_years, len(aug_features),
    )
    return X_train, X_test, y_train, y_test, site_gl_baseline, aug_features


# ============================================================
# Per-channel tuning + training
# ============================================================

def _tune_with_optuna(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_trials: int,
    channel: str,
) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    y_train_lg = _logit(y_train)
    y_test_lg  = _logit(y_test)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "enable_categorical": True,
            "random_state": 42,
            "n_estimators": 500,
            "early_stopping_rounds": 30,
            "eval_metric": _prob_mae,
            "callbacks": [XGBoostPruningCallback(trial, "validation_0-prob_mae")],
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train_lg, eval_set=[(X_test, y_test_lg)], verbose=False)
        preds = np.clip(_sigmoid(model.predict(X_test)), 0.0, 1.0)
        return float(mean_absolute_error(y_test, preds))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5),
        study_name=f"xgb_stage3_share_{channel}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(
        "Optuna Stage 3 channel '%s' complete: best MAE=%.4f, params=%s",
        channel, study.best_value, study.best_params,
    )
    return study.best_params


def _train_final_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    best_params: dict,
) -> tuple[XGBRegressor, dict]:
    y_train_lg = _logit(y_train)
    y_test_lg  = _logit(y_test)


    if y_test_lg.size > 0:
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=2000,
            tree_method="hist",
            enable_categorical=True,
            random_state=42,
            early_stopping_rounds=50,
            eval_metric=_prob_mae,
            **best_params,
        )
        model.fit(X_train, y_train_lg, eval_set=[(X_test, y_test_lg)], verbose=False)
    else:
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=2000,
            tree_method="hist",
            enable_categorical=True,
            random_state=42,
            **best_params,
        )
        model.fit(X_train, y_train_lg, verbose=False)

    return model


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


def _load_model_from_s3(bucket: str, key: str) -> XGBRegressor:
    s3_client = boto3.client("s3")
    try:
        with tempfile.TemporaryFile() as fp:
            s3_client.download_fileobj(bucket, key, fp)
            fp.seek(0)
            model = joblib.load(fp)
            if not isinstance(model, XGBRegressor):
                raise ValueError(f"Object loaded from s3://{bucket}/{key} is not an XGBRegressor")
            logger.info("Channel-share model loaded from s3://%s/%s", bucket, key)
            return model
    except Exception:
        logger.exception("Failed to load channel-share model from s3://%s/%s", bucket, key)
        raise


# ============================================================
# Pipeline step
# ============================================================

@enforce({
    # ContextKeys.DF_RECOVERY_PREPROCESSED: Requires(),  # not enforced; read_from_key makes it optional
    ContextKeys.SHARE_MODELS: Sequence(Defines(strict=True), Locks(strict=True)),
})
class TrainPerChannelShareRegressors(PipelineStep):
    """Pipeline step that trains (and optionally tunes) the Stage 3 per-channel share regressors.

    For each of the 4 consolidated recovery channels, fits an XGBRegressor on
    logit(share) where share = prob_<channel> / prob_recovered, on rows with
    prob_recovered > 0. When tune=True, a separate Optuna study is run per
    channel; when tune=False, the baked-in _DEFAULT_CHANNEL_PARAMS are used.

    When read_from_key is provided, all 4 models are loaded from S3 using
    read_from_key as the prefix (e.g. "model/recovery_channel_share_softmax")
    and no training occurs.

    Each trained model carries `site_gl_baseline_`, `aug_features_`, `channel_`,
    and `metrics_` so that downstream inference can recover everything from the
    single artifact. Trained models are saved to S3 under save_to_prefix
    (one .joblib per channel) and stored in context under
    ContextKeys.SHARE_MODELS as a dict keyed by channel name.

    Softmax normalisation across channels and combination with the Stage-1
    p_recovered_hat prediction happen at inference, not in this step.
    """

    def __init__(
        self,
        tune: bool = False,
        n_trials: int = 50,
        train_years: list[int] | None = None,
        test_years: list[int] | None = None,
        read_from_key: str | None = None,
        save_to_prefix: str | None = None,
    ):
        self.tune = tune
        self.n_trials = n_trials
        self.train_years = train_years if train_years is not None else DEFAULT_TRAIN_YEARS
        self.test_years = test_years if test_years is not None else DEFAULT_TEST_YEARS
        self.read_from_key = read_from_key
        self.save_to_prefix = save_to_prefix or SHARE_MODELS_S3_PREFIX

    def __call__(self, context: Context) -> Context:
        if self.read_from_key is not None:
            prefix = self.read_from_key.rstrip("/")
            share_models: dict[str, XGBRegressor] = {}
            for i, channel in enumerate(RECOVERY_CHANNELS, start=1):
                logger.info(
                    "[%d/%d] Loading pre-trained share model for channel '%s'",
                    i, len(RECOVERY_CHANNELS), channel,
                )
                share_models[channel] = _load_model_from_s3(S3_BUCKET, f"{prefix}/{channel}_share.joblib")
        else:
            df = context[ContextKeys.DF_RECOVERY_PREPROCESSED]

            feature_cols = _resolve_feature_cols(df)
            logger.info(
                "Per-channel share regressor: %d / %d base features resolved against frame schema",
                len(feature_cols), len(_BASE_FEATURE_COLS),
            )

            share_models = {}
            save_prefix = self.save_to_prefix.rstrip("/")
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

                if self.tune:
                    best_params = _tune_with_optuna(
                        X_train, X_test, y_train, y_test,
                        n_trials=self.n_trials,
                        channel=channel,
                    )
                else:
                    best_params = _DEFAULT_CHANNEL_PARAMS[channel]

                model = _train_final_model(
                    X_train, X_test, y_train, y_test, best_params,
                )
                model.site_gl_baseline_ = site_gl_baseline
                model.aug_features_ = aug_features
                model.channel_ = channel


                _save_model_to_s3(model, S3_BUCKET, f"{save_prefix}/{channel}_share.joblib")

                share_models[channel] = model

        context[ContextKeys.SHARE_MODELS] = share_models
        context.lock(ContextKeys.SHARE_MODELS)

        return context
