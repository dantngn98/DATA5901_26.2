# standard
import logging

# third-party
import numpy as np
import optuna
import pandas as pd
import polars as pl
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# local
from src.config import (
    RecoverySchema,
    RECOVERY_FUNNEL_CONSOLIDATED_RECOVERY_TYPES,
    CATEGORICAL_COLUMNS, PER_TYPE_REG_FEATURE_COLUMNS,
    RECOVERY_RATE_TARGET_COLUMN, PER_TYPE_TARGET_COLUMN_DICT,
    PER_TYPE_REG_DEFAULT_PARAMS_DICT,
    ContextKeys
)
from src.pipeline import Context, enforce
from src.pipeline.conditions import Defines, Locks, Sequence
from src.pipeline.types import PipelineStep
from src.util import cast_categoricals, load_joblib_from_s3, write_joblib_to_s3, S3Path


logger = logging.getLogger(__name__)

# ============================================================
# Pipeline step
# ============================================================

@enforce({
    # ContextKeys.DF_RECOVERY_PREPROCESSED: Requires(),  # not enforced; read_from_key makes it optional
    ContextKeys.SHARE_MODELS: Sequence(Defines(strict=True), Locks(strict=True)),
})
class TrainPerChannelShareRegressors(PipelineStep):
    """Pipeline step that trains (and optionally tunes) the Stage 3 per-channel share regressors.

    For each of the 5 consolidated recovery channels, fits an XGBRegressor on
    logit(share) where share = prob_<channel> / prob_recovered, on rows with
    prob_recovered > 0. When tune=True, a separate Optuna study is run per
    channel; when tune=False, the baked-in _DEFAULT_CHANNEL_PARAMS are used.

    When load_from is provided, all 5 models are loaded from S3 and no training occurs.

    Each trained model carries `site_gl_baseline_`, `channel_`, and `metrics_`
    so that downstream inference can recover everything from the single artifact
    Trained models are saved to S3 under save_to (one .joblib per channel) and stored
    in context under ContextKeys.SHARE_MODELS as a dict keyed by channel name.

    Softmax normalisation across channels and combination with the Stage-1
    p_recovered_hat prediction happen at inference, not in this step.
    """

    def __init__(
        self,
        train_years,
        test_years,
        tune: bool = False,
        n_trials: int = 50,
        load_from: dict[str, S3Path] | None = None,  # consolidated_recovery_type -> joblib file
        save_to: dict[str, S3Path] | None = None,
    ):
        # load from/save to should occur for all or none of the (non-sales) recovery types
        if load_from is not None and\
           (load_recovery_types := set(load_from.keys())) != RECOVERY_FUNNEL_CONSOLIDATED_RECOVERY_TYPES:
            raise ValueError(
                f"load_from keys should be recovery funnel types {RECOVERY_FUNNEL_CONSOLIDATED_RECOVERY_TYPES} "
                f"but got {load_recovery_types}"
            )
        if save_to is not None and\
           (save_recovery_types := set(save_to.keys())) != RECOVERY_FUNNEL_CONSOLIDATED_RECOVERY_TYPES:
            raise ValueError(
                f"save_to keys should be recovery funnel types {RECOVERY_FUNNEL_CONSOLIDATED_RECOVERY_TYPES} "
                f"but got {save_recovery_types}"
            )
        if load_from is not None and save_to is not None:
            logger.warning(f"both loading and saving model (is this intentional?): '{load_from}' -> '{save_to}'")
        
        self.tune = tune
        self.n_trials = n_trials
        self.train_years = train_years
        self.test_years = test_years
        self.load_from = load_from
        self.save_to = save_to

    def __call__(self, context: Context) -> Context:
        if self.load_from is not None:
            share_models = {}
            for i, (consolidated_recovery_type, s3_path) in enumerate(self.load_from.items(), start=1):
                logger.info(
                    f"{i}/{len(RECOVERY_FUNNEL_CONSOLIDATED_RECOVERY_TYPES)} loading regression model "
                    f"for recovery type {consolidated_recovery_type} from '{s3_path}'"
                )
                model = load_joblib_from_s3(s3_path)
                logger.info(f"loaded {type(model)} object")
                assert isinstance(model, XGBRegressor)
                share_models[consolidated_recovery_type] = model
        else:
            df = context[ContextKeys.DF_RECOVERY_PREPROCESSED]

            share_models = {}
            for i, consolidated_recovery_type in enumerate(RECOVERY_FUNNEL_CONSOLIDATED_RECOVERY_TYPES, start=1):
                logger.info(
                    "[%d/%d] Training share regressor for channel '%s'",
                    i, len(RECOVERY_FUNNEL_CONSOLIDATED_RECOVERY_TYPES), consolidated_recovery_type,
                )

                X_train, X_test, y_train, y_test, site_gl_baseline = (
                    _build_channel_splits(
                        df,
                        target_col=PER_TYPE_TARGET_COLUMN_DICT[consolidated_recovery_type],
                        train_years=self.train_years,
                        test_years=self.test_years
                    )
                )

                if self.tune:
                    best_params = _tune_with_optuna(
                        X_train, X_test, y_train, y_test,
                        n_trials=self.n_trials,
                        channel=consolidated_recovery_type,
                    )
                else:
                    best_params = PER_TYPE_REG_DEFAULT_PARAMS_DICT[consolidated_recovery_type]

                model = _train_final_model(
                    X_train, X_test, y_train, y_test, best_params,
                )
                model.site_gl_baseline_ = site_gl_baseline
                model.channel_ = consolidated_recovery_type

                if self.save_to is not None:
                    write_joblib_to_s3(model, self.save_to[consolidated_recovery_type])

                share_models[consolidated_recovery_type] = model

        context[ContextKeys.SHARE_MODELS] = share_models
        context.lock(ContextKeys.SHARE_MODELS)

        return context


# ============================================================
# Transform helpers
# ============================================================

_EPS = 1e-7

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

def _compute_site_gl_share_baseline(df_train_nz: pl.DataFrame) -> pl.DataFrame:
    return (
        df_train_nz
        .group_by([RecoverySchema.HASHED_FC, RecoverySchema.GL_PRODUCT_GROUP])
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
    test_years: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pl.DataFrame]:
    """Filter to recovered subset, build per-channel share + baseline, return matrices."""
    df_train_nz = (
        df
        .filter(pl.col(RecoverySchema.YEAR).is_in(train_years))
        .filter(pl.col(RECOVERY_RATE_TARGET_COLUMN) > 0)
        .with_columns((pl.col(target_col) / pl.col(RECOVERY_RATE_TARGET_COLUMN)).alias("_share"))
    )
    df_test_nz = (
        df
        .filter(pl.col(RecoverySchema.YEAR).is_in(test_years))
        .filter(pl.col(RECOVERY_RATE_TARGET_COLUMN) > 0)
        .with_columns((pl.col(target_col) / pl.col(RECOVERY_RATE_TARGET_COLUMN)).alias("_share"))
    )

    site_gl_baseline = _compute_site_gl_share_baseline(df_train_nz)

    df_train_nz = df_train_nz.join(
        site_gl_baseline, on=[RecoverySchema.HASHED_FC, RecoverySchema.GL_PRODUCT_GROUP], how="left"
    )
    df_test_nz = df_test_nz.join(
        site_gl_baseline, on=[RecoverySchema.HASHED_FC, RecoverySchema.GL_PRODUCT_GROUP], how="left"
    )

    features = tuple(PER_TYPE_REG_FEATURE_COLUMNS - {target_col})
    X_train = cast_categoricals(_polars_to_pandas_safe(df_train_nz.select(features)), CATEGORICAL_COLUMNS)
    X_test  = cast_categoricals(_polars_to_pandas_safe(df_test_nz.select(features)), CATEGORICAL_COLUMNS)

    y_train = df_train_nz["_share"].to_numpy()
    y_test  = df_test_nz["_share"].to_numpy()

    logger.info(
        "Channel '%s' splits: train_nz=%d (%s), test_nz=%d (%s)",
        target_col, len(df_train_nz), train_years, len(df_test_nz), test_years,
    )
    return X_train, X_test, y_train, y_test, site_gl_baseline


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
) -> XGBRegressor:
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
