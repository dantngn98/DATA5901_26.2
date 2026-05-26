# standard
import logging

# third-party
import numpy as np
import optuna
import polars as pl
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# local
from src.config import (
    CATEGORICAL_COLUMNS,
    RECOVERY_RATE_CLF_FEATURE_COLUMNS, RECOVERY_RATE_REG_DEFAULT_PARAMS,
    ContextKeys
)
from src.pipeline import Context, enforce
from src.pipeline.conditions import Defines, Locks, Sequence
from src.pipeline.types import PipelineStep
from src.util import cast_categoricals, load_joblib_from_s3, write_joblib_to_s3, S3Path

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

_BASELINE_COLS: list[str] = [
    "site_gl_mean_rate",
    "site_gl_std_rate",
    "site_gl_n_nonzero_weeks",
]

_EXTENDED_FEATURE_COLS: list[str] = RECOVERY_RATE_CLF_FEATURE_COLUMNS + _BASELINE_COLS

_EPS = 1e-6

# ============================================================
# Pipeline step
# ============================================================

@enforce({
    # ContextKeys.DF_RECOVERY_PREPROCESSED: Requires(),  # not enforced; read_from makes it optional
    ContextKeys.REG_MODEL: Sequence(Defines(strict=True), Locks(strict=True)),
})
class TrainRegressor(PipelineStep):
    """
    Define the regression model for predicting recovery rates, either by loading a pre-trained model from S3 or 
    by training a new XGBoost regressor on the training data output by the preprocessing step.
    Operates on non-zero recovery rows only. Computes site - GL baseline features
    from training data, persists them on the model object as `site_gl_baseline_` and
    `baseline_priors_`, saves to S3, and
    stores the model in context under ContextKeys.REG_MODEL.
    """

    def __init__(
        self,
        train_years,
        test_years,
        tune: bool = False,
        n_trials: int = 50,
        holdout_frac: float = 0.05,
        holdout_seed: int = 42,
        read_from: S3Path | None = None,
        save_to: S3Path | None = None,
    ):
        self.train_years = train_years
        self.test_years = test_years
        self.tune = tune
        self.n_trials = n_trials
        self.holdout_frac = holdout_frac
        self.holdout_seed = holdout_seed
        self.read_from = read_from
        self.save_to = save_to

    def __call__(self, context: Context) -> Context:
        if self.read_from_key is not None:
            logger.info(f"loading regressor from '{self.read_from.uri}'")
            model = load_joblib_from_s3(self.read_from)
            logger.info(f"loaded {type(model)} object")
            assert isinstance(model, XGBRegressor)
        else:
            df = context[ContextKeys.DF_RECOVERY_PREPROCESSED]

            X_train, X_val, y_train, y_val, site_gl_baseline, priors = _build_train_val_splits(
                df,
                train_years=self.train_years,
                test_years=self.test_years,
                holdout_frac=self.holdout_frac,
                holdout_seed=self.holdout_seed,
            )

            if self.tune:
                best_params = _tune_with_optuna(
                    X_train, X_val, y_train, y_val,
                    n_trials=self.n_trials,
                )
            else:
                best_params = RECOVERY_RATE_REG_DEFAULT_PARAMS

            model = _train_final_model(X_train, X_val, y_train, y_val, best_params)

            # Persist baseline data on the model so downstream steps and evaluation
            # notebooks can load it without re-reading training data.
            model.site_gl_baseline_ = site_gl_baseline
            model.baseline_priors_ = priors

        if self.save_to is not None and self.save_to != self.read_from:
            logger.info(f"saving binary classifier to '{self.save_to.uri}'")
            write_joblib_to_s3(model, self.save_to)

        context[ContextKeys.REG_MODEL] = model
        context.lock(ContextKeys.REG_MODEL)

        return context


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


def _compute_weights(y: np.ndarray) -> np.ndarray:
    """Sample weights emphasising the 10-60% recovery rate band."""
    w = np.ones(len(y))
    w[(y >= 0.1) & (y < 0.3)] = 3.0
    w[(y >= 0.3) & (y < 0.6)] = 8.0
    w[(y >= 0.6)] = 2.0
    return w


# ============================================================
# Baseline helpers (site × GL mean/std/count features)
# ============================================================

def _compute_site_gl_baseline(
    df_train_nz: pl.DataFrame,
    target_col: str = "prob_recovered",
) -> pl.DataFrame:
    return (
        df_train_nz
        .group_by(["hashed_fc", "gl_product_group"])
        .agg([
            pl.col(target_col).mean().alias("site_gl_mean_rate"),
            pl.col(target_col).std().alias("site_gl_std_rate"),
            pl.col(target_col).count().alias("site_gl_n_nonzero_weeks"),
        ])
    )


def _build_baseline_priors(
    df_train_nz: pl.DataFrame,
    target_col: str = "prob_recovered",
) -> dict:
    gl_baseline = (
        df_train_nz
        .group_by("gl_product_group")
        .agg([
            pl.col(target_col).mean().alias("gl_mean_rate"),
            pl.col(target_col).std().alias("gl_std_rate"),
        ])
    )
    site_baseline = (
        df_train_nz
        .group_by("hashed_fc")
        .agg([
            pl.col(target_col).mean().alias("site_mean_rate"),
            pl.col(target_col).std().alias("site_std_rate"),
        ])
    )
    return {
        "gl_baseline": gl_baseline,
        "site_baseline": site_baseline,
        "global_mean_rate": float(df_train_nz[target_col].mean()),
        "global_std_rate": float(df_train_nz[target_col].std()),
    }


def _holdout_site_gl_baseline(
    site_gl_baseline: pl.DataFrame,
    holdout_frac: float = 0.05,
    seed: int = 42,
) -> pl.DataFrame:
    """Remove `holdout_frac` of site-GL pairs so XGBoost learns splits on prior-filled rows."""
    holdout_pairs = (
        site_gl_baseline
        .select(["hashed_fc", "gl_product_group"])
        .sample(fraction=holdout_frac, seed=seed)
        .with_columns(pl.lit(True).alias("_held_out"))
    )
    return (
        site_gl_baseline
        .join(holdout_pairs, on=["hashed_fc", "gl_product_group"], how="left")
        .filter(pl.col("_held_out").is_null())
        .drop("_held_out")
    )


def _fill_site_gl_baseline(df: pl.DataFrame, priors: dict) -> pl.DataFrame:
    """Fill NaN baseline cols using GL -> site -> global fallback hierarchy."""
    df = df.join(priors["gl_baseline"], on="gl_product_group", how="left")
    df = df.join(priors["site_baseline"], on="hashed_fc", how="left")
    df = df.with_columns([
        pl.coalesce([
            pl.col("site_gl_mean_rate"),
            pl.col("gl_mean_rate"),
            pl.col("site_mean_rate"),
            pl.lit(priors["global_mean_rate"]),
        ]).alias("site_gl_mean_rate"),
        pl.coalesce([
            pl.col("site_gl_std_rate"),
            pl.col("gl_std_rate"),
            pl.col("site_std_rate"),
            pl.lit(priors["global_std_rate"]),
        ]).alias("site_gl_std_rate"),
        pl.col("site_gl_n_nonzero_weeks").fill_null(0).alias("site_gl_n_nonzero_weeks"),
    ])
    return df.drop(["gl_mean_rate", "gl_std_rate", "site_mean_rate", "site_std_rate"])


# ============================================================
# Private helpers
# ============================================================

def _build_train_val_splits(
    df: pl.DataFrame,
    train_years: list[int],
    test_years: list[int],
    holdout_frac: float,
    holdout_seed: int,
    target_col: str = "prob_recovered",
):
    """Filter to non-zero rows, compute and join baselines, return feature matrices."""
    df_train = df.filter(pl.col("year").is_in(train_years))
    df_val = df.filter(pl.col("year").is_in(test_years))

    df_train_nz = df_train.filter(pl.col(target_col) > 0)
    df_val_nz = df_val.filter(pl.col(target_col) > 0)

    logger.info(
        "Stage 2 split: train_nz=%d rows (%s), val_nz=%d rows (%s)",
        len(df_train_nz), train_years, len(df_val_nz), test_years,
    )

    site_gl_baseline = _compute_site_gl_baseline(df_train_nz, target_col)
    priors = _build_baseline_priors(df_train_nz, target_col)

    site_gl_baseline_seen = _holdout_site_gl_baseline(
        site_gl_baseline, holdout_frac=holdout_frac, seed=holdout_seed
    )
    n_held = len(site_gl_baseline) - len(site_gl_baseline_seen)
    logger.info(
        "Holdout site-GL pairs: %d / %d (%.1f%%)",
        n_held, len(site_gl_baseline), holdout_frac * 100,
    )

    df_train_nz = df_train_nz.join(
        site_gl_baseline_seen, on=["hashed_fc", "gl_product_group"], how="left"
    )
    df_train_nz = _fill_site_gl_baseline(df_train_nz, priors)

    df_val_nz = df_val_nz.join(
        site_gl_baseline, on=["hashed_fc", "gl_product_group"], how="left"
    )
    n_unseen = df_val_nz.filter(pl.col("site_gl_mean_rate").is_null()).height
    logger.info(
        "Val rows with unseen site-GL (pre-fill): %d (%.1f%%)",
        n_unseen, n_unseen / len(df_val_nz) * 100 if len(df_val_nz) else 0,
    )
    df_val_nz = _fill_site_gl_baseline(df_val_nz, priors)

    X_train = cast_categoricals(df_train_nz.select(_EXTENDED_FEATURE_COLS).to_pandas(), CATEGORICAL_COLUMNS)
    X_val = cast_categoricals(df_val_nz.select(_EXTENDED_FEATURE_COLS).to_pandas(), CATEGORICAL_COLUMNS)

    y_train = df_train_nz[target_col].to_pandas().values
    y_val = df_val_nz[target_col].to_pandas().values

    return X_train, X_val, y_train, y_val, site_gl_baseline, priors


def _tune_with_optuna(
    X_train,
    X_val,
    y_train: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    y_train_logit = _logit(y_train)
    y_val_logit = _logit(y_val)
    w_train = _compute_weights(y_train)

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
        model.fit(
            X_train, y_train_logit,
            sample_weight=w_train,
            eval_set=[(X_val, y_val_logit)],
            verbose=False,
        )
        preds = np.clip(_sigmoid(model.predict(X_val)), 0.0, 1.0)
        return float(mean_absolute_error(y_val, preds))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5),
        study_name="xgb_stage2_regressor",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(
        "Optuna Stage 2 complete: best MAE=%.4f, params=%s",
        study.best_value, study.best_params,
    )
    return study.best_params


def _train_final_model(
    X_train,
    X_val,
    y_train: np.ndarray,
    y_val: np.ndarray,
    best_params: dict,
) -> XGBRegressor:
    y_train_logit = _logit(y_train)
    y_val_logit = _logit(y_val)
    w_train = _compute_weights(y_train)

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
    model.fit(
        X_train, y_train_logit,
        sample_weight=w_train,
        eval_set=[(X_val, y_val_logit)],
        verbose=100,
    )
    logger.info("Stage 2 final model: best_iteration=%d", model.best_iteration)
    return model