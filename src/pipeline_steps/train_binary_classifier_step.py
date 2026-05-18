# standard
import logging
import tempfile

# third-party
import boto3
import joblib
import numpy as np
import optuna
import pandas as pd
import polars as pl
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

# local
from src.config import CLF_MODEL_S3_KEY, ContextKeys, S3_BUCKET
from src.pipeline import Context, enforce
from src.pipeline.conditions import Defines, Locks, Sequence
from src.pipeline.types import PipelineStep
from src.util import load

logger = logging.getLogger(__name__)

# ============================================================
# Feature and category constants
# ============================================================

_CAT_COLS: list[str] = [
    "hashed_fc",
    "gl_product_group",
    "country",
    "country_state",
    "site_type",
    "site_category",
]

_gl_composition_cols: list[str] = [
    "share_food", "share_non_food", "share_pet_food",
    "share_RETAIL", "share_FBA", "share_hazmat",
]

_gl_volume_cols: list[str] = [
    "units_total", "cogs_total", "weight_total",
    "avg_cogs_per_unit", "avg_weight_per_unit", "cogs_per_unit_weight",
]

_gl_at_site_cols: list[str] = [
    "site_units_share_week", "site_weight_share_week",
]

_site_context_cols: list[str] = [
    "site_units_total_week", "site_weight_total_week",
    "site_type", "site_category", "country", "country_state",
]

_temporal_site_context_cols: list[str] = [
    "site_units_total_week_lag_1w",
    "site_units_total_week_lag_4w",
    "site_units_total_week_lag_12w",
    "site_units_total_week_lag_13w",
    "site_units_total_week_lag_52w",
    "site_weight_total_week_lag_1w",
    "site_weight_total_week_lag_4w",
    "site_weight_total_week_lag_12w",
    "site_weight_total_week_lag_13w",
    "site_weight_total_week_lag_52w",
    "site_prob_recovered_week_lag_1w",
    "site_prob_recovered_week_lag_4w",
    "site_prob_recovered_week_lag_12w",
    "site_prob_recovered_week_lag_13w",
    "site_prob_recovered_week_lag_52w",
    "site_prob_recovered_week_rolling_4w",
    "site_prob_recovered_week_rolling_12w",
    "site_prob_recovered_week_rolling_26w",
    "site_prob_recovered_week_rolling_52w",
]

_calendar_cols: list[str] = ["month", "week"]

_temporal_composition_cols: list[str] = [
    "share_RETAIL_lag_1w",
    "share_RETAIL_lag_4w",
    "share_RETAIL_lag_12w",
    "share_RETAIL_lag_13w",
    "share_RETAIL_lag_52w",
    "share_FBA_lag_1w",
    "share_FBA_lag_4w",
    "share_FBA_lag_12w",
    "share_FBA_lag_13w",
    "share_FBA_lag_52w",
    "share_hazmat_lag_1w",
    "share_hazmat_lag_4w",
    "share_hazmat_lag_12w",
    "share_hazmat_lag_13w",
    "share_hazmat_lag_52w",
    "share_food_lag_1w",
    "share_food_lag_4w",
    "share_food_lag_12w",
    "share_food_lag_13w",
    "share_food_lag_52w",
    "share_non_food_lag_1w",
    "share_non_food_lag_4w",
    "share_non_food_lag_12w",
    "share_non_food_lag_13w",
    "share_non_food_lag_52w",
    "share_pet_food_lag_1w",
    "share_pet_food_lag_4w",
    "share_pet_food_lag_12w",
    "share_pet_food_lag_13w",
    "share_pet_food_lag_52w",
    "share_food_rolling_4w",
    "share_food_rolling_12w",
    "share_non_food_rolling_4w",
    "share_non_food_rolling_12w",
    "share_pet_food_rolling_4w",
    "share_pet_food_rolling_12w",
    "share_RETAIL_rolling_4w",
    "share_RETAIL_rolling_12w",
    "share_FBA_rolling_4w",
    "share_FBA_rolling_12w",
    "share_hazmat_rolling_4w",
    "share_hazmat_rolling_12w",
    "share_food_rolling_26w",
    "share_food_rolling_52w",
    "share_non_food_rolling_26w",
    "share_non_food_rolling_52w",
    "share_pet_food_rolling_26w",
    "share_pet_food_rolling_52w",
    "share_RETAIL_rolling_26w",
    "share_RETAIL_rolling_52w",
    "share_FBA_rolling_26w",
    "share_FBA_rolling_52w",
    "share_hazmat_rolling_26w",
    "share_hazmat_rolling_52w",
    "share_RETAIL_ewma_5a",
    "share_RETAIL_ewma_1a",
    "share_FBA_ewma_5a",
    "share_FBA_ewma_1a",
    "share_hazmat_ewma_5a",
    "share_hazmat_ewma_1a",
    "share_food_ewma_5a",
    "share_food_ewma_1a",
    "share_non_food_ewma_5a",
    "share_non_food_ewma_1a",
    "share_pet_food_ewma_5a",
    "share_pet_food_ewma_1a",
]

_temporal_volume_cols: list[str] = [
    "units_total_lag_1w",
    "units_total_lag_4w",
    "units_total_lag_12w",
    "units_total_lag_13w",
    "units_total_lag_52w",
    "cogs_total_lag_1w",
    "cogs_total_lag_4w",
    "cogs_total_lag_12w",
    "cogs_total_lag_13w",
    "cogs_total_lag_52w",
    "weight_total_lag_1w",
    "weight_total_lag_4w",
    "weight_total_lag_12w",
    "weight_total_lag_13w",
    "weight_total_lag_52w",
    "units_total_rolling_4w",
    "units_total_rolling_12w",
    "cogs_total_rolling_4w",
    "cogs_total_rolling_12w",
    "weight_total_rolling_4w",
    "weight_total_rolling_12w",
    "units_total_rolling_26w",
    "units_total_rolling_52w",
    "cogs_total_rolling_26w",
    "cogs_total_rolling_52w",
    "weight_total_rolling_26w",
    "weight_total_rolling_52w",
    "units_total_ewma_5a",
    "units_total_ewma_1a",
    "cogs_total_ewma_5a",
    "cogs_total_ewma_1a",
    "weight_total_ewma_5a",
    "weight_total_ewma_1a",
]

_temporal_probability_cols: list[str] = [
    "prob_recovered_lag_1w",
    "prob_recovered_lag_4w",
    "prob_recovered_lag_12w",
    "prob_recovered_lag_13w",
    "prob_recovered_lag_52w",
    "prob_recovered_rolling_26w",
    "prob_recovered_rolling_52w",
    "prob_recovered_rolling_4w",
    "prob_recovered_rolling_12w",
    "prob_recovered_ewma_5a",
    "prob_recovered_ewma_1a",
]

_FEATURE_COLS: list[str] = (
    _gl_composition_cols
    + _gl_volume_cols
    + _gl_at_site_cols
    + _site_context_cols
    + _temporal_site_context_cols
    + _calendar_cols
    + _temporal_composition_cols
    + _temporal_volume_cols
    + _temporal_probability_cols
)

assert len(_FEATURE_COLS) == 151, (
    f"Expected 151 features, got {len(_FEATURE_COLS)}. "
    "Check for duplicates or missing columns in the sublists."
)

_DEFAULT_CLF_PARAMS: dict = {
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

# TODO: config
_DEFAULT_TRAIN_YEARS: list[int] = [2022, 2023, 2024]
_DEFAULT_TEST_YEARS: list[int] = [2025]


# ============================================================
# Private helpers
# ============================================================

def _cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in _CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _build_train_val_splits(
    df: pl.DataFrame,
    train_years: list[int],
    test_years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, float]:
    df_train = df.filter(pl.col("year").is_in(train_years))
    df_val = df.filter(pl.col("year").is_in(test_years))

    X_train = _cast_categoricals(df_train.select(_FEATURE_COLS).to_pandas())
    X_val = _cast_categoricals(df_val.select(_FEATURE_COLS).to_pandas())

    y_train = (df_train["prob_recovered"].to_pandas().values > 0).astype(np.float32)
    y_val = (df_val["prob_recovered"].to_pandas().values > 0).astype(np.float32)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    if n_pos == 0:
        raise ValueError(
            f"No positive samples in training split (years={train_years}). "
            "Check train_years and data coverage."
        )
    scale_pos_weight = n_neg / n_pos
    logger.info(
        "Stage 1 split: train=%d rows (%s), val=%d rows (%s), "
        "n_neg=%d, n_pos=%d, scale_pos_weight=%.2f",
        len(df_train), train_years, len(df_val), test_years,
        n_neg, n_pos, scale_pos_weight,
    )
    return X_train, X_val, y_train, y_val, scale_pos_weight


def _tune_with_optuna(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
    n_trials: int,
) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "enable_categorical": True,
            "random_state": 42,
            "n_estimators": 500,
            "early_stopping_rounds": 30,
            "eval_metric": ["aucpr", "logloss"],
            "scale_pos_weight": scale_pos_weight,
            "callbacks": [XGBoostPruningCallback(trial, "validation_0-logloss")],
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "max_delta_step":   trial.suggest_int("max_delta_step", 0, 10),
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        p_val = model.predict_proba(X_val)[:, 1]
        return -float(average_precision_score(y_val, p_val))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5),
        study_name="xgb_stage1_classifier",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(
        "Optuna Stage 1 complete: best AUC-PR=%.4f, params=%s",
        -study.best_value, study.best_params,
    )
    return study.best_params


def _train_final_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
    best_params: dict,
) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=2000,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        enable_categorical=True,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric=["auc", "aucpr", "logloss"],
        **best_params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    logger.info("Stage 1 final model: best_iteration=%d", model.best_iteration)
    return model

# TODO: implement these as load/write utils
def _save_clf_to_s3(model: XGBClassifier, bucket: str, key: str) -> None:
    s3_client = boto3.client("s3")
    try:
        with tempfile.TemporaryFile() as fp:
            joblib.dump(model, fp)
            fp.seek(0)
            s3_client.put_object(Body=fp.read(), Bucket=bucket, Key=key)
            logger.info("Classifier saved to s3://%s/%s", bucket, key)
    except Exception:
        logger.exception("Failed to save classifier to s3://%s/%s", bucket, key)
        raise

def _load_clf_from_s3(bucket: str, key: str) -> XGBClassifier:
    s3_client = boto3.client("s3")
    try:
        with tempfile.TemporaryFile() as fp:
            s3_client.download_fileobj(bucket, key, fp)
            fp.seek(0)

            model = joblib.load(fp)

            if not isinstance(model, XGBClassifier):
                raise ValueError("Object loaded from s3://%s/%s is not an instance of XGBClassifier")

            logger.info("Classifier loaded from s3://%s/%s", bucket, key)
            return model

    except Exception:
        logger.exception(
            "Failed to load classifier from s3://%s/%s",
            bucket,
            key,
        )
        raise


# ============================================================
# Pipeline step
# ============================================================

@enforce({
    # ContextKeys.DF_RECOVERY_PREPROCESSED: Requires(),  # not enforced; read_from_key makes it optional
    ContextKeys.CLF_MODEL: Sequence(Defines(strict=True), Locks(strict=True)),
})
class TrainBinaryClassifier(PipelineStep):
    """
    Define the binary classification model, either by loading a pre-trained model from S3 or 
    by training a new XGBoost classifier on the training data output by the preprocessing step.
    
    Trains a binary classifier predicting P(prob_recovered > 0), saves the model
    to S3, and stores it in context under ContextKeys.CLF_MODEL.
    """

    def __init__(
        self,
        tune: bool = False,
        n_trials: int = 50,
        train_years: list[int] | None = None,
        test_years: list[int] | None = None,
        read_from_key: str | None = None,
        save_to_key: str | None = None,
    ):
        self.tune = tune
        self.n_trials = n_trials
        self.train_years = train_years if train_years is not None else _DEFAULT_TRAIN_YEARS
        self.test_years = test_years if test_years is not None else _DEFAULT_TEST_YEARS
        self.read_from_key = read_from_key
        self.save_to_key = save_to_key

    def __call__(self, context: Context) -> Context:
        if self.read_from_key is not None:
            model = _load_clf_from_s3(S3_BUCKET, self.read_from_key)
        else:
            df = context[ContextKeys.DF_RECOVERY_PREPROCESSED]

            X_train, X_val, y_train, y_val, scale_pos_weight = _build_train_val_splits(
                df, self.train_years, self.test_years
            )

            if self.tune:
                best_params = _tune_with_optuna(
                    X_train, X_val, y_train, y_val,
                    scale_pos_weight=scale_pos_weight,
                    n_trials=self.n_trials,
                )
            else:
                best_params = _DEFAULT_CLF_PARAMS

            model = _train_final_model(
                X_train, X_val, y_train, y_val,
                scale_pos_weight=scale_pos_weight,
                best_params=best_params,
            )

        if self.save_to_key is not None:
            _save_clf_to_s3(model, S3_BUCKET, self.save_to_key)

        context[ContextKeys.CLF_MODEL] = model
        context.lock(ContextKeys.CLF_MODEL)

        return context
