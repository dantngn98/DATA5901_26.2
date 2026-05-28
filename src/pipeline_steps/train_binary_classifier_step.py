# standard
import logging

# third-party
import numpy as np
import optuna
import pandas as pd
import polars as pl
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

# local
from src.config import (
    RecoverySchema,
    CATEGORICAL_COLUMNS,
    RECOVERY_RATE_CLF_FEATURE_COLUMNS, RECOVERY_RATE_TARGET_COLUMN, RECOVERY_RATE_CLF_DEFAULT_PARAMS, 
    ContextKeys
)
from src.pipeline import Context, enforce
from src.pipeline.conditions import Defines, Locks, Sequence
from src.pipeline.types import PipelineStep
from src.util import (
    load_joblib_from_s3, write_joblib_to_s3,
    cast_categoricals,
    S3Path
)


logger = logging.getLogger(__name__)

# ============================================================
# Pipeline step
# ============================================================

@enforce({
    # ContextKeys.DF_RECOVERY_PREPROCESSED: Requires(),  # not enforced; read_from makes it optional
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
        train_years: list[int],
        test_years: list[int],
        tune: bool = False,
        n_trials: int = 50,
        load_from: S3Path | None = None,
        save_to: S3Path | None = None,
    ):
        if load_from is not None and save_to is not None:
            logger.warning(f"both loading and saving model (is this intentional?): '{load_from}' -> '{save_to}'")
        
        self.train_years = train_years
        self.test_years = test_years
        self.tune = tune
        self.n_trials = n_trials
        self.load_from = load_from
        self.save_to = save_to

    def __call__(self, context: Context) -> Context:
        if self.load_from is not None:
            logger.info(f"loading recovery binary classifier from '{self.load_from.uri}'")
            model = load_joblib_from_s3(self.load_from)
            logger.info(f"loaded {type(model)} object")
            assert isinstance(model, XGBClassifier)
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
                best_params = RECOVERY_RATE_CLF_DEFAULT_PARAMS

            model = _train_final_model(
                X_train, X_val, y_train, y_val,
                scale_pos_weight=scale_pos_weight,
                best_params=best_params,
            )

        if self.save_to is not None:
            logger.info(f"saving recovery binary classifier to '{self.save_to.uri}'")
            write_joblib_to_s3(model, self.save_to)

        context[ContextKeys.CLF_MODEL] = model
        context.lock(ContextKeys.CLF_MODEL)

        return context


# ============================================================
# Private helpers
# ============================================================

def _build_train_val_splits(
    df: pl.DataFrame,
    train_years: list[int],
    test_years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, float]:
    df_train = df.filter(pl.col(RecoverySchema.YEAR).is_in(train_years))
    df_val = df.filter(pl.col(RecoverySchema.YEAR).is_in(test_years))

    feature_columns = tuple(RECOVERY_RATE_CLF_FEATURE_COLUMNS)
    X_train = cast_categoricals(df_train.select(feature_columns).to_pandas(), CATEGORICAL_COLUMNS)
    X_val = cast_categoricals(df_val.select(feature_columns).to_pandas(), CATEGORICAL_COLUMNS)

    y_train = (df_train[RECOVERY_RATE_TARGET_COLUMN].to_numpy() > 0).astype(np.float32)
    y_val = (df_val[RECOVERY_RATE_TARGET_COLUMN].to_numpy() > 0).astype(np.float32)

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
