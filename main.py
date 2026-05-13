# standard
import argparse
import logging

# local
from src.pipeline import Context, Pipeline
from src.pipeline_steps import (
    Load,
    Preprocess,
    TrainBinaryClassifier, TrainRegressor,
    TrainPerChannelShareRegressors,
    Predict
)


logger = logging.getLogger(__name__)


# === CONSTANTS ===


# === CLI ===

parser = argparse.ArgumentParser(description="...")
parser.add_argument("example", help="example parameter")


# === parameterized context, pipeline construction ===

def init_context(args: argparse.Namespace) -> Context:
    return Context()

def construct_pipeline(args: argparse.Namespace) -> Pipeline:
    """
    general param categories: data sourcing, behavior parameterization

    1. Load
        - params: source data filepath(s)
    2. Preprocess
        - params:
            - optional saved preprocessed data location
            - optional write location
            - preprocessing parameters (e.g., # lag weeks)
    3. Train Binary Classifier, Regressor, Per-Channel Share Regressors
        - params:
            - optinal read from/write to locations (i.e., read saved model instead of re-training)
            - whether to tune model vs. use defaults
            - train parameters (e.g., train-test split, n_trials)
    4. Predict
        - params:
            - optional read files for preprocessed data and models (TODO: these should be handled in previous steps)
            - subset to predict on (sites, GL groups, years, weeks)
            - SHAP parameters

    """

    # steps can be omitted by passing None
    # e.g., Load() if not load_preprocessed_data else None
    return Pipeline(
        Load(),
        Preprocess(),
        TrainBinaryClassifier(),
        TrainRegressor(),
        TrainPerChannelShareRegressors(),
        Predict(),
        # Report()
    )

# === main ===

def main(args: argparse.Namespace):
    context = init_context(args)
    pipeline = construct_pipeline(args)
    result = pipeline(context)  # context -> pipeline -> result
    ...
    # either do something with result here or incorporate into pipeline
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
