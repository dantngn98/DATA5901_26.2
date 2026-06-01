# Recovery Funnel Prediction Pipeline

This project predicts the **proportion of products that enter the recovery funnel** (returned to vendor, donated, disposed of, etc.) for a given fulfillment site, GL product group, and week.

The entry point is `main.py`. All modelling logic is wired together through a lightweight pipeline framework that lives in `src/pipeline/` and `src/pipeline_steps/`.

---

## Pipeline Framework

Three abstractions drive everything.

### `Pipeline`

An ordered sequence of steps executed one after another. You construct it by passing step objects and then call it like a function:

```python
result_context = pipeline(context)
```

Pipelines can be extended or concatenated:

```python
pipeline_a + pipeline_b          # new combined Pipeline
pipeline_a.extend(pipeline_b)    # mutate in place
```

Passing `None` in place of a step silently skips it. This is the idiomatic way to make steps optional from `main.py`:

```python
Pipeline(
    Load() if not use_cached_data else None,
    Preprocess(),
    ...
)
```

### `Context`

A shared dictionary that flows through each step. Every step reads its inputs from context and writes its outputs back to context. Keys are plain strings, defined as constants in `ContextKeys` (see `src/config.py`) so nothing is hard-coded across files.

Variables can be **locked** after writing:

```python
context.lock(ContextKeys.CLF_MODEL)
```

A locked variable cannot be overwritten downstream, which prevents accidental mutation of a trained model or processed dataset later in the pipeline.

### `PipelineStep` and `@enforce`

Any callable with the signature `(context: Context) -> Context` qualifies as a pipeline step. All steps in `src/pipeline_steps/` are classes that implement `__call__`.

Steps declare their expectations with the `@enforce` decorator:

```python
@enforce({
    ContextKeys.DF_RECOVERY_PREPROCESSED: Requires(),
    ContextKeys.CLF_MODEL: Sequence(Defines(strict=True), Locks(strict=True)),
})
class TrainBinaryClassifier(PipelineStep):
    ...
```

This says: "before running, `DF_RECOVERY_PREPROCESSED` must exist in context; after running, `CLF_MODEL` must have been defined and locked." If either condition is violated, a `RuntimeError` is raised immediately, making misconfiguration easy to catch.

---

## Constructing a Pipeline (`main.py`)

`main.py` defines two functions:

- **`init_context(args)`** — creates the initial empty `Context`.
- **`construct_pipeline(args)`** — instantiates each step with its parameters and returns a `Pipeline`.

A full end-to-end run looks like:

```python
pipeline = Pipeline(
    Load(...),
    Preprocess(...),
    TrainBinaryClassifier(...),
    TrainRegressor(...),
    TrainPerChannelShareRegressors(...),
    Predict(...),
    Report(),
)

context = Context()
result = pipeline(context)
```

Each step is configured at construction time through its `__init__` parameters. The sections below describe what each step does and which parameters matter.

---

## Pipeline Steps

### 1. `Load` — `src/pipeline_steps/load_step.py`

Reads raw recovery data from disk into context.

**Writes:** `ContextKeys.DF_RECOVERY_LOADED`

| Parameter | Description |
|-----------|-------------|
| `source` | A single file path or a list of paths (CSV or parquet). Multiple files are concatenated. |

Skip this step if you are reading from a pre-saved preprocessed file instead (see `Preprocess.read_from`).

---

### 2. `Preprocess` — `src/pipeline_steps/preprocess_step.py`

Transforms raw row-level recovery records into the modelling-ready feature matrix. The sequence is:

1. Filter to recognised recovery types and consolidate categories.
2. Aggregate to site x GL product group x week.
3. Compute unit/cost/weight distribution features and GL composition shares.
4. Add ISO week and site-week context features.
5. Generate temporal features: lag, rolling mean/std, and EWMA over configurable windows.
6. Drop any columns not consumed by the models.

**Reads:** `ContextKeys.DF_RECOVERY_LOADED` (or `read_from` path directly)
**Writes:** `ContextKeys.DF_RECOVERY_PREPROCESSED`

| Parameter | Description |
|-----------|-------------|
| `read_from` | Path to a pre-saved preprocessed parquet file. If provided, skips all computation and loads directly. |
| `write_to` | Path to cache the output. Useful when preprocessing is slow and you want to reuse the result. |
| `n_lag_weeks` | Number of weekly lag features to generate (default defined in `config.py`). |

---

### 3. `TrainBinaryClassifier` — `src/pipeline_steps/train_binary_classifier_step.py`

**Stage 1 model.** Trains an `XGBClassifier` to predict whether any recovery occurs at all: *P(recovery_rate > 0)*. Class imbalance is handled automatically via `scale_pos_weight`. When `tune=True`, Optuna searches for hyperparameters using AUC-PR as the objective.

**Reads:** `ContextKeys.DF_RECOVERY_PREPROCESSED`
**Writes:** `ContextKeys.CLF_MODEL`

| Parameter | Description |
|-----------|-------------|
| `train_years` | List of years used for training (e.g. `[2022, 2023, 2024]`). |
| `test_years` | List of years used for validation during training. |
| `tune` | If `True`, run Optuna hyperparameter search instead of using defaults. |
| `n_trials` | Number of Optuna trials (only used when `tune=True`). |
| `load_from` | S3 path to a saved `.joblib` model. If provided, training is skipped entirely. |
| `save_to` | S3 path to save the trained model after fitting. |

---

### 4. `TrainRegressor` — `src/pipeline_steps/train_regressor_step.py`

**Stage 2 model.** Trains an `XGBRegressor` to predict the expected recovery rate on rows where recovery is non-zero: *E(rate | rate > 0)*. The model operates in logit space and outputs are back-transformed via sigmoid at inference. Sample weights emphasise the 10-60% rate band, which is the hardest to predict accurately.

Two extra artefacts are attached directly to the model object before saving:

- `model.site_gl_baseline_` — per-site x GL mean/std/count statistics computed from training non-zero rows. Used as additional features at inference.
- `model.baseline_priors_` — GL-level, site-level, and global fallback rates for site-GL pairs not seen during training.

This means downstream steps only need the model object; no training data is required at inference time.

**Reads:** `ContextKeys.DF_RECOVERY_PREPROCESSED`
**Writes:** `ContextKeys.REG_MODEL`

Same parameters as `TrainBinaryClassifier` (`train_years`, `test_years`, `tune`, `n_trials`, `load_from`, `save_to`).

---

### 5. `TrainPerChannelShareRegressors` — `src/pipeline_steps/train_per_channel_share_step.py`

**Stage 3 models.** Trains one `XGBRegressor` per recovery channel (Donations, Liquidations, Return to Vendor, Warehouse Deals/GR, Disposal), each predicting the channel's share of total recovery in logit space. The five models are stored together as a `dict[channel_name -> model]`.

At inference (`Predict`), share predictions are softmax-normalised across channels and multiplied by the overall combined rate to produce absolute per-channel predictions.

**Reads:** `ContextKeys.DF_RECOVERY_PREPROCESSED`
**Writes:** `ContextKeys.SHARE_MODELS`

| Parameter | Description |
|-----------|-------------|
| `train_years` / `test_years` | Same meaning as the other training steps. |
| `tune` / `n_trials` | Optuna tuning toggle and trial count. |
| `load_from` | `dict[channel_name -> S3Path]` — one path per channel. All five must be supplied together or not at all. |
| `save_to` | `dict[channel_name -> S3Path]` — one path per channel. |

---

### 6. `Predict` — `src/pipeline_steps/predict_step.py`

Runs all three model stages end-to-end on the preprocessed data and assembles the output DataFrame.

- **Stage 1** produces *p_nonzero*: probability that any recovery occurs.
- **Stage 2** produces *e_rate*: expected rate given recovery occurs.
- **Combined rate**: `p_nonzero x e_rate`.
- **Stage 3** produces per-channel absolute rates: normalised shares x combined rate.

The output DataFrame includes identifiers (site, GL, year, week), all stage predictions, ground-truth columns when available, absolute errors, rate/volume diagnostic buckets, and optional SHAP attribution columns.

**Reads:** `ContextKeys.DF_RECOVERY_PREPROCESSED`, `ContextKeys.CLF_MODEL`, `ContextKeys.REG_MODEL`, `ContextKeys.SHARE_MODELS`
**Writes:** `ContextKeys.PREDICTIONS`

| Parameter | Description |
|-----------|-------------|
| `years` | Restrict predictions to these years. Defaults to the most recent year in the data. |
| `sites` | Restrict to a list of site IDs (`hashed_fc`). `None` means all sites. |
| `gl_groups` | Restrict to a list of GL product groups. `None` means all groups. |
| `weeks` | Restrict to specific ISO week numbers. `None` means all weeks. |
| `predict_channels` | Whether to run Stage 3 and include per-channel predictions. |
| `run_shap` | Run SHAP decomposition on every row. Only set to `True` when the dataset is small — this is very slow on large inputs. |
| `save_to` | File path to write the predictions DataFrame. |

---

### 7. `Report` — `src/pipeline_steps/report_step.py`

Logs a concise performance summary to the terminal. Does not write to context or produce any files.

Output includes:
- Overall MAE, plus MAE broken down by rate bucket (zero / 0-10% / 10-30% / 30-60% / >60%) and by volume bucket.
- Top-N GL product groups ranked by mean predicted recovery rate.
- Top-N sites ranked by mean predicted recovery rate.

Ground-truth actuals are shown alongside predictions when they are present in the predictions DataFrame.

**Reads:** `ContextKeys.PREDICTIONS`

| Parameter | Description |
|-----------|-------------|
| `top_n` | How many GL groups and sites to surface in the highest-risk lists (default 5). |

---

## Configuration (`src/config.py`)

`config.py` is the single place to change data locations, feature sets, and model defaults. You should not need to edit individual step files for routine configuration changes.

### S3 paths

Constants like `DATA_S3_URI` and `MODEL_S3_URI` point to where data and trained models are stored on S3. Change these to redirect the pipeline to a different dataset or a different set of saved models.

### Feature lists

Three constants define exactly which columns are fed to each model stage:

- `RECOVERY_RATE_CLF_FEATURE_COLUMNS` — features for Stage 1 (classifier).
- `RECOVERY_RATE_REG_FEATURE_COLUMNS` — features for Stage 2 (regressor), including baseline columns.
- `PER_TYPE_REG_FEATURE_COLUMNS` — features for Stage 3 (per-channel share regressors).

Because all step files import these lists directly, editing a list in `config.py` automatically affects both training and inference. There is no duplication to keep in sync.

### Default hyperparameters

- `CLF_DEFAULT_PARAMS` — used by `TrainBinaryClassifier` when `tune=False`.
- `REG_DEFAULT_PARAMS` — used by `TrainRegressor` when `tune=False`.
- `PER_TYPE_REG_DEFAULT_PARAMS_DICT` — per-channel defaults used by `TrainPerChannelShareRegressors` when `tune=False`.

Edit these to lock in a known-good configuration without running an Optuna search.

### `ContextKeys`

String constants for every variable that passes through the pipeline (`DF_RECOVERY_LOADED`, `CLF_MODEL`, `PREDICTIONS`, etc.). All step files import from here so that key names are never duplicated as raw strings.

### Schema and category constants

`RecoverySchema` defines column names; `ConsolidatedRecoveryTypes` defines the five recovery channel names; normalisation dicts (`NORMALIZED_MACRO_CATEGORY_DICT`, etc.) map raw category labels to the standardised values used during preprocessing.

---

## Typical Workflows

**Full run from raw data, default hyperparameters:**

```python
Pipeline(
    Load(source="path/to/raw.parquet"),
    Preprocess(write_to="path/to/preprocessed.parquet"),
    TrainBinaryClassifier(train_years=[2022, 2023, 2024], test_years=[2025]),
    TrainRegressor(train_years=[2022, 2023, 2024], test_years=[2025]),
    TrainPerChannelShareRegressors(train_years=[2022, 2023, 2024], test_years=[2025]),
    Predict(years=[2025]),
    Report(),
)
```

**Load pre-saved models and run inference only:**

```python
Pipeline(
    Load(source="path/to/raw.parquet"),
    Preprocess(read_from="path/to/preprocessed.parquet"),
    TrainBinaryClassifier(load_from=S3Path("s3://bucket/clf.joblib")),
    TrainRegressor(load_from=S3Path("s3://bucket/reg.joblib")),
    TrainPerChannelShareRegressors(load_from={ch: S3Path(...) for ch in channels}),
    Predict(years=[2025], save_to="path/to/predictions.parquet"),
    Report(),
)
```

**Skip a step by passing `None`:**

```python
Pipeline(
    None,  # skip Load — Preprocess reads from read_from instead
    Preprocess(read_from="path/to/preprocessed.parquet"),
    ...
)
```
