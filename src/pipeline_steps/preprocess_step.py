# standard
import logging
from operator import eq, ne
from os import PathLike
from typing import Any, Callable

# third-party
import numpy as np
import polars as pl

# local
from src.config import (
    RecoverySchema,
    NORMALIZED_MACRO_CATEGORY_DICT, NORMALIZED_PRODUCT_TYPES_DICT,
    RECOVERY_TYPES_TO_KEEP,
    ConsolidatedRecoveryTypes, CONSOLIDATED_RECOVERY_TYPES, CONSOLIDATED_RECOVERY_TYPE_DICT,
    LAG_WEEKS, ROLLING_WEEKS, ROLLING_WEEKS_LONG, EWMA_ALPHAS,
    ContextKeys
)
from src.pipeline import Context, enforce
from src.pipeline.types import PipelineStep
from src.pipeline.conditions import Defines, Deletes, Locks, Sequence
from src.util import load_dataframe, write_dataframe


logger = logging.getLogger(__name__)


# ============================================================
# PIPELINE STEP
# ============================================================

@enforce({
    # ContextKeys.DF_RECOVERY_LOADED: Requires(),           # not needed if loading from saved preprocessed data
    ContextKeys.DF_RECOVERY_LOADED: Deletes(strict=False),  # delete if exists to free up memory
    ContextKeys.DF_RECOVERY_PREPROCESSED: Sequence(Defines(strict=True), Locks(strict=True))
})
class Preprocess(PipelineStep):
    """
    Defines the preprocessed recovery DataFrame, either by loading previously saved preprocessed
    data or by processing the raw recovery data passed by the Load step. Optionally writes
    preprocessed data to an output parquet file.
    """

    def __init__(
            self,
            load_from: str | PathLike[str] | None = None,  # 
            save_to: str | PathLike[str] | None = None
        ):
        if load_from is not None and save_to is not None:
            logger.warning(f"both loading and saving preprocessed data (is this intentional?): '{load_from}' -> '{save_to}'")
        self.load_from = load_from
        self.save_to = save_to
    
    def __call__(self, context: Context) -> Context:
        if self.load_from is not None:  # read saved preprocessed data if provided
            logger.info(f"loading preprocessed data from '{self.load_from}'")
            df_recovery = load_dataframe(self.load_from)
        else:  # otherwise process the raw data from the Load step
            logger.info("preprocessing data from Load step")
            df_recovery = (
                context[ContextKeys.DF_RECOVERY_LOADED]
                .pipe(_pre_cleaning)
                .pipe(
                    _aggregation,
                    groupby=[
                        RecoverySchema.HASHED_FC, RecoverySchema.YEAR, RecoverySchema.MONTH,
                        RecoverySchema.WEEK, RecoverySchema.GL_PRODUCT_GROUP
                    ]
                )
                .pipe(_unit_distribution_features)      # X% of units are Y
                .pipe(_recovery_distribution_features)  # X% of recovered units are recovery_type Y
                .pipe(_iso_week)
                .pipe(
                    _site_week_features,
                    groupby=[RecoverySchema.HASHED_FC, "week_date"])
                .pipe(                                  # e.g., X% of site-week units are GL Y
                    _gl_share_features,
                )
                .pipe(_other_non_temporal_features)
                .pipe(_round_decimal_columns, decimals=6)  # TODO: is this necessary to keep?
                .pipe(
                    _temporal_features,
                    groupby = [RecoverySchema.HASHED_FC, RecoverySchema.GL_PRODUCT_GROUP],
                    lag_weeks = LAG_WEEKS,
                    rolling_weeks = ROLLING_WEEKS,
                    rolling_weeks_long = ROLLING_WEEKS_LONG,
                    ewma_alphas = EWMA_ALPHAS
                )
                .pipe(_post_cleaning)
            )
        
        if ContextKeys.DF_RECOVERY_LOADED in context:  # clean up raw loaded data if exists
            context.unlock(ContextKeys.DF_RECOVERY_LOADED, strict=False)
            del context[ContextKeys.DF_RECOVERY_LOADED]
        
        if self.save_to is not None:  # write preprocessed data
            logger.info(f"saving preprocessed data to '{self.save_to}'")
            write_dataframe(df_recovery, self.save_to)
        
        context[ContextKeys.DF_RECOVERY_PREPROCESSED] = df_recovery
        context.lock(ContextKeys.DF_RECOVERY_PREPROCESSED)

        return context


# ============================================================
# (PRIVATE) HELPER FUNCTIONS
# ============================================================

def _pre_cleaning(df: pl.DataFrame) -> pl.DataFrame:
    # drop columns with all null values
    all_null_columns = (
        df
        .select(pl.all().is_null().all())
        .unpivot()
        .filter(pl.col("value") == True)
        .select("variable")
        .to_series()
        .to_list()
    )
    df = df.drop(all_null_columns)

    # filter out C-Returns and Bintools Theft
    df = df.filter(pl.col(RecoverySchema.RECOVERY_TYPE).isin(RECOVERY_TYPES_TO_KEEP))

    # consolidate recovery types
    df = df.with_columns(
        pl.col(RecoverySchema.RECOVERY_TYPE).replace(CONSOLIDATED_RECOVERY_TYPE_DICT)
    )

    # drop marketplace_id
    df = df.drop([RecoverySchema.MARKETPLACE_ID])  # marketplace_id redundant (same as country)

    # drop missing gl_product_group
    df = (
        df
        .filter(pl.col(RecoverySchema.GL_PRODUCT_GROUP).is_not_null())
        .filter(pl.col(RecoverySchema.GL_PRODUCT_GROUP) != -1)
    )

    # create target variable
    df = df.with_columns(
        pl.when(pl.col(RecoverySchema.RECOVERY_TYPE) == ConsolidatedRecoveryTypes.SALES)
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias("recovery")
    )

    # reorder/select
    df = df.select([
        RecoverySchema.HASHED_FC,
        RecoverySchema.YEAR,
        RecoverySchema.MONTH,
        RecoverySchema.WEEK,
        RecoverySchema.GL_PRODUCT_GROUP,
        RecoverySchema.PRODUCT_TYPE,
        RecoverySchema.MACRO_CATEGORY,
        RecoverySchema.ITEM_DISPOSITION_CODE,
        RecoverySchema.APPLICATION_NAME,
        RecoverySchema.IS_STRANDED,
        RecoverySchema.REASON_CODE,
        RecoverySchema.REASON_CODE_STRANDED,
        RecoverySchema.STRANDED_POTENTIAL_ISSUE,
        RecoverySchema.IS_HAZMAT,
        RecoverySchema.UNITS,
        RecoverySchema.COGS,
        RecoverySchema.WEIGHT,
        RecoverySchema.COUNTRY,
        RecoverySchema.COUNTRY_STATE,
        RecoverySchema.ZIP_CODE,
        RecoverySchema.SITE_TYPE,
        RecoverySchema.SITE_CATEGORY,
        "recovery",
        RecoverySchema.RECOVERY_TYPE
    ])

    return df
    
def _aggregation(df: pl.DataFrame, groupby: str | list[str]) -> pl.DataFrame:
    return (
        df
        .group_by(groupby)
        .agg([
            pl.len().alias("num_records"),

            pl.col(RecoverySchema.UNITS).sum().alias(f"units_total"),
            pl.col(RecoverySchema.COGS).sum().alias("cogs_total"),
            pl.col(RecoverySchema.WEIGHT).sum().alias("weight_total"),

            # site characteristics (take the first value since they should be the same across the group)
            *[
                pl.first(field)
                for field in (
                    RecoverySchema.COUNTRY, RecoverySchema.COUNTRY_STATE, RecoverySchema.ZIP_CODE,
                    RecoverySchema.SITE_TYPE, RecoverySchema.SITE_CATEGORY
                )
            ],

            # unit counts
            *_conditional_sums(  # units_retail, units_fba
                filter_col=RecoverySchema.MACRO_CATEGORY,
                filter_value_to_str=NORMALIZED_MACRO_CATEGORY_DICT,
                sum_col=RecoverySchema.UNITS,
                alias_prefix="units"
            ).values(),
            *_conditional_sums(  # units_food, units_non_food, units_pet_food
                filter_col=RecoverySchema.PRODUCT_TYPE,
                filter_value_to_str=NORMALIZED_PRODUCT_TYPES_DICT,
                sum_col=RecoverySchema.UNITS,
                alias_prefix="units"
            ).values(),
            _conditional_sum(  # units_recovered (i.e., units not sold)
                filter_col=RecoverySchema.RECOVERY_TYPE,
                filter_value=ConsolidatedRecoveryTypes.SALES,
                sum_col="units",
                alias="units_recovered",
                comparator=ne
            ),
            *_conditional_sums(  # units_sales, units_return_to_vendor, ...
                filter_col=RecoverySchema.RECOVERY_TYPE,
                filter_value_to_str={t: t for t in CONSOLIDATED_RECOVERY_TYPES},  # we use consolidated names after pre-cleaning
                sum_col=RecoverySchema.UNITS,
                alias_prefix="units"
            ).values(),
            _conditional_sum(RecoverySchema.IS_HAZMAT, "Y", RecoverySchema.UNITS, f"units_hazmat"),
        ])
    )

def _unit_distribution_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        # macro category: share_retail, share_fba
        # product type: share_food, share_non_food, share_pet_food
        # hazmat: share_hazmat
        *[
            _safe_division(f"units_{c}", "units_total", f"share_{c}")
            for c in list(NORMALIZED_MACRO_CATEGORY_DICT.values()) +
                     list(NORMALIZED_PRODUCT_TYPES_DICT.values()) +
                     ["hazmat"]
        ],

        # recovered units
        *[  # prob_sales, prob_return_to_vendor, prob_warehouse_deals_and_g&r, prob_liquidations, prob_donations, prob_disposal
            _safe_division(f"units_{consolidated_recovery_type}", "units_total", f"prob_{consolidated_recovery_type}")
            for consolidated_recovery_type in CONSOLIDATED_RECOVERY_TYPES
        ],
        _safe_division("units_recovered", "units_total", "prob_recovered"),
    ])

def _recovery_distribution_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        *[  # share_return_to_vendor, share_warehouse_deals_and_g, share_donations, share_liquidations, share_disposal
            # (e.g., share_return_to_vendor = P(Return to Vendor | ~Sale))
            _safe_division(f"units_{consolidated_recovery_type}", "units_recovered", f"share_{consolidated_recovery_type}")
            for consolidated_recovery_type in CONSOLIDATED_RECOVERY_TYPES
        ]
    ])

def _iso_week(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.date(pl.col(RecoverySchema.YEAR), 1, 4)  # first ISO week always contains Jan 4th
        .dt.truncate("1w")                          # truncate to the Monday of that week
        .dt.offset_by(                              # week w = week 1 shifted by (w-1) weeks
            (pl.col(RecoverySchema.WEEK).cast(pl.Int32) - 1).cast(pl.Utf8) + "w"
        )
        .alias("week_date")
    )

def _site_week_features(df: pl.DataFrame, groupby: list[str]) -> pl.DataFrame:
    return df.with_columns([
        *[
            pl.sum(col).over(groupby).alias(f"site_{col}_week")
            for col in ["units_total", "weight_total", "units_recovered"] + [
                f"units_{consolidated_recovery_type}"
                for consolidated_recovery_type in CONSOLIDATED_RECOVERY_TYPES
            ]
        ]
    ])

def _gl_share_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (pl.col("units_total") / pl.col("site_units_total_week")).alias("site_units_share_week"),
        (pl.col("weight_total") / pl.col("site_weight_total_week")).alias("site_weight_share_week"),
        (pl.col("units_recovered") / pl.col("site_units_recovered_week")).fill_nan(0).alias("site_recovered_share_week"),

        # Overall recovery probability at site level
        (pl.col("site_units_recovered_week") / pl.col("site_units_total_week")).alias("site_prob_recovered_week"),

        # Per consolidated channel probability at site level
        *[
            (
                pl.col(f"site_units_{consolidated_recovery_type}_week") / pl.col("site_units_total_week")
            ).alias(f"site_prob_{consolidated_recovery_type}_week")
            for consolidated_recovery_type in CONSOLIDATED_RECOVERY_TYPES
        ]
    ])

def _other_non_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        _safe_division("cogs_total", "units_total", "avg_cogs_per_unit"),
        _safe_division("weight_total", "units_total", "avg_weight_per_unit"),
        _safe_division("cogs_total", "weight_total", "cogs_per_unit_weight")
    ])

def _round_decimal_columns(df: pl.DataFrame, decimals: int) -> pl.DataFrame:
    return df.with_columns(
        pl.col(pl.Float32, pl.Float64).round(decimals)
    )

def _temporal_features(
    df: pl.DataFrame,
    groupby: list[str],
    lag_weeks: list[int],
    rolling_weeks: list[int],
    rolling_weeks_long: list[int],
    ewma_alphas: list[float]
) -> pl.DataFrame:
    # === construct week grid ====
    full_weeks = (
        df
        .group_by(groupby)
        .agg([
            pl.min("week_date").alias("start"),
            pl.max("week_date").alias("end"),
        ])
        .with_columns(
            pl.date_ranges("start", "end", interval="1w").alias("week_date")
        )
        .explode("week_date")
    )

    df_full_weeks = (
        full_weeks
        .join(df, on=groupby + ["week_date"], how="left")
        .sort(groupby + ["week_date"])
    )

    # === lag, rolling, and EWMA features of numeric columns ===
    exclude_cols = {
        RecoverySchema.HASHED_FC, RecoverySchema.GL_PRODUCT_GROUP,
        "start", "end", "week_date",
        RecoverySchema.YEAR, RecoverySchema.MONTH, RecoverySchema.WEEK,
        RecoverySchema.COUNTRY, RecoverySchema.COUNTRY_STATE, RecoverySchema.ZIP_CODE,
        RecoverySchema.SITE_TYPE, RecoverySchema.SITE_CATEGORY, 
        "num_records"
    }
    feature_cols = set(_numeric_columns(df))-exclude_cols

    # == lag features ==
    lag_exprs = [
        pl.col(col).shift(n).over(groupby).alias(f"{col}_lag_{n}w")
        for col in feature_cols
        for n in lag_weeks
    ]

    # == Short rolling (require at least half the window) ==
    rolling_exprs = [
        pl.col(col).shift(1).rolling_mean(window_size=n, min_samples=max(1, n // 2)).over(groupby).alias(f"{col}_rolling_{n}w")
        for col in feature_cols
        for n in rolling_weeks
    ]

    # == Long rolling (lenient, min 4 periods) ==
    rolling_exprs_long = [
        pl.col(col).shift(1).rolling_mean(window_size=n, min_samples=4).over(groupby).alias(f"{col}_rolling_{n}w")
        for col in feature_cols
        for n in rolling_weeks_long
    ]

    # == EWMA (fast: alpha=0.5, slow: alpha=0.1) ==
    ewma_exprs = [
        pl.col(col).shift(1).ewm_mean(alpha=alpha, min_periods=4).over(groupby).alias(f"{col}_ewma_{alpha}")
        for col in feature_cols
        for alpha in ewma_alphas
    ]

    df_full_weeks  = (  # apply in batches to avoid memory pressure
        df_full_weeks
        .with_columns(lag_exprs)
        .with_columns(rolling_exprs)
        .with_columns(rolling_exprs_long)
        .with_columns(ewma_exprs)
    )

    # === drop buffer weeks ===
    df_full_weeks = df_full_weeks.drop_nulls(subset=["week"])

    # === sin and cos week transformation ===
    period = 365.25/52
    df_full_weeks = df_full_weeks.with_columns(
        (np.sin(2 * np.pi * pl.col(RecoverySchema.WEEK) / period)).alias("week_sin"),
        (np.cos(2 * np.pi * pl.col(RecoverySchema.WEEK) / period)).alias("week_cos")
    )

    return df_full_weeks

def _post_cleaning(df: pl.DataFrame) -> pl.DataFrame:
    # TODO: select only the features used in the models
    ...


# e.g., SUM(sum_col WHERE filter_col = filter_value)
def _conditional_sum(
    filter_col: str,
    filter_value: Any,
    sum_col: str,
    alias: str,
    comparator: Callable[[pl.Expr, Any], pl.Expr] = eq
) -> pl.Expr:
    return (
        pl.when(comparator(pl.col(filter_col), filter_value))
        .then(pl.col(sum_col))
        .otherwise(0)
        .sum()
        .alias(alias)
    )

# e.g., units_retail = SUM(units WHERE MACRO_CATEGORY = "RETAIL"), units_fba = SUM(units WHERE MACRO_CATEGORY = "FBA"), ...
def _conditional_sums(
      filter_col: str,
      filter_value_to_str: dict[Any, str],  # value -> string representation for column
      sum_col: str,
      alias_prefix: str
) -> dict[str, pl.Expr]:
    return {
        f"{alias_prefix}_{filter_value_string}": _conditional_sum(
            filter_col,
            filter_value,
            sum_col,
            alias = f"{alias_prefix}_{filter_value_string}"
        )
        for filter_value, filter_value_string in filter_value_to_str.items()
    }

def _safe_division(numerator_col: str, denominator_col: str, alias: str) -> pl.Expr:
    numerator = pl.col(numerator_col)
    denominator = pl.col(denominator_col)
    safe_denominator = pl.when(denominator != 0).then(denominator).otherwise(None)
    return (numerator/safe_denominator).alias(alias)

def _numeric_columns(df: pl.DataFrame) -> list[str]:
    return  [
        col for col in df.columns
        if df[col].dtype in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        )
    ]
