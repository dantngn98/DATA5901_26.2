# standard
from operator import eq, ne
from typing import Any, Callable

# third-party
import numpy as np
import polars as pl

# local
from src.config import ContextKeys
from src.pipeline import Context, enforce
from src.pipeline.types import PipelineStep
from src.pipeline.conditions import Defines, Locks, Sequence
from src.util import load

@enforce({
    # ContextKeys.DF_RECOVERY_LOADED: Requires,  # not needed if loading saved preprocessed data
    ContextKeys.DF_RECOVERY_PREPROCESSED: Sequence(Defines(strict=True), Locks(strict=True))
})
class Preprocess(PipelineStep):
    """
    Read raw recovery data from csv files, perform data cleaning,
    group by hashed_fc-GL-week, and construct features.
    Return the preprocessed DataFrame in context and optionally write to parquet for reuse in training and prediction steps.
    """
    def __init__(self, read_from: str | None = None, write_to: str | None = None):
        self.read_from = read_from
        self.write_to = write_to
    
    def __call__(self, context: Context) -> Context:
        if self.read_from is not None:
            df_recovery = load(self.read_from)
        else:
            df_recovery = (
                context[ContextKeys.DF_RECOVERY_LOADED]
                .pipe(_pre_cleaning)
                .pipe(_aggregation, groupby=["hashed_fc", "year", "month", "week", "gl_product_group"])
                .pipe(_unit_distribution_features)      # X% of units are Y
                .pipe(_recovery_distribution_features)  # X% of recovered units are recovery_type Y
                .pipe(_iso_week)
                .pipe(_site_week_features)
                .pipe(_gl_share_features)               # e.g., X% of site-week units are GL Y
                .pipe(_other_non_temporal_features)
                .pipe(_round_decimal_columns, decimals=6)
                .pipe(
                    _temporal_features,
                    lag_weeks = [1, 4, 12, 13, 52],
                    rolling_weeks = [4, 12],
                    rolling_weeks_long = [26, 52],
                    ewma_alphas = [0.5, 0.1] 
                )
            )

        if self.write_to is not None and self.write_to != self.read_from:  # ! string equality != path equality
            df_recovery.write_parquet(self.write_to)
        
        context[ContextKeys.DF_RECOVERY_PREPROCESSED] = df_recovery
        context.lock(ContextKeys.DF_RECOVERY_PREPROCESSED)

        return context


def _pre_cleaning(df: pl.DataFrame) -> pl.DataFrame:
    # drop columns with all null values
    all_null_columns = (
        df.select(pl.all().is_null().all())
        .unpivot()
        .filter(pl.col("value") == True)
        .select("variable")
        .to_series()
        .to_list()
    )
    df = df.drop(all_null_columns)

    # filter out C-Returns
    df = df.filter(pl.col("recovery_type") != "C-Returns")

    # drop marketplace_id
    df = df.drop(["marketplace_id"])  # marketplace_id redundant (same as country)

    # drop missing gl_product_group
    df = (
        df
        .filter(pl.col("gl_product_group").is_not_null())
        .filter(pl.col("gl_product_group") != -1)
    )

    # create target variable
    df = df.with_columns(
        pl.when(pl.col("recovery_type") == "Sales").then(pl.lit(0))
        .otherwise(pl.lit(1)).alias("recovery")
    )

    # reorder/select
    df = df.select([
        "hashed_fc",
        "year",
        "month",
        "week",
        "gl_product_group",
        "product_type",
        "macro_category",
        "item_disposition_code",
        "reason_code",
        "application_name",
        "is_stranded",
        "reason_code_type",
        "reason_code_stranded",
        "stranded_potential_issue",
        "is_hazmat",
        "units",
        "cogs",
        "weight",
        "country",
        "country_state",
        "zip_code",
        "site_type",
        "site_category",
        "recovery",
        "recovery_type"
    ])

    return df
    
def _aggregation(df: pl.DataFrame, groupby: list[str]) -> pl.DataFrame:
    return (
        df
        .group_by(groupby)
        .agg([
            pl.len().alias("num_records"),

            pl.col("units").sum().alias("units_total"),
            pl.col("cogs").sum().alias("cogs_total"),
            pl.col("weight").sum().alias("weight_total"),

            # site characteristics (take the first value since they should be the same across the group)
            pl.first("country"),
            pl.first("country_state"),
            pl.first("zip_code"),
            pl.first("site_type"),
            pl.first("site_category"),


            # unit counts
            _conditional_sum("macro_category", "RETAIL", "units", "units_RETAIL"),  # macro_category = "RETAIL"
            _conditional_sum("macro_category", "FBA", "units", "units_FBA"),        # macro_category = "FBA"

            _conditional_sum("is_hazmat", "Y", "units", "units_hazmat"),  # is_hazmat = "Y"

            _conditional_sum("product_type", "Food", "units", "units_food"),          # product_type = "Food"
            _conditional_sum("product_type", "Non Food", "units", "units_non_food"),  # product_type = "Non Food"
            _conditional_sum("product_type", "Pet Food", "units", "units_pet_food"),  # product_type = "Pet Food"

            _conditional_sum("recovery_type", "Sales", "units", "units_sales"),                                        # recovery_type = "Sales"
            _conditional_sum("recovery_type", "Sales", "units", "units_recovered", comparator=ne),                    # recovery_type != "Sales"
            _conditional_sum("recovery_type", "Return to Vendor", "units", "units_return_to_vendor"),                  # recovery_type = "Return to Vendor"
            _conditional_sum("recovery_type", "Remove Return", "units", "units_remove_return"),                        # recovery_type = "Remove Return"
            _conditional_sum("recovery_type", "Warehouse Deals and G&R", "units", "units_warehouse_deals_and_gr"),     # recovery_type = "Warehouse Deals and G&R"
            _conditional_sum("recovery_type", "Donations", "units", "units_donations"),                                # recovery_type = "Donations"
            _conditional_sum("recovery_type", "Bintool Donations", "units", "units_bintool_donations"),                # recovery_type = "Bintool Donations"
            _conditional_sum("recovery_type", "Liquidations", "units", "units_liquidations"),                          # recovery_type = "Liquidations"
            _conditional_sum("recovery_type", "Bintool Remove Liquidate", "units", "units_bintool_remove_liquidate"),  # recovery_type = "Bintool Remove Liquidate"
            _conditional_sum("recovery_type", "Remove Liquidate", "units", "units_remove_liquidate"),                  # recovery_type = "Remove Liquidate"
            _conditional_sum("recovery_type", "Bintool Theft", "units", "units_bintool_theft")                         # recovery_type = "Bintool Theft"
        ])
    )

def _unit_distribution_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        # macro category
        _safe_division("units_RETAIL", "units_total", "share_RETAIL"),
        _safe_division("units_FBA", "units_total", "share_FBA"),

        # hazmat
        _safe_division("units_hazmat", "units_total", "share_hazmat"),

        # product type
        _safe_division("units_food", "units_total", "share_food"),
        _safe_division("units_non_food", "units_total", "share_non_food"),
        _safe_division("units_pet_food", "units_total", "share_pet_food"),

        # recovered units
        _safe_division("units_sales", "units_total", "prob_sales"),                                        # P(Sales)
        _safe_division("units_recovered", "units_total", "prob_recovered"),                                # P(~Sales) = 1-P(Sales) = SUM of below
        _safe_division("units_return_to_vendor", "units_total", "prob_return_to_vendor"),                  # P(Return to Vendor)
        _safe_division("units_warehouse_deals_and_gr", "units_total", "prob_warehouse_deals_and_gr"),      # P(Warehouse Deals and G&R)
        _safe_division("units_remove_return", "units_total", "prob_remove_return"),                        # P(Remove Return)
        _safe_division("units_donations", "units_total", "prob_donations"),                                # P(Donations)
        _safe_division("units_bintool_donations", "units_total", "prob_bintool_donations"),                # P(Bintool Donations)
        _safe_division("units_liquidations", "units_total", "prob_liquidations"),                          # P(Liquidations)
        _safe_division("units_bintool_remove_liquidate", "units_total", "prob_bintool_remove_liquidate"),  # P(Bintool Remove Liquidate)
        _safe_division("units_remove_liquidate", "units_total", "prob_remove_liquidate"),                  # P(Remove Liquidate)
        _safe_division("units_bintool_theft", "units_total", "prob_bintool_theft")                         # P(Bintool Theft)
    ])

def _recovery_distribution_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        _safe_division("units_sales", "units_recovered", "share_sales"),                                        # |Sales|/|~Sales|
        _safe_division("units_return_to_vendor", "units_recovered", "_share_return_to_vendor"),                 # P(Return to Vendor | ~Sale)
        _safe_division("units_warehouse_deals_and_gr", "units_recovered", "share_warehouse_deals_and_gr"),      # P(Warehouse Deals and G&R | ~Sale)
        _safe_division("units_remove_return", "units_recovered", "share_remove_return"),                        # P(Remove Return | ~Sale)
        _safe_division("units_donations", "units_recovered", "share_donations"),                                # P(Donations | ~Sale)
        _safe_division("units_bintool_donations", "units_recovered", "share_bintool_donations"),                # P(Bintool Donations | ~Sale)
        _safe_division("units_liquidations", "units_recovered", "share_liquidations"),                          # P(Liquidations | ~Sale)
        _safe_division("units_bintool_remove_liquidate", "units_recovered", "share_bintool_remove_liquidate"),  # P(Bintool Remove Liquidate | ~Sale)
        _safe_division("units_remove_liquidate", "units_recovered", "share_remove_liquidate"),                  # P(Remove Liquidate | ~Sale)
        _safe_division("units_bintool_theft", "units_recovered", "share_bintool_theft")                         # P(Bintool Theft | ~Sale)
    ])

def _iso_week(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.date(pl.col("year"), 1, 4)
        .dt.truncate("1w")
        .dt.offset_by(
            f"{(pl.col("week").cast(pl.Int32) - 1).cast(pl.Utf8)}w"
        )
        .alias("week_date")
    )

def _site_week_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.sum("units_total").over(["hashed_fc","week_date"]).alias("site_units_total_week"),
        pl.sum("weight_total").over(["hashed_fc","week_date"]).alias("site_weight_total_week"),
        pl.sum("units_recovered").over(["hashed_fc","week_date"]).alias("site_units_recovered_week"),

        # Recovery type site totals
        pl.sum("units_remove_return").over(["hashed_fc","week_date"]).alias("site_units_remove_return_week"),
        pl.sum("units_bintool_donations").over(["hashed_fc","week_date"]).alias("site_units_bintool_donations_week"),
        pl.sum("units_donations").over(["hashed_fc","week_date"]).alias("site_units_donations_week"),
        pl.sum("units_warehouse_deals_and_gr").over(["hashed_fc","week_date"]).alias("site_units_warehouse_deals_and_gr_week"),
        pl.sum("units_liquidations").over(["hashed_fc","week_date"]).alias("site_units_liquidations_week"),
        pl.sum("units_sales").over(["hashed_fc","week_date"]).alias("site_units_sales_week"),
        pl.sum("units_return_to_vendor").over(["hashed_fc","week_date"]).alias("site_units_return_to_vendor_week"),
        pl.sum("units_bintool_theft").over(["hashed_fc","week_date"]).alias("site_units_bintool_theft_week"),
        pl.sum("units_remove_liquidate").over(["hashed_fc","week_date"]).alias("site_units_remove_liquidate_week"),
        pl.sum("units_bintool_remove_liquidate").over(["hashed_fc","week_date"]).alias("site_units_bintool_remove_liquidate_week")
    ])

def _gl_share_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (pl.col("units_total") / pl.col("site_units_total_week")).alias("site_units_share_week"),
        (pl.col("weight_total") / pl.col("site_weight_total_week")).alias("site_weight_share_week"),
        (pl.col("units_recovered") / pl.col("site_units_recovered_week")).fill_nan(0).alias("site_recovered_share_week"),

        # Overall recovery probability at site level
        (pl.col("site_units_recovered_week") / pl.col("site_units_total_week")).alias("site_prob_recovered_week"),

        # Per recovery outcome probability at site level
        (pl.col("site_units_remove_return_week") / pl.col("site_units_total_week")).alias("site_prob_remove_return_week"),
        (pl.col("site_units_bintool_donations_week") / pl.col("site_units_total_week")).alias("site_prob_bintool_donations_week"),
        (pl.col("site_units_donations_week") / pl.col("site_units_total_week")).alias("site_prob_donations_week"),
        (pl.col("site_units_warehouse_deals_and_gr_week") / pl.col("site_units_total_week")).alias("site_prob_warehouse_deals_and_gr_week"),
        (pl.col("site_units_liquidations_week") / pl.col("site_units_total_week")).alias("site_prob_liquidations_week"),
        (pl.col("site_units_sales_week") / pl.col("site_units_total_week")).alias("site_prob_sales_week"),
        (pl.col("site_units_return_to_vendor_week") / pl.col("site_units_total_week")).alias("site_prob_return_to_vendor_week"),
        (pl.col("site_units_bintool_theft_week") / pl.col("site_units_total_week")).alias("site_prob_bintool_theft_week"),
        (pl.col("site_units_remove_liquidate_week") / pl.col("site_units_total_week")).alias("site_prob_remove_liquidate_week"),
        (pl.col("site_units_bintool_remove_liquidate_week") / pl.col("site_units_total_week")).alias("site_prob_bintool_remove_liquidate_week"),
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
    lag_weeks: list[int],
    rolling_weeks: list[int],
    rolling_weeks_long: list[int],
    ewma_alphas: list[float]
) -> pl.DataFrame:
    groupby = ["hashed_fc", "gl_product_group"]

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
        "hashed_fc", "gl_product_group",
        "start", "end", "week_date",
        "year", "month", "week",
        "country", "country_state", "zip_code",
        "site_type", "site_category", "num_records"
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
        pl.col(col).shift(1).ewm_mean(alpha=alpha, min_periods=4).over(groupby).alias(f"{col}_ewma_{str(alpha).replace('0.', '')}a")
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
        (np.sin(2 * np.pi * pl.col("week") / period)).alias("week_sin"),
        (np.cos(2 * np.pi * pl.col("week") / period)).alias("week_cos")
    )

    return df_full_weeks


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

def _safe_division(numerator: pl.Expr, denominator: pl.Expr, alias: str) -> pl.Expr:
    denominator_ = pl.when(denominator != 0).then(denominator).otherwise(None)
    return (numerator/denominator_).alias(alias)

def _numeric_columns(df: pl.DataFrame) -> list[str]:
    return  [
        col for col in df.columns
        if df[col].dtype in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        )
    ]
