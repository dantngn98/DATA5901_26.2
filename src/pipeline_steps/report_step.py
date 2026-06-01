# standard
import logging

# third-party
import pandas as pd

# local
from src.config import ContextKeys, RecoverySchema
from src.pipeline import Context, enforce
from src.pipeline.conditions import Requires
from src.pipeline.types import PipelineStep


logger = logging.getLogger(__name__)

_RATE_BUCKET_ORDER   = ["zero", "0-10%", "10-30%", "30-60%", ">60%"]
_VOLUME_BUCKET_ORDER = ["<10", "10-100", "100-1k", ">1k"]


# ============================================================
# Pipeline step
# ============================================================

@enforce({
    ContextKeys.PREDICTIONS: Requires(),
})
class Report(PipelineStep):
    """Reads ContextKeys.PREDICTIONS and logs a terminal summary.

    Outputs:
      - Overall model performance (MAE, by rate bucket, by volume bucket)
      - Highest-risk GL product groups by mean predicted recovery rate
      - Highest-risk sites by mean predicted recovery rate
    """

    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    def __call__(self, context: Context) -> Context:
        df = context[ContextKeys.PREDICTIONS].to_pandas()
        logger.info("Building report from %d prediction rows", len(df))

        mae = _mae_summary(df)
        _log_performance(mae)

        gl_tbl   = _top_n_by_rate(df, RecoverySchema.GL_PRODUCT_GROUP, self.top_n)
        site_tbl = _top_n_by_rate(df, RecoverySchema.HASHED_FC, self.top_n)
        _log_highest_risk(gl_tbl, site_tbl)

        return context


# ============================================================
# Performance summary
# ============================================================

def _mae_summary(df: pd.DataFrame) -> dict:
    if "abs_err" not in df.columns:
        return {}

    overall = df["abs_err"].mean()

    def _bucket_table(df: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
        grp = (
            df.groupby(col)["abs_err"]
            .agg(n_rows="count", mean_abs_err="mean", median_abs_err="median")
            .reset_index()
        )
        grp[col] = pd.Categorical(grp[col], categories=order, ordered=True)
        return grp.sort_values(col).reset_index(drop=True)

    return {
        "overall": overall,
        "by_rate_bucket":   _bucket_table(df, "rate_bucket",   _RATE_BUCKET_ORDER),
        "by_volume_bucket": _bucket_table(df, "volume_bucket", _VOLUME_BUCKET_ORDER),
    }


def _log_performance(summary: dict) -> None:
    if not summary:
        return

    lines = [
        "",
        "=== Model Performance ===",
        f"Overall MAE: {summary['overall']:.4f}",
        "",
        "MAE by Rate Bucket:",
    ]
    for _, row in summary["by_rate_bucket"].iterrows():
        lines.append(
            f"  {row['rate_bucket']:<8} | n={int(row['n_rows']):>5} "
            f"| mean={row['mean_abs_err']:.4f} | median={row['median_abs_err']:.4f}"
        )
    lines += ["", "MAE by Volume Bucket:"]
    for _, row in summary["by_volume_bucket"].iterrows():
        lines.append(
            f"  {row['volume_bucket']:<8} | n={int(row['n_rows']):>5} "
            f"| mean={row['mean_abs_err']:.4f} | median={row['median_abs_err']:.4f}"
        )
    logger.info("\n".join(lines))


# ============================================================
# Highest-risk groups
# ============================================================

def _top_n_by_rate(df: pd.DataFrame, group_col: str, n: int) -> pd.DataFrame:
    agg: dict[str, tuple] = {"combined_rate": ("combined_rate", "mean")}
    if "prob_recovered" in df.columns:
        agg["prob_recovered"] = ("prob_recovered", "mean")
    result = (
        df.groupby(group_col)
        .agg(**agg)
        .reset_index()
        .sort_values("combined_rate", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    result.rename(columns={"combined_rate": "mean_pred_rate"}, inplace=True)
    if "prob_recovered" in result.columns:
        result.rename(columns={"prob_recovered": "mean_actual_rate"}, inplace=True)
    return result


def _log_highest_risk(gl_tbl: pd.DataFrame, site_tbl: pd.DataFrame) -> None:
    has_actual_gl   = "mean_actual_rate" in gl_tbl.columns
    has_actual_site = "mean_actual_rate" in site_tbl.columns

    def _format_row(rank: int, name: str, pred: float, actual: float | None) -> str:
        actual_part = f"  (actual={actual:.4f})" if actual is not None else ""
        return f"  {rank}. {name:<30}  mean_pred={pred:.4f}{actual_part}"

    lines = ["", "=== Highest-Risk GL Product Groups ==="]
    for i, row in gl_tbl.iterrows():
        actual = row["mean_actual_rate"] if has_actual_gl else None
        lines.append(_format_row(i + 1, row[RecoverySchema.GL_PRODUCT_GROUP], row["mean_pred_rate"], actual))

    lines += ["", "=== Highest-Risk Sites ==="]
    for i, row in site_tbl.iterrows():
        actual = row["mean_actual_rate"] if has_actual_site else None
        lines.append(_format_row(i + 1, row[RecoverySchema.HASHED_FC], row["mean_pred_rate"], actual))

    logger.info("\n".join(lines))
