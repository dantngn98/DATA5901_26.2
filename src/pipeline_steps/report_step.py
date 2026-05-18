# standard
import base64
import io
import logging

# third-party
import boto3
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# local
from src.config import ContextKeys, REPORT_S3_KEY, S3_BUCKET
from src.pipeline import Context, enforce
from src.pipeline.conditions import Defines, Locks, Requires, Sequence
from src.pipeline.types import PipelineStep
from src.pipeline_steps.train_per_channel_share_step import RECOVERY_CHANNELS

logger = logging.getLogger(__name__)

_CHANNEL_LABELS: dict[str, str] = {
    "prob_donations":               "Donations",
    "prob_liquidations":            "Liquidations",
    "prob_return_to_vendor":        "Return to Vendor",
    "prob_warehouse_deals_and_gr":  "Warehouse Deals / GR",
}

_RATE_BUCKET_ORDER = ["zero", "0-10%", "10-30%", "30-60%", ">60%"]
_VOLUME_BUCKET_ORDER = ["<10", "10-100", "100-1k", ">1k"]


# ============================================================
# Utility helpers
# ============================================================

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _img_tag(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;"/>'


def _df_to_html_table(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    rows = ["<table>", "<thead><tr>"]
    for col in df.columns:
        rows.append(f"<th>{col}</th>")
    rows.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        rows.append("<tr>")
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                cell = f"{val:{float_fmt}}"
                # colour-code MAE columns
                if "mae" in col.lower():
                    if val < 0.05:
                        style = "background:#d4edda;"
                    elif val < 0.10:
                        style = "background:#fff3cd;"
                    else:
                        style = "background:#f8d7da;"
                    rows.append(f'<td style="{style}">{cell}</td>')
                    continue
            else:
                cell = str(val)
            rows.append(f"<td>{cell}</td>")
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


# ============================================================
# Section 1 — Model performance summary
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


def _render_mae_section(summary: dict) -> str:
    if not summary:
        return ""
    html = [
        "<section>",
        "<h2>Model Performance Summary</h2>",
        f"<p><strong>Overall MAE:</strong> {summary['overall']:.4f}</p>",
        "<h3>MAE by Rate Bucket</h3>",
        _df_to_html_table(summary["by_rate_bucket"]),
        "<h3>MAE by Volume Bucket</h3>",
        _df_to_html_table(summary["by_volume_bucket"]),
        "</section>",
    ]
    return "\n".join(html)


# ============================================================
# Section 2 — Top recovery performers
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


def _render_performers_section(df: pd.DataFrame, top_n: int) -> str:
    gl_tbl   = _top_n_by_rate(df, "gl_product_group", top_n)
    site_tbl = _top_n_by_rate(df, "hashed_fc", top_n)
    html = [
        "<section>",
        f"<h2>Highest-Risk Groups (Elevated Recovery Rate)</h2>",
        "<p>A higher recovery rate indicates more product being returned, donated, or disposed of "
        "rather than sold. The groups below warrant the most attention.</p>",
        f"<h3>GL Product Groups</h3>",
        _df_to_html_table(gl_tbl),
        f"<h3>Sites</h3>",
        _df_to_html_table(site_tbl),
        "</section>",
    ]
    return "\n".join(html)


# ============================================================
# Section 3 — Weekly trend charts
# ============================================================

def _chart_weekly_single(
    df: pd.DataFrame,
    group_col: str,
    group_val: str,
) -> str:
    subset = df[df[group_col] == group_val]
    weekly = subset.groupby("week")["combined_rate"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(weekly["week"], weekly["combined_rate"], marker="o", color="#1f77b4", label="Predicted")
    if "prob_recovered" in subset.columns:
        gt = subset.groupby("week")["prob_recovered"].mean().reset_index()
        ax.plot(gt["week"], gt["prob_recovered"], marker="o", linestyle="--",
                color="#ff7f0e", label="Actual")
        ax.legend(fontsize=7)
    ax.set_xlabel("Week", fontsize=8)
    ax.set_ylabel("Recovery rate", fontsize=8)
    ax.set_title(group_val, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _charts_weekly_individual(
    df: pd.DataFrame,
    group_col: str,
    top_n: int,
) -> list[tuple[str, str]]:
    top_vals = (
        df.groupby(group_col)["combined_rate"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    return [(val, _chart_weekly_single(df, group_col, val)) for val in top_vals]


def _chart_weekly_channels(df: pd.DataFrame) -> str | None:
    pred_cols = [f"pred_{ch}" for ch in RECOVERY_CHANNELS if f"pred_{ch}" in df.columns]
    if not pred_cols:
        return None

    weekly = df.groupby("week")[pred_cols].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 4))
    for col in pred_cols:
        channel_key = col.removeprefix("pred_")
        label = _CHANNEL_LABELS.get(channel_key, channel_key)
        ax.plot(weekly["week"], weekly[col], marker="o", label=label)
    ax.set_xlabel("Week")
    ax.set_ylabel("Predicted channel rate")
    ax.set_title("Predicted Recovery Rate per Channel by Week")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _render_individual_charts(charts: list[tuple[str, str]]) -> str:
    items = "".join(
        f'<figure style="margin:0;">'
        f'<img src="data:image/png;base64,{b64}" style="width:100%;border:1px solid #ddd;border-radius:4px;"/>'
        f'<figcaption style="font-size:.8rem;text-align:center;color:#555;">{label}</figcaption>'
        f'</figure>'
        for label, b64 in charts
    )
    return f'<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;margin-bottom:1.5rem;">{items}</div>'


def _render_trend_section(df: pd.DataFrame, top_n: int) -> str:
    gl_charts   = _charts_weekly_individual(df, "gl_product_group", top_n)
    site_charts = _charts_weekly_individual(df, "hashed_fc", top_n)
    b64_channels = _chart_weekly_channels(df)

    html = [
        "<section>",
        "<h2>Recovery Rate Trend over Week</h2>",
        f"<h3>Top {top_n} GL Product Groups</h3>",
        _render_individual_charts(gl_charts),
        f"<h3>Top {top_n} Sites</h3>",
        _render_individual_charts(site_charts),
    ]
    if b64_channels:
        html += ["<h3>Per-Channel Breakdown</h3>", _img_tag(b64_channels)]
    html.append("</section>")
    return "\n".join(html)


# ============================================================
# Section 4 — SHAP attribution
# ============================================================

def _shap_available(df: pd.DataFrame) -> bool:
    return (
        "shap_baseline_rate" in df.columns
        and df["shap_baseline_rate"].notna().any()
    )


def _chart_shap_attribution(df: pd.DataFrame, top_n: int = 5) -> str:
    site_shap = (
        df.groupby("hashed_fc")
        .agg(
            mean_rate=("combined_rate", "mean"),
            baseline=("shap_baseline_rate", "mean"),
            deviation=("shap_deviation_contribution", "mean"),
        )
        .reset_index()
        .sort_values("mean_rate", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(site_shap))
    ax.bar(x, site_shap["baseline"],  label="Baseline rate",          color="#4c72b0")
    ax.bar(x, site_shap["deviation"], bottom=site_shap["baseline"],
           label="Deviation contribution", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(site_shap["hashed_fc"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Recovery rate (SHAP decomposition)")
    ax.set_title(f"Top {top_n} Highest-Risk Sites — SHAP Baseline vs Deviation")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _render_shap_section(df: pd.DataFrame, top_n: int) -> str:
    if not _shap_available(df):
        return ""

    b64 = _chart_shap_attribution(df, min(top_n, 5))

    # Callout: site with highest positive deviation
    site_dev = (
        df.groupby("hashed_fc")["shap_deviation_contribution"].mean()
        .sort_values(ascending=False)
    )
    best_site = site_dev.index[0]
    best_dev  = site_dev.iloc[0]

    html = [
        "<section>",
        "<h2>SHAP Attribution: Baseline vs Current-Week Deviation</h2>",
        "<p>The stacked bars show how much of each site's recovery rate comes from its "
        "historical site-GL baseline (blue) versus signals from the current week (orange).</p>",
        _img_tag(b64),
        f"<p><strong>Callout:</strong> Site <code>{best_site}</code> has the largest "
        f"current-week deviation above its historical baseline (<strong>+{best_dev:.4f}</strong>), "
        "indicating it is at <strong>heightened risk</strong> this period beyond what its "
        "site-GL history would predict.</p>",
        "</section>",
    ]
    return "\n".join(html)


# ============================================================
# Full report assembly
# ============================================================

_CSS = """
<style>
  body { font-family: Arial, sans-serif; margin: 2rem; color: #333; }
  h1   { color: #1a1a2e; border-bottom: 2px solid #1a1a2e; padding-bottom: .3rem; }
  h2   { color: #16213e; margin-top: 2rem; }
  h3   { color: #0f3460; }
  table{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; font-size: .9rem; }
  th, td { border: 1px solid #ccc; padding: .4rem .7rem; text-align: left; }
  th   { background: #e8eaf6; }
  tr:nth-child(even) { background: #f9f9f9; }
  img  { margin: .5rem 0 1rem; border: 1px solid #ddd; border-radius: 4px; }
  section { margin-bottom: 2.5rem; }
  p    { line-height: 1.5; }
</style>
"""


def _build_report(df: pd.DataFrame, top_n: int) -> str:
    mae_summary = _mae_summary(df)

    sections = [
        _render_mae_section(mae_summary),
        _render_performers_section(df, top_n),
        _render_trend_section(df, top_n),
        _render_shap_section(df, top_n),
    ]

    body = "\n".join(s for s in sections if s)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Recovery Funnel Report</title>
  {_CSS}
</head>
<body>
  <h1>Recovery Funnel — Model Insights Report</h1>
  {body}
</body>
</html>"""


# ============================================================
# S3 persistence
# ============================================================

def _save_html_to_s3(html: str, s3_uri: str) -> None:
    path = s3_uri.removeprefix("s3://")
    bucket, _, key = path.partition("/")
    s3_client = boto3.client("s3")
    body = html.encode("utf-8")
    s3_client.put_object(Body=body, Bucket=bucket, Key=key, ContentType="text/html")
    logger.info("Report saved to s3://%s/%s", bucket, key)


# ============================================================
# Pipeline step
# ============================================================

@enforce({
    ContextKeys.PREDICTIONS: Requires(),
    ContextKeys.REPORT: Sequence(Defines(strict=True), Locks(strict=True)),
})
class Report(PipelineStep):
    """Reads ContextKeys.PREDICTIONS and produces a self-contained HTML report.

    Sections:
      1. Model performance summary (overall MAE, MAE by rate/volume bucket)
      2. Highest-risk GL groups and sites by mean predicted recovery rate (high rate = bad outcome)
      3. Individual recovery-rate trend charts per GL group and per site (top_n each) + per-channel
      4. SHAP baseline-vs-deviation attribution for highest-risk sites (skipped when SHAP not run)

    The HTML string is stored in ContextKeys.REPORT and optionally uploaded to S3.
    """

    def __init__(
        self,
        top_n: int = 10,
        save_to: str | None = None,
    ):
        self.top_n = top_n
        self.save_to = save_to

    def __call__(self, context: Context) -> Context:
        df = context[ContextKeys.PREDICTIONS].to_pandas()
        logger.info("Building report from %d prediction rows", len(df))

        html = _build_report(df, self.top_n)

        if self.save_to is not None:
            _save_html_to_s3(html, self.save_to)

        context[ContextKeys.REPORT] = html
        context.lock(ContextKeys.REPORT)

        return context
