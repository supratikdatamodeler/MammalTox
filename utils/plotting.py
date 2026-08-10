"""Simple plotting helpers for dataset statistics."""

from __future__ import annotations

import pandas as pd

try:
    import plotly.express as px
except Exception:  # pragma: no cover - fallback when plotly is unavailable
    px = None


def class_distribution_percentage_table(stats: pd.DataFrame) -> pd.DataFrame:
    """Return per-species class counts and percentages."""
    if stats.empty:
        return pd.DataFrame(
            columns=[
                "species",
                "total_count",
                "non_toxic_count",
                "toxic_count",
                "non_toxic_percent",
                "toxic_percent",
            ]
        )

    summary = pd.DataFrame(
        {
            "species": stats["species"],
            "total_count": pd.to_numeric(stats["total_compound_count"], errors="coerce"),
            "non_toxic_count": pd.to_numeric(stats["total_class_0"], errors="coerce"),
            "toxic_count": pd.to_numeric(stats["total_class_1"], errors="coerce"),
        }
    )
    summary["non_toxic_percent"] = (
        summary["non_toxic_count"].div(summary["total_count"]).mul(100).round(2)
    )
    summary["toxic_percent"] = summary["toxic_count"].div(summary["total_count"]).mul(100).round(2)
    return summary


def class_distribution_figure(stats: pd.DataFrame):
    """Create a percentage-based class-distribution bar chart."""
    if stats.empty or px is None:
        return None

    summary = class_distribution_percentage_table(stats)
    if summary.empty:
        return None

    plot_df = summary.melt(
        id_vars="species",
        value_vars=["non_toxic_percent", "toxic_percent"],
        var_name="class",
        value_name="percentage",
    )
    plot_df["class"] = plot_df["class"].map(
        {
            "non_toxic_percent": "Low toxicity",
            "toxic_percent": "High toxicity",
        }
    )
    fig = px.bar(
        plot_df,
        x="species",
        y="percentage",
        color="class",
        barmode="group",
        title="Class Distribution by Species (%)",
        labels={
            "species": "Species",
            "percentage": "Percentage of compounds (%)",
            "class": "Class",
        },
        category_orders={"species": summary["species"].astype(str).tolist()},
        color_discrete_map={
            "Low toxicity": "#4C78A8",
            "High toxicity": "#72B7B2",
        },
    )
    fig.update_traces(
        marker_line_color="white",
        marker_line_width=0.8,
        hovertemplate="%{x}<br>%{fullData.name}: %{y:.2f}%<extra></extra>",
    )
    fig.update_yaxes(range=[0, 100], ticksuffix="%", gridcolor="#E7EAED")
    fig.update_xaxes(showgrid=False)
    fig.update_layout(
        template="plotly_white",
        legend_title_text="",
        height=460,
        margin={"l": 70, "r": 30, "t": 70, "b": 70},
        font={"family": "Arial, sans-serif", "size": 13, "color": "#25313B"},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left", "x": 0},
    )
    return fig
