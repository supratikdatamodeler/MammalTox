"""Scientific PCA and SHAP figures for MammalTox results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
import shap


POSITIVE_COLOR = "#C2546D"
NEGATIVE_COLOR = "#3478A8"
TEXT_COLOR = "#25313B"
GRID_COLOR = "#E7EAED"


def _is_shap_math_label(value: str) -> bool:
    compact = re.sub(r"[\s$\\{}]", "", str(value)).lower()
    return (
        "f(x)" in compact
        or "e[f(x)]" in compact
        or "expectedvalue" in compact
    )


def _hide_shap_math_labels(fig) -> None:
    """Hide only SHAP function/expectation notation from a local plot."""
    for text in list(fig.texts):
        if _is_shap_math_label(text.get_text()):
            text.set_visible(False)
    for axis in fig.axes:
        for text in list(axis.texts):
            if _is_shap_math_label(text.get_text()):
                text.set_visible(False)
        for label in [axis.xaxis.label, axis.yaxis.label, axis.title]:
            if _is_shap_math_label(label.get_text()):
                label.set_text("")
        for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            if _is_shap_math_label(label.get_text()):
                label.set_visible(False)
    # SHAP uses two overlaid top x-axes for the output and expected-value
    # annotations. Their formula and numeric suffixes are separate tick objects.
    for annotation_axis in fig.axes[1:]:
        annotation_axis.set_xticks([])
        annotation_axis.xaxis.set_visible(False)


def pca_chemical_space_figure(artifact: dict, endpoint_label: str):
    if not artifact or artifact.get("error"):
        return None
    training = np.asarray(artifact["training_coordinates"], dtype=float)
    classes = np.asarray(artifact["training_classes"])
    queries = np.asarray(artifact["query_coordinates"], dtype=float)
    query_ids = list(artifact["query_ids"])
    explained = np.asarray(artifact["explained_variance"], dtype=float) * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=training[classes == 0, 0],
            y=training[classes == 0, 1],
            mode="markers",
            name="Training Low toxicity",
            marker={"size": 7, "color": "#2E8B57", "opacity": 0.42},
            hovertemplate="Training Low toxicity<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=training[classes == 1, 0],
            y=training[classes == 1, 1],
            mode="markers",
            name="Training High toxicity",
            marker={"size": 7, "color": "#C43C39", "opacity": 0.42},
            hovertemplate="Training High toxicity<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
        )
    )
    if len(queries) == 1:
        fig.add_trace(
            go.Scatter(
                x=queries[:, 0],
                y=queries[:, 1],
                mode="markers",
                name="Submitted queries",
                marker={
                    "size": 19,
                    "symbol": "star",
                    "color": "#1464B4",
                    "line": {"color": "white", "width": 1.5},
                },
                customdata=np.asarray(query_ids, dtype=object).reshape(-1, 1),
                hovertemplate="Query: %{customdata[0]}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
            )
        )
    else:
        show_labels = len(queries) <= 12
        fig.add_trace(
            go.Scatter(
                x=queries[:, 0],
                y=queries[:, 1],
                mode="markers+text" if show_labels else "markers",
                text=[str(query_id) for query_id in query_ids] if show_labels else None,
                textposition="top right",
                textfont={"size": 11, "color": TEXT_COLOR},
                name="Submitted queries",
                marker={
                    "size": 11,
                    "symbol": "circle",
                    "color": "#1464B4",
                    "line": {"color": "white", "width": 1.2},
                },
                customdata=np.asarray(query_ids, dtype=object).reshape(-1, 1),
                hovertemplate="Query: %{customdata[0]}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
            )
        )

    fig.add_hline(y=0, line={"color": "#D9DEE3", "width": 1})
    fig.add_vline(x=0, line={"color": "#D9DEE3", "width": 1})
    fig.update_layout(
        title={
            "text": f"{endpoint_label} training chemical space",
            "x": 0.01,
            "xanchor": "left",
            "font": {"family": "Arial, sans-serif", "size": 18, "color": TEXT_COLOR},
        },
        xaxis_title=f"PC1 ({explained[0]:.1f}% explained variance)",
        yaxis_title=f"PC2 ({explained[1]:.1f}% explained variance)",
        template="plotly_white",
        height=640,
        margin={"l": 70, "r": 30, "t": 80, "b": 90},
        font={"family": "Arial, sans-serif", "size": 13, "color": TEXT_COLOR},
        legend={"orientation": "h", "yanchor": "top", "y": -0.16, "xanchor": "left", "x": 0},
        hoverlabel={"font_size": 12},
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig


def local_shap_figure(artifact: dict, query_index: int = 0, max_display: int = 12):
    """Render an official SHAP explanation in the deployed model's native output space."""
    if not artifact or artifact.get("error"):
        return None
    values = np.asarray(artifact["values"], dtype=float)[query_index]
    data = np.asarray(artifact["data"], dtype=float)[query_index]
    base_value = float(np.asarray(artifact["base_values"], dtype=float)[query_index])
    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=data,
        feature_names=list(artifact["feature_names"]),
    )
    plt.figure(figsize=(11.2, 7.6), dpi=300, facecolor="white")
    shap.plots.waterfall(explanation, max_display=min(max_display, len(values)), show=False)
    fig = plt.gcf()
    fig.set_size_inches(11.2, 7.6, forward=True)
    fig.set_dpi(300)
    axis = fig.axes[0]
    order = np.argsort(-np.abs(values))
    display_count = min(max_display, len(values))
    display_values = list(values[order[: max(display_count - 1, 0)]])
    if display_count:
        display_values.append(float(values[order[max(display_count - 1, 0) :]].sum()))
    contribution_labels = [
        text
        for text in axis.texts
        if text.get_text().strip().startswith(("+", "-", "−"))
    ]
    contribution_labels.sort(key=lambda text: text.get_position()[1], reverse=True)
    for label, value in zip(contribution_labels, display_values):
        label.set_text(f"{value:+.4f}")
        label.set_fontsize(8)
    _hide_shap_math_labels(fig)
    axis.tick_params(axis="x", labelsize=9, colors=TEXT_COLOR)
    axis.tick_params(axis="y", labelsize=9, colors=TEXT_COLOR)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.6, zorder=0)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.86, bottom=0.10)
    return fig


def batch_shap_figure(artifact: dict, max_display: int = 15):
    if not artifact or artifact.get("error"):
        return None
    values = np.asarray(artifact["values"], dtype=float)
    data = pd.DataFrame(artifact["data"], columns=artifact["feature_names"])
    output_space = str(artifact.get("output_space", "probability"))
    output_label = "probability" if output_space == "probability" else "model score"
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    ):
        plt.figure(figsize=(10.2, 7.4), dpi=300, facecolor="white")
        shap.summary_plot(
            values,
            data,
            feature_names=list(artifact["feature_names"]),
            plot_type="dot",
            max_display=max_display,
            color=plt.get_cmap("coolwarm"),
            show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(10.2, 7.4, forward=True)
        fig.set_dpi(300)
        axis = fig.axes[0]
        axis.axvline(0, color=TEXT_COLOR, linewidth=1.0, zorder=1)
        axis.set_title("")
        axis.set_xlabel(f"SHAP contribution to High toxicity {output_label}")
        axis.grid(axis="x", color=GRID_COLOR, linewidth=0.7, zorder=0)
        axis.tick_params(colors=TEXT_COLOR)
        if len(fig.axes) > 1:
            color_axis = fig.axes[-1]
            color_axis.set_ylabel("Feature value", fontsize=10, labelpad=8)
            color_axis.tick_params(labelsize=9)
        fig.subplots_adjust(left=0.31, right=0.94, bottom=0.11, top=0.91)
    return fig
