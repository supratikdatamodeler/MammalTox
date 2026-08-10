"""Full-source final-model performance data and scientific figures."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)

from .model_loading import (
    load_model_from_path,
    load_preprocessor_from_path,
    prepare_model_input,
)


TRAIN_COLOR = "#24485F"
TEST_COLOR = "#B78B42"
CLASS_0_COLOR = "#527C78"
CLASS_1_COLOR = "#A45B64"
CURVE_COLOR = "#B4475A"
PR_COLOR = "#326D8B"
GRID_COLOR = "#E7EAED"
TEXT_COLOR = "#25313B"
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "MCC"]


def _column_by_terms(frame: pd.DataFrame, terms: tuple[str, ...]) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    for term in terms:
        if term in normalized:
            return normalized[term]
    for column in frame.columns:
        compact = str(column).strip().lower().replace(" ", "_")
        if any(term in compact for term in terms):
            return str(column)
    return None


def _numeric_frame(frame: pd.DataFrame, descriptors: tuple[str, ...]) -> pd.DataFrame:
    missing = [descriptor for descriptor in descriptors if descriptor not in frame.columns]
    if missing:
        raise ValueError(f"Required descriptors are missing: {', '.join(missing[:12])}")
    numeric = frame.loc[:, list(descriptors)].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"NaN or infinite values occur in required descriptors: {', '.join(bad[:12])}")
    return numeric


def _labels(frame: pd.DataFrame) -> np.ndarray:
    column = _column_by_terms(frame, ("class", "label", "target"))
    if column is None:
        raise ValueError("Class/label/target column not found")
    labels = pd.to_numeric(frame[column], errors="coerce")
    if labels.isna().any() or not set(labels.astype(int).unique()).issubset({0, 1}):
        raise ValueError(f"Target column {column} is not a complete binary 0/1 target")
    return labels.astype(int).to_numpy()


def _positive_index(model) -> int:
    classes = list(getattr(model, "classes_", [0, 1]))
    return classes.index(1) if 1 in classes else len(classes) - 1


def _preprocessing_method(model, preprocessor_path: str) -> str:
    """Describe external or estimator-embedded training standardization."""
    if str(preprocessor_path or "").strip():
        return "Training-derived standardization"
    embedded_estimator = getattr(model, "estimator", None)
    embedded_steps = getattr(embedded_estimator, "steps", [])
    if any(type(step).__name__ == "StandardScaler" for _, step in embedded_steps):
        return "Training-derived standardization"
    return "No separate preprocessing step"


def _predict(model, preprocessor, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    inputs = prepare_model_input(model, frame, preprocessor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predicted = np.asarray(model.predict(inputs), dtype=int)
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(inputs), dtype=float)
            scores = probabilities[:, _positive_index(model)]
        elif hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(inputs), dtype=float).reshape(-1)
        else:
            raise ValueError("Saved model provides neither predict_proba nor decision_function")
    return predicted, scores


def _metrics(labels: np.ndarray, predicted: np.ndarray) -> tuple[float, ...]:
    return (
        accuracy_score(labels, predicted),
        precision_score(labels, predicted, zero_division=0),
        recall_score(labels, predicted, zero_division=0),
        f1_score(labels, predicted, zero_division=0),
        matthews_corrcoef(labels, predicted),
    )


def _p_ld50_values(
    selected_train: pd.DataFrame,
    full_test: pd.DataFrame,
    full_dataset: pd.DataFrame,
) -> pd.DataFrame:
    id_source = _column_by_terms(full_dataset, ("taid", "compound_id", "id"))
    pld50_source = _column_by_terms(
        full_dataset,
        ("pld50", "pld50_exp", "experimental_pld50", "transformed_ld50"),
    )
    train_id = _column_by_terms(selected_train, ("taid", "compound_id", "id"))
    test_id = _column_by_terms(full_test, ("taid", "compound_id", "id"))
    if not all([id_source, pld50_source, train_id, test_id]):
        return pd.DataFrame(columns=["Subset", "pLD50"])
    lookup = full_dataset[[id_source, pld50_source]].copy()
    lookup.columns = ["Identifier", "pLD50"]
    lookup["Identifier"] = lookup["Identifier"].astype(str)
    lookup["pLD50"] = pd.to_numeric(lookup["pLD50"], errors="coerce")
    rows = []
    for subset, frame, identifier_column in (
        ("Training", selected_train, train_id),
        ("Test", full_test, test_id),
    ):
        identifiers = frame[[identifier_column]].copy()
        identifiers.columns = ["Identifier"]
        identifiers["Identifier"] = identifiers["Identifier"].astype(str)
        merged = identifiers.merge(lookup, on="Identifier", how="left")
        for value in merged["pLD50"].dropna():
            rows.append({"Subset": subset, "pLD50": float(value)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=24)
def load_endpoint_source_map(path: str, deployment_version: str = "") -> pd.DataFrame:
    del deployment_version
    return pd.read_csv(path).fillna("")


@st.cache_data(show_spinner=False, max_entries=24)
def prepare_model_performance_data(
    model_id: str,
    model_path: str,
    preprocessor_path: str,
    selected_train_path: str,
    full_test_path: str,
    full_dataset_path: str,
    tuning_results_path: str,
    performance_results_path: str,
    fiore_output_path: str,
    descriptors: tuple[str, ...],
    app_root: str,
    deployment_version: str = "",
) -> dict:
    """Reproduce endpoint performance from full train/test source matrices."""
    del deployment_version
    try:
        model = load_model_from_path(model_path, app_root)
        preprocessor = load_preprocessor_from_path(preprocessor_path, app_root)
        selected_train = pd.read_csv(selected_train_path, low_memory=False)
        full_test = pd.read_csv(full_test_path, low_memory=False)
        full_dataset = pd.read_csv(full_dataset_path, low_memory=False, encoding_errors="replace")
        train_x = _numeric_frame(selected_train, descriptors)
        test_x = _numeric_frame(full_test, descriptors)
        train_y = _labels(selected_train)
        test_y = _labels(full_test)
        train_predicted, train_scores = _predict(model, preprocessor, train_x)
        test_predicted, test_scores = _predict(model, preprocessor, test_x)

        train_metrics = _metrics(train_y, train_predicted)
        test_metrics = _metrics(test_y, test_predicted)
        metric_data = pd.DataFrame(
            {
                "Metric": METRIC_NAMES,
                "Training": train_metrics,
                "Test": test_metrics,
            }
        )
        fpr, tpr, _ = roc_curve(test_y, test_scores, pos_label=1)
        precision, recall, _ = precision_recall_curve(test_y, test_scores, pos_label=1)
        auc_value = roc_auc_score(test_y, test_scores)
        train_auc_value = roc_auc_score(train_y, train_scores)
        ap_value = average_precision_score(test_y, test_scores)
        confusion = confusion_matrix(test_y, test_predicted, labels=[0, 1])
        pld50_data = _p_ld50_values(selected_train, full_test, full_dataset)

        tuning = pd.read_csv(tuning_results_path, low_memory=False)
        tuning["number"] = pd.to_numeric(tuning["number"], errors="coerce")
        tuning["value"] = pd.to_numeric(tuning["value"], errors="coerce")
        if "state" in tuning.columns:
            tuning = tuning[tuning["state"].astype(str).str.upper().eq("COMPLETE")]
        tuning = tuning.dropna(subset=["number", "value"]).sort_values("number")
        tuning["best_so_far"] = tuning["value"].cummax()
        preprocessing_method = _preprocessing_method(model, preprocessor_path)

        class_data = pd.DataFrame(
            {
                "Subset": ["Training", "Training", "Test", "Test"],
                "Class": ["Low toxicity", "High toxicity"] * 2,
                "Count": [
                    int(np.sum(train_y == 0)),
                    int(np.sum(train_y == 1)),
                    int(np.sum(test_y == 0)),
                    int(np.sum(test_y == 1)),
                ],
            }
        )
        source_summary = pd.DataFrame(
            [
                {
                    "Data component": "Training set",
                    "Records": len(selected_train),
                    "Description": "Selected Mordred descriptors used for model development.",
                },
                {
                    "Data component": "Test set",
                    "Records": len(full_test),
                    "Description": "Complete test set used for final model evaluation.",
                },
                {
                    "Data component": "Curated endpoint dataset",
                    "Records": len(full_dataset),
                    "Description": "Full curated endpoint dataset used in model development.",
                },
                {
                    "Data component": "Optimization history",
                    "Records": f"{len(tuning)} trials",
                    "Description": "Completed model-optimization trials.",
                },
                {
                    "Data component": "Preprocessing method",
                    "Records": "Not applicable",
                    "Description": preprocessing_method,
                },
            ]
        )
        source_summary["Records"] = source_summary["Records"].astype(str)
        availability = pd.DataFrame(
            [
                {
                    "Visualization": "Test-set ROC curve",
                    "Status": "Available",
                    "Notes": f"Generated using the complete test set ({len(test_y)} compounds); ROC-AUC {auc_value:.4f}.",
                },
                {
                    "Visualization": "Test-set confusion matrix",
                    "Status": "Available",
                    "Notes": f"Contains predictions for all {int(confusion.sum())} test compounds.",
                },
                {
                    "Visualization": "Test-set precision-recall curve",
                    "Status": "Available",
                    "Notes": f"Generated using the complete test set; average precision {ap_value:.4f}.",
                },
                {
                    "Visualization": "Training/test-set pLD50 distribution",
                    "Status": "Available" if not pld50_data.empty else "Unavailable",
                    "Notes": (
                        "Training-set and test-set identifiers were joined to real curated pLD50 values."
                        if not pld50_data.empty
                        else "A pLD50 column or joinable compound identifier was not found."
                    ),
                },
                {
                    "Visualization": "Optimization history",
                    "Status": "Available" if not tuning.empty else "Unavailable",
                    "Notes": (
                        f"Based on {len(tuning)} completed trials with recorded objective and cumulative-best values."
                        if not tuning.empty
                        else "No complete per-trial objective values were found."
                    ),
                },
            ]
        )
        return {
            "model_id": model_id,
            "train_count": len(train_y),
            "test_count": len(test_y),
            "descriptor_count": len(descriptors),
            "metric_data": metric_data,
            "class_data": class_data,
            "roc": {"fpr": fpr, "tpr": tpr, "auc": auc_value},
            "train_roc_auc": train_auc_value,
            "precision_recall": {
                "precision": precision,
                "recall": recall,
                "average_precision": ap_value,
                "prevalence": float(np.mean(test_y)),
            },
            "confusion": confusion,
            "prediction_output": (
                "High toxicity probability"
                if hasattr(model, "predict_proba")
                else "High toxicity model score"
            ),
            "preprocessing_method": preprocessing_method,
            "pld50_data": pld50_data,
            "tuning": tuning,
            "source_summary": source_summary,
            "availability": availability,
            "error": "",
        }
    except Exception as exc:
        return {
            "model_id": model_id,
            "availability": pd.DataFrame(
                [
                    {
                        "Visualization": "Model performance visualizations",
                        "Status": "Unavailable",
                        "Notes": "Visualization data could not be prepared in this environment.",
                    }
                ]
            ),
            "source_summary": pd.DataFrame(),
            "error": "Model performance visualizations are currently unavailable for this endpoint.",
            "debug_error": str(exc),
        }


def final_model_evaluation_figure(data: dict, endpoint_label: str, model_name: str):
    if not data or data.get("error"):
        return None
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Test-set ROC curve",
            "Test-set confusion matrix",
            "Test-set precision-recall curve",
            "Final model performance",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.14,
    )
    roc_data = data["roc"]
    fig.add_trace(
        go.Scatter(
            x=roc_data["fpr"], y=roc_data["tpr"], mode="lines",
            name=f"ROC (AUC = {roc_data['auc']:.4f})",
            line={"color": CURVE_COLOR, "width": 3},
            hovertemplate="False positive rate: %{x:.3f}<br>True positive rate: %{y:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Chance",
            line={"color": "#87939C", "width": 1.5, "dash": "dash"},
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    matrix = np.asarray(data["confusion"], dtype=int)
    row_percent = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1) * 100
    cell_text = np.asarray(
        [[f"{matrix[row, column]}<br>{row_percent[row, column]:.1f}%" for column in range(2)] for row in range(2)]
    )
    interpretations = np.asarray(
        [
            ["Correct prediction: Low toxicity", "False positive"],
            ["False negative", "Correct prediction: High toxicity"],
        ]
    )
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=["Predicted Low toxicity", "Predicted High toxicity"],
            y=["Actual Low toxicity", "Actual High toxicity"],
            colorscale="Blues",
            text=cell_text,
            texttemplate="%{text}",
            textfont={"size": 15},
            customdata=interpretations,
            hovertemplate="%{customdata}<br>Count: %{z}<extra></extra>",
            colorbar={"title": "Count", "len": 0.34, "thickness": 14, "x": 1.01, "y": 0.82},
            showscale=True,
            name="Test-set confusion matrix",
        ),
        row=1, col=2,
    )

    pr_data = data["precision_recall"]
    fig.add_trace(
        go.Scatter(
            x=pr_data["recall"], y=pr_data["precision"], mode="lines",
            name=f"Precision-recall (AP = {pr_data['average_precision']:.4f})",
            line={"color": PR_COLOR, "width": 3},
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[pr_data["prevalence"]] * 2, mode="lines",
            name=f"Prevalence = {pr_data['prevalence']:.3f}",
            line={"color": "#87939C", "width": 1.5, "dash": "dash"},
            hoverinfo="skip",
        ),
        row=2, col=1,
    )

    metrics = data["metric_data"]
    metric_labels = metrics["Metric"].replace({"ROC-AUC": "ROC–AUC"})
    for subset, display_name, color in (
        ("Training", "Training set", TRAIN_COLOR),
        ("Test", "Test set", TEST_COLOR),
    ):
        fig.add_trace(
            go.Bar(
                x=metric_labels, y=metrics[subset], name=display_name,
                marker_color=color, text=[f"{value:.3f}" for value in metrics[subset]],
                textposition="outside", cliponaxis=False,
                hovertemplate=f"{display_name}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
            ),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="False positive rate", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="True positive rate", range=[0, 1.02], row=1, col=1)
    fig.update_xaxes(title_text="Predicted class", row=1, col=2)
    fig.update_yaxes(title_text="True class", autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="Recall", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="Precision", range=[0, 1.02], row=2, col=1)
    fig.update_yaxes(title_text="Score", range=[0, 1.1], row=2, col=2)
    fig.update_layout(
        title={"text": f"{endpoint_label} | {model_name} final-model evaluation", "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        barmode="group",
        height=850,
        margin={"l": 70, "r": 80, "t": 100, "b": 80},
        font={"family": "Arial, sans-serif", "size": 12, "color": TEXT_COLOR},
        legend={"orientation": "h", "yanchor": "top", "y": -0.12, "xanchor": "left", "x": 0},
        hoverlabel={"font_size": 12},
    )
    fig.update_annotations(font={"size": 15, "color": TEXT_COLOR})
    return fig


def final_model_development_figure(data: dict, endpoint_label: str, model_name: str):
    if not data or data.get("error"):
        return None
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Training-set and test-set pLD50 distribution", "Optimization history"),
        horizontal_spacing=0.12,
    )

    distribution = data["pld50_data"]
    if distribution.empty:
        class_data = data["class_data"]
        for class_label, color in (
            ("Low toxicity", CLASS_0_COLOR),
            ("High toxicity", CLASS_1_COLOR),
        ):
            values = [int(class_data[(class_data["Subset"] == subset) & (class_data["Class"] == class_label)]["Count"].iloc[0]) for subset in ["Training", "Test"]]
            fig.add_trace(
                go.Bar(x=["Training set", "Test set"], y=values, name=class_label, marker_color=color),
                row=1, col=1,
            )
        fig.layout.annotations[0].text = "Training-set and test-set class distribution"
        fig.update_yaxes(title_text="Compound count", row=1, col=1)
    else:
        distribution_colors = {
            "Training": "#607886",
            "Test": "#B38B45",
        }
        for subset in ("Training", "Test"):
            values = distribution.loc[distribution["Subset"].eq(subset), "pLD50"]
            fig.add_trace(
                go.Histogram(
                    x=values, nbinsx=20, histnorm="probability density",
                    name=f"{subset} set (n={len(values)})",
                    marker={
                        "color": distribution_colors[subset],
                        "line": {"color": "#FFFFFF", "width": 0.45},
                    },
                    opacity=0.86,
                    hovertemplate="pLD50: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
                ),
                row=1, col=1,
            )
        fig.update_xaxes(title_text="pLD50", row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)

    tuning = data["tuning"]
    fig.layout.annotations[1].text = f"Optimization history ({len(tuning)} completed trials)"
    fig.add_trace(
        go.Scatter(
            x=tuning["number"], y=tuning["value"], mode="lines",
            name="Trial objective", line={"color": "#8DA0AA", "width": 1.2}, opacity=0.7,
            legend="legend2",
            hovertemplate="Trial %{x:.0f}<br>Recorded objective: %{y:.4f}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=tuning["number"], y=tuning["best_so_far"], mode="lines",
            name="Best so far", line={"color": CURVE_COLOR, "width": 3},
            legend="legend2",
            hovertemplate="Trial %{x:.0f}<br>Best recorded objective: %{y:.4f}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Optimization trial number", row=1, col=2)
    fig.update_yaxes(title_text="Recorded objective value", row=1, col=2)

    fig.update_layout(
        title={
            "text": f"{endpoint_label} | {model_name} model-development evidence",
            "x": 0.01,
            "xanchor": "left",
            "font": {"color": "#111111"},
        },
        template="plotly_white",
        barmode="group",
        height=520,
        margin={"l": 70, "r": 40, "t": 100, "b": 120},
        font={"family": "Arial, sans-serif", "size": 12, "color": "#111111"},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.22,
            "xanchor": "center",
            "x": 0.235,
            "font": {"color": "#111111"},
        },
        legend2={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.22,
            "xanchor": "center",
            "x": 0.765,
            "font": {"color": "#111111"},
        },
        hoverlabel={"font_size": 12},
    )
    axis_style = {
        "color": "#111111",
        "gridcolor": GRID_COLOR,
        "linecolor": "#111111",
        "showline": True,
        "tickcolor": "#111111",
        "tickfont": {"color": "#111111"},
        "ticks": "outside",
        "title_font": {"color": "#111111"},
        "zerolinecolor": GRID_COLOR,
    }
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_annotations(font={"size": 15, "color": "#111111"})
    return fig
