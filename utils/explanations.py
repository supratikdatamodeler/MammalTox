"""SHAP explanations for the actual deployed MammalTox estimators."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import warnings

import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.model_selection import StratifiedShuffleSplit

from .model_loading import (
    load_model_from_path,
    load_preprocessor_from_path,
    model_feature_names,
    model_input,
    prepare_model_input,
)


LOGGER = logging.getLogger("mammaltox.shap")

SHAP_BACKGROUND_SIZE = 100
SHAP_BACKGROUND_RANDOM_SEED = 42


TREE_MODEL_NAMES = {
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
    "LGBMClassifier",
}


@dataclass
class ShapResource:
    kind: str
    explainer: object
    model: object
    descriptors: tuple[str, ...]
    positive_class_index: int
    fallback_reason: str
    background_size: int
    preprocessor: object | None
    background_values: np.ndarray
    background_classes: np.ndarray
    background_outputs: np.ndarray
    output_space: str


def _positive_class_index(model) -> int:
    classes = list(getattr(model, "classes_", [0, 1]))
    return classes.index(1) if 1 in classes else len(classes) - 1


def _predict_toxic_output(
    model,
    preprocessor,
    descriptors: tuple[str, ...],
    values: np.ndarray,
) -> tuple[np.ndarray, str]:
    frame = pd.DataFrame(np.asarray(values, dtype=float), columns=list(descriptors))
    inputs = prepare_model_input(model, frame, preprocessor)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(inputs)
        return (
            np.asarray(probabilities, dtype=float)[:, _positive_class_index(model)],
            "probability",
        )
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(inputs), dtype=float)
        if scores.ndim == 1:
            classes = list(getattr(model, "classes_", [0, 1]))
            positive_scores = scores if classes[-1] == 1 else -scores
        else:
            positive_scores = scores[:, _positive_class_index(model)]
        return np.asarray(positive_scores, dtype=float).reshape(-1), "model score"
    raise ValueError("The deployed model provides neither predict_proba nor decision_function")


def _representative_background(
    training: pd.DataFrame,
    descriptors: tuple[str, ...],
    background_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Select a reproducible, training-set-stratified SHAP background."""
    numeric = training.loc[:, list(descriptors)].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    valid = ~numeric.isna().any(axis=1)
    numeric = numeric.loc[valid]
    class_column = next(
        (column for column in training.columns if str(column).strip().lower() in {"class", "label", "target"}),
        None,
    )
    classes = (
        pd.to_numeric(training.loc[valid, class_column], errors="coerce").fillna(-1).astype(int)
        if class_column
        else pd.Series(-1, index=numeric.index, dtype=int)
    )
    if numeric.empty:
        raise ValueError("SHAP background data contain no complete descriptor rows")

    sample_size = min(background_size, len(numeric))
    if sample_size == len(numeric):
        background = numeric.copy()
    else:
        usable_classes = classes.isin({0, 1})
        class_counts = classes.loc[usable_classes].value_counts()
        can_stratify = (
            usable_classes.all()
            and len(class_counts) > 1
            and int(class_counts.min()) >= 2
            and sample_size >= len(class_counts)
        )
        if can_stratify:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=sample_size,
                random_state=SHAP_BACKGROUND_RANDOM_SEED,
            )
            selected_positions, _ = next(splitter.split(numeric, classes.to_numpy()))
            background = numeric.iloc[np.sort(selected_positions)]
        else:
            background = numeric.sample(
                n=sample_size,
                random_state=SHAP_BACKGROUND_RANDOM_SEED,
            ).sort_index()
    return background, classes.loc[background.index].to_numpy(dtype=int)


@st.cache_resource(show_spinner=False, max_entries=24)
def load_shap_resource(
    model_id: str,
    model_path: str,
    preprocessor_path: str,
    training_path: str,
    descriptors: tuple[str, ...],
    app_root: str,
    background_size: int,
) -> ShapResource:
    """Cache a model-appropriate SHAP explainer and representative background."""
    model = load_model_from_path(model_path, app_root)
    preprocessor = load_preprocessor_from_path(preprocessor_path, app_root)
    training = pd.read_csv(training_path, low_memory=False)
    background, background_classes = _representative_background(
        training, descriptors, background_size
    )
    positive_index = _positive_class_index(model)
    background_values = background.to_numpy(dtype=float)
    background_outputs, output_space = _predict_toxic_output(
        model, preprocessor, descriptors, background_values
    )

    fallback_reason = ""
    if type(model).__name__ in TREE_MODEL_NAMES and preprocessor is None:
        try:
            explainer = shap.TreeExplainer(model)
            probe = background.iloc[:1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer.shap_values(model_input(model, probe), check_additivity=False)
            return ShapResource(
                "tree",
                explainer,
                model,
                descriptors,
                positive_index,
                "",
                len(background),
                preprocessor,
                background_values,
                background_classes,
                background_outputs,
                output_space,
            )
        except Exception as exc:
            fallback_reason = f"TreeExplainer was incompatible ({exc})."

    predictor = lambda values: _predict_toxic_output(
        model, preprocessor, descriptors, values
    )[0]
    try:
        masker = shap.maskers.Independent(background_values, max_samples=len(background_values))
        explainer = shap.Explainer(
            predictor,
            masker,
            algorithm="permutation",
            feature_names=list(descriptors),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer(
                background_values[:1],
                max_evals=2 * len(descriptors) + 1,
                silent=True,
            )
        note = " ".join(
            part
            for part in [
                fallback_reason,
                (
                    "PermutationExplainer is used with predict_proba for High toxicity."
                    if output_space == "probability"
                    else "PermutationExplainer is used with the High toxicity score."
                ),
            ]
            if part
        )
        return ShapResource(
            "permutation",
            explainer,
            model,
            descriptors,
            positive_index,
            note,
            len(background),
            preprocessor,
            background_values,
            background_classes,
            background_outputs,
            output_space,
        )
    except Exception as exc:
        note = " ".join(
            part
            for part in [
                fallback_reason,
                f"PermutationExplainer was incompatible ({exc}); KernelExplainer is used.",
            ]
            if part
        )
        explainer = shap.KernelExplainer(predictor, background_values)
        return ShapResource(
            "kernel",
            explainer,
            model,
            descriptors,
            positive_index,
            note,
            len(background),
            preprocessor,
            background_values,
            background_classes,
            background_outputs,
            output_space,
        )


def _select_positive_output(values, expected_value, positive_index: int) -> tuple[np.ndarray, float]:
    if isinstance(values, list):
        selected = np.asarray(values[positive_index], dtype=float)
    else:
        selected = np.asarray(values, dtype=float)
        if selected.ndim == 3:
            selected = selected[:, :, positive_index]

    expected = np.asarray(expected_value, dtype=float)
    if expected.ndim == 0:
        base_value = float(expected)
    else:
        flat = expected.reshape(-1)
        base_value = float(flat[positive_index] if len(flat) > positive_index else flat[-1])
    return selected, base_value


@st.cache_data(show_spinner=False, max_entries=48)
def calculate_shap_values(
    model_id: str,
    model_path: str,
    preprocessor_path: str,
    training_path: str,
    descriptors: tuple[str, ...],
    app_root: str,
    query_values: tuple[tuple[float, ...], ...],
    query_ids: tuple[str, ...],
    background_size: int = SHAP_BACKGROUND_SIZE,
) -> dict:
    """Calculate High toxicity SHAP values for unchanged query inputs."""
    if not query_values:
        return {"error": "SHAP calculation failed: no successful query records are available"}

    try:
        resource = load_shap_resource(
            model_id,
            model_path,
            preprocessor_path,
            training_path,
            descriptors,
            app_root,
            background_size,
        )
        query = pd.DataFrame(query_values, columns=list(descriptors), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if resource.kind == "tree":
                raw_values = resource.explainer.shap_values(
                    model_input(resource.model, query), check_additivity=False
                )
                values, base_value = _select_positive_output(
                    raw_values, resource.explainer.expected_value, resource.positive_class_index
                )
                base_values = np.repeat(base_value, len(query))
            elif resource.kind == "permutation":
                explanation = resource.explainer(
                    query.to_numpy(dtype=float),
                    max_evals=min(320, 2 * len(descriptors) + 64),
                    silent=True,
                )
                values = np.asarray(explanation.values, dtype=float)
                base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
                if len(base_values) == 1:
                    base_values = np.repeat(base_values[0], len(query))
            else:
                sample_count = min(160, 2 * len(descriptors) + 64)
                raw_values = resource.explainer.shap_values(
                    query.to_numpy(dtype=float), nsamples=sample_count, silent=True
                )
                values, base_value = _select_positive_output(
                    raw_values, resource.explainer.expected_value, resource.positive_class_index
                )
                base_values = np.repeat(base_value, len(query))
        if values.shape != query.shape:
            raise ValueError(f"unexpected SHAP output shape {values.shape}; expected {query.shape}")
        query_outputs, output_space = _predict_toxic_output(
            resource.model,
            resource.preprocessor,
            descriptors,
            query.to_numpy(dtype=float),
        )
        reconstructed = np.asarray(base_values, dtype=float) + values.sum(axis=1)
        reconstruction_error = np.abs(reconstructed - query_outputs)
        transformed_background = prepare_model_input(
            resource.model,
            pd.DataFrame(resource.background_values, columns=list(descriptors)),
            resource.preprocessor,
        )
        training_columns = list(pd.read_csv(training_path, nrows=0).columns)
        missing_descriptors = [name for name in descriptors if name not in training_columns]
        model_names = model_feature_names(resource.model)
        feature_order_match = model_names == list(descriptors) if model_names else (
            getattr(resource.model, "n_features_in_", len(descriptors)) == len(descriptors)
        )
        diagnostics = {
            "model_filename": model_path,
            "preprocessor_filename": preprocessor_path or "None",
            "descriptor_list_filename": training_path,
            "expected_feature_count": int(getattr(resource.model, "n_features_in_", len(descriptors))),
            "supplied_feature_count": len(descriptors),
            "feature_order_match": bool(feature_order_match),
            "missing_descriptors": missing_descriptors,
            "extra_descriptors": [],
            "raw_descriptor_min": float(np.min(resource.background_values)),
            "raw_descriptor_max": float(np.max(resource.background_values)),
            "transformed_descriptor_min": float(np.min(transformed_background)),
            "transformed_descriptor_max": float(np.max(transformed_background)),
            "background_sample_size": resource.background_size,
            "background_class_0": int(np.sum(resource.background_classes == 0)),
            "background_class_1": int(np.sum(resource.background_classes == 1)),
            "background_output_min": float(np.min(resource.background_outputs)),
            "background_output_max": float(np.max(resource.background_outputs)),
            "background_output_mean": float(np.mean(resource.background_outputs)),
            "query_wrapper_outputs": query_outputs.tolist(),
            "shap_base_values": np.asarray(base_values, dtype=float).tolist(),
            "shap_contribution_sums": values.sum(axis=1).tolist(),
            "reconstructed_outputs": reconstructed.tolist(),
            "reconstruction_errors": reconstruction_error.tolist(),
            "output_space": output_space,
        }
        saturation_warning = ""
        if np.ptp(resource.background_outputs) < 1e-6 or np.max(np.abs(values)) < 1e-8:
            saturation_warning = (
                "Meaningful local SHAP contributions are limited because the model output is saturated "
                "for this query and reference background."
            )
        if model_id == "rat_mlp":
            LOGGER.info("Rat IV MLP SHAP diagnostics: %s", diagnostics)
        return {
            "values": values,
            "base_values": base_values,
            "data": query.to_numpy(dtype=float),
            "feature_names": list(descriptors),
            "query_ids": list(query_ids),
            "explainer_type": {
                "tree": "TreeExplainer",
                "permutation": "PermutationExplainer",
                "kernel": "KernelExplainer",
            }[resource.kind],
            "fallback_reason": resource.fallback_reason,
            "background_size": resource.background_size,
            "query_outputs": query_outputs,
            "query_probabilities": query_outputs if output_space == "probability" else np.asarray([]),
            "reconstructed_outputs": reconstructed,
            "reconstruction_errors": reconstruction_error,
            "output_space": output_space,
            "diagnostics": diagnostics,
            "saturation_warning": saturation_warning,
            "error": "",
        }
    except Exception as exc:
        return {"error": f"SHAP calculation failed: {exc}"}
