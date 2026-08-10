"""Model loading and feature-name helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from .metadata_loading import APP_ROOT, resolve_app_path


GUINEA_PIG_PICKLE_MESSAGE = (
    "Guinea Pig model could not be loaded in this environment. Please verify the active "
    "MLP pickle or re-export/retrain this model in a compatible environment."
)


class ModelLoadError(RuntimeError):
    """Raised when a saved model cannot be loaded."""


@st.cache_resource(show_spinner=False, max_entries=64)
def load_model_from_path(model_path: str, root: str | None = None):
    """Load a joblib/pickle model from an app-relative path."""
    base = Path(root) if root else APP_ROOT
    resolved = resolve_app_path(model_path, base)
    if not resolved.exists():
        raise ModelLoadError(f"Model file not found: {model_path}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(resolved)
    except Exception as exc:
        error_text = str(exc)
        if "guinea_pig" in model_path and (
            "MT19937" in error_text or "BitGenerator" in error_text or "legacy MT19937 state" in error_text
        ):
            raise ModelLoadError(f"{GUINEA_PIG_PICKLE_MESSAGE} Technical details: {error_text}") from exc
        raise ModelLoadError(f"Model file could not be loaded: {exc}") from exc


@st.cache_resource(show_spinner=False, max_entries=32)
def load_preprocessor_from_path(preprocessor_path: str, root: str | None = None):
    """Load an optional app-relative scaler or preprocessor artifact."""
    if not str(preprocessor_path or "").strip():
        return None
    base = Path(root) if root else APP_ROOT
    resolved = resolve_app_path(preprocessor_path, base)
    if not resolved.exists():
        raise ModelLoadError(f"Preprocessor file not found: {preprocessor_path}")
    try:
        return joblib.load(resolved)
    except Exception as exc:
        raise ModelLoadError(f"Preprocessor file could not be loaded: {exc}") from exc


def model_feature_names(model) -> list[str] | None:
    """Return model.feature_names_in_ when available."""
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return None
    return [str(name) for name in list(names)]


def resolve_required_descriptors(
    model,
    model_id: str,
    selected_descriptors_by_model: dict[str, Sequence[str]],
) -> list[str]:
    """Prefer model feature names, otherwise use metadata descriptors."""
    feature_names = model_feature_names(model)
    if feature_names:
        return feature_names
    descriptors = selected_descriptors_by_model.get(model_id, [])
    return [str(desc) for desc in descriptors]


def model_input(model, descriptor_frame: pd.DataFrame):
    """Preserve feature names when the saved estimator was fitted with them."""
    if model_feature_names(model):
        return descriptor_frame
    return descriptor_frame.to_numpy(dtype=float)


def prepare_model_input(model, descriptor_frame: pd.DataFrame, preprocessor=None):
    """Apply endpoint preprocessing exactly once before model prediction."""
    if preprocessor is None:
        return model_input(model, descriptor_frame)
    if hasattr(model, "steps") and len(getattr(model, "steps", [])) > 1:
        raise ModelLoadError(
            "Both an external preprocessor and a model Pipeline were configured; refusing to preprocess twice"
        )
    transformed = preprocessor.transform(descriptor_frame.to_numpy(dtype=float))
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=float)
    if not np.isfinite(transformed).all():
        raise ModelLoadError("Saved preprocessing produced NaN or infinite model inputs")
    return transformed


def transform_reference_space(
    model,
    descriptor_frame: pd.DataFrame,
    external_preprocessor=None,
) -> tuple[np.ndarray, object | None]:
    """Apply the configured external or Pipeline preprocessor to a reference space."""
    if external_preprocessor is not None:
        transformed = prepare_model_input(model, descriptor_frame, external_preprocessor)
        return transformed, external_preprocessor
    if hasattr(model, "steps") and len(getattr(model, "steps", [])) > 1:
        preprocessor = model[:-1]
        transformed = preprocessor.transform(descriptor_frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed, dtype=float), preprocessor
    return descriptor_frame.to_numpy(dtype=float), None
