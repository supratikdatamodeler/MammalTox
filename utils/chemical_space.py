"""Endpoint-specific training space for PCA and applicability-domain analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .model_loading import (
    load_model_from_path,
    load_preprocessor_from_path,
    transform_reference_space,
)


@dataclass
class TrainingSpace:
    descriptors: tuple[str, ...]
    training_raw: pd.DataFrame
    training_scaled: np.ndarray
    classes: np.ndarray
    identifiers: list[str]
    scaler: StandardScaler | None
    preprocessor: object | None
    neighbors: NearestNeighbors
    neighbor_count: int
    ad_threshold: float
    pca: PCA
    training_pca: np.ndarray
    preprocessing_note: str


def _column_by_name(frame: pd.DataFrame, names: set[str]) -> str | None:
    for column in frame.columns:
        if str(column).strip().lower() in names:
            return str(column)
    return None


@st.cache_resource(show_spinner=False, max_entries=24)
def load_training_space(
    model_id: str,
    model_path: str,
    preprocessor_path: str,
    training_path: str,
    descriptors: tuple[str, ...],
    app_root: str,
) -> TrainingSpace:
    """Fit training-only scaling, kNN AD reference, and two-component PCA."""
    del model_id
    training = pd.read_csv(training_path, low_memory=False)
    missing = [feature for feature in descriptors if feature not in training.columns]
    if missing:
        raise ValueError(f"Training reference is missing required descriptors: {', '.join(missing[:12])}")

    class_column = _column_by_name(training, {"class", "label", "target"})
    if class_column is None:
        raise ValueError("Training reference does not contain a Class column")
    id_column = _column_by_name(training, {"taid", "id", "compound_id", "compound id", "name"})

    raw = training.loc[:, list(descriptors)].apply(pd.to_numeric, errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan)
    valid_mask = ~raw.isna().any(axis=1)
    raw = raw.loc[valid_mask].reset_index(drop=True)
    if len(raw) < 3:
        raise ValueError("Training reference has fewer than three complete descriptor rows")

    classes = pd.to_numeric(training.loc[valid_mask, class_column], errors="coerce").fillna(-1).astype(int).to_numpy()
    if id_column:
        identifiers = training.loc[valid_mask, id_column].astype(str).tolist()
    else:
        identifiers = [f"Training_{index + 1:05d}" for index in range(len(raw))]

    model = load_model_from_path(model_path, app_root)
    external_preprocessor = load_preprocessor_from_path(preprocessor_path, app_root)
    transformed, preprocessor = transform_reference_space(model, raw, external_preprocessor)
    if not np.isfinite(transformed).all():
        raise ValueError("Saved preprocessing produced NaN or infinite training values")

    if preprocessor is None:
        scaler: StandardScaler | None = StandardScaler()
        scaled = scaler.fit_transform(transformed)
        preprocessing_note = (
            "PCA and applicability-domain reference data use training-derived standardization; "
            "model prediction inputs remain unchanged."
        )
    else:
        scaler = None
        scaled = transformed
        preprocessing_note = (
            "PCA and applicability-domain reference data use the endpoint's training-derived preprocessing."
        )

    neighbor_count = min(5, len(scaled) - 1)
    training_neighbors = NearestNeighbors(n_neighbors=neighbor_count + 1).fit(scaled)
    training_distances = training_neighbors.kneighbors(scaled, return_distance=True)[0][:, 1:]
    mean_training_distances = training_distances.mean(axis=1)
    threshold = float(np.percentile(mean_training_distances, 95))
    query_neighbors = NearestNeighbors(n_neighbors=neighbor_count).fit(scaled)

    pca = PCA(n_components=2, random_state=42)
    training_pca = pca.fit_transform(scaled)
    return TrainingSpace(
        descriptors=descriptors,
        training_raw=raw,
        training_scaled=scaled,
        classes=classes,
        identifiers=identifiers,
        scaler=scaler,
        preprocessor=preprocessor,
        neighbors=query_neighbors,
        neighbor_count=neighbor_count,
        ad_threshold=threshold,
        pca=pca,
        training_pca=training_pca,
        preprocessing_note=preprocessing_note,
    )


def transform_queries(space: TrainingSpace, descriptor_frame: pd.DataFrame) -> np.ndarray:
    raw = descriptor_frame.loc[:, list(space.descriptors)]
    if space.preprocessor is not None:
        transformed = space.preprocessor.transform(raw.to_numpy(dtype=float))
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=float)
    else:
        transformed = raw.to_numpy(dtype=float)
    if space.scaler is not None:
        transformed = space.scaler.transform(transformed)
    if not np.isfinite(transformed).all():
        raise ValueError("Query preprocessing produced NaN or infinite values")
    return transformed


def applicability_statuses(space: TrainingSpace, descriptor_frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    transformed = transform_queries(space, descriptor_frame)
    distances = space.neighbors.kneighbors(transformed, return_distance=True)[0].mean(axis=1)
    labels = ["In AD" if distance <= space.ad_threshold else "Out of AD" for distance in distances]
    return labels, distances


def project_queries(space: TrainingSpace, descriptor_frame: pd.DataFrame) -> np.ndarray:
    return space.pca.transform(transform_queries(space, descriptor_frame))


def pca_artifact(space: TrainingSpace, query_frame: pd.DataFrame, query_ids: list[str]) -> dict:
    query_coordinates = project_queries(space, query_frame)
    return {
        "training_coordinates": space.training_pca.copy(),
        "training_classes": space.classes.copy(),
        "query_coordinates": query_coordinates,
        "query_ids": list(query_ids),
        "explained_variance": space.pca.explained_variance_ratio_.copy(),
        "ad_threshold": space.ad_threshold,
        "neighbor_count": space.neighbor_count,
        "preprocessing_note": space.preprocessing_note,
        "error": "",
    }
