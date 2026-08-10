"""Metadata loading helpers for MammalTox."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_COLUMNS = [
    "model_id",
    "species",
    "route",
    "endpoint",
    "model_name",
    "model_path",
    "selected_descriptor_source",
    "train_file",
    "test_file",
    "active",
]

PERFORMANCE_COLUMNS = [
    "model_id",
    "species",
    "route",
    "endpoint",
    "model_name",
    "selected_descriptor_count",
    "training_compound_count",
    "test_compound_count",
    "Training Accuracy",
    "Training Precision",
    "Training Recall",
    "Training F1",
    "Training MCC",
    "Test Accuracy",
    "Test Precision",
    "Test Recall",
    "Test F1",
    "Test MCC",
    "ROC-AUC",
]

DATASET_STAT_COLUMNS = [
    "species",
    "route",
    "endpoint",
    "total_compound_count",
    "source_total_compound_count",
    "total_class_0",
    "total_class_1",
    "training_compound_count",
    "test_compound_count",
    "train_class_0",
    "train_class_1",
    "test_class_0",
    "test_class_1",
    "notes",
]

NON_DESCRIPTOR_COLUMNS = {
    "taid",
    "id",
    "name",
    "iupac name",
    "pubchem cid",
    "canonical smiles",
    "smiles",
    "inchikey",
    "class",
    "pld50",
    "pld50 (molar kg unit)",
    "toxicity value(mg/kg)",
    "toxicity value(g/kg)",
    "toxicity value",
    "g/kg",
    "mw",
    "ld50 (mol/kg)",
    "mol/kg",
    "toxicity molar kg unit",
    "toxicity value (g/kg)",
    "toxicity value (mg/kg)",
    "",
}


def resolve_app_path(relative_path: str | Path, root: Path | None = None) -> Path:
    """Resolve a registry-relative path inside the application directory."""
    root = root or APP_ROOT
    path = Path(str(relative_path))
    if path.is_absolute():
        return path
    return root / path


def _read_csv(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(columns))
    return pd.read_csv(path, low_memory=False)


def _coerce_active(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def generate_registry_from_models(root: Path | None = None) -> pd.DataFrame:
    """Fallback registry scanner used when metadata/model_registry.csv is absent."""
    root = root or APP_ROOT
    rows = []
    for model_file in sorted((root / "models").glob("**/*.pkl")):
        species = model_file.parent.name.replace("_", " ").title()
        model_name = model_file.name.replace("_final_model.pkl", "")
        model_id = f"{model_file.parent.name}_{model_name.lower()}"
        rows.append(
            {
                "model_id": model_id,
                "species": species,
                "route": "Not available",
                "endpoint": "Acute toxicity LD50 classification",
                "model_name": model_name,
                "model_path": model_file.relative_to(root).as_posix(),
                "selected_descriptor_source": "",
                "train_file": "",
                "test_file": "",
                "active": True,
            }
        )
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def load_registry(root: Path | None = None) -> pd.DataFrame:
    """Load the model registry, or scan models as a fallback."""
    root = root or APP_ROOT
    path = root / "metadata" / "model_registry.csv"
    if not path.exists():
        return generate_registry_from_models(root)
    registry = pd.read_csv(path, low_memory=False)
    for col in REGISTRY_COLUMNS:
        if col not in registry.columns:
            registry[col] = "" if col != "active" else True
    registry["active"] = _coerce_active(registry["active"])
    return registry


def descriptor_columns_from_file(path: Path) -> list[str]:
    """Infer descriptor columns from a feature-selected training file."""
    if not path.exists():
        return []
    header = pd.read_csv(path, nrows=0).columns
    return [str(col) for col in header if str(col).strip().lower() not in NON_DESCRIPTOR_COLUMNS]


def load_selected_descriptors(root: Path | None = None) -> dict[str, list[str]]:
    """Return selected descriptor names keyed by model_id."""
    root = root or APP_ROOT
    selected_path = root / "metadata" / "selected_descriptors.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path, low_memory=False)
        if {"model_id", "descriptor_name"}.issubset(selected.columns):
            if "descriptor_order" in selected.columns:
                selected = selected.sort_values(["model_id", "descriptor_order"])
            return {
                model_id: group["descriptor_name"].astype(str).tolist()
                for model_id, group in selected.groupby("model_id", sort=False)
            }

    descriptors_by_model: dict[str, list[str]] = {}
    registry = load_registry(root)
    for _, row in registry.iterrows():
        source = str(row.get("selected_descriptor_source", "")).strip()
        if source:
            descriptors_by_model[str(row["model_id"])] = descriptor_columns_from_file(
                resolve_app_path(source, root)
            )
    return descriptors_by_model


def load_model_performance(root: Path | None = None) -> pd.DataFrame:
    root = root or APP_ROOT
    return _read_csv(root / "metadata" / "model_performance.csv", PERFORMANCE_COLUMNS)


def load_dataset_statistics(root: Path | None = None) -> pd.DataFrame:
    root = root or APP_ROOT
    return _read_csv(root / "metadata" / "dataset_statistics.csv", DATASET_STAT_COLUMNS)


def active_models(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return registry
    return registry[registry["active"]].copy()


def model_label(row: pd.Series) -> str:
    route = row.get("route", "Not available")
    return f"{row.get('species', 'Unknown')} | {row.get('model_name', 'Model')} | {route}"


def merge_model_information(registry: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame(columns=PERFORMANCE_COLUMNS)
    base = registry.copy()
    if performance.empty:
        for col in PERFORMANCE_COLUMNS:
            if col not in base.columns:
                base[col] = "Not available"
        return base
    merged = base.merge(
        performance.drop(columns=[c for c in ["species", "route", "endpoint", "model_name"] if c in performance.columns]),
        on="model_id",
        how="left",
    )
    for col in PERFORMANCE_COLUMNS:
        if col not in merged.columns:
            merged[col] = "Not available"
    return merged
