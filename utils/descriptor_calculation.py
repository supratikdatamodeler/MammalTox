"""RDKit validation and Mordred descriptor calculation."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd
import streamlit as st

try:
    from rdkit import Chem
    from mordred import Calculator, descriptors

    IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only when deps missing
    Chem = None
    Calculator = None
    descriptors = None
    IMPORT_ERROR = exc


class DescriptorCalculationError(RuntimeError):
    """Raised when descriptor calculation cannot run."""


class DescriptorSelectionError(RuntimeError):
    """Raised when required model descriptors are unavailable."""


def canonicalize_smiles(smiles: str) -> tuple[str | None, object | None, str | None]:
    """Validate and canonicalize a SMILES string."""
    if IMPORT_ERROR is not None:
        return None, None, f"RDKit/Mordred import failed: {IMPORT_ERROR}"
    value = "" if smiles is None else str(smiles).strip()
    if not value:
        return None, None, "Missing SMILES value"
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        return None, None, "Invalid SMILES"
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), mol, None


@lru_cache(maxsize=1)
def _calculator():
    if IMPORT_ERROR is not None:
        raise DescriptorCalculationError(f"RDKit/Mordred import failed: {IMPORT_ERROR}")
    return Calculator(descriptors, ignore_3D=True)


@lru_cache(maxsize=1)
def _official_descriptor_description_lookup() -> dict[str, str]:
    """Read descriptor definitions only from installed Mordred descriptor objects."""
    if IMPORT_ERROR is not None:
        return {}
    lookup: dict[str, str] = {}
    for descriptor in _calculator().descriptors:
        try:
            description = descriptor.description()
        except Exception:
            description = None
        lookup[str(descriptor)] = (
            str(description).strip()
            if description is not None and str(description).strip()
            else "Official description unavailable."
        )
    return lookup


def official_descriptor_descriptions(descriptor_names: Sequence[str]) -> list[str]:
    """Return official Mordred descriptions in the caller's exact descriptor order."""
    lookup = _official_descriptor_description_lookup()
    return [
        lookup.get(str(name), "Official description unavailable.")
        for name in descriptor_names
    ]


@st.cache_data(show_spinner=False, max_entries=32)
def _calculate_mordred_descriptors_cached(
    smiles_values: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate Mordred descriptors for a sequence of SMILES.

    Returns a descriptor dataframe indexed like the input sequence and a status
    dataframe with canonical SMILES and per-row validation errors.
    """
    statuses = []
    mols = []
    valid_indices = []
    for idx, smiles in enumerate(smiles_values):
        canonical, mol, error = canonicalize_smiles(smiles)
        statuses.append(
            {
                "input_index": idx,
                "Input SMILES": smiles,
                "Canonical SMILES": canonical,
                "descriptor_error": error,
            }
        )
        if mol is not None:
            valid_indices.append(idx)
            mols.append(mol)

    status_df = pd.DataFrame(statuses).set_index("input_index", drop=False)
    descriptor_df = pd.DataFrame(index=range(len(smiles_values)))
    if not mols:
        return descriptor_df, status_df

    try:
        calculated = _calculator().pandas(mols, quiet=True, nproc=1)
    except Exception as exc:
        status_df.loc[valid_indices, "descriptor_error"] = f"Mordred descriptor calculation failed: {exc}"
        return descriptor_df, status_df

    calculated.columns = [str(col) for col in calculated.columns]
    calculated.index = valid_indices
    descriptor_df = descriptor_df.join(calculated, how="left")
    return descriptor_df, status_df


def calculate_mordred_descriptors(smiles_values: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate and cache the Mordred descriptors used by deployed models."""
    values = tuple("" if value is None else str(value) for value in smiles_values)
    descriptors_frame, status_frame = _calculate_mordred_descriptors_cached(values)
    return descriptors_frame.copy(), status_frame.copy()


def select_required_descriptors(
    descriptor_df: pd.DataFrame, required_descriptors: Sequence[str]
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Select and numeric-clean model descriptors.

    Missing descriptor columns fail the entire model selection. Missing,
    infinite, or non-numeric values fail only the affected input rows.
    """
    required = [str(desc) for desc in required_descriptors]
    missing = [desc for desc in required if desc not in descriptor_df.columns]
    if missing:
        if len(missing) == 1:
            raise DescriptorSelectionError(f"Required descriptor {missing[0]} is missing")
        preview = ", ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f", and {len(missing) - 20} more"
        raise DescriptorSelectionError(f"Required Mordred descriptors are missing: {preview}{suffix}")

    selected = descriptor_df.loc[:, required].copy()
    numeric = selected.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    row_errors: dict[int, str] = {}
    missing_mask = numeric.isna()
    for idx, row in missing_mask.iterrows():
        if row.any():
            bad = row[row].index.astype(str).tolist()
            preview = ", ".join(bad[:12])
            suffix = "" if len(bad) <= 12 else f", and {len(bad) - 12} more"
            row_errors[int(idx)] = f"Descriptor values are NaN or infinite: {preview}{suffix}"
    return numeric, row_errors
