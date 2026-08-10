"""Training-derived descriptor-space applicability-domain helpers."""

from __future__ import annotations

import pandas as pd

from .chemical_space import TrainingSpace, applicability_statuses


def applicability_domain_details(
    space: TrainingSpace, descriptor_row: pd.Series
) -> tuple[str, float, float]:
    """Return the existing AD label, query distance, and training-derived cutoff."""
    frame = descriptor_row.to_frame().T
    labels, distances = applicability_statuses(space, frame)
    return labels[0], float(distances[0]), float(space.ad_threshold)


def check_applicability_domain(space: TrainingSpace, descriptor_row: pd.Series) -> str:
    """Classify one query with the endpoint training-derived kNN threshold."""
    label, _, _ = applicability_domain_details(space, descriptor_row)
    return label
