"""Shared task taxonomy and task-ratio validation for M1--M4."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


TASK_TYPES = ("training", "inference", "other", "unclassified")

_PUBLIC_TASK_TYPE_ALIASES = {
    "training": "training",
    "train": "training",
    "inference": "inference",
    "online_inference": "inference",
    "offline_inference": "inference",
    "other": "other",
    "dev": "other",
    "development": "other",
    "unclassified": "unclassified",
    "unknown": "unclassified",
}


def _normalize_task_label(value: object) -> str:
    if pd.isna(value):
        return "unclassified"
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _PUBLIC_TASK_TYPE_ALIASES.get(label, "unclassified")


def task_type_ids(values: pd.Series) -> np.ndarray:
    """Map public trace labels to integer IDs in ``TASK_TYPES`` order."""
    id_by_type = {task_type: index for index, task_type in enumerate(TASK_TYPES)}
    return values.map(_normalize_task_label).map(id_by_type).to_numpy(dtype=np.int64)


def build_country_task_ratio_table(
    countries: Sequence[str],
    task_ratio_by_country: Optional[Mapping[str, Mapping[str, float]]] = None,
    infer_ratio_by_country: Optional[Mapping[str, float]] = None,
    default_p_infer: float = 0.75,
    default_p_other: float = 0.05,
) -> np.ndarray:
    """Build and validate country task shares in ``TASK_TYPES`` order."""
    countries = list(countries)
    country_set = set(countries)
    for name, value in {
        "default_p_infer": default_p_infer,
        "default_p_other": default_p_other,
    }.items():
        if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if default_p_infer + default_p_other > 1.0:
        raise ValueError("default_p_infer + default_p_other must not exceed 1.")

    ratios = np.tile(
        np.array(
            [1.0 - default_p_infer - default_p_other, default_p_infer, default_p_other, 0.0],
            dtype=float,
        ),
        (len(countries), 1),
    )
    country_index = {country: index for index, country in enumerate(countries)}

    if infer_ratio_by_country is not None:
        unknown = sorted(set(infer_ratio_by_country) - country_set)
        if unknown:
            raise ValueError(f"infer_ratio_by_country contains unknown countries: {unknown}")
        for country, value in infer_ratio_by_country.items():
            value = float(value)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0 - default_p_other:
                raise ValueError(
                    f"Inference ratio for '{country}' must be in [0, {1.0 - default_p_other}], "
                    f"got {value}"
                )
            row = country_index[country]
            ratios[row, TASK_TYPES.index("training")] = 1.0 - value - default_p_other
            ratios[row, TASK_TYPES.index("inference")] = value

    if task_ratio_by_country is not None:
        unknown = sorted(set(task_ratio_by_country) - country_set)
        if unknown:
            raise ValueError(f"task_ratio_by_country contains unknown countries: {unknown}")
        for country, task_ratios in task_ratio_by_country.items():
            unknown_tasks = sorted(set(task_ratios) - set(TASK_TYPES))
            if unknown_tasks:
                raise ValueError(
                    f"Task ratios for '{country}' contain unknown task types: {unknown_tasks}"
                )
            missing_tasks = [task_type for task_type in TASK_TYPES if task_type not in task_ratios]
            if missing_tasks:
                raise ValueError(
                    f"Task ratios for '{country}' must define all task types; missing: {missing_tasks}"
                )
            values = np.array([float(task_ratios[task_type]) for task_type in TASK_TYPES])
            if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError(f"Task ratios for '{country}' must all be finite and in [0, 1].")
            if not np.isclose(values.sum(), 1.0, atol=1e-8):
                raise ValueError(
                    f"Task ratios for '{country}' must sum to 1, got {values.sum():.8f}."
                )
            ratios[country_index[country]] = values

    return ratios
