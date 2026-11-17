"""Shared geographic boundary for the 24 major AI countries in this study."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import pandas as pd

from dataset.Installed_capacity_data import (
    DEFAULT_AI_CAPACITY_FACTORS, DEFAULT_COUNTRIES, IT_RATIO, STUDY_SCOPE_LABEL,
)


AGGREGATE_SCOPE = "selected_countries"
LEGACY_CALIBRATION = "legacy_global_reference_fitted_to_selected_country_sum"
PROVIDED_CALIBRATION = "provided_common_factors_see_calibration_record"


def resolve_countries(countries: Optional[Sequence[str]]) -> list[str]:
    selected = list(DEFAULT_COUNTRIES if countries is None else countries)
    if not selected:
        raise ValueError("countries must not be empty.")
    if len(selected) != len(set(selected)):
        raise ValueError("countries must contain unique country identifiers.")
    unknown = sorted(set(selected) - set(DEFAULT_COUNTRIES))
    if unknown:
        raise ValueError(f"Countries outside the study data boundary: {unknown}")
    shares = [float(IT_RATIO[country]) for country in selected]
    if any(not math.isfinite(share) or share < 0 for share in shares):
        raise ValueError("IT capacity shares must be finite and non-negative.")
    if not 0 < math.fsum(shares) <= 1:
        raise ValueError("Selected IT capacity shares must sum to a value in (0, 1].")
    return selected


def build_boundary_metadata(
    countries: Optional[Sequence[str]] = None,
    ai_capacity_factors: Optional[Mapping[int, float]] = None,
) -> pd.DataFrame:
    selected = resolve_countries(countries)
    coverage = math.fsum(float(IT_RATIO[country]) for country in selected)
    full_study = set(selected) == set(DEFAULT_COUNTRIES)
    return pd.DataFrame([{
        "accounting_scope": AGGREGATE_SCOPE,
        "scope_label": STUDY_SCOPE_LABEL if full_study else "subset of the 24 major AI study countries",
        "country_count": len(selected),
        "countries": "|".join(selected),
        "it_capacity_share_sum": coverage,
        "unmodelled_source_it_capacity_share": 1.0 - coverage,
        "capacity_shares_renormalized": False,
        "residual_region_modelled": False,
        "energy_coverage_fraction": None,
        "energy_coverage_status": "not_established_by_capacity_shares",
        "country_selection_basis": "availability_of_source_based_country_capacity_and_grid_data",
        "calibration_basis": LEGACY_CALIBRATION if ai_capacity_factors is None else PROVIDED_CALIBRATION,
    }])


def annotate_selected_aggregate(frame: pd.DataFrame, boundary: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("accounting_scope", "scope_label", "country_count", "it_capacity_share_sum"):
        result[column] = boundary.iloc[0][column]
    return result


def resolve_ai_capacity_factors(requested_years, overrides=None):
    source = DEFAULT_AI_CAPACITY_FACTORS if overrides is None else overrides
    factors = {year: float(source[year]) for year in requested_years}
    if any(not math.isfinite(value) or not 0 < value <= 1 for value in factors.values()):
        raise ValueError("AI capacity factors must be finite and in (0, 1].")
    return factors


def capacity_factor_table(year_start: int, years: int,
                          overrides: Optional[Mapping[int, float]] = None) -> pd.DataFrame:
    source = resolve_ai_capacity_factors(range(year_start, year_start + years), overrides)
    return pd.DataFrame([{"year": year, "ai_capacity_factor": float(source[year]),
                          "calibration_basis": LEGACY_CALIBRATION if overrides is None else PROVIDED_CALIBRATION}
                         for year in range(year_start, year_start + years)])


def load_ai_capacity_factors(
    path: Union[str, Path], countries: Optional[Sequence[str]] = None,
) -> dict[int, float]:
    """Load plain factors or a versioned boundary-analysis calibration record."""
    record = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(record, dict):
        raise ValueError("Capacity factor JSON must contain an object.")
    if "ai_capacity_factors" in record:
        if record.get("smoke_test_only", False):
            raise ValueError("Smoke-test calibration factors cannot be used for analysis results.")
        if record.get("accounting_scope") != AGGREGATE_SCOPE:
            raise ValueError("Calibration record must use the selected-country accounting scope.")
        if set(record.get("countries", [])) != set(resolve_countries(countries)):
            raise ValueError("Calibration and requested country boundaries do not match.")
        values = record["ai_capacity_factors"]
    else:
        values = record
    if not isinstance(values, dict) or not values:
        raise ValueError("Capacity factor mapping must not be empty.")
    factors = {}
    for year, value in values.items():
        if isinstance(value, bool):
            raise ValueError(f"Invalid capacity factor for {year}: {value}")
        factor = float(value)
        if not math.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError(f"Capacity factor for {year} must be finite and in (0, 1].")
        factors[int(year)] = factor
    if len(factors) != len(values):
        raise ValueError("Capacity factor years must be unique after integer conversion.")
    return factors
