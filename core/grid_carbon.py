"""Shared validation and construction rules for stylized hourly grid inputs.

g CO2eq/kWh and kg CO2eq/MWh have the same numerical value. The annual
Factors.py table is a DIRECT-emissions reference only. No life-cycle fallback
or life-cycle annual closure is inferred from that table.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.Factors import get_carbon_factor

COUNTRY_DIRS = {"United_Kingdom": "Great Britain"}
SCOPES = ("direct", "life_cycle")
CI_COLUMNS = {
    "direct": "Carbon intensity gCO2eq/kWh (direct)",
    "life_cycle": "Carbon intensity gCO2eq/kWh (Life cycle)",
}
MEAN_ATOL = 1e-6


def calendar_hours(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", inclusive="left", freq="h", tz="UTC")


def calendar_status(timestamps: pd.DatetimeIndex, year: int) -> str:
    expected = calendar_hours(year)
    if timestamps.equals(expected):
        return "full_calendar"
    if len(expected) == 8784 and timestamps.equals(expected[:8760]):
        return "legacy_8760_truncated_leap_year"
    return "invalid_calendar"


def intensity_column(columns, scope: str) -> str:
    if scope not in SCOPES:
        raise ValueError(f"Carbon scope must be one of {SCOPES}.")
    token = "life cycle" if scope == "life_cycle" else "direct"
    matches = [c for c in columns if "carbon intensity" in c.lower()
               and re.search(r"\b" + token + r"\b", c.lower())]
    if len(matches) != 1:
        raise ValueError(f"Expected one carbon intensity column for {scope}, found {matches}.")
    column = matches[0]
    unit = column.lower().replace("₂", "2")
    if not re.search(r"(?<![a-z])(?:g\s*co2(?:eq|e)?\s*/\s*kwh|kg\s*co2(?:eq|e)?\s*/\s*mwh)(?![a-z])", unit):
        raise ValueError(f"Unsupported or missing carbon-intensity units: {column}")
    return column


def find_hourly_file(directory, country: str, policy: str, year: int):
    folder = Path(directory) / COUNTRY_DIRS.get(country, country)
    matches = sorted(folder.glob(f"*-{policy}-{year}-hourly.csv"))
    if len(matches) > 1:
        raise ValueError(f"Expected one hourly carbon file for {country}/{policy}/{year}, got {len(matches)}.")
    return matches[0] if matches else None


def read_hourly_file(path, year: int, scope: str, *, allow_legacy: bool = False, frame=None):
    # Shared experiments can parse and hash the same byte buffer only once.
    frame = pd.read_csv(path) if frame is None else frame
    columns = [c for c in frame if "datetime" in c.lower()]
    if len(columns) != 1:
        raise ValueError(f"Expected one datetime column: {path}")
    time_column = columns[0]
    # Naive timestamps are accepted only when the column explicitly declares UTC.
    text = frame[time_column].astype(str)
    if "utc" not in time_column.lower() and not text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all():
        raise ValueError(f"Timestamps must declare UTC or an explicit offset: {path}")
    timestamps = pd.DatetimeIndex(pd.to_datetime(text, utc=True, errors="coerce"))
    if timestamps.isna().any() or timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError(f"Invalid, duplicate or unordered timestamps: {path}")
    status = calendar_status(timestamps, year)
    if status == "invalid_calendar" or (status != "full_calendar" and not allow_legacy):
        raise ValueError(f"Incomplete UTC calendar for {year} ({status}): {path}")
    values = pd.to_numeric(frame[intensity_column(frame.columns, scope)], errors="coerce").to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"Carbon factors must be finite, non-negative and nonempty: {path}")
    return timestamps, values


def validate_direct_mean(values, country: str, policy: str, year: int) -> None:
    target = get_carbon_factor(policy, country, year)
    if not np.isclose(np.mean(values), target, rtol=1e-9, atol=MEAN_ATOL):
        raise ValueError(f"Direct hourly mean does not close to annual input for {country}/{policy}/{year}: "
                         f"mean={np.mean(values):.12g}, target={target:.12g}.")


def transform_shape(values, *, target_mean=None, variability_scale=1.0, phase_hours=0):
    """Stress-test a shape; clip negatives, then restore its specified mean."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Shape must be a finite non-negative one-dimensional series.")
    if not np.isfinite(variability_scale) or variability_scale < 0:
        raise ValueError("Variability scale must be finite and non-negative.")
    if not isinstance(phase_hours, (int, np.integer)):
        raise ValueError("Phase shift must be an integer number of UTC hours.")
    mean = float(values.mean())
    target = mean if target_mean is None else float(target_mean)
    if not np.isfinite(target) or target < 0:
        raise ValueError("Target mean must be finite and non-negative.")
    if target == 0:
        return np.zeros_like(values)
    if mean <= 0:
        raise ValueError("A zero template cannot be normalized to a positive target.")
    transformed = np.maximum(mean + variability_scale * (np.roll(values, phase_hours) - mean), 0)
    return transformed * (target / transformed.mean())


def map_template_to_year(timestamps, values, year: int):
    """Match UTC month/day/hour; use February 28 for an absent February 29."""
    target = calendar_hours(year)
    keys = list(zip(timestamps.month, timestamps.day, timestamps.hour))
    if len(set(keys)) != len(keys):
        raise ValueError("Template must contain unique month/day/hour keys from one year.")
    lookup = dict(zip(keys, np.asarray(values, dtype=float)))
    output = []
    for month, day, hour in zip(target.month, target.day, target.hour):
        key = (month, day, hour)
        if key not in lookup and month == 2 and day == 29:
            key = (2, 28, hour)
        if key not in lookup:
            raise ValueError(f"Template is missing UTC calendar key {key}.")
        output.append(lookup[key])
    return target, np.asarray(output)


def matching_diagnostics(energy, factors, annual_energy: float) -> dict:
    """Separate mean-CI emissions from load-CI covariance at fixed electricity."""
    energy, factors = np.asarray(energy, dtype=float), np.asarray(factors, dtype=float)
    if energy.ndim != 1 or energy.shape != factors.shape or not energy.size:
        raise ValueError("Energy and CI must be nonempty aligned hourly vectors.")
    mean = float(factors.mean())
    emissions = float(np.dot(energy, factors) / 1000)
    covariance = float(np.mean((energy - energy.mean()) * (factors - mean)))
    baseline = float(annual_energy * mean / 1000)
    matching = emissions - baseline
    return {
        "hourly_mean_ci_kg_per_mwh": mean,
        "load_weighted_ci_kg_per_mwh": emissions * 1000 / annual_energy if annual_energy else np.nan,
        "hourly_energy_sum_mwh": float(energy.sum()),
        "annual_energy_mwh": float(annual_energy),
        "energy_closure_residual_mwh": float(energy.sum() - annual_energy),
        "mean_ci_carbon_tco2": baseline,
        "matched_carbon_tco2": emissions,
        "load_ci_covariance": covariance,
        "hourly_matching_effect_tco2": matching,
        "covariance_identity_residual_tco2": matching - len(energy) * covariance / 1000,
    }
