"""One-at-a-time controls for author-defined annual comparator assumptions.

The hourly endpoint, capacity factors, PUE and activity sequence remain fixed.
These deterministic contrasts do not validate either endpoint against metered
energy and their envelope is not a statistical confidence interval.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import pandas as pd

from core.accounting_boundary import resolve_countries
from core.annual_activity import (
    AnnualActivityConfig, REFERENCE_TASK_RATIOS, REFERENCE_TASK_UTILIZATION,
    REFERENCE_IDLE_POWER_RATE, REFERENCE_MAX_POWER_RATE,
)
from core.annual_cpu_model import run_annual_cpu_model


REFERENCE_DEFAULTS = {
    "default_p_infer": REFERENCE_TASK_RATIOS["inference"],
    "default_p_other": REFERENCE_TASK_RATIOS["other"],
    "u_train": REFERENCE_TASK_UTILIZATION["training"],
    "u_infer": REFERENCE_TASK_UTILIZATION["inference"],
    "u_other": REFERENCE_TASK_UTILIZATION["other"],
    "idle_power_rate": REFERENCE_IDLE_POWER_RATE,
    "max_power_rate": REFERENCE_MAX_POWER_RATE,
}
DEFAULT_REFERENCE_LEVELS = {
    "default_p_infer": (0.60, 0.75, 0.90),
    "u_train": (0.64, 0.80, 0.96),
    "u_infer": (0.40, 0.50, 0.60),
    "u_other": (0.40, 0.50, 0.60),
}
KEYS = ["scenario", "policy", "year", "country"]
METRICS = ["installed_it_mw", "it_energy_mwh", "facility_energy_mwh", "carbon_tco2"]
INTERPRETATION = "author_assumption_stress_not_prediction_accuracy_or_confidence_interval"


def validate_reference_sensitivity(config=None):
    """Reject misspelled, nonphysical or baseline-free ranges before any I/O."""
    config = {} if config is None else config
    if not isinstance(config, Mapping) or set(config) - {"enabled", "levels"}:
        raise ValueError("Reference sensitivity accepts only enabled and levels.")
    enabled = config.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("Reference sensitivity enabled must be a boolean.")
    supplied = config.get("levels", {})
    if not isinstance(supplied, Mapping) or set(supplied) - set(DEFAULT_REFERENCE_LEVELS):
        raise ValueError("Unknown annual reference sensitivity parameter.")
    levels = {}
    for parameter, values in {**DEFAULT_REFERENCE_LEVELS, **supplied}.items():
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            raise ValueError(f"Reference levels for {parameter} require at least two values.")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in values):
            raise ValueError(f"Reference levels for {parameter} must be finite numbers.")
        upper = 1. - REFERENCE_DEFAULTS["default_p_other"] if parameter == "default_p_infer" else 1.
        if any(not 0. <= v <= upper for v in values):
            raise ValueError(f"Reference levels for {parameter} must be in [0, {upper}].")
        if len(set(values)) != len(values) or REFERENCE_DEFAULTS[parameter] not in values:
            raise ValueError(f"Reference levels for {parameter} require unique values including the baseline.")
        levels[parameter] = list(values)
    return {"enabled": enabled, "levels": levels}


def reference_cases(config=None):
    config = validate_reference_sensitivity(config)
    if not config["enabled"]:
        return []
    cases = [{"case": "baseline", "varied_parameter": "none", "parameter_value": np.nan,
              "parameter_level": "Base", **REFERENCE_DEFAULTS}]
    for parameter, values in config["levels"].items():
        for number, value in enumerate(values):
            if value == REFERENCE_DEFAULTS[parameter]:
                continue  # One shared baseline, rather than four duplicated runs.
            cases.append({"case": f"{parameter}_{number}", "varied_parameter": parameter,
                          "parameter_value": float(value),
                          "parameter_level": "Low" if value < REFERENCE_DEFAULTS[parameter] else "High",
                          **REFERENCE_DEFAULTS, parameter: float(value)})
    for case in cases:
        case.update(training_task_share=1. - case["default_p_infer"] - case["default_p_other"],
                    inference_task_share=case["default_p_infer"], other_task_share=case["default_p_other"],
                    unclassified_task_share=0., interpretation=INTERPRETATION,
                    range_basis="author_defined_stress_levels_not_empirical_parameter_bounds")
    return cases


def _percent(actual, reference):
    actual, reference = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
    return np.divide((actual - reference) * 100., reference,
                     out=np.full(actual.shape, np.nan), where=reference != 0.)


def annual_reference_controls(reference, *, countries=None, year_start=2026, years=5,
                              scenario="Base", policy="CP", activity_config=None,
                              ai_capacity_factors=None, pue_scale=1., config=None,
                              annual_runner=None):
    """Reuse a country/year hourly endpoint; run only the inexpensive annual model.

    A different coverage, capacity or PUE is a failed comparison, not a model
    effect. Input frames are never mutated. Missing/zero carbon denominators
    yield undefined percentage gaps (NaN), never infinite apparent accuracy.
    """
    cases = reference_cases(config)
    if not cases:
        return {}
    countries = resolve_countries(countries)
    if isinstance(years, bool) or not isinstance(years, int) or years <= 0:
        raise ValueError("Reference control years must be a positive integer.")
    if isinstance(year_start, bool) or not isinstance(year_start, int) or not 2025 <= year_start <= year_start + years - 1 <= 2030:
        raise ValueError("Reference control years must be within 2025–2030.")
    required = KEYS + METRICS
    if not isinstance(reference, pd.DataFrame) or not set(required).issubset(reference.columns):
        raise ValueError(f"Hourly reference requires country/year keys and {METRICS}.")
    requested_years = list(range(year_start, year_start + years))
    ref = reference.loc[(reference["scenario"] == scenario) & (reference["policy"] == policy)
                        & reference["year"].isin(requested_years)
                        & reference["country"].isin(countries), required].copy()
    expected = {(scenario, policy, year, country) for year in requested_years for country in countries}
    if ref.duplicated(KEYS).any() or set(ref[KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError("Hourly reference must contain exactly one row per requested country/year.")
    values = ref[METRICS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (ref[METRICS[:3]] <= 0).any().any():
        raise ValueError("Hourly reference requires finite nonnegative values and positive capacity/energy.")
    activity = activity_config or AnnualActivityConfig()
    activity.validate()
    runner = annual_runner or run_annual_cpu_model
    country_frames, checks = [], []
    for case in cases:
        output = runner(renewable_energy_policy=policy, scenarios=[scenario], countries=countries,
                        year_start=year_start, years=years, activity_config=activity,
                        ai_capacity_factors=ai_capacity_factors, pue_scale=pue_scale,
                        save_outputs=False, verbose=False,
                        **{name: case[name] for name in REFERENCE_DEFAULTS})["annual_summary"]
        pair = output[required].merge(ref, on=KEYS, how="outer", validate="one_to_one",
                                      suffixes=("_annual", "_reference"), indicator=True)
        if not pair.pop("_merge").eq("both").all():
            raise ValueError("Annual and hourly reference coverage differ.")
        annual_values = pair[[f"{name}_annual" for name in METRICS]].to_numpy(dtype=float)
        if not np.isfinite(annual_values).all() or (annual_values < 0).any():
            raise ValueError("Annual reference control produced invalid energy or emissions.")
        capacity_residual = pair["installed_it_mw_annual"] - pair["installed_it_mw_reference"]
        annual_pue = pair["facility_energy_mwh_annual"] / pair["it_energy_mwh_annual"]
        reference_pue = pair["facility_energy_mwh_reference"] / pair["it_energy_mwh_reference"]
        capacity_ok = np.allclose(pair["installed_it_mw_annual"], pair["installed_it_mw_reference"], rtol=1e-10, atol=1e-6)
        pue_ok = np.allclose(annual_pue, reference_pue, rtol=1e-10, atol=1e-10)
        if not capacity_ok or not pue_ok:
            raise ValueError("Annual reference controls require identical capacity factors and PUE at both endpoints.")
        checks.append({"case": case["case"], "country_year_count": len(pair),
                       "max_capacity_residual_mw": float(capacity_residual.abs().max()),
                       "max_pue_residual": float((annual_pue - reference_pue).abs().max()),
                       "hourly_reference_held_fixed": True, "capacity_factors_recalibrated": False,
                       "status": "PASS", "interpretation": INTERPRETATION})
        for name in ("case", "varied_parameter", "parameter_value", "parameter_level"):
            pair[name] = case[name]
        country_frames.append(pair)
    country = pd.concat(country_frames, ignore_index=True)
    summary = country.groupby(["case", "scenario", "policy", "year"], as_index=False, sort=False)[
        [f"{metric}_{endpoint}" for endpoint in ("annual", "reference") for metric in METRICS]
    ].sum()
    for frame in (country, summary):
        for name, metric in (("energy", "facility_energy_mwh"), ("carbon", "carbon_tco2")):
            frame[f"{name}_gap_pct_of_reference"] = _percent(frame[f"{metric}_annual"], frame[f"{metric}_reference"])
        keys = KEYS if "country" in frame.columns else KEYS[:-1]
        baseline = frame.loc[frame["case"] == "baseline", keys + ["facility_energy_mwh_annual", "carbon_tco2_annual"]]
        baseline = baseline.rename(columns={"facility_energy_mwh_annual": "baseline_energy_mwh",
                                            "carbon_tco2_annual": "baseline_carbon_tco2"})
        joined = frame.merge(baseline, on=keys, how="left", validate="many_to_one", sort=False)
        frame["annual_energy_change_pct_of_baseline"] = _percent(joined["facility_energy_mwh_annual"], joined["baseline_energy_mwh"])
        frame["annual_carbon_change_pct_of_baseline"] = _percent(joined["carbon_tco2_annual"], joined["baseline_carbon_tco2"])
        frame["interpretation"] = INTERPRETATION
        frame["activity_mode"] = activity.mode
        frame["hourly_reference_held_fixed"] = True
        frame["capacity_factors_recalibrated"] = False
    envelope = summary.groupby(["scenario", "policy", "year"], as_index=False).agg(
        minimum_energy_gap_pct=("energy_gap_pct_of_reference", "min"),
        maximum_energy_gap_pct=("energy_gap_pct_of_reference", "max"),
        minimum_carbon_gap_pct=("carbon_gap_pct_of_reference", "min"),
        maximum_carbon_gap_pct=("carbon_gap_pct_of_reference", "max"),
        case_count=("case", "nunique"))
    envelope["interpretation"] = "range_across_tested_OAT_cases_not_joint_uncertainty_or_confidence_interval"
    metadata = pd.DataFrame([{"reviewer_comments": "R1.01|R2.03", "scenario": scenario, "policy": policy,
        "year_start": year_start, "years": years, "country_count": len(countries),
        "case_count": len(cases), "activity_mode": activity.mode, "pue_scale": pue_scale,
        "trace_reads": 0, "hourly_model_runs": 0, "annual_model_runs": len(cases),
        "task_share_rule": "fixed_other_share_and_zero_unclassified; training=1-inference-other",
        "calibration_rule": "common_existing_factors_fixed_for_every_case_no_recalibration",
        "interpretation": INTERPRETATION}])
    return {"W04_Reference_Cases": pd.DataFrame(cases), "W04_Reference_Country": country,
            "W04_Reference_Summary": summary, "W04_Reference_Envelope": envelope,
            "W04_Reference_Checks": pd.DataFrame(checks), "W04_Reference_Metadata": metadata}
