"""Generate the calculation results required by Figure 3.

Figure 3 compares the conventional annual CPU baseline with the trace-driven,
component-level hourly workload-resolved model. An additive bridge identifies
how the result changes as the annual representation is progressively replaced
by the hourly representation:

1. component power configuration under the legacy task/load assumptions;
2. trace-derived mean workload instead of fixed task/load assumptions;
3. hourly workload variability instead of a flat trace-derived workload; and
4. hourly carbon matching instead of annual-average carbon factors.

The four signed effects are calculated in the annual-to-hourly direction and
must sum to the direct hourly-minus-annual difference. All calculations are fixed to Base-CP
for 2026--2030 and the default 24-country boundary.  The workbook also reports
the peak/ramp information that an annual model cannot represent.  The script
contains no plotting code and does not export intermediate calculation or
validation tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.annual_cpu_model import run_annual_cpu_model
from core.hourly_workload_resolved_model import (
    RESOURCES,
    WorkloadProfile,
    build_workload_profile,
    run_workload_component_footprint,
)
from core.task_model import TASK_TYPES
from dataset.Factors import get_carbon_factor
from dataset.Installed_capacity_data import DEFAULT_COUNTRIES


DEFAULT_OUTPUT = ROOT_DIR / "results" / "figure3_data.xlsx"
FIGURE_SCENARIO = "Base"
FIGURE_POLICY = "CP"
FIGURE_YEAR_START = 2026
FIGURE_YEAR_END = 2030
FIGURE_YEARS = FIGURE_YEAR_END - FIGURE_YEAR_START + 1
DETAIL_YEAR = 2030
POLICIES = (FIGURE_POLICY,)
MODEL_ORDER = ("annual_cpu_model", "hourly_workload_resolved_model")

ANNUAL_BASELINE_VARIANT = "annual_cpu_baseline"
POWER_CONFIGURATION_VARIANT = "component_power_with_legacy_load"
MEAN_WORKLOAD_VARIANT = "flat_trace_annual_carbon"
HOURLY_WORKLOAD_VARIANT = "hourly_trace_annual_carbon"
HOURLY_REFERENCE_VARIANT = "hourly_trace_hourly_carbon"
VARIANT_ORDER = (
    ANNUAL_BASELINE_VARIANT,
    POWER_CONFIGURATION_VARIANT,
    MEAN_WORKLOAD_VARIANT,
    HOURLY_WORKLOAD_VARIANT,
    HOURLY_REFERENCE_VARIANT,
)

EFFECT_STEPS = (
    (
        "power_configuration",
        ANNUAL_BASELINE_VARIANT,
        POWER_CONFIGURATION_VARIANT,
        (
            "Replace the legacy aggregate utilization-power equation with the "
            "component power configuration while retaining the annual CPU "
            "model's task shares "
            "and fixed utilization assumptions."
        ),
    ),
    (
        "mean_workload_representation",
        POWER_CONFIGURATION_VARIANT,
        MEAN_WORKLOAD_VARIANT,
        (
            "Replace the annual CPU model's task shares and fixed utilization assumptions with the "
            "trace-derived mean task-resource workload; keep the workload flat "
            "over time and retain annual-average carbon factors."
        ),
    ),
    (
        "load_temporal_variability",
        MEAN_WORKLOAD_VARIANT,
        HOURLY_WORKLOAD_VARIANT,
        (
            "Replace the flat trace-derived workload with the observed hourly "
            "trace shape while retaining annual-average carbon factors."
        ),
    ),
    (
        "hourly_carbon_matching",
        HOURLY_WORKLOAD_VARIANT,
        HOURLY_REFERENCE_VARIANT,
        (
            "Replace annual-average carbon factors with hourly carbon factors "
            "while retaining the full trace-driven energy profile."
        ),
    ),
)

COUNTRY_ISO3 = {
    "USA": "USA",
    "China": "CHN",
    "Japan": "JPN",
    "France": "FRA",
    "India": "IND",
    "Singapore": "SGP",
    "Canada": "CAN",
    "Germany": "DEU",
    "United_Kingdom": "GBR",
    "Australia": "AUS",
    "Italy": "ITA",
    "South_Korea": "KOR",
    "South_Africa": "ZAF",
    "Ireland": "IRL",
    "UAE": "ARE",
    "Brazil": "BRA",
    "Israel": "ISR",
    "Netherlands": "NLD",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Belgium": "BEL",
    "Norway": "NOR",
    "Poland": "POL",
    "Switzerland": "CHE",
}

DATA_YEAR_START = 2025
LEGACY_TASK_RATIOS = {
    "training": 0.20,
    "inference": 0.75,
    "other": 0.05,
    "unclassified": 0.0,
}
LEGACY_TASK_UTILIZATION = {
    "training": 0.80,
    "inference": 0.50,
    "other": 0.50,
    "unclassified": 0.50,
}
LEGACY_TRAINING_ACTIVITY_START = 0.90
LEGACY_TRAINING_ACTIVITY_TARGET = 0.925
LEGACY_INFERENCE_ACTIVITY_START = 0.50
LEGACY_INFERENCE_ACTIVITY_TARGET = 0.70

RESULT_COLUMNS = (
    "variant",
    "model",
    "scenario",
    "policy",
    "year",
    "country",
    "facility_energy_mwh",
    "power_twh",
    "load_weighted_carbon_factor_kg_per_mwh",
    "carbon_tco2",
    "carbon_mtco2",
)


def _relative_difference(
    values: Union[pd.Series, np.ndarray],
    reference: Union[pd.Series, np.ndarray],
) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    return np.divide(
        (values_array - reference_array) * 100.0,
        reference_array,
        out=np.full(values_array.shape, np.nan, dtype=float),
        where=reference_array != 0,
    )


def _recalculate_summary_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["power_twh"] = result["facility_energy_mwh"] / 1e6
    result["carbon_mtco2"] = result["carbon_tco2"] / 1e6
    result["load_weighted_carbon_factor_kg_per_mwh"] = np.divide(
        result["carbon_tco2"].to_numpy(dtype=float) * 1000.0,
        result["facility_energy_mwh"].to_numpy(dtype=float),
        out=np.full(len(result), np.nan),
        where=result["facility_energy_mwh"].to_numpy(dtype=float) != 0,
    )
    return result


def _standardize_result(
    frame: pd.DataFrame,
    *,
    variant: str,
    model: str,
    policy: Optional[str] = None,
) -> pd.DataFrame:
    result = frame.copy()
    if policy is not None:
        result["policy"] = policy
    if "policy" not in result:
        raise ValueError("A policy column or explicit policy is required.")
    result["variant"] = variant
    result["model"] = model
    result = _recalculate_summary_metrics(result)
    return result[list(RESULT_COLUMNS)]


def _derive_annual_carbon_variant(
    energy_frame: pd.DataFrame,
    *,
    variant: str,
    policy: str,
) -> pd.DataFrame:
    """Apply annual factors to an existing energy result without changing energy."""
    result = energy_frame.copy()
    result["policy"] = policy
    result["carbon_tco2"] = (
        result["facility_energy_mwh"].to_numpy(dtype=float)
        * np.array(
            [
                get_carbon_factor(policy, country, int(year))
                for country, year in zip(result["country"], result["year"])
            ],
            dtype=float,
        )
        / 1000.0
    )
    return _standardize_result(
        result,
        variant=variant,
        model="hourly_workload_resolved_model",
        policy=policy,
    )


def _copy_profile_with_load(
    profile: WorkloadProfile,
    load: np.ndarray,
) -> WorkloadProfile:
    load_array = np.asarray(load, dtype=np.float64)
    if load_array.shape != profile.load.shape:
        raise ValueError(
            "Replacement workload must match the source profile shape: "
            f"expected {profile.load.shape}, got {load_array.shape}."
        )
    if np.any(~np.isfinite(load_array)) or np.any(load_array < 0):
        raise ValueError("Replacement workload must be finite and non-negative.")
    return WorkloadProfile(
        interval_index=profile.interval_index.copy(),
        interval_hours=float(profile.interval_hours),
        load=load_array.copy(),
        trace_capacity=np.asarray(profile.trace_capacity, dtype=float).copy(),
        task_counts=np.asarray(profile.task_counts).copy(),
        task_type_summary=profile.task_type_summary.copy(),
        relative_time=bool(profile.relative_time),
        trace_capacity_source=tuple(profile.trace_capacity_source),
    )


def _flat_workload_profile(profile: WorkloadProfile) -> WorkloadProfile:
    """Flatten each task-resource series while preserving its total load."""
    mean_load = profile.load.mean(axis=2, keepdims=True)
    flat_load = np.broadcast_to(mean_load, profile.load.shape).copy()
    if not np.allclose(
        flat_load.sum(axis=2),
        profile.load.sum(axis=2),
        rtol=1e-12,
        atol=1e-9,
    ):
        raise AssertionError("Flat workload does not preserve task-resource totals.")
    return _copy_profile_with_load(profile, flat_load)


def _legacy_activity(year: int, model_year_end: int) -> np.ndarray:
    """Return the annual CPU model's task activity coefficients for one year."""
    native_year_count = model_year_end - DATA_YEAR_START + 1
    year_index = year - DATA_YEAR_START
    if native_year_count <= 0 or not 0 <= year_index < native_year_count:
        raise ValueError(
            f"Legacy activity year {year} must be within "
            f"{DATA_YEAR_START}-{model_year_end}."
        )
    training = LEGACY_TRAINING_ACTIVITY_START + (
        LEGACY_TRAINING_ACTIVITY_TARGET - LEGACY_TRAINING_ACTIVITY_START
    ) / native_year_count * year_index
    inference = LEGACY_INFERENCE_ACTIVITY_START + (
        LEGACY_INFERENCE_ACTIVITY_TARGET - LEGACY_INFERENCE_ACTIVITY_START
    ) / native_year_count * year_index
    return np.array(
        [
            training if task == "training" else inference
            for task in TASK_TYPES
        ],
        dtype=float,
    )


def _legacy_assumption_profile(
    profile: WorkloadProfile,
    *,
    year: int,
    model_year_end: int,
) -> WorkloadProfile:
    """Build a flat profile with the annual CPU model's task/load assumptions.

    The normalized load for every resource equals the annual-model task ratio multiplied
    by its annual activity and assumed utilization.  Passing this profile
    through the component model changes the power layer while retaining the annual
    task and utilization assumptions.
    """
    ratios = np.array([LEGACY_TASK_RATIOS[task] for task in TASK_TYPES], dtype=float)
    utilization = np.array(
        [LEGACY_TASK_UTILIZATION[task] for task in TASK_TYPES], dtype=float
    )
    normalized_task_load = ratios * utilization * _legacy_activity(
        year, model_year_end
    )
    load = (
        normalized_task_load[:, None, None]
        * np.asarray(profile.trace_capacity, dtype=float)[None, :, None]
        * np.ones((1, len(RESOURCES), profile.n_intervals), dtype=float)
    )
    return _copy_profile_with_load(profile, load)


def _run_hourly_variant(
    *,
    variant: str,
    policy: str,
    scenario: str,
    countries: Sequence[str],
    year_start: int,
    years: int,
    workload_profile_path: Union[str, Path],
    server_profile_path: Optional[Union[str, Path]],
    hourly_carbon_factors_dir: Optional[Union[str, Path]],
    hourly_carbon_scope: str,
    hourly_carbon_fallback_to_annual: bool,
    capacity_quantile: float,
    max_resource_utilization: float,
    max_intervals: Optional[int],
    profile: WorkloadProfile,
    verbose: bool,
    return_hourly: bool = False,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    result = run_workload_component_footprint(
        renewable_energy_policy=policy,
        scenarios=[scenario],
        years=years,
        countries=countries,
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        year_start=year_start,
        save_outputs=False,
        verbose=verbose,
        capacity_quantile=capacity_quantile,
        max_resource_utilization=max_resource_utilization,
        hourly_carbon_factors_dir=hourly_carbon_factors_dir,
        hourly_carbon_scope=hourly_carbon_scope,
        hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
        save_hourly_outputs=False,
        max_intervals=max_intervals,
        workload_profile=profile,
    )
    annual = _standardize_result(
        result["annual_summary"],
        variant=variant,
        model="hourly_workload_resolved_model",
        policy=policy,
    )
    if not return_hourly:
        return annual

    hourly = result["hourly_carbon"][[
        "scenario",
        "year",
        "country",
        "timestamp_utc",
        "facility_energy_mwh",
    ]].copy()
    if hourly.empty:
        raise AssertionError("The requested hourly energy table is empty.")
    return annual, hourly


def _validate_variant_coverage(
    results: pd.DataFrame,
    *,
    scenarios: Sequence[str],
    policies: Sequence[str],
    countries: Sequence[str],
    year_start: int,
    years: int,
) -> None:
    keys = ["variant", "scenario", "policy", "year", "country"]
    duplicates = results.duplicated(keys, keep=False)
    if duplicates.any():
        sample = results.loc[duplicates, keys].head().to_dict("records")
        raise AssertionError(f"Duplicate Figure 3 variant results: {sample}")

    expected_rows = (
        len(VARIANT_ORDER)
        * len(scenarios)
        * len(policies)
        * len(countries)
        * years
    )
    if len(results) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} Figure 3 variant rows, got {len(results)}."
        )
    expected_years = set(range(year_start, year_start + years))
    if set(results["year"].astype(int)) != expected_years:
        raise AssertionError("Figure 3 variants do not cover all requested years.")


def _collect_variant_results(
    *,
    scenarios: Sequence[str],
    countries: Sequence[str],
    year_start: int,
    years: int,
    workload_profile_path: Union[str, Path],
    server_profile_path: Optional[Union[str, Path]],
    hourly_carbon_factors_dir: Union[str, Path],
    hourly_carbon_scope: str,
    hourly_carbon_fallback_to_annual: bool,
    capacity_quantile: float,
    max_resource_utilization: float,
    max_intervals: Optional[int],
    profile: WorkloadProfile,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_frames: list[pd.DataFrame] = []
    base_cp_hourly: Optional[pd.DataFrame] = None
    model_year_end = year_start + years - 1

    for policy in POLICIES:
        if verbose:
            print(f"[figure3] Calculating annual CPU baseline for {policy}.", flush=True)
        annual_cpu_result = run_annual_cpu_model(
            renewable_energy_policy=policy,
            scenarios=scenarios,
            years=years,
            countries=countries,
            year_start=year_start,
            save_outputs=False,
            verbose=False,
        )
        all_frames.append(
            _standardize_result(
                annual_cpu_result["annual_summary"],
                variant=ANNUAL_BASELINE_VARIANT,
                model="annual_cpu_model",
            )
        )

    flat_profile = _flat_workload_profile(profile)
    flat_energy_frames = []
    legacy_energy_frames = []

    for scenario in scenarios:
        if verbose:
            print(
                f"[figure3] Calculating flat trace workload for {scenario}.",
                flush=True,
            )
        flat_energy_frames.append(
            _run_hourly_variant(
                variant=MEAN_WORKLOAD_VARIANT,
                policy=FIGURE_POLICY,
                scenario=scenario,
                countries=countries,
                year_start=year_start,
                years=years,
                workload_profile_path=workload_profile_path,
                server_profile_path=server_profile_path,
                hourly_carbon_factors_dir=None,
                hourly_carbon_scope=hourly_carbon_scope,
                hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
                capacity_quantile=capacity_quantile,
                max_resource_utilization=max_resource_utilization,
                max_intervals=max_intervals,
                profile=flat_profile,
                verbose=False,
            )
        )

        for year in range(year_start, year_start + years):
            if verbose:
                print(
                    f"[figure3] Calculating component power with annual-model load for "
                    f"{scenario} x {year}.",
                    flush=True,
                )
            legacy_profile = _legacy_assumption_profile(
                profile,
                year=year,
                model_year_end=model_year_end,
            )
            legacy_energy_frames.append(
                _run_hourly_variant(
                    variant=POWER_CONFIGURATION_VARIANT,
                    policy=FIGURE_POLICY,
                    scenario=scenario,
                    countries=countries,
                    year_start=year,
                    years=1,
                    workload_profile_path=workload_profile_path,
                    server_profile_path=server_profile_path,
                    hourly_carbon_factors_dir=None,
                    hourly_carbon_scope=hourly_carbon_scope,
                    hourly_carbon_fallback_to_annual=(
                        hourly_carbon_fallback_to_annual
                    ),
                    capacity_quantile=capacity_quantile,
                    max_resource_utilization=max_resource_utilization,
                    max_intervals=max_intervals,
                    profile=legacy_profile,
                    verbose=False,
                )
            )

    flat_energy = pd.concat(flat_energy_frames, ignore_index=True)
    legacy_energy = pd.concat(legacy_energy_frames, ignore_index=True)
    for policy in POLICIES:
        all_frames.append(
            _derive_annual_carbon_variant(
                legacy_energy,
                variant=POWER_CONFIGURATION_VARIANT,
                policy=policy,
            )
        )
        all_frames.append(
            _derive_annual_carbon_variant(
                flat_energy,
                variant=MEAN_WORKLOAD_VARIANT,
                policy=policy,
            )
        )

    for policy in POLICIES:
        for scenario in scenarios:
            if verbose:
                print(
                    f"[figure3] Calculating the hourly reference for {scenario} x {policy}.",
                    flush=True,
                )
            capture_hourly = (
                policy == FIGURE_POLICY and scenario == FIGURE_SCENARIO
            )
            hourly_output = _run_hourly_variant(
                variant=HOURLY_REFERENCE_VARIANT,
                policy=policy,
                scenario=scenario,
                countries=countries,
                year_start=year_start,
                years=years,
                workload_profile_path=workload_profile_path,
                server_profile_path=server_profile_path,
                hourly_carbon_factors_dir=hourly_carbon_factors_dir,
                hourly_carbon_scope=hourly_carbon_scope,
                hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
                capacity_quantile=capacity_quantile,
                max_resource_utilization=max_resource_utilization,
                max_intervals=max_intervals,
                profile=profile,
                verbose=False,
                return_hourly=capture_hourly,
            )
            if capture_hourly:
                if not isinstance(hourly_output, tuple):
                    raise AssertionError("Base-CP hourly output was not captured.")
                hourly_reference, base_cp_hourly = hourly_output
            else:
                if isinstance(hourly_output, tuple):
                    raise AssertionError("Unexpected hourly output for non-Base-CP run.")
                hourly_reference = hourly_output
            all_frames.append(
                _derive_annual_carbon_variant(
                    hourly_reference,
                    variant=HOURLY_WORKLOAD_VARIANT,
                    policy=policy,
                )
            )
            all_frames.append(hourly_reference)

    results = pd.concat(all_frames, ignore_index=True)
    _validate_variant_coverage(
        results,
        scenarios=scenarios,
        policies=POLICIES,
        countries=countries,
        year_start=year_start,
        years=years,
    )
    variant_order = pd.CategoricalDtype(VARIANT_ORDER, ordered=True)
    results["variant"] = results["variant"].astype(variant_order)
    if base_cp_hourly is None:
        raise AssertionError("Base-CP is required for Figure 3 hourly diagnostics.")
    sorted_results = results.sort_values(
        ["scenario", "policy", "year", "country", "variant"],
        ignore_index=True,
    )
    return sorted_results, base_cp_hourly


def _global_model_comparison(variant_results: pd.DataFrame) -> pd.DataFrame:
    selected = variant_results[
        variant_results["variant"].isin(
            (ANNUAL_BASELINE_VARIANT, HOURLY_REFERENCE_VARIANT)
        )
    ]
    keys = ["scenario", "policy", "year"]
    global_results = selected.groupby(
        ["variant", "model", *keys], as_index=False, observed=True
    )[["facility_energy_mwh", "carbon_tco2"]].sum()
    global_results = _recalculate_summary_metrics(global_results)

    reference = global_results.loc[
        global_results["variant"] == HOURLY_REFERENCE_VARIANT,
        keys + ["facility_energy_mwh", "carbon_tco2"],
    ].rename(
        columns={
            "facility_energy_mwh": "hourly_reference_facility_energy_mwh",
            "carbon_tco2": "hourly_reference_carbon_tco2",
        }
    )
    result = global_results.merge(reference, on=keys, validate="many_to_one")
    result["energy_difference_vs_hourly_reference_mwh"] = (
        result["facility_energy_mwh"]
        - result["hourly_reference_facility_energy_mwh"]
    )
    result["energy_difference_vs_hourly_reference_pct"] = _relative_difference(
        result["facility_energy_mwh"],
        result["hourly_reference_facility_energy_mwh"],
    )
    result["carbon_difference_vs_hourly_reference_tco2"] = (
        result["carbon_tco2"] - result["hourly_reference_carbon_tco2"]
    )
    result["carbon_difference_vs_hourly_reference_pct"] = _relative_difference(
        result["carbon_tco2"], result["hourly_reference_carbon_tco2"]
    )
    model_order = pd.CategoricalDtype(MODEL_ORDER, ordered=True)
    result["model"] = result["model"].astype(model_order)
    return result.sort_values(
        ["scenario", "policy", "year", "model"], ignore_index=True
    )


def _country_effect_decomposition(variant_results: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "policy", "year", "country"]
    energy = variant_results.pivot(
        index=keys, columns="variant", values="facility_energy_mwh"
    )
    carbon = variant_results.pivot(
        index=keys, columns="variant", values="carbon_tco2"
    )
    missing = [variant for variant in VARIANT_ORDER if variant not in energy.columns]
    if missing:
        raise AssertionError(f"Missing bridge variants: {missing}")

    records = []
    key_frame = energy.index.to_frame(index=False)
    for order, (effect, source, target, definition) in enumerate(EFFECT_STEPS, start=1):
        frame = key_frame.copy()
        frame["effect_order"] = order
        frame["effect"] = effect
        frame["source_variant"] = source
        frame["target_variant"] = target
        frame["effect_definition"] = definition
        frame["source_facility_energy_mwh"] = energy[source].to_numpy(dtype=float)
        frame["target_facility_energy_mwh"] = energy[target].to_numpy(dtype=float)
        frame["source_carbon_tco2"] = carbon[source].to_numpy(dtype=float)
        frame["target_carbon_tco2"] = carbon[target].to_numpy(dtype=float)
        frame["energy_effect_mwh"] = (
            frame["target_facility_energy_mwh"]
            - frame["source_facility_energy_mwh"]
        )
        frame["carbon_effect_tco2"] = (
            frame["target_carbon_tco2"] - frame["source_carbon_tco2"]
        )
        frame["hourly_reference_facility_energy_mwh"] = energy[
            HOURLY_REFERENCE_VARIANT
        ].to_numpy(dtype=float)
        frame["hourly_reference_carbon_tco2"] = carbon[
            HOURLY_REFERENCE_VARIANT
        ].to_numpy(dtype=float)
        frame["annual_baseline_facility_energy_mwh"] = energy[
            ANNUAL_BASELINE_VARIANT
        ].to_numpy(dtype=float)
        frame["annual_baseline_carbon_tco2"] = carbon[
            ANNUAL_BASELINE_VARIANT
        ].to_numpy(dtype=float)
        frame["total_hourly_minus_annual_energy_mwh"] = (
            frame["hourly_reference_facility_energy_mwh"]
            - frame["annual_baseline_facility_energy_mwh"]
        )
        frame["total_hourly_minus_annual_carbon_tco2"] = (
            frame["hourly_reference_carbon_tco2"]
            - frame["annual_baseline_carbon_tco2"]
        )
        records.append(frame)

    result = pd.concat(records, ignore_index=True)
    result["energy_effect_pct_of_hourly_reference"] = np.divide(
        result["energy_effect_mwh"].to_numpy(dtype=float) * 100.0,
        result["hourly_reference_facility_energy_mwh"].to_numpy(dtype=float),
        out=np.full(len(result), np.nan),
        where=result["hourly_reference_facility_energy_mwh"].to_numpy(dtype=float)
        != 0,
    )
    result["carbon_effect_pct_of_hourly_reference"] = np.divide(
        result["carbon_effect_tco2"].to_numpy(dtype=float) * 100.0,
        result["hourly_reference_carbon_tco2"].to_numpy(dtype=float),
        out=np.full(len(result), np.nan),
        where=result["hourly_reference_carbon_tco2"].to_numpy(dtype=float) != 0,
    )

    group_keys = ["scenario", "policy", "year", "country"]
    result["sum_of_energy_effects_mwh"] = result.groupby(group_keys)[
        "energy_effect_mwh"
    ].transform("sum")
    result["sum_of_carbon_effects_tco2"] = result.groupby(group_keys)[
        "carbon_effect_tco2"
    ].transform("sum")
    result["energy_decomposition_check_mwh"] = (
        result["sum_of_energy_effects_mwh"]
        - result["total_hourly_minus_annual_energy_mwh"]
    )
    result["carbon_decomposition_check_tco2"] = (
        result["sum_of_carbon_effects_tco2"]
        - result["total_hourly_minus_annual_carbon_tco2"]
    )
    return result.sort_values(
        ["scenario", "policy", "country", "year", "effect_order"],
        ignore_index=True,
    )


def _annual_average_global_effects(effect_results: pd.DataFrame) -> pd.DataFrame:
    """Sum the 24 countries by year, then average the annual bridge effects."""
    annual_keys = [
        "scenario",
        "policy",
        "year",
        "effect_order",
        "effect",
        "source_variant",
        "target_variant",
        "effect_definition",
    ]
    annual = effect_results.groupby(
        annual_keys,
        as_index=False,
        sort=False,
    ).agg(
        countries_aggregated=("country", "nunique"),
        source_facility_energy_mwh=("source_facility_energy_mwh", "sum"),
        target_facility_energy_mwh=("target_facility_energy_mwh", "sum"),
        energy_effect_mwh=("energy_effect_mwh", "sum"),
        source_carbon_tco2=("source_carbon_tco2", "sum"),
        target_carbon_tco2=("target_carbon_tco2", "sum"),
        carbon_effect_tco2=("carbon_effect_tco2", "sum"),
        hourly_reference_facility_energy_mwh=(
            "hourly_reference_facility_energy_mwh",
            "sum",
        ),
        hourly_reference_carbon_tco2=("hourly_reference_carbon_tco2", "sum"),
        total_hourly_minus_annual_energy_mwh=(
            "total_hourly_minus_annual_energy_mwh",
            "sum",
        ),
        total_hourly_minus_annual_carbon_tco2=(
            "total_hourly_minus_annual_carbon_tco2",
            "sum",
        ),
    )
    country_counts = annual["countries_aggregated"].unique()
    if len(country_counts) != 1:
        raise AssertionError(
            "Figure 3 bridge uses inconsistent country counts: "
            f"{sorted(country_counts.tolist())}."
        )

    average_keys = [column for column in annual_keys if column != "year"]
    averaged = annual.groupby(
        average_keys,
        as_index=False,
        sort=False,
    ).agg(
        year_start=("year", "min"),
        year_end=("year", "max"),
        years_averaged=("year", "nunique"),
        countries_aggregated=("countries_aggregated", "min"),
        annual_avg_source_facility_energy_mwh=(
            "source_facility_energy_mwh",
            "mean",
        ),
        annual_avg_target_facility_energy_mwh=(
            "target_facility_energy_mwh",
            "mean",
        ),
        annual_avg_energy_effect_mwh=("energy_effect_mwh", "mean"),
        annual_avg_source_carbon_tco2=("source_carbon_tco2", "mean"),
        annual_avg_target_carbon_tco2=("target_carbon_tco2", "mean"),
        annual_avg_carbon_effect_tco2=("carbon_effect_tco2", "mean"),
        annual_avg_hourly_reference_facility_energy_mwh=(
            "hourly_reference_facility_energy_mwh",
            "mean",
        ),
        annual_avg_hourly_reference_carbon_tco2=(
            "hourly_reference_carbon_tco2",
            "mean",
        ),
        annual_avg_total_hourly_minus_annual_energy_mwh=(
            "total_hourly_minus_annual_energy_mwh",
            "mean",
        ),
        annual_avg_total_hourly_minus_annual_carbon_tco2=(
            "total_hourly_minus_annual_carbon_tco2",
            "mean",
        ),
    )
    averaged["energy_effect_pct_of_hourly_reference"] = np.divide(
        averaged["annual_avg_energy_effect_mwh"].to_numpy(dtype=float) * 100.0,
        averaged[
            "annual_avg_hourly_reference_facility_energy_mwh"
        ].to_numpy(dtype=float),
        out=np.full(len(averaged), np.nan),
        where=averaged[
            "annual_avg_hourly_reference_facility_energy_mwh"
        ].to_numpy(dtype=float)
        != 0,
    )
    averaged["carbon_effect_pct_of_hourly_reference"] = np.divide(
        averaged["annual_avg_carbon_effect_tco2"].to_numpy(dtype=float) * 100.0,
        averaged["annual_avg_hourly_reference_carbon_tco2"].to_numpy(dtype=float),
        out=np.full(len(averaged), np.nan),
        where=averaged["annual_avg_hourly_reference_carbon_tco2"].to_numpy(dtype=float)
        != 0,
    )
    bridge_keys = ["scenario", "policy"]
    averaged["sum_of_energy_effects_mwh"] = averaged.groupby(bridge_keys)[
        "annual_avg_energy_effect_mwh"
    ].transform("sum")
    averaged["sum_of_carbon_effects_tco2"] = averaged.groupby(bridge_keys)[
        "annual_avg_carbon_effect_tco2"
    ].transform("sum")
    averaged["energy_decomposition_check_mwh"] = (
        averaged["sum_of_energy_effects_mwh"]
        - averaged["annual_avg_total_hourly_minus_annual_energy_mwh"]
    )
    averaged["carbon_decomposition_check_tco2"] = (
        averaged["sum_of_carbon_effects_tco2"]
        - averaged["annual_avg_total_hourly_minus_annual_carbon_tco2"]
    )
    return averaged.sort_values(
        ["scenario", "policy", "effect_order"],
        ignore_index=True,
    )


def _power_metrics(hourly_power_mw: np.ndarray) -> dict[str, float]:
    values = np.asarray(hourly_power_mw, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Hourly power must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Hourly power must be finite and non-negative.")

    mean_power = float(values.mean())
    differences = np.diff(values)
    max_ramp_up = max(float(differences.max()), 0.0) if len(differences) else 0.0
    max_ramp_down = min(float(differences.min()), 0.0) if len(differences) else 0.0
    max_abs_ramp = float(np.abs(differences).max()) if len(differences) else 0.0
    return {
        "mean_power_mw": mean_power,
        "peak_power_mw": float(values.max()),
        "p99_power_mw": float(np.quantile(values, 0.99)),
        "peak_to_mean_ratio": (
            float(values.max() / mean_power) if mean_power > 0 else np.nan
        ),
        "max_ramp_up_mw_per_h": max_ramp_up,
        "max_ramp_down_mw_per_h": max_ramp_down,
        "max_abs_ramp_mw_per_h": max_abs_ramp,
        "max_abs_ramp_pct_of_mean": (
            max_abs_ramp * 100.0 / mean_power if mean_power > 0 else np.nan
        ),
    }


def _peak_ramp_table(
    variant_results: pd.DataFrame,
    base_cp_hourly: pd.DataFrame,
) -> pd.DataFrame:
    """Return plotting data for the flat and hourly workload profiles."""
    is_base_cp = (variant_results["scenario"] == FIGURE_SCENARIO) & (
        variant_results["policy"] == FIGURE_POLICY
    )
    annual = variant_results.loc[
        is_base_cp
        & variant_results["variant"].isin(
            (MEAN_WORKLOAD_VARIANT, HOURLY_REFERENCE_VARIANT)
        ),
        ["variant", "year", "country", "facility_energy_mwh"],
    ].copy()
    hourly = base_cp_hourly.copy()
    hourly["year"] = hourly["year"].astype(int)
    hourly_coverage = hourly.groupby(
        ["year", "country"],
        as_index=False,
    ).agg(
        hourly_rows=("timestamp_utc", "size"),
        unique_hours=("timestamp_utc", "nunique"),
    )
    invalid_coverage = hourly_coverage.loc[
        (hourly_coverage["hourly_rows"] != 8760)
        | (hourly_coverage["unique_hours"] != 8760)
    ]
    if not invalid_coverage.empty:
        sample = invalid_coverage.head().to_dict("records")
        raise AssertionError(
            "Figure 3 operating profiles must contain exactly 8,760 unique "
            f"hours per country-year; invalid rows: {sample}."
        )

    for year, year_group in hourly.groupby("year", sort=True):
        country_profiles = year_group.pivot(
            index="timestamp_utc",
            columns="country",
            values="facility_energy_mwh",
        ).sort_index()
        if country_profiles.isna().any().any():
            raise AssertionError(
                f"Figure 3 hourly country profiles are incomplete for {year}."
            )
        profile_values = country_profiles.to_numpy(dtype=float)
        profile_means = profile_values.mean(axis=0, keepdims=True)
        if np.any(profile_means <= 0):
            raise AssertionError(
                f"Figure 3 hourly country profiles have non-positive means for {year}."
            )
        normalized_profiles = profile_values / profile_means
        if not np.allclose(
            normalized_profiles,
            normalized_profiles[:, [0]],
            rtol=1e-9,
            atol=1e-9,
        ):
            max_deviation = float(
                np.abs(normalized_profiles - normalized_profiles[:, [0]]).max()
            )
            raise AssertionError(
                "Figure 3 country profiles do not share one relative hourly "
                f"shape for {year}; maximum normalized deviation={max_deviation:.3e}."
            )

    hourly_annual = hourly.groupby(
        ["year", "country"], as_index=False, sort=True
    )["facility_energy_mwh"].sum()
    hourly_reference_annual = annual.loc[
        annual["variant"] == HOURLY_REFERENCE_VARIANT,
        ["year", "country", "facility_energy_mwh"],
    ]
    energy_check = hourly_reference_annual.merge(
        hourly_annual,
        on=["year", "country"],
        how="outer",
        suffixes=("_annual", "_hourly"),
        indicator=True,
        validate="one_to_one",
    )
    if (energy_check["_merge"] != "both").any() or not np.allclose(
        energy_check["facility_energy_mwh_annual"],
        energy_check["facility_energy_mwh_hourly"],
        rtol=1e-10,
        atol=1e-5,
    ):
        raise AssertionError("Hourly energy does not reproduce its annual total.")

    records: list[dict[str, object]] = []

    def add_profile_pair(
        *,
        scope: str,
        country: str,
        year: int,
        trace_values: np.ndarray,
        flat_annual_energy_mwh: float,
    ) -> None:
        n_hours = len(trace_values)
        profiles = (
            (
                "flat_trace_mean_workload",
                np.full(n_hours, flat_annual_energy_mwh / n_hours, dtype=float),
            ),
            ("hourly_trace_workload", trace_values),
        )
        for profile_name, values in profiles:
            records.append(
                {
                    "scenario": FIGURE_SCENARIO,
                    "policy": FIGURE_POLICY,
                    "scope": scope,
                    "country": country,
                    "year": int(year),
                    "profile": profile_name,
                    "hours": n_hours,
                    **_power_metrics(values),
                }
            )

    flat_annual = annual.loc[annual["variant"] == MEAN_WORKLOAD_VARIANT]
    for year, trace_group in hourly.groupby("year", sort=True):
        global_trace = (
            trace_group.groupby("timestamp_utc", sort=True)["facility_energy_mwh"]
            .sum()
            .to_numpy(dtype=float)
        )
        global_flat_energy = float(
            flat_annual.loc[flat_annual["year"] == year, "facility_energy_mwh"].sum()
        )
        add_profile_pair(
            scope="aggregate",
            country="ALL",
            year=int(year),
            trace_values=global_trace,
            flat_annual_energy_mwh=global_flat_energy,
        )

    detail_year = DETAIL_YEAR
    if detail_year not in set(hourly["year"]):
        raise AssertionError(
            f"Figure 3d requires hourly results for {detail_year}."
        )
    detail_hourly = hourly.loc[hourly["year"] == detail_year]
    for country, trace_group in detail_hourly.groupby("country", sort=True):
        trace_values = trace_group.sort_values("timestamp_utc")[
            "facility_energy_mwh"
        ].to_numpy(dtype=float)
        country_flat_energy = float(
            flat_annual.loc[
                (flat_annual["year"] == detail_year)
                & (flat_annual["country"] == country),
                "facility_energy_mwh",
            ].iloc[0]
        )
        add_profile_pair(
            scope="country",
            country=COUNTRY_ISO3[country],
            year=detail_year,
            trace_values=trace_values,
            flat_annual_energy_mwh=country_flat_energy,
        )

    result = pd.DataFrame.from_records(records)
    flat = result["profile"] == "flat_trace_mean_workload"
    if not np.allclose(result.loc[flat, "peak_to_mean_ratio"], 1.0) or not np.allclose(
        result.loc[flat, "max_abs_ramp_mw_per_h"], 0.0
    ):
        raise AssertionError("The flat workload peak/ramp reference is not flat.")
    return result.sort_values(
        ["scope", "year", "country", "profile"], ignore_index=True
    )


def _figure3a_output_table(global_comparison: pd.DataFrame) -> pd.DataFrame:
    is_base_cp = (global_comparison["scenario"] == FIGURE_SCENARIO) & (
        global_comparison["policy"] == FIGURE_POLICY
    )
    selected = global_comparison.loc[
        is_base_cp,
        [
            "scenario",
            "policy",
            "year",
            "model",
            "facility_energy_mwh",
            "carbon_tco2",
        ],
    ].copy()
    wide = selected.pivot(
        index=["scenario", "policy", "year"],
        columns="model",
        values=["facility_energy_mwh", "carbon_tco2"],
    )
    required = [
        (metric, model)
        for metric in ("facility_energy_mwh", "carbon_tco2")
        for model in MODEL_ORDER
    ]
    missing = [column for column in required if column not in wide.columns]
    if missing:
        raise AssertionError(f"Figure 3a is missing model metrics: {missing}")

    result = wide.index.to_frame(index=False)
    annual_energy = wide[
        ("facility_energy_mwh", "annual_cpu_model")
    ].to_numpy(dtype=float)
    hourly_energy = wide[
        ("facility_energy_mwh", "hourly_workload_resolved_model")
    ].to_numpy(dtype=float)
    annual_carbon = wide[("carbon_tco2", "annual_cpu_model")].to_numpy(
        dtype=float
    )
    hourly_carbon = wide[
        ("carbon_tco2", "hourly_workload_resolved_model")
    ].to_numpy(dtype=float)
    result["annual_cpu_facility_energy_twh"] = annual_energy / 1e6
    result["hourly_workload_resolved_facility_energy_twh"] = hourly_energy / 1e6
    result[
        "annual_vs_hourly_facility_energy_difference_pct"
    ] = _relative_difference(
        annual_energy,
        hourly_energy,
    )
    result["annual_cpu_carbon_mtco2"] = annual_carbon / 1e6
    result["hourly_workload_resolved_carbon_mtco2"] = hourly_carbon / 1e6
    result["annual_vs_hourly_carbon_difference_pct"] = _relative_difference(
        annual_carbon,
        hourly_carbon,
    )
    return result.sort_values("year", ignore_index=True)


def _figure3b_output_table(
    annual_average_global_effects: pd.DataFrame,
) -> pd.DataFrame:
    is_base_cp = (
        annual_average_global_effects["scenario"] == FIGURE_SCENARIO
    ) & (
        annual_average_global_effects["policy"] == FIGURE_POLICY
    )
    result = annual_average_global_effects.loc[
        is_base_cp,
        [
            "scenario",
            "policy",
            "year_start",
            "year_end",
            "years_averaged",
            "countries_aggregated",
            "effect_order",
            "effect",
            "effect_definition",
            "annual_avg_energy_effect_mwh",
            "energy_effect_pct_of_hourly_reference",
            "annual_avg_carbon_effect_tco2",
            "carbon_effect_pct_of_hourly_reference",
            "annual_avg_total_hourly_minus_annual_energy_mwh",
            "annual_avg_total_hourly_minus_annual_carbon_tco2",
        ],
    ].copy()
    result["annual_avg_energy_effect_twh"] = (
        result["annual_avg_energy_effect_mwh"] / 1e6
    )
    result["annual_avg_carbon_effect_mtco2"] = (
        result["annual_avg_carbon_effect_tco2"] / 1e6
    )
    result["annual_avg_total_hourly_minus_annual_energy_twh"] = (
        result["annual_avg_total_hourly_minus_annual_energy_mwh"] / 1e6
    )
    result["annual_avg_total_hourly_minus_annual_carbon_mtco2"] = (
        result["annual_avg_total_hourly_minus_annual_carbon_tco2"] / 1e6
    )
    columns = [
        "scenario",
        "policy",
        "year_start",
        "year_end",
        "years_averaged",
        "countries_aggregated",
        "effect_order",
        "effect",
        "effect_definition",
        "annual_avg_energy_effect_twh",
        "energy_effect_pct_of_hourly_reference",
        "annual_avg_carbon_effect_mtco2",
        "carbon_effect_pct_of_hourly_reference",
        "annual_avg_total_hourly_minus_annual_energy_twh",
        "annual_avg_total_hourly_minus_annual_carbon_mtco2",
    ]
    return result[columns].sort_values("effect_order", ignore_index=True)


def _figure3c_output_table(peak_ramp: pd.DataFrame) -> pd.DataFrame:
    result = peak_ramp.loc[peak_ramp["scope"] == "aggregate"].copy()
    result["mean_power_gw"] = result["mean_power_mw"] / 1000.0
    result["peak_power_gw"] = result["peak_power_mw"] / 1000.0
    result["p99_power_gw"] = result["p99_power_mw"] / 1000.0
    result["max_abs_ramp_gw_per_h"] = (
        result["max_abs_ramp_mw_per_h"] / 1000.0
    )
    columns = [
        "scenario",
        "policy",
        "year",
        "profile",
        "hours",
        "mean_power_gw",
        "peak_power_gw",
        "p99_power_gw",
        "max_abs_ramp_gw_per_h",
    ]
    profile_order = result["profile"].map(
        {"flat_trace_mean_workload": 0, "hourly_trace_workload": 1}
    )
    result = result.assign(_profile_order=profile_order).sort_values(
        ["year", "_profile_order"],
        ignore_index=True,
    )
    return result[columns]


def _figure3d_output_table(peak_ramp: pd.DataFrame) -> pd.DataFrame:
    is_country_hourly = (peak_ramp["scope"] == "country") & (
        peak_ramp["profile"] == "hourly_trace_workload"
    )
    result = peak_ramp.loc[is_country_hourly].copy()
    if set(result["year"].astype(int)) != {DETAIL_YEAR}:
        raise AssertionError(f"Figure 3d must contain only {DETAIL_YEAR} results.")
    result["peak_power_gw"] = result["peak_power_mw"] / 1000.0
    result["max_abs_ramp_gw_per_h"] = (
        result["max_abs_ramp_mw_per_h"] / 1000.0
    )
    result = result.sort_values(
        ["peak_power_gw", "country"],
        ascending=[False, True],
        ignore_index=True,
    )
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    columns = [
        "rank",
        "scenario",
        "policy",
        "year",
        "country",
        "profile",
        "hours",
        "peak_power_gw",
        "max_abs_ramp_gw_per_h",
    ]
    return result[columns]


def _write_workbook(output_path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                    max(max_length + 2, 10), 48
                )


def generate_figure3_data(
    output_path: Union[str, Path] = DEFAULT_OUTPUT,
    countries: Sequence[str] = DEFAULT_COUNTRIES,
    year_start: int = FIGURE_YEAR_START,
    years: int = FIGURE_YEARS,
    workload_profile_path: Union[str, Path] = ROOT_DIR / "dataset",
    server_profile_path: Optional[Union[str, Path]] = None,
    hourly_carbon_factors_dir: Union[str, Path] = ROOT_DIR
    / "dataset"
    / "EM-CPNDCNZ",
    hourly_carbon_scope: str = "direct",
    hourly_carbon_fallback_to_annual: bool = True,
    capacity_quantile: float = 0.96,
    max_resource_utilization: float = 1.0,
    max_intervals: Optional[int] = None,
    verbose: bool = True,
) -> Path:
    """Calculate fixed Base-CP Figure 3 panels a-d and write one workbook."""
    countries = list(countries)
    year_end = year_start + years - 1
    if year_start != FIGURE_YEAR_START or year_end != FIGURE_YEAR_END:
        raise ValueError(
            "Figure 3 is fixed to "
            f"{FIGURE_YEAR_START}-{FIGURE_YEAR_END}; got {year_start}-{year_end}."
        )
    expected_countries = set(DEFAULT_COUNTRIES)
    if len(countries) != len(DEFAULT_COUNTRIES) or set(countries) != expected_countries:
        raise ValueError(
            "Figure 3 is fixed to the complete 24-country boundary."
        )
    unmapped_countries = [
        country for country in countries if country not in COUNTRY_ISO3
    ]
    if unmapped_countries:
        raise ValueError(f"Missing ISO3 labels for countries: {unmapped_countries}")

    if verbose:
        print("[figure3] Building the shared GPU workload profile.", flush=True)
    profile = build_workload_profile(
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        capacity_quantile=capacity_quantile,
        max_intervals=max_intervals,
        verbose=verbose,
    )

    variant_results, base_cp_hourly = _collect_variant_results(
        scenarios=[FIGURE_SCENARIO],
        countries=countries,
        year_start=year_start,
        years=years,
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        hourly_carbon_factors_dir=hourly_carbon_factors_dir,
        hourly_carbon_scope=hourly_carbon_scope,
        hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
        capacity_quantile=capacity_quantile,
        max_resource_utilization=max_resource_utilization,
        max_intervals=max_intervals,
        profile=profile,
        verbose=verbose,
    )
    global_comparison = _global_model_comparison(variant_results)
    country_effects = _country_effect_decomposition(variant_results)
    annual_average_effects = _annual_average_global_effects(country_effects)
    peak_ramp = _peak_ramp_table(variant_results, base_cp_hourly)

    max_energy_check = float(
        annual_average_effects["energy_decomposition_check_mwh"].abs().max()
    )
    max_carbon_check = float(
        annual_average_effects["carbon_decomposition_check_tco2"].abs().max()
    )
    if max_energy_check > 1e-6 or max_carbon_check > 1e-6:
        raise AssertionError(
            "Annual-to-hourly effect bridge failed to close: "
            f"energy={max_energy_check:.6g} MWh, "
            f"carbon={max_carbon_check:.6g} tCO2."
        )

    sheets = {
        "Fig3a_Model_Comparison": _figure3a_output_table(global_comparison),
        "Fig3b_Effect_Decomposition": _figure3b_output_table(
            annual_average_effects
        ),
        "Fig3c_Operating_Shape": _figure3c_output_table(peak_ramp),
        "Fig3d_Country_Peak_Ramp": _figure3d_output_table(peak_ramp),
    }
    resolved_output = Path(output_path)
    _write_workbook(resolved_output, sheets)
    if verbose:
        print(
            f"[figure3] Effect bridge closed with maximum checks "
            f"{max_energy_check:.3e} MWh and {max_carbon_check:.3e} tCO2."
        )
        print(f"[figure3] Excel workbook saved to {resolved_output.resolve()}.")
    return resolved_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Excel calculation results required by Figure 3."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workload-profile-path", type=Path, default=ROOT_DIR / "dataset"
    )
    parser.add_argument("--server-profile-path", type=Path, default=None)
    parser.add_argument(
        "--hourly-carbon-factors-dir",
        type=Path,
        default=ROOT_DIR / "dataset" / "EM-CPNDCNZ",
    )
    parser.add_argument(
        "--hourly-carbon-scope", choices=("direct", "life_cycle"), default="direct"
    )
    parser.add_argument("--strict-hourly-carbon", action="store_true")
    parser.add_argument("--capacity-quantile", type=float, default=0.96)
    parser.add_argument("--max-resource-utilization", type=float, default=1.0)
    parser.add_argument(
        "--max-intervals",
        type=int,
        default=None,
        help="Debugging only; omit for formal paper results.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_figure3_data(
        output_path=args.output,
        workload_profile_path=args.workload_profile_path,
        server_profile_path=args.server_profile_path,
        hourly_carbon_factors_dir=args.hourly_carbon_factors_dir,
        hourly_carbon_scope=args.hourly_carbon_scope,
        hourly_carbon_fallback_to_annual=not args.strict_hourly_carbon,
        capacity_quantile=args.capacity_quantile,
        max_resource_utilization=args.max_resource_utilization,
        max_intervals=args.max_intervals,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
