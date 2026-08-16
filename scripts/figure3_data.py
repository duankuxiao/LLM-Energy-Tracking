"""Generate the calculation results required by Figure 3.

Figure 3 compares only the legacy annual aggregate baseline (M1) with the
trace-driven, GPU-aware hourly model (M4).  Intermediate numbered models are
not constructed.  Instead, an additive bridge identifies how the result
changes as M1 is progressively replaced by the M4 representation:

1. component power configuration under the legacy task/load assumptions;
2. trace-derived mean workload instead of fixed task/load assumptions;
3. hourly workload variability instead of a flat trace-derived workload; and
4. hourly carbon matching instead of annual-average carbon factors.

The four signed effects are calculated in the M1-to-M4 direction and must sum
to the direct M4-minus-M1 difference.  The script evaluates the complete
4-demand-scenario x 3-grid-policy space for 2026--2030, writes one Excel
workbook with one worksheet per Figure 3 panel, and contains no plotting code.
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

from core.m1_annual_cpu_model import run_m1_annual_cpu_model
from core.m4_hourly_gpu_model import (
    RESOURCES,
    WorkloadProfile,
    build_workload_profile,
    run_workload_component_footprint,
)
from core.task_model import TASK_TYPES
from dataset.Factors import get_carbon_factor
from dataset.Installed_capacity_data import DEFAULT_COUNTRIES


DEFAULT_OUTPUT = ROOT_DIR / "results" / "figure3_data.xlsx"
DEFAULT_SCENARIOS = ("Base", "Lift-Off", "High Efficiency", "Headwinds")
POLICIES = ("CP", "NDC", "NZ")
MODEL_ORDER = ("M1", "M4")

M1_VARIANT = "M1_baseline"
POWER_CONFIGURATION_VARIANT = "M4_power_with_legacy_load"
MEAN_WORKLOAD_VARIANT = "M4_flat_trace_annual_carbon"
HOURLY_WORKLOAD_VARIANT = "M4_trace_annual_carbon"
M4_VARIANT = "M4_trace_hourly_carbon"
VARIANT_ORDER = (
    M1_VARIANT,
    POWER_CONFIGURATION_VARIANT,
    MEAN_WORKLOAD_VARIANT,
    HOURLY_WORKLOAD_VARIANT,
    M4_VARIANT,
)

EFFECT_STEPS = (
    (
        "power_configuration",
        M1_VARIANT,
        POWER_CONFIGURATION_VARIANT,
        (
            "Replace the legacy aggregate utilization-power equation with the "
            "M4 component power configuration while retaining M1 task shares "
            "and fixed utilization assumptions."
        ),
    ),
    (
        "mean_workload_representation",
        POWER_CONFIGURATION_VARIANT,
        MEAN_WORKLOAD_VARIANT,
        (
            "Replace M1 task shares and fixed utilization assumptions with the "
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
        M4_VARIANT,
        (
            "Replace annual-average carbon factors with hourly carbon factors "
            "while retaining the full trace-driven M4 energy profile."
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
        model="M4",
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
    """Return the M1 task activity coefficients for one calendar year."""
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
    """Build a flat profile with the task/load assumptions used by M1.

    The normalized load for every resource equals the M1 task ratio multiplied
    by its annual activity and assumed utilization.  Passing this profile
    through M4 changes the power/configuration layer while retaining the M1
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


def _run_m4_variant(
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
) -> pd.DataFrame:
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
    return _standardize_result(
        result["annual_summary"],
        variant=variant,
        model="M4",
        policy=policy,
    )


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
) -> pd.DataFrame:
    all_frames: list[pd.DataFrame] = []
    model_year_end = year_start + years - 1

    for policy in POLICIES:
        if verbose:
            print(f"[figure3] Calculating M1 baseline for {policy}.", flush=True)
        m1_result = run_m1_annual_cpu_model(
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
                m1_result["annual_summary"],
                variant=M1_VARIANT,
                model="M1",
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
            _run_m4_variant(
                variant=MEAN_WORKLOAD_VARIANT,
                policy="CP",
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
                    f"[figure3] Calculating M4 power with legacy load for "
                    f"{scenario} x {year}.",
                    flush=True,
                )
            legacy_profile = _legacy_assumption_profile(
                profile,
                year=year,
                model_year_end=model_year_end,
            )
            legacy_energy_frames.append(
                _run_m4_variant(
                    variant=POWER_CONFIGURATION_VARIANT,
                    policy="CP",
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
                    f"[figure3] Calculating full M4 for {scenario} x {policy}.",
                    flush=True,
                )
            m4_hourly = _run_m4_variant(
                variant=M4_VARIANT,
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
            )
            all_frames.append(
                _derive_annual_carbon_variant(
                    m4_hourly,
                    variant=HOURLY_WORKLOAD_VARIANT,
                    policy=policy,
                )
            )
            all_frames.append(m4_hourly)

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
    return results.sort_values(
        ["scenario", "policy", "year", "country", "variant"],
        ignore_index=True,
    )


def _global_model_comparison(variant_results: pd.DataFrame) -> pd.DataFrame:
    selected = variant_results[variant_results["variant"].isin((M1_VARIANT, M4_VARIANT))]
    keys = ["scenario", "policy", "year"]
    global_results = selected.groupby(
        ["variant", "model", *keys], as_index=False, observed=True
    )[["facility_energy_mwh", "carbon_tco2"]].sum()
    global_results = _recalculate_summary_metrics(global_results)

    reference = global_results.loc[
        global_results["variant"] == M4_VARIANT,
        keys + ["facility_energy_mwh", "carbon_tco2"],
    ].rename(
        columns={
            "facility_energy_mwh": "m4_facility_energy_mwh",
            "carbon_tco2": "m4_carbon_tco2",
        }
    )
    result = global_results.merge(reference, on=keys, validate="many_to_one")
    result["energy_difference_vs_m4_mwh"] = (
        result["facility_energy_mwh"] - result["m4_facility_energy_mwh"]
    )
    result["energy_difference_vs_m4_pct"] = _relative_difference(
        result["facility_energy_mwh"], result["m4_facility_energy_mwh"]
    )
    result["carbon_difference_vs_m4_tco2"] = (
        result["carbon_tco2"] - result["m4_carbon_tco2"]
    )
    result["carbon_difference_vs_m4_pct"] = _relative_difference(
        result["carbon_tco2"], result["m4_carbon_tco2"]
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
        frame["m4_facility_energy_mwh"] = energy[M4_VARIANT].to_numpy(dtype=float)
        frame["m4_carbon_tco2"] = carbon[M4_VARIANT].to_numpy(dtype=float)
        frame["m1_facility_energy_mwh"] = energy[M1_VARIANT].to_numpy(dtype=float)
        frame["m1_carbon_tco2"] = carbon[M1_VARIANT].to_numpy(dtype=float)
        frame["total_m4_minus_m1_energy_mwh"] = (
            frame["m4_facility_energy_mwh"] - frame["m1_facility_energy_mwh"]
        )
        frame["total_m4_minus_m1_carbon_tco2"] = (
            frame["m4_carbon_tco2"] - frame["m1_carbon_tco2"]
        )
        records.append(frame)

    result = pd.concat(records, ignore_index=True)
    result["energy_effect_pct_of_m4"] = np.divide(
        result["energy_effect_mwh"].to_numpy(dtype=float) * 100.0,
        result["m4_facility_energy_mwh"].to_numpy(dtype=float),
        out=np.full(len(result), np.nan),
        where=result["m4_facility_energy_mwh"].to_numpy(dtype=float) != 0,
    )
    result["carbon_effect_pct_of_m4"] = np.divide(
        result["carbon_effect_tco2"].to_numpy(dtype=float) * 100.0,
        result["m4_carbon_tco2"].to_numpy(dtype=float),
        out=np.full(len(result), np.nan),
        where=result["m4_carbon_tco2"].to_numpy(dtype=float) != 0,
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
        - result["total_m4_minus_m1_energy_mwh"]
    )
    result["carbon_decomposition_check_tco2"] = (
        result["sum_of_carbon_effects_tco2"]
        - result["total_m4_minus_m1_carbon_tco2"]
    )
    return result.sort_values(
        ["scenario", "policy", "country", "year", "effect_order"],
        ignore_index=True,
    )


def _annual_average_country_effects(effect_results: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "scenario",
        "policy",
        "country",
        "effect_order",
        "effect",
        "source_variant",
        "target_variant",
        "effect_definition",
    ]
    averaged = effect_results.groupby(keys, as_index=False, sort=False).agg(
        year_start=("year", "min"),
        year_end=("year", "max"),
        years_averaged=("year", "nunique"),
        annual_avg_source_facility_energy_mwh=("source_facility_energy_mwh", "mean"),
        annual_avg_target_facility_energy_mwh=("target_facility_energy_mwh", "mean"),
        annual_avg_energy_effect_mwh=("energy_effect_mwh", "mean"),
        annual_avg_source_carbon_tco2=("source_carbon_tco2", "mean"),
        annual_avg_target_carbon_tco2=("target_carbon_tco2", "mean"),
        annual_avg_carbon_effect_tco2=("carbon_effect_tco2", "mean"),
        annual_avg_m4_facility_energy_mwh=("m4_facility_energy_mwh", "mean"),
        annual_avg_m4_carbon_tco2=("m4_carbon_tco2", "mean"),
        annual_avg_total_m4_minus_m1_energy_mwh=(
            "total_m4_minus_m1_energy_mwh",
            "mean",
        ),
        annual_avg_total_m4_minus_m1_carbon_tco2=(
            "total_m4_minus_m1_carbon_tco2",
            "mean",
        ),
    )
    averaged["energy_effect_pct_of_m4"] = np.divide(
        averaged["annual_avg_energy_effect_mwh"].to_numpy(dtype=float) * 100.0,
        averaged["annual_avg_m4_facility_energy_mwh"].to_numpy(dtype=float),
        out=np.full(len(averaged), np.nan),
        where=averaged["annual_avg_m4_facility_energy_mwh"].to_numpy(dtype=float)
        != 0,
    )
    averaged["carbon_effect_pct_of_m4"] = np.divide(
        averaged["annual_avg_carbon_effect_tco2"].to_numpy(dtype=float) * 100.0,
        averaged["annual_avg_m4_carbon_tco2"].to_numpy(dtype=float),
        out=np.full(len(averaged), np.nan),
        where=averaged["annual_avg_m4_carbon_tco2"].to_numpy(dtype=float) != 0,
    )
    bridge_keys = ["scenario", "policy", "country"]
    averaged["sum_of_energy_effects_mwh"] = averaged.groupby(bridge_keys)[
        "annual_avg_energy_effect_mwh"
    ].transform("sum")
    averaged["sum_of_carbon_effects_tco2"] = averaged.groupby(bridge_keys)[
        "annual_avg_carbon_effect_tco2"
    ].transform("sum")
    averaged["energy_decomposition_check_mwh"] = (
        averaged["sum_of_energy_effects_mwh"]
        - averaged["annual_avg_total_m4_minus_m1_energy_mwh"]
    )
    averaged["carbon_decomposition_check_tco2"] = (
        averaged["sum_of_carbon_effects_tco2"]
        - averaged["annual_avg_total_m4_minus_m1_carbon_tco2"]
    )
    averaged["country"] = [COUNTRY_ISO3[country] for country in averaged["country"]]
    return averaged.sort_values(
        ["scenario", "policy", "country", "effect_order"],
        ignore_index=True,
    )


def _country_m1_m4_comparison(variant_results: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "policy", "year", "country"]
    selected = variant_results[variant_results["variant"].isin((M1_VARIANT, M4_VARIANT))]
    energy = selected.pivot(
        index=keys, columns="variant", values="facility_energy_mwh"
    ).reset_index()
    carbon = selected.pivot(
        index=keys, columns="variant", values="carbon_tco2"
    ).reset_index()
    result = energy.merge(carbon, on=keys, suffixes=("_energy", "_carbon"))
    result = result.rename(
        columns={
            f"{M1_VARIANT}_energy": "m1_facility_energy_mwh",
            f"{M4_VARIANT}_energy": "m4_facility_energy_mwh",
            f"{M1_VARIANT}_carbon": "m1_carbon_tco2",
            f"{M4_VARIANT}_carbon": "m4_carbon_tco2",
        }
    )
    result["m1_vs_m4_energy_error_pct"] = _relative_difference(
        result["m1_facility_energy_mwh"], result["m4_facility_energy_mwh"]
    )
    result["m1_vs_m4_carbon_error_pct"] = _relative_difference(
        result["m1_carbon_tco2"], result["m4_carbon_tco2"]
    )
    result["m1_minus_m4_carbon_tco2"] = (
        result["m1_carbon_tco2"] - result["m4_carbon_tco2"]
    )
    return result.sort_values(keys, ignore_index=True)


def _failure_risk_table(
    country_comparison: pd.DataFrame,
    failure_thresholds_pct: Sequence[float],
) -> pd.DataFrame:
    records = []
    group_keys = ["scenario", "policy", "year"]
    for key, group in country_comparison.groupby(group_keys, sort=True):
        scenario, policy, year = key
        carbon_error = group["m1_vs_m4_carbon_error_pct"]
        energy_error = group["m1_vs_m4_energy_error_pct"]
        absolute_carbon_error = carbon_error.abs()
        absolute_energy_error = energy_error.abs()
        for threshold in failure_thresholds_pct:
            carbon_failed = absolute_carbon_error > threshold
            energy_failed = absolute_energy_error > threshold
            either_failed = carbon_failed | energy_failed
            country_count = len(group)
            records.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "year": int(year),
                    "failure_threshold_pct": float(threshold),
                    "country_count": country_count,
                    "carbon_failure_country_count": int(carbon_failed.sum()),
                    "carbon_failure_share_pct": float(carbon_failed.mean() * 100.0),
                    "energy_failure_country_count": int(energy_failed.sum()),
                    "energy_failure_share_pct": float(energy_failed.mean() * 100.0),
                    "either_failure_country_count": int(either_failed.sum()),
                    "either_failure_share_pct": float(either_failed.mean() * 100.0),
                    "mean_signed_carbon_error_pct": float(carbon_error.mean()),
                    "mean_absolute_carbon_error_pct": float(
                        absolute_carbon_error.mean()
                    ),
                    "median_absolute_carbon_error_pct": float(
                        absolute_carbon_error.median()
                    ),
                    "p95_absolute_carbon_error_pct": float(
                        absolute_carbon_error.quantile(0.95)
                    ),
                    "maximum_absolute_carbon_error_pct": float(
                        absolute_carbon_error.max()
                    ),
                    "mean_absolute_energy_error_pct": float(
                        absolute_energy_error.mean()
                    ),
                    "net_m1_minus_m4_carbon_tco2": float(
                        group["m1_minus_m4_carbon_tco2"].sum()
                    ),
                    "total_absolute_m1_minus_m4_carbon_tco2": float(
                        group["m1_minus_m4_carbon_tco2"].abs().sum()
                    ),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        ["year", "policy", "scenario", "failure_threshold_pct"],
        ignore_index=True,
    )


def _figure3a_output_table(global_comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "scenario",
        "policy",
        "year",
        "facility_energy_mwh",
        "power_twh",
        "energy_difference_vs_m4_mwh",
        "energy_difference_vs_m4_pct",
        "carbon_tco2",
        "carbon_mtco2",
        "carbon_difference_vs_m4_tco2",
        "carbon_difference_vs_m4_pct",
    ]
    is_base_cp = (global_comparison["scenario"] == "Base") & (
        global_comparison["policy"] == "CP"
    )
    return global_comparison.loc[is_base_cp, columns].reset_index(drop=True)


def _figure3b_output_table(
    annual_average_country_effects: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "scenario",
        "policy",
        "year_start",
        "year_end",
        "years_averaged",
        "country",
        "effect_order",
        "effect",
        "source_variant",
        "target_variant",
        "effect_definition",
        "annual_avg_energy_effect_mwh",
        "energy_effect_pct_of_m4",
        "annual_avg_carbon_effect_tco2",
        "carbon_effect_pct_of_m4",
        "annual_avg_total_m4_minus_m1_energy_mwh",
        "annual_avg_total_m4_minus_m1_carbon_tco2",
        "sum_of_energy_effects_mwh",
        "sum_of_carbon_effects_tco2",
        "energy_decomposition_check_mwh",
        "carbon_decomposition_check_tco2",
    ]
    return annual_average_country_effects[columns].copy()


def _figure3c_output_table(failure_risk: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "policy",
        "year",
        "failure_threshold_pct",
        "carbon_failure_country_count",
        "carbon_failure_share_pct",
        "mean_absolute_carbon_error_pct",
        "p95_absolute_carbon_error_pct",
        "maximum_absolute_carbon_error_pct",
    ]
    return failure_risk[columns].copy()


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
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    countries: Sequence[str] = DEFAULT_COUNTRIES,
    year_start: int = 2026,
    years: int = 5,
    failure_thresholds_pct: Sequence[float] = (5.0, 10.0, 20.0),
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
    """Calculate all Figure 3 panel data and write one Excel workbook."""
    scenarios = list(scenarios)
    countries = list(countries)
    failure_thresholds_pct = [float(value) for value in failure_thresholds_pct]
    if any(value <= 0 for value in failure_thresholds_pct):
        raise ValueError("All failure thresholds must be positive percentages.")

    if verbose:
        print("[figure3] Building the shared GPU workload profile.", flush=True)
    profile = build_workload_profile(
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        capacity_quantile=capacity_quantile,
        max_intervals=max_intervals,
        verbose=verbose,
    )

    variant_results = _collect_variant_results(
        scenarios=scenarios,
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
    annual_average_effects = _annual_average_country_effects(country_effects)
    country_comparison = _country_m1_m4_comparison(variant_results)
    failure_risk = _failure_risk_table(
        country_comparison, failure_thresholds_pct
    )

    max_energy_check = float(
        annual_average_effects["energy_decomposition_check_mwh"].abs().max()
    )
    max_carbon_check = float(
        annual_average_effects["carbon_decomposition_check_tco2"].abs().max()
    )
    if max_energy_check > 1e-6 or max_carbon_check > 1e-6:
        raise AssertionError(
            "M1-to-M4 effect bridge failed to close: "
            f"energy={max_energy_check:.6g} MWh, "
            f"carbon={max_carbon_check:.6g} tCO2."
        )

    sheets = {
        "Fig3a_Model_Comparison": _figure3a_output_table(global_comparison),
        "Fig3b_Effect_Decomposition": _figure3b_output_table(
            annual_average_effects
        ),
        "Fig3c_Failure_Risk": _figure3c_output_table(failure_risk),
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
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--countries", nargs="+", default=list(DEFAULT_COUNTRIES))
    parser.add_argument("--year-start", type=int, default=2026)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument(
        "--failure-thresholds-pct",
        nargs="+",
        type=float,
        default=[5.0, 10.0, 20.0],
    )
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
        scenarios=args.scenarios,
        countries=args.countries,
        year_start=args.year_start,
        years=args.years,
        failure_thresholds_pct=args.failure_thresholds_pct,
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
