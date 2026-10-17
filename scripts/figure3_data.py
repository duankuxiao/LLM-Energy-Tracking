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
contains no plotting code. W04 configuration, activity controls and source
tables accompany the panel data so representation differences are inspectable.
W06 adds seam-aware ramps, shared-shape diagnostics and fixed-electricity
load/CI timing stresses, reusing the captured hourly reference.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.annual_cpu_model import run_annual_cpu_model
from core.annual_activity import (
    ACTIVITY_MODES, ACTIVITY_YEAR_START, ACTIVITY_YEAR_END, AnnualActivityConfig,
    REFERENCE_TASK_RATIOS, REFERENCE_TASK_UTILIZATION, reference_parameter_table,
)
from core.grid_carbon import calendar_hours
from core.temporal_robustness import power_metrics, shared_shape_diagnostics, timing_stress_tables
from core.accounting_boundary import build_boundary_metadata, capacity_factor_table, load_ai_capacity_factors
from core.hourly_workload_resolved_model import (
    RESOURCES,
    WorkloadProfile,
    build_workload_profile,
    run_workload_component_footprint,
)
from core.task_model import TASK_TYPES
from dataset.Factors import get_carbon_factor
from dataset.Installed_capacity_data import DEFAULT_COUNTRIES, COUNTRY_ISO3


from core.research_io import write_workbook
from scripts.figure_contract import prepare_figure_sources

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

DATA_YEAR_START = 2025
LEGACY_TASK_RATIOS = REFERENCE_TASK_RATIOS
LEGACY_TASK_UTILIZATION = REFERENCE_TASK_UTILIZATION

RESULT_COLUMNS = (
    "variant",
    "model",
    "scenario",
    "policy",
    "year",
    "country",
    "installed_it_mw",
    "it_energy_mwh",
    "cpu_utilization",
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
    if "cpu_utilization" not in result:
        result["cpu_utilization"] = (result["avg_cpu_utilization"] if "avg_cpu_utilization" in result
                                     else result["composite_utilization"])
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
    return profile.with_load(flat_load)


def _legacy_activity(year: int, activity_config: Optional[AnnualActivityConfig] = None) -> np.ndarray:
    """Return the annual CPU model's task activity coefficients for one year."""
    return (activity_config or AnnualActivityConfig()).vector(year)


def _legacy_assumption_profile(
    profile: WorkloadProfile,
    *,
    year: int,
    activity_config: Optional[AnnualActivityConfig] = None,
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
    normalized_task_load = ratios * utilization * _legacy_activity(year, activity_config)
    load = (
        normalized_task_load[:, None, None]
        * np.asarray(profile.trace_capacity, dtype=float)[None, :, None]
        * np.ones((1, len(RESOURCES), profile.n_intervals), dtype=float)
    )
    return profile.with_load(load)


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
    ai_capacity_factors: Optional[Mapping[int, float]] = None,
    model_runner=None,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]]:
    result = (model_runner or run_workload_component_footprint)(
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
        ai_capacity_factors=ai_capacity_factors,
    )
    annual = _standardize_result(
        result["annual_summary"],
        variant=variant,
        model="hourly_workload_resolved_model",
        policy=policy,
    )
    if not return_hourly:
        return annual

    hourly = result["hourly_carbon"].copy()
    if hourly.empty:
        raise AssertionError("The requested hourly energy table is empty.")
    temporal = {"W06_Trace_Alignment": result["temporal_alignment"].copy(),
                "W06_Baseline_Peak_Ramp": result["temporal_metrics"].copy(),
                "W06_Baseline_Seams": result["trace_seams"].copy()}
    return annual, hourly, result["grid_carbon_audit"].copy(), temporal


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
    ai_capacity_factors: Optional[Mapping[int, float]] = None,
    activity_config: Optional[AnnualActivityConfig] = None,
    experiment_context=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    all_frames: list[pd.DataFrame] = []
    base_cp_hourly: Optional[pd.DataFrame] = None
    base_cp_grid_audit: Optional[pd.DataFrame] = None
    base_cp_temporal: dict[str, pd.DataFrame] = {}

    for policy in POLICIES:
        if verbose:
            print(f"[figure3] Calculating annual CPU baseline for {policy}.", flush=True)
        annual_cpu_result = (experiment_context.run_annual if experiment_context is not None else run_annual_cpu_model)(
            ai_capacity_factors=ai_capacity_factors,
            renewable_energy_policy=policy,
            scenarios=scenarios,
            years=years,
            countries=countries,
            year_start=year_start,
            save_outputs=False,
            verbose=False,
            activity_config=activity_config,
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
                model_runner=experiment_context.run_hourly if experiment_context is not None else None,
                ai_capacity_factors=ai_capacity_factors,
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
                activity_config=activity_config,
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
                    model_runner=experiment_context.run_hourly if experiment_context is not None else None,
                    ai_capacity_factors=ai_capacity_factors,
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
                model_runner=experiment_context.run_hourly if experiment_context is not None else None,
                verbose=False,
                return_hourly=capture_hourly,
                ai_capacity_factors=ai_capacity_factors,
            )
            if capture_hourly:
                if not isinstance(hourly_output, tuple):
                    raise AssertionError("Base-CP hourly output was not captured.")
                hourly_reference, base_cp_hourly, base_cp_grid_audit, base_cp_temporal = hourly_output
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
    if base_cp_hourly is None or base_cp_grid_audit is None:
        raise AssertionError("Base-CP is required for Figure 3 hourly diagnostics.")
    sorted_results = results.sort_values(
        ["scenario", "policy", "year", "country", "variant"],
        ignore_index=True,
    )
    return sorted_results, base_cp_hourly, base_cp_grid_audit, base_cp_temporal


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


def _w04_endpoint_pair(annual: pd.DataFrame, component: pd.DataFrame, case: str) -> pd.DataFrame:
    """Report signed annual-minus-component gaps, with explicit denominators."""
    annual, component = annual.copy(), component.copy()
    for endpoint in (annual, component):
        if "cpu_utilization" not in endpoint:
            endpoint["cpu_utilization"] = (endpoint["avg_cpu_utilization"] if "avg_cpu_utilization" in endpoint
                                           else endpoint["composite_utilization"])
    keys = ["scenario", "policy", "year", "country"]
    metrics = ["installed_it_mw", "it_energy_mwh", "facility_energy_mwh", "carbon_tco2", "cpu_utilization"]
    frames = []
    for scope in ("country", "selected_countries"):
        left, right = annual[keys + metrics].copy(), component[keys + metrics].copy()
        if scope == "selected_countries":
            def aggregate(source):
                source = source.copy()
                source["cpu_active_capacity_mw"] = source["cpu_utilization"] * source["installed_it_mw"]
                summed = source.groupby(keys[:-1], as_index=False, observed=True)[metrics[:-1] + ["cpu_active_capacity_mw"]].sum()
                summed["cpu_utilization"] = summed.pop("cpu_active_capacity_mw") / summed["installed_it_mw"]
                return summed
            left, right = aggregate(left), aggregate(right)
            left["country"], right["country"] = "SELECTED_COUNTRIES", "SELECTED_COUNTRIES"
        pair = left.merge(right, on=keys, how="outer", suffixes=("_annual", "_component"),
                          validate="one_to_one", indicator=True)
        if not pair["_merge"].eq("both").all():
            raise ValueError(f"W04 endpoint coverage differs for {case}.")
        pair = pair.drop(columns="_merge")
        if not np.allclose(pair["installed_it_mw_annual"], pair["installed_it_mw_component"], rtol=1e-10, atol=1e-6):
            raise ValueError(f"W04 endpoint capacity boundaries differ for {case}.")
        pair["cpu_utilization_difference"] = pair["cpu_utilization_annual"] - pair["cpu_utilization_component"]
        if case.startswith("common_mean") and not np.allclose(
            pair["cpu_utilization_annual"], pair["cpu_utilization_component"], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Common CPU activity is not matched between endpoints for {case}.")
        for metric, unit in (("facility_energy_mwh", "mwh"), ("carbon_tco2", "tco2")):
            left_values = pair[f"{metric}_annual"].to_numpy(dtype=float)
            right_values = pair[f"{metric}_component"].to_numpy(dtype=float)
            name = "energy" if unit == "mwh" else "carbon"
            pair[f"annual_minus_component_{name}_{unit}"] = left_values - right_values
            pair[f"{name}_gap_pct_of_component"] = _relative_difference(left_values, right_values)
            pair[f"{name}_gap_pct_of_annual"] = np.divide(
                (left_values - right_values) * 100., left_values,
                out=np.full(len(pair), np.nan), where=left_values != 0)
        for endpoint in ("annual", "component"):
            capacity = pair[f"installed_it_mw_{endpoint}"].to_numpy(dtype=float)
            for metric in ("it_energy_mwh", "facility_energy_mwh", "carbon_tco2"):
                pair[f"{endpoint}_{metric}_per_installed_it_mw"] = np.divide(
                    pair[f"{metric}_{endpoint}"].to_numpy(dtype=float), capacity,
                    out=np.full(len(pair), np.nan), where=capacity > 0)
            pair[f"{endpoint}_mean_it_power_per_installed_it_mw"] = pair[f"{endpoint}_it_energy_mwh_per_installed_it_mw"] / 8760.
        pair["case"], pair["scope"] = case, scope
        pair["estimand"] = "conditional_representation_difference_not_observed_prediction_error"
        frames.append(pair)
    return pd.concat(frames, ignore_index=True)


def _annual_activity_controls(
    *, variant_results: pd.DataFrame, profile: WorkloadProfile,
    countries: Sequence[str], year_start: int, years: int,
    activity_config: AnnualActivityConfig,
    workload_profile_path: Union[str, Path], server_profile_path: Optional[Union[str, Path]],
    capacity_quantile: float, max_resource_utilization: float, max_intervals: Optional[int],
    ai_capacity_factors: Optional[Mapping[int, float]],
    experiment_context=None,
) -> dict[str, pd.DataFrame]:
    """Separate activity ramps from power representation without recalibration.

    Common-mean controls use realized, capped task-resource means. The annual
    aggregate equation represents the common CPU channel; the component model
    additionally represents GPU/memory/storage channels, which are disclosed.
    Both common-mean endpoints use ANNUAL direct CI and flat workloads.
    """
    requested_years = list(range(year_start, year_start + years))
    reference = variant_results.loc[variant_results["variant"] == HOURLY_REFERENCE_VARIANT].copy()
    pairs, configuration, loads, factor_frames = [], [], [], []
    for mode in ACTIVITY_MODES:
        config = replace(activity_config, mode=mode)
        factor_frames.append(config.table(range(ACTIVITY_YEAR_START, ACTIVITY_YEAR_END + 1)))
        if mode == activity_config.mode:
            annual = variant_results.loc[variant_results["variant"] == ANNUAL_BASELINE_VARIANT].copy()
        else:
            annual = (experiment_context.run_annual if experiment_context is not None else run_annual_cpu_model)(
                renewable_energy_policy=FIGURE_POLICY, scenarios=[FIGURE_SCENARIO],
                years=years, year_start=year_start, countries=countries,
                activity_config=config, ai_capacity_factors=ai_capacity_factors,
                save_outputs=False, verbose=False)["annual_summary"]
        case = f"{mode}_baseline"
        pairs.append(_w04_endpoint_pair(annual, reference, case))
        configuration.append({
            "case": case, "annual_activity": mode, "annual_load": "reference_task_shares_and_utilizations",
            "component_activity": "unchanged_trace", "component_load": "full_hourly_trace",
            "annual_carbon": "annual_direct", "component_carbon": "hourly_direct_or_explicit_fallback",
            "difference_dimensions": "power_configuration; workload_assumptions; temporal_shape; CI_matching",
            "capacity_calibration": "same_factor_set_in_both_endpoints_no_recalibration",
        })

    normalized = profile.load / profile.trace_capacity[None, :, None]
    totals = normalized.sum(axis=0)
    cap_scale = np.minimum(1., np.divide(max_resource_utilization, totals,
                                        out=np.ones_like(totals), where=totals > 0))
    common_mean = (normalized * cap_scale[None, :, :]).mean(axis=2)
    cpu_id = RESOURCES.index("cpu")
    unity = AnnualActivityConfig(mode="fixed", training_start=1., training_target=1.,
                                 inference_start=1., inference_target=1.)
    calendar = replace(activity_config, mode="calendar")
    anchor = calendar.vector(ACTIVITY_YEAR_START)
    if np.any(anchor <= 0):
        raise ValueError("Symmetric relative activity controls require positive 2025 activity anchors.")
    for case in ("common_mean_fixed_activity", "common_mean_symmetric_trend"):
        annual_frames, component_frames = [], []
        for year in requested_years:
            multiplier = (np.ones(len(TASK_TYPES)) if case == "common_mean_fixed_activity" else
                          calendar.vector(year) / anchor)
            requested = common_mean * multiplier[:, None]
            resource_totals = requested.sum(axis=0)
            clipping = np.minimum(1., np.divide(max_resource_utilization, resource_totals,
                                               out=np.ones_like(resource_totals), where=resource_totals > 0))
            realized = requested * clipping[None, :]
            cpu_total = float(realized[:, cpu_id].sum())
            cpu_utilization = float(np.clip(cpu_total, 0., max_resource_utilization))
            cpu_shares = (realized[:, cpu_id] / cpu_total if cpu_total > 0 else
                          np.array([REFERENCE_TASK_RATIOS[task] for task in TASK_TYPES]))
            task_ratios = dict(zip(TASK_TYPES, cpu_shares))
            annual_frames.append((experiment_context.run_annual if experiment_context is not None else run_annual_cpu_model)(
                renewable_energy_policy=FIGURE_POLICY, scenarios=[FIGURE_SCENARIO],
                years=1, year_start=year, countries=countries, activity_config=unity,
                task_ratio_by_country={country: task_ratios for country in countries},
                u_train=cpu_utilization, u_infer=cpu_utilization, u_other=cpu_utilization,
                ai_capacity_factors=ai_capacity_factors, save_outputs=False, verbose=False)["annual_summary"])
            flat_load = np.broadcast_to(
                realized[:, :, None] * profile.trace_capacity[None, :, None], profile.load.shape).copy()
            flat_profile = profile.with_load(flat_load)
            component_frames.append(_run_hourly_variant(
                variant=case, policy=FIGURE_POLICY, scenario=FIGURE_SCENARIO, countries=countries,
                year_start=year, years=1, workload_profile_path=workload_profile_path,
                server_profile_path=server_profile_path, hourly_carbon_factors_dir=None,
                hourly_carbon_scope="direct", hourly_carbon_fallback_to_annual=False,
                capacity_quantile=capacity_quantile, max_resource_utilization=max_resource_utilization,
                max_intervals=max_intervals, profile=flat_profile, verbose=False,
                ai_capacity_factors=ai_capacity_factors,
                model_runner=experiment_context.run_hourly if experiment_context is not None else None))
            for task_id, task in enumerate(TASK_TYPES):
                for resource_id, resource in enumerate(RESOURCES):
                    loads.append({
                        "case": case, "year": year, "task_type": task, "resource": resource,
                        "base_realized_normalized_mean": common_mean[task_id, resource_id],
                        "imposed_task_multiplier": multiplier[task_id],
                        "requested_normalized_mean": requested[task_id, resource_id],
                        "resource_cap_scale": clipping[resource_id],
                        "realized_normalized_mean": realized[task_id, resource_id],
                        "represented_in_annual_aggregate": resource == "cpu",
                        "represented_in_component_model": True,
                    })
        pairs.append(_w04_endpoint_pair(pd.concat(annual_frames, ignore_index=True),
                                        pd.concat(component_frames, ignore_index=True), case))
        configuration.append({
            "case": case, "annual_activity": "common_realized_mean_CPU_contributions",
            "annual_load": "flat_common_CPU_mean", "component_activity": case,
            "component_load": "flat_common_task_resource_means",
            "annual_carbon": "annual_direct", "component_carbon": "annual_direct",
            "difference_dimensions": "aggregate_CPU_power_vs_component_power; additional_non_CPU_channels",
            "capacity_calibration": "same_factor_set_in_both_endpoints_no_recalibration",
        })
    comparison = pd.concat(pairs, ignore_index=True).sort_values(
        ["case", "scope", "scenario", "policy", "country", "year"], ignore_index=True)
    group = comparison.groupby(["case", "scope", "scenario", "policy", "country"], sort=False)
    for metric in ("annual_minus_component_energy_mwh", "annual_minus_component_carbon_tco2"):
        comparison[f"{metric}_change_since_first_year"] = comparison[metric] - group[metric].transform("first")
    comparison["gap_evolution_start_year"] = year_start
    comparison["installed_capacity_ratio_since_first_year"] = (
        comparison["installed_it_mw_annual"] / group["installed_it_mw_annual"].transform("first"))
    for endpoint in ("annual", "component"):
        metric = f"{endpoint}_mean_it_power_per_installed_it_mw"
        comparison[f"{metric}_change_since_first_year"] = comparison[metric] - group[metric].transform("first")
    baseline = comparison.loc[comparison["case"].isin([f"{mode}_baseline" for mode in ACTIVITY_MODES])].copy()
    keys = ["scope", "scenario", "policy", "year", "country"]
    fixed = baseline.loc[baseline["case"] == "fixed_baseline", keys + ["facility_energy_mwh_annual", "carbon_tco2_annual"]].rename(
        columns={"facility_energy_mwh_annual": "fixed_annual_energy_mwh", "carbon_tco2_annual": "fixed_annual_carbon_tco2"})
    baseline = baseline.merge(fixed, on=keys, validate="many_to_one")
    baseline["annual_energy_change_vs_fixed_mwh"] = baseline["facility_energy_mwh_annual"] - baseline["fixed_annual_energy_mwh"]
    baseline["annual_energy_change_pct_of_fixed_annual"] = _relative_difference(
        baseline["facility_energy_mwh_annual"], baseline["fixed_annual_energy_mwh"])
    baseline["annual_carbon_change_vs_fixed_tco2"] = baseline["carbon_tco2_annual"] - baseline["fixed_annual_carbon_tco2"]
    baseline["annual_carbon_change_pct_of_fixed_annual"] = _relative_difference(
        baseline["carbon_tco2_annual"], baseline["fixed_annual_carbon_tco2"])
    legacy = baseline.loc[baseline["case"] == "legacy_six_year_baseline", keys + ["facility_energy_mwh_annual", "carbon_tco2_annual"]].rename(
        columns={"facility_energy_mwh_annual": "legacy_annual_energy_mwh", "carbon_tco2_annual": "legacy_annual_carbon_tco2"})
    baseline = baseline.merge(legacy, on=keys, validate="many_to_one")
    baseline["annual_energy_change_vs_legacy_mwh"] = baseline["facility_energy_mwh_annual"] - baseline["legacy_annual_energy_mwh"]
    baseline["annual_energy_change_pct_of_legacy_annual"] = _relative_difference(
        baseline["facility_energy_mwh_annual"], baseline["legacy_annual_energy_mwh"])
    baseline["annual_carbon_change_vs_legacy_tco2"] = baseline["carbon_tco2_annual"] - baseline["legacy_annual_carbon_tco2"]
    baseline["annual_carbon_change_pct_of_legacy_annual"] = _relative_difference(
        baseline["carbon_tco2_annual"], baseline["legacy_annual_carbon_tco2"])
    return {"W04_Control_Comparison": comparison, "W04_Configuration_Matrix": pd.DataFrame(configuration),
            "W04_Activity_Factors": pd.concat(factor_frames, ignore_index=True),
            "W04_Common_Mean_Load": pd.DataFrame(loads), "W04_Baseline_Impact": baseline}


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


def _peak_ramp_table(
    variant_results: pd.DataFrame,
    base_cp_hourly: pd.DataFrame,
    allow_legacy_calendar: bool = True,
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
    expected_hours = hourly_coverage["year"].map(lambda year: len(calendar_hours(int(year))))
    full = (hourly_coverage["hourly_rows"] == expected_hours) & (hourly_coverage["unique_hours"] == expected_hours)
    legacy = (allow_legacy_calendar & (expected_hours == 8784)
              & hourly_coverage["hourly_rows"].eq(8760) & hourly_coverage["unique_hours"].eq(8760))
    invalid_coverage = hourly_coverage.loc[~(full | legacy)]
    if not invalid_coverage.empty:
        sample = invalid_coverage.head().to_dict("records")
        raise AssertionError(
            "Figure 3 operating profiles must contain a complete UTC calendar "
            f"hours per country-year; invalid rows: {sample}."
        )

    shape_by_year = {}
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
        shape_by_year[int(year)] = shared_shape_diagnostics(profile_values.T)

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
        seam_before: np.ndarray,
        discontinuity_before: np.ndarray,
        sum_of_country_peaks_mw: float = np.nan,
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
            is_trace = profile_name == "hourly_trace_workload"
            seams = seam_before if is_trace else np.zeros(n_hours, dtype=bool)
            artifacts = discontinuity_before if is_trace else np.zeros(n_hours, dtype=bool)
            records.append(
                {
                    "scenario": FIGURE_SCENARIO,
                    "policy": FIGURE_POLICY,
                    "scope": scope,
                    "country": country,
                    "year": int(year),
                    "profile": profile_name,
                    "hours": n_hours,
                    "calendar_status": ("full_calendar" if n_hours == len(calendar_hours(int(year)))
                                        else "legacy_8760_truncated_leap_year"),
                    **(shape_by_year[year] if is_trace else {
                        **shape_by_year[year], "all_positive_country_shapes_identical": True,
                        "maximum_normalized_shape_deviation": 0.}),
                    "power_unit": "MW", "ramp_unit": "MW h^-1",
                    "ramp_definition": "Maximum absolute hourly ramp",
                    "peak_construction": "sum_country_power_at_same_UTC_hour_then_max" if scope == "aggregate" else "own_country_hourly_maximum",
                    "sum_of_individual_country_peaks_mw": sum_of_country_peaks_mw if is_trace else (flat_annual_energy_mwh / n_hours if scope == "aggregate" else np.nan),
                    **power_metrics(values, seams, artifacts),
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
            seam_before=trace_group.groupby("timestamp_utc", sort=True)["trace_repeat_seam_before"].any().to_numpy(),
            discontinuity_before=trace_group.groupby("timestamp_utc", sort=True)["alignment_discontinuity_before"].any().to_numpy(),
            sum_of_country_peaks_mw=float(trace_group.groupby("country")["facility_energy_mwh"].max().sum()),
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
            seam_before=trace_group.sort_values("timestamp_utc")["trace_repeat_seam_before"].to_numpy(dtype=bool),
            discontinuity_before=trace_group.sort_values("timestamp_utc")["alignment_discontinuity_before"].to_numpy(dtype=bool),
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


def _plot_ramp_columns(result: pd.DataFrame) -> list[str]:
    metrics = ("max_abs_ramp_excluding_seams", "max_abs_ramp_excluding_alignment",
               "p95_abs_ramp", "p99_abs_ramp", "p95_abs_ramp_excluding_seams",
               "p99_abs_ramp_excluding_seams")
    for metric in metrics:
        result[f"{metric}_gw_per_h"] = result[f"{metric}_mw_per_h"] / 1000.
    result["power_unit"] = "GW"
    result["ramp_unit"] = "GW h^-1"
    result["ramp_definition"] = "Maximum absolute hourly ramp"
    return [f"{metric}_gw_per_h" for metric in metrics] + [
        "calendar_status", "repeat_seam_count", "alignment_discontinuity_count",
        "raw_max_occurs_at_repeat_seam", "raw_max_occurs_at_alignment_discontinuity",
        "power_unit", "ramp_unit", "ramp_definition", "peak_construction",
        "all_positive_country_shapes_identical", "maximum_normalized_shape_deviation"]


def _figure3c_output_table(peak_ramp: pd.DataFrame) -> pd.DataFrame:
    result = peak_ramp.loc[peak_ramp["scope"] == "aggregate"].copy()
    result["mean_power_gw"] = result["mean_power_mw"] / 1000.0
    result["peak_power_gw"] = result["peak_power_mw"] / 1000.0
    result["p99_power_gw"] = result["p99_power_mw"] / 1000.0
    result["max_abs_ramp_gw_per_h"] = (
        result["max_abs_ramp_mw_per_h"] / 1000.0
    )
    extra_columns = _plot_ramp_columns(result)
    result["sum_of_individual_country_peaks_gw"] = result["sum_of_individual_country_peaks_mw"] / 1000.
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
        "sum_of_individual_country_peaks_gw",
    ] + extra_columns
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
    extra_columns = _plot_ramp_columns(result)
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
    ] + extra_columns
    return result[columns]


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
    ai_capacity_factors: Optional[Mapping[int, float]] = None,
    activity_config: Optional[AnnualActivityConfig] = None,
    experiment_context=None,
) -> Path:
    """Calculate fixed Base-CP Figure 3 panels a-d and write one workbook."""
    countries = list(countries)
    activity_config = activity_config or AnnualActivityConfig()
    activity_config.validate()
    if hourly_carbon_scope != "direct":
        raise ValueError("Figure 3 requires direct factors in both endpoints; no comparable annual life-cycle table is supplied.")
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
        print("[figure3] Reusing the prepared workload profile." if experiment_context is not None else "[figure3] Building the workload profile.", flush=True)
    profile = experiment_context.profile if experiment_context is not None else build_workload_profile(
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        capacity_quantile=capacity_quantile,
        max_intervals=max_intervals,
        verbose=verbose,
    )

    variant_results, base_cp_hourly, grid_audit, temporal_baseline = _collect_variant_results(
        experiment_context=experiment_context,
        ai_capacity_factors=ai_capacity_factors,
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
        activity_config=activity_config,
    )
    global_comparison = _global_model_comparison(variant_results)
    country_effects = _country_effect_decomposition(variant_results)
    annual_average_effects = _annual_average_global_effects(country_effects)
    peak_ramp = _peak_ramp_table(variant_results, base_cp_hourly,
                                allow_legacy_calendar=hourly_carbon_fallback_to_annual)
    if verbose:
        print("[figure3] Reusing hourly output for W06 fixed-electricity timing controls.", flush=True)
    timing_controls = timing_stress_tables(base_cp_hourly, policy=FIGURE_POLICY, carbon_scope=hourly_carbon_scope)
    if verbose:
        print("[figure3] Calculating W04 fixed-activity, historical and common-load controls.", flush=True)
    controls = _annual_activity_controls(
        experiment_context=experiment_context,
        variant_results=variant_results, profile=profile, countries=countries,
        year_start=year_start, years=years, activity_config=activity_config,
        workload_profile_path=workload_profile_path, server_profile_path=server_profile_path,
        capacity_quantile=capacity_quantile, max_resource_utilization=max_resource_utilization,
        max_intervals=max_intervals, ai_capacity_factors=ai_capacity_factors)

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
        "W04_Run_Metadata": pd.DataFrame([{
            "annual_activity_mode": activity_config.mode,
            "activity_anchor_start_year": ACTIVITY_YEAR_START, "activity_anchor_end_year": ACTIVITY_YEAR_END,
            "trace_intervals": profile.n_intervals, "trace_hours": profile.trace_hours,
            "max_intervals": max_intervals, "trace_truncated_by_max_intervals": max_intervals is not None,
            "grid_input_directory": str(Path(hourly_carbon_factors_dir).resolve()),
            "strict_hourly_carbon": not hourly_carbon_fallback_to_annual,
            "capacity_quantile": capacity_quantile, "max_resource_utilization": max_resource_utilization,
            "calibration_rule": "same_capacity_factors_for_all_cases_no_recalibration",
            "estimand": "conditional_representation_difference_not_observed_accuracy",
            "bridge_interpretation": "ordered_configuration_changes_not_unique_causal_attribution",
            "energy_convention": "8760_hour_annualized_trace_energy_distributed_over_CI_calendar",
        }]),
        "W04_Parameter_Assumptions": reference_parameter_table(activity_config),
        "W04_Grid_Input_Audit": grid_audit,
        **temporal_baseline,
        **timing_controls,
        **controls,
        "Source_Bridge_Variants": variant_results,
        "Source_Country_Bridge": country_effects,
        "Accounting_Boundary": build_boundary_metadata(countries, ai_capacity_factors),
        "AI_Capacity_Factors": capacity_factor_table(year_start, years, ai_capacity_factors),
        "Fig3a_Model_Comparison": _figure3a_output_table(global_comparison),
        "Fig3b_Effect_Decomposition": _figure3b_output_table(
            annual_average_effects
        ),
        "Fig3c_Operating_Shape": _figure3c_output_table(peak_ramp),
        "Fig3d_Country_Peak_Ramp": _figure3d_output_table(peak_ramp),
    }
    resolved_output = Path(output_path)
    if experiment_context is not None:
        sheets["Shared_Experiment_Configuration"] = experiment_context.metadata_table()
    sheets = prepare_figure_sources(3, sheets, carbon_scope=hourly_carbon_scope)
    write_workbook(resolved_output, sheets)
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
    parser.add_argument("--ai-capacity-factors-file", type=Path, default=None)
    parser.add_argument("--annual-activity-mode", choices=ACTIVITY_MODES, default="calendar",
                        help="Fixed 2025-2030 calendar, fixed 2025 activity, or historical six-year denominator.")
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
        activity_config=AnnualActivityConfig(mode=args.annual_activity_mode),
        ai_capacity_factors=(load_ai_capacity_factors(args.ai_capacity_factors_file)
                             if args.ai_capacity_factors_file else None),
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
