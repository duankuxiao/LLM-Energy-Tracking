"""Generate the hourly-model sensitivity results required by Figure 4a.

The script performs one-at-a-time sensitivity experiments for six core
dimensions that affect the paper's hourly workload-resolved energy and carbon results:

1. GPU idle-power ratio;
2. training and inference GPU utilization;
3. training/inference workload mix;
4. GPU share of the IT full-power budget;
5. hourly carbon-intensity variability; and
6. pre-power resource-load shape at fixed source task/resource totals.

W06 adds pre-power resource-shape controls and reuses baseline hourly output
for fixed-electricity timing controls. W07 adds CPU/GPU unit-wattage mechanisms,
CPU budget/idle controls, response exponents and accounting diagnostics.
Only calculation results are produced.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.energy_attribution import IDLE_ALLOCATION_RULES
from core.power_parameter_audit import load_hardware_config
from core.accounting_boundary import build_boundary_metadata, capacity_factor_table, load_ai_capacity_factors, resolve_countries
from core.hourly_workload_resolved_model import (
    RESOURCES,
    Alibaba2026TraceConfig,
    HardwarePowerConfig,
    WorkloadProfile,
    build_workload_profile,
    run_workload_component_footprint,
)
from core.task_model import TASK_TYPES
from core.temporal_robustness import timing_stress_tables
from dataset.Installed_capacity_data import DEFAULT_COUNTRIES


from core.research_io import write_workbook
from scripts.figure_contract import prepare_figure_sources
from core.experiment_config import SENSITIVITY_DEFAULTS

DEFAULT_OUTPUT = ROOT_DIR / "results" / "figure4_data.xlsx"


def _scale_task_gpu_utilization(
    profile: WorkloadProfile,
    task_type: str,
    scale: float,
) -> WorkloadProfile:
    """Scale one task type's GPU load without changing other resources."""
    if task_type not in ("training", "inference"):
        raise ValueError("GPU-utilization sensitivity supports training or inference.")
    if scale <= 0:
        raise ValueError("GPU-utilization scale must be positive.")

    load = np.array(profile.load, copy=True)
    task_id = TASK_TYPES.index(task_type)
    gpu_id = RESOURCES.index("gpu")
    load[task_id, gpu_id, :] *= float(scale)
    return profile.with_load(load)


def _resource_shape_control(profile: WorkloadProfile, scale: float) -> WorkloadProfile:
    """Preserve each source task/resource total before capacity clipping.

    Installed capacity, trace-capacity denominators and hardware stay fixed.
    Realized resource-hours and electricity must be measured after clipping
    and nonlinear power conversion; neither is assumed to remain constant.
    """
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("Resource variability must be finite and non-negative.")
    baseline = np.asarray(profile.load, dtype=float)
    mean = baseline.mean(axis=-1, keepdims=True)
    changed = np.maximum(mean + scale * (baseline - mean), 0.)
    source_total = baseline.sum(axis=-1, keepdims=True)
    changed_total = changed.sum(axis=-1, keepdims=True)
    changed *= np.divide(source_total, changed_total, out=np.zeros_like(changed_total), where=changed_total > 0)
    if not np.allclose(changed.sum(axis=-1), baseline.sum(axis=-1), rtol=1e-10, atol=1e-6):
        raise ValueError("Resource-shape control must preserve every task/resource source total.")
    return profile.with_load(changed)


def _classified_gpu_shares(profile: WorkloadProfile) -> np.ndarray:
    """Return training, inference and other shares of classified GPU load."""
    gpu_id = RESOURCES.index("gpu")
    classified_ids = [TASK_TYPES.index(name) for name in ("training", "inference", "other")]
    gpu_load = profile.load[classified_ids, gpu_id, :].sum(axis=1)
    total = float(gpu_load.sum())
    if total <= 0:
        raise ValueError("The workload profile has no classified GPU load.")
    shares = gpu_load / total
    if np.any(shares <= 0):
        raise ValueError(
            "Task-mix sensitivity requires positive training, inference and other GPU loads."
        )
    return shares


def _change_inference_share(
    profile: WorkloadProfile,
    inference_share_shift: float,
) -> tuple[WorkloadProfile, float]:
    """Change classified inference share while preserving annual resource totals.

    The other-task share remains at its baseline value.  Training absorbs the
    opposite change in inference share.  Each classified task retains its own
    temporal shape, while the annual total of every resource channel is
    restored to its baseline value.  Unclassified workload is unchanged.
    """
    classified_names = ("training", "inference", "other")
    classified_ids = [TASK_TYPES.index(name) for name in classified_names]
    baseline_shares = _classified_gpu_shares(profile)
    other_share = float(baseline_shares[2])
    target_inference_share = float(baseline_shares[1] + inference_share_shift)
    target_inference_share = float(
        np.clip(target_inference_share, 1e-6, 1.0 - other_share - 1e-6)
    )
    target_training_share = 1.0 - other_share - target_inference_share
    target_shares = np.array(
        [target_training_share, target_inference_share, other_share], dtype=float
    )

    load = np.array(profile.load, copy=True)
    for local_id, task_id in enumerate(classified_ids):
        load[task_id, :, :] *= target_shares[local_id] / baseline_shares[local_id]

    # Restore each resource's annual classified total.  This isolates task
    # structure and temporal shape from a change in total resource demand.
    for resource_id in range(len(RESOURCES)):
        baseline_total = float(
            profile.load[classified_ids, resource_id, :].sum()
        )
        changed_total = float(load[classified_ids, resource_id, :].sum())
        if baseline_total > 0 and changed_total > 0:
            load[classified_ids, resource_id, :] *= baseline_total / changed_total

    if not np.allclose(load.sum(axis=(0, 2)), profile.load.sum(axis=(0, 2)), rtol=1e-10, atol=1e-6):
        raise ValueError("Task-mix reallocation must preserve every source resource total.")
    if not np.array_equal(load[TASK_TYPES.index("unclassified")], profile.load[TASK_TYPES.index("unclassified")]):
        raise ValueError("Task-mix reallocation must retain unclassified activity.")

    return profile.with_load(load), target_inference_share


def _hardware_with_gpu_idle_ratio(
    baseline: HardwarePowerConfig,
    idle_ratio: float,
) -> HardwarePowerConfig:
    if not 0 <= idle_ratio <= 1:
        raise ValueError("GPU idle-power ratio must be in [0, 1].")
    return replace(
        baseline,
        gpu_idle_power_w=baseline.gpu_full_power_w * float(idle_ratio),
    )


def _hardware_with_power_share(baseline, component, share):
    """Change one full-power budget share and normalize the other components."""
    scale = (1. - share) / (1. - getattr(baseline, f"{component}_power_share"))
    changes = {f"{name}_power_share": getattr(baseline, f"{name}_power_share") * scale
               for name in ("cpu", "gpu", "memory", "storage", "it_fan") if name != component}
    config = replace(baseline, **{f"{component}_power_share": share, **changes})
    config.validate()
    return config


def _hardware_with_unit_power_scale(baseline, resource, scale, fixed_idle_ratio):
    if resource not in ("cpu", "gpu") or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Unit power scale must be positive for CPU or GPU.")
    full = "cpu_full_power_w_per_core" if resource == "cpu" else "gpu_full_power_w"
    idle = "cpu_idle_power_w_per_core" if resource == "cpu" else "gpu_idle_power_w"
    changes = {full: getattr(baseline, full) * scale}
    if fixed_idle_ratio:
        changes[idle] = getattr(baseline, idle) * scale
    config = replace(baseline, **changes)
    config.validate()
    return config


def _run_hourly_model_case(
    *,
    scenario: str,
    policy: str,
    year_start: int,
    years: int,
    countries: Sequence[str],
    workload_profile_path: Union[str, Path],
    server_profile_path: Optional[Union[str, Path]],
    hourly_carbon_factors_dir: Union[str, Path],
    hourly_carbon_scope: str,
    hourly_carbon_fallback_to_annual: bool,
    capacity_quantile: float,
    max_resource_utilization: float,
    max_intervals: Optional[int],
    workload_profile: WorkloadProfile,
    hardware_config: HardwarePowerConfig,
    carbon_variability_scale: float,
    verbose: bool,
    ai_capacity_factors: Optional[Mapping[int, float]] = None,
    idle_allocation_rule: str = "hourly_driver",
    collect_timing_stress: bool = False,
    model_runner=None,
) -> dict[str, object]:
    result = (model_runner or run_workload_component_footprint)(
        renewable_energy_policy=policy, scenarios=[scenario], years=years, countries=countries,
        workload_profile_path=workload_profile_path, server_profile_path=server_profile_path,
        year_start=year_start, save_outputs=False, verbose=verbose, hardware_config=hardware_config,
        capacity_quantile=capacity_quantile, max_resource_utilization=max_resource_utilization,
        hourly_carbon_factors_dir=hourly_carbon_factors_dir, hourly_carbon_scope=hourly_carbon_scope,
        hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual, save_hourly_outputs=False,
        max_intervals=max_intervals, workload_profile=workload_profile,
        ai_capacity_factors=ai_capacity_factors, idle_allocation_rule=idle_allocation_rule,
        carbon_variability_scale=carbon_variability_scale)

    annual = result["annual_summary"]
    annual_global = annual.groupby("year", as_index=False)[
        ["facility_energy_mwh", "carbon_tco2"]
    ].sum()
    if len(annual_global) != years:
        raise RuntimeError(
            "Unexpected number of annual hourly-model sensitivity results: "
            f"expected {years}, received {len(annual_global)}."
        )
    facility_energy_mwh = float(annual_global["facility_energy_mwh"].mean())
    carbon_tco2 = float(annual_global["carbon_tco2"].mean())
    components = result["component_energy"]
    total_it = float(components["it_energy_mwh"].sum())
    matrix = result["task_component_energy"]
    execution = result["task_execution"]
    classified = execution.loc[(execution["resource"] == "gpu") & (execution["task_type"] != "unclassified")]
    classified_gpu = float(classified["realized_resource_hours"].sum())
    return {
        "facility_energy_mwh": facility_energy_mwh,
        "carbon_tco2": carbon_tco2,
        "configured_system_W_per_GPU": hardware_config.gpu_full_power_w / hardware_config.gpu_power_share,
        "gpu_share_of_total_it_energy_pct": float(components.loc[components["component"] == "gpu", "it_energy_mwh"].sum()) / total_it * 100. if total_it > 0 else np.nan,
        "unclassified_share_of_total_it_energy_pct": float(matrix.loc[matrix["task_type"] == "unclassified", "it_energy_mwh"].sum()) / total_it * 100. if total_it > 0 else np.nan,
        "idle_share_of_total_it_energy_pct": float(components["idle_it_energy_mwh"].sum()) / total_it * 100. if total_it > 0 else np.nan,
        "idle_pool_share_of_total_it_energy_pct": float(components["unallocated_idle_it_energy_mwh"].sum()) / total_it * 100. if total_it > 0 else np.nan,
        "realized_classified_inference_gpu_share": float(classified.loc[classified["task_type"] == "inference", "realized_resource_hours"].sum()) / classified_gpu if classified_gpu > 0 else np.nan,
        "tables": {name: result[name] for name in (
            "annual_summary", "component_energy", "task_component_energy_all_rules", "idle_pool",
            "allocation_audit", "energy_conservation", "resource_audit", "global_resource_audit",
            "power_nonlinearity", "grid_carbon_audit", "resource_proxy_metadata",
            "temporal_alignment", "temporal_metrics", "trace_seams", "capacity_parameter_audit",
            "hardware_parameters", "parameter_dependencies", "mechanism_conservation")},
        "timing_stress": timing_stress_tables(result["hourly_carbon"], policy=policy, carbon_scope=hourly_carbon_scope) if collect_timing_stress else {},
    }


def _result_record(
    *,
    experiment_order: int,
    sensitivity_group: str,
    parameter_name: str,
    level: str,
    parameter_value: float,
    parameter_unit: str,
    baseline_parameter_value: float,
    scenario: str,
    policy: str,
    year_start: int,
    year_end: int,
    years_averaged: int,
    country_count: int,
    result: dict[str, object],
    baseline_result: dict[str, object],
    experiment_note: str,
    control_kind: str = "sensitivity",
) -> dict[str, object]:
    record: dict[str, object] = {
        "experiment_order": experiment_order,
        "sensitivity_group": sensitivity_group,
        "parameter_name": parameter_name,
        "level": level,
        "is_baseline_level": level == "Base",
        "parameter_value": parameter_value,
        "parameter_unit": parameter_unit,
        "baseline_parameter_value": baseline_parameter_value,
        "scenario": scenario,
        "policy": policy,
        "year_start": year_start,
        "year_end": year_end,
        "years_averaged": years_averaged,
        "country_count": country_count,
        "experiment_note": experiment_note,
        "control_kind": control_kind,
    }
    for metric in (
        "facility_energy_mwh",
        "carbon_tco2",
    ):
        baseline_value = baseline_result[metric]
        difference = result[metric] - baseline_value
        record[f"annual_mean_{metric}"] = result[metric]
        record[f"baseline_annual_mean_{metric}"] = baseline_value
        record[f"{metric}_change_vs_baseline_pct"] = (
            difference / baseline_value * 100.0
            if baseline_value != 0
            else np.nan
        )
    for metric in ("gpu_share_of_total_it_energy_pct", "unclassified_share_of_total_it_energy_pct",
                   "idle_share_of_total_it_energy_pct", "idle_pool_share_of_total_it_energy_pct",
                   "realized_classified_inference_gpu_share"):
        record[metric] = result[metric]
    record["configured_system_W_per_GPU"] = result["configured_system_W_per_GPU"]
    return record


def _level_label(value: float, baseline: float) -> str:
    if np.isclose(value, baseline, rtol=1e-12, atol=1e-15):
        return "Base"
    return "Low" if value < baseline else "High"


def generate_figure4a_data(
    output_path: Union[str, Path] = DEFAULT_OUTPUT,
    scenario: str = "Base",
    policy: str = "CP",
    year_start: int = 2026,
    years: int = 5,
    countries: Sequence[str] = DEFAULT_COUNTRIES,
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
    idle_allocation_rule: str = "hourly_driver",
    hardware_config: Optional[HardwarePowerConfig] = None,
    experiment_context=None,
    sensitivity_config: Optional[Mapping[str, Sequence[float]]] = None,
) -> Path:
    """Reuse one trace profile for sensitivity, W06 timing and W07 mechanism controls."""
    countries = resolve_countries(countries)
    if idle_allocation_rule not in IDLE_ALLOCATION_RULES:
        raise ValueError(f"idle_allocation_rule must be one of {IDLE_ALLOCATION_RULES}.")
    if years <= 0:
        raise ValueError("years must be a positive integer.")
    year_end = year_start + years - 1
    sensitivity = {**SENSITIVITY_DEFAULTS, **(sensitivity_config or {})}
    if set(sensitivity) - set(SENSITIVITY_DEFAULTS):
        raise ValueError("Unknown Figure 4 sensitivity range.")
    for name, values in sensitivity.items():
        baseline = 0. if name == "task_mix_inference_shifts" else 1.
        if not values or len(set(values)) != len(values) or not np.isfinite(values).all() or baseline not in values:
            raise ValueError(f"Sensitivity {name} must contain unique finite values and baseline {baseline}.")
        if baseline == 1. and min(values) < 0 or name in ("unit_power_scales", "dynamic_exponent_scales", "component_budget_scales") and min(values) <= 0:
            raise ValueError(f"Invalid negative/zero range for {name}.")
    baseline_hardware = hardware_config or (experiment_context.config.hardware if experiment_context is not None else HardwarePowerConfig())
    baseline_hardware.validate()
    if not 0 < baseline_hardware.cpu_power_share < 1 or not 0 < baseline_hardware.gpu_power_share < 1:
        raise ValueError("W07 CPU/GPU budget controls require positive CPU/GPU shares below one.")
    baseline_idle_ratio = (
        baseline_hardware.gpu_idle_power_w
        / baseline_hardware.gpu_full_power_w
    )

    if verbose:
        print("[figure4a] Reusing the prepared workload profile." if experiment_context is not None else "[figure4a] Building the workload profile.", flush=True)
    baseline_profile = experiment_context.profile if experiment_context is not None else build_workload_profile(
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        capacity_quantile=capacity_quantile,
        trace_config=Alibaba2026TraceConfig(),
        max_intervals=max_intervals,
        verbose=verbose,
    )
    baseline_inference_share = float(_classified_gpu_shares(baseline_profile)[1])

    common_run = {
        "model_runner": experiment_context.run_hourly if experiment_context is not None else None,
        "idle_allocation_rule": idle_allocation_rule,
        "ai_capacity_factors": ai_capacity_factors,
        "scenario": scenario,
        "policy": policy,
        "year_start": year_start,
        "years": years,
        "countries": countries,
        "workload_profile_path": workload_profile_path,
        "server_profile_path": server_profile_path,
        "hourly_carbon_factors_dir": hourly_carbon_factors_dir,
        "hourly_carbon_scope": hourly_carbon_scope,
        "hourly_carbon_fallback_to_annual": hourly_carbon_fallback_to_annual,
        "capacity_quantile": capacity_quantile,
        "max_resource_utilization": max_resource_utilization,
        "max_intervals": max_intervals,
        "verbose": verbose,
    }

    if verbose:
        print("[figure4a] Calculating the shared baseline result.", flush=True)
    baseline_result = _run_hourly_model_case(
        **common_run,
        workload_profile=baseline_profile,
        hardware_config=baseline_hardware,
        carbon_variability_scale=1.0,
        collect_timing_stress=True,
    )

    records: list[dict[str, object]] = []
    experiment_order = 0
    diagnostics = {name: [] for name in baseline_result["tables"] if name != "resource_proxy_metadata"}
    resource_changes = []
    control_checks = []
    temporal_changes = []
    parameter_changes = []
    mechanism_controls = []

    def collect_diagnostics(result, order, parameter_name, level, value, control_kind="sensitivity"):
        tag = {"experiment_order": order, "parameter_name": parameter_name, "level": level,
               "parameter_value": value, "control_kind": control_kind}
        for name in diagnostics:
            table = result["tables"][name].copy()
            for key, item in tag.items():
                table[key] = item
            diagnostics[name].append(table)
        keys = ["scenario", "policy", "year", "resource"]
        metrics = ["source_trace_resource_hours", "normalized_source_activity_hours", "after_global_cap_resource_hours",
                   "realized_resource_hours", "normalized_realized_activity_hours", "global_clipped_resource_hours", "country_clipped_resource_hours"]
        changes = result["tables"]["global_resource_audit"][keys + metrics].merge(
            baseline_result["tables"]["global_resource_audit"][keys + metrics],
            on=keys, suffixes=("_case", "_baseline"), validate="one_to_one")
        for metric in metrics:
            changes[f"{metric}_difference"] = changes[f"{metric}_case"] - changes[f"{metric}_baseline"]
        for key, item in tag.items():
            changes[key] = item
        resource_changes.append(changes)
        timing_keys = ["scenario", "policy", "year", "scope", "country"]
        timing_metrics = ["peak_power_mw", "peak_to_mean_ratio", "max_abs_ramp_mw_per_h",
                          "max_abs_ramp_excluding_seams_mw_per_h", "p99_abs_ramp_excluding_seams_mw_per_h",
                          "configured_facility_idle_floor_mw", "aligned_mean_idle_power_mw", "aligned_idle_energy_share"]
        timing = result["tables"]["temporal_metrics"][timing_keys + timing_metrics].merge(
            baseline_result["tables"]["temporal_metrics"][timing_keys + timing_metrics],
            on=timing_keys, suffixes=("_case", "_baseline"), validate="one_to_one")
        for metric in timing_metrics:
            timing[f"{metric}_difference"] = timing[f"{metric}_case"] - timing[f"{metric}_baseline"]
        for key, item in tag.items():
            timing[key] = item
        temporal_changes.append(timing)
        source_preserved = bool(np.allclose(changes["source_trace_resource_hours_case"], changes["source_trace_resource_hours_baseline"], rtol=1e-10, atol=1e-6))
        realized_preserved = bool(np.allclose(changes["normalized_realized_activity_hours_case"], changes["normalized_realized_activity_hours_baseline"], rtol=1e-10, atol=1e-6))
        annual_keys = ["scenario", "year", "country"]
        energy = result["tables"]["annual_summary"][annual_keys + ["facility_energy_mwh"]].merge(
            baseline_result["tables"]["annual_summary"][annual_keys + ["facility_energy_mwh"]],
            on=annual_keys, suffixes=("_case", "_baseline"), validate="one_to_one")
        energy_preserved = bool(np.allclose(energy["facility_energy_mwh_case"], energy["facility_energy_mwh_baseline"], rtol=1e-10, atol=1e-6))
        ci_keys = ["scenario", "policy", "year", "country", "scope"]
        ci = result["tables"]["grid_carbon_audit"][ci_keys + ["hourly_mean_ci_kg_per_mwh"]].merge(
            baseline_result["tables"]["grid_carbon_audit"][ci_keys + ["hourly_mean_ci_kg_per_mwh"]],
            on=ci_keys, suffixes=("_case", "_baseline"), validate="one_to_one")
        ci_preserved = bool(np.allclose(ci["hourly_mean_ci_kg_per_mwh_case"], ci["hourly_mean_ci_kg_per_mwh_baseline"], rtol=1e-10, atol=1e-6))
        if parameter_name == "hourly_carbon_variability_scale" and not (energy_preserved and ci_preserved):
            raise ValueError("CI-only sensitivity must preserve every country/year's electricity and mean CI.")
        if parameter_name in ("classified_inference_gpu_load_share", "pre_power_resource_variability_scale") and not source_preserved:
            raise ValueError("Task-mix / resource-shape sensitivity failed to preserve source resource totals.")
        parameter_keys = ["scenario", "policy", "year", "country", "resource"]
        parameter_metrics = ["model_resource_capacity", "trace_resource_capacity_at_quantile", "full_unit_power_w", "component_budget_mw", "idle_fraction",
                             "mean_realized_utilization", "peak_realized_utilization", "dynamic_response_exponent"]
        parameters = result["tables"]["capacity_parameter_audit"][parameter_keys + parameter_metrics].merge(
            baseline_result["tables"]["capacity_parameter_audit"][parameter_keys + parameter_metrics],
            on=parameter_keys, suffixes=("_case", "_baseline"), validate="one_to_one")
        for metric in parameter_metrics:
            parameters[f"{metric}_difference"] = parameters[f"{metric}_case"] - parameters[f"{metric}_baseline"]
        for key, item in tag.items():
            parameters[key] = item
        parameter_changes.append(parameters)
        normalized_utilization_preserved = all(np.allclose(parameters[f"{metric}_case"], parameters[f"{metric}_baseline"], rtol=1e-10, atol=1e-10)
                                              for metric in ("mean_realized_utilization", "peak_realized_utilization"))
        component_keys = ["scenario", "policy", "year", "country", "component"]
        component_energy = result["tables"]["component_energy"][component_keys + ["it_energy_mwh"]].merge(
            baseline_result["tables"]["component_energy"][component_keys + ["it_energy_mwh"]],
            on=component_keys, suffixes=("_case", "_baseline"), validate="one_to_one")
        component_energy_preserved = bool(np.allclose(component_energy["it_energy_mwh_case"], component_energy["it_energy_mwh_baseline"], rtol=1e-10, atol=1e-6))
        carbon = result["tables"]["annual_summary"][annual_keys + ["carbon_tco2"]].merge(
            baseline_result["tables"]["annual_summary"][annual_keys + ["carbon_tco2"]],
            on=annual_keys, suffixes=("_case", "_baseline"), validate="one_to_one")
        carbon_preserved = bool(np.allclose(carbon["carbon_tco2_case"], carbon["carbon_tco2_baseline"], rtol=1e-10, atol=1e-6))
        inverse_scale_applicable = parameter_name.endswith("unit_power_scale_fixed_idle_ratio") or parameter_name.endswith("unit_power_scale_fixed_idle_watts")
        inverse_capacity_pass = inverse_resource_pass = None
        if inverse_scale_applicable:
            resource = parameter_name.split("_", 1)[0]
            target = parameters.loc[parameters["resource"].eq(resource)]
            native = changes.loc[changes["resource"].eq(resource)]
            inverse_capacity_pass = bool(np.allclose(target["model_resource_capacity_case"] * value, target["model_resource_capacity_baseline"], rtol=1e-10, atol=1e-6))
            inverse_resource_pass = bool(np.allclose(native["realized_resource_hours_case"] * value, native["realized_resource_hours_baseline"], rtol=1e-10, atol=1e-6))
            if not (inverse_capacity_pass and inverse_resource_pass and normalized_utilization_preserved):
                raise ValueError("Unit-wattage control failed its capacity/resource inverse-scaling mechanism.")
        negative_pass = component_energy_preserved and energy_preserved and carbon_preserved and realized_preserved and normalized_utilization_preserved
        if control_kind == "negative_fixed_idle_ratio" and not negative_pass:
            raise ValueError("Fixed-idle/full unit-power negative control changed energy, carbon or normalized utilization.")
        mechanism_controls.append({**tag, "source_resource_totals_preserved": source_preserved,
                                   "normalized_utilization_summaries_preserved": normalized_utilization_preserved,
                                   "normalized_realized_activity_preserved": realized_preserved,
                                   "component_IT_electricity_preserved": component_energy_preserved,
                                   "country_year_facility_electricity_preserved": energy_preserved,
                                   "country_year_carbon_preserved": carbon_preserved,
                                   "capacity_inverse_unit_power_scale_pass": inverse_capacity_pass,
                                   "realized_resource_inverse_unit_power_scale_pass": inverse_resource_pass,
                                   "response_detected_in_component_energy": not component_energy_preserved,
                                   "response_detected_in_facility_energy": not energy_preserved,
                                   "negative_invariance_pass": negative_pass if control_kind == "negative_fixed_idle_ratio" else None,
                                   "baseline_level": level == "Base", "external_accuracy_validation": False,
                                   "configured_system_W_per_GPU_case": result["configured_system_W_per_GPU"],
                                   "configured_system_W_per_GPU_baseline": baseline_result["configured_system_W_per_GPU"],
                                   "conservation_checks_all_pass": result["tables"]["mechanism_conservation"]["status"].eq("PASS").all()})
        control_checks.append({**tag, "source_resource_totals_preserved": source_preserved,
                               "normalized_realized_activity_preserved": realized_preserved,
                               "country_year_electricity_preserved": energy_preserved, "country_year_mean_CI_preserved": ci_preserved,
                               "control_stage": "pre_power_resource_shape" if parameter_name == "pre_power_resource_variability_scale" else "model_parameter_or_source_load",
                               "electricity_change_not_inferred_from_resource_conservation": True})

    collect_diagnostics(baseline_result, 0, "baseline", "Base", np.nan)

    def add_experiment(
        *,
        sensitivity_group: str,
        parameter_name: str,
        values: Sequence[float],
        baseline_parameter_value: float,
        parameter_unit: str,
        profile_builder,
        hardware_builder,
        carbon_variability_builder,
        experiment_note: str,
        control_kind: str = "sensitivity",
    ) -> None:
        nonlocal experiment_order
        for value in values:
            experiment_order += 1
            level = _level_label(float(value), baseline_parameter_value)
            if level == "Base":
                result = baseline_result
            else:
                if verbose:
                    print(
                        f"[figure4a] {parameter_name}: {value:g} {parameter_unit}.",
                        flush=True,
                    )
                result = _run_hourly_model_case(
                    **common_run,
                    workload_profile=profile_builder(float(value)),
                    hardware_config=hardware_builder(float(value)),
                    carbon_variability_scale=carbon_variability_builder(
                        float(value)
                    ),
                )
            collect_diagnostics(result, experiment_order, parameter_name, level, float(value), control_kind)
            records.append(
                _result_record(
                    experiment_order=experiment_order,
                    sensitivity_group=sensitivity_group,
                    parameter_name=parameter_name,
                    level=level,
                    parameter_value=float(value),
                    parameter_unit=parameter_unit,
                    baseline_parameter_value=baseline_parameter_value,
                    scenario=scenario,
                    policy=policy,
                    year_start=year_start,
                    year_end=year_end,
                    years_averaged=years,
                    country_count=len(countries),
                    result=result,
                    baseline_result=baseline_result,
                    experiment_note=experiment_note,
                    control_kind=control_kind,
                )
            )

    unchanged_profile = lambda _value: baseline_profile
    unchanged_hardware = lambda _value: baseline_hardware
    unchanged_variability = lambda _value: 1.0

    for resource in ("cpu", "gpu"):
        for fixed_ratio in (True, False):
            constraint = "fixed_idle_ratio" if fixed_ratio else "fixed_idle_watts"
            full_w = baseline_hardware.cpu_full_power_w_per_core if resource == "cpu" else baseline_hardware.gpu_full_power_w
            idle_w = baseline_hardware.cpu_idle_power_w_per_core if resource == "cpu" else baseline_hardware.gpu_idle_power_w
            valid_scales = [scale for scale in sensitivity["unit_power_scales"] if fixed_ratio or full_w * scale >= idle_w]
            add_experiment(
                sensitivity_group=f"{resource.upper()} unit wattage mechanism",
                parameter_name=f"{resource}_unit_power_scale_{constraint}",
                values=valid_scales, baseline_parameter_value=1., parameter_unit="multiple_of_full_unit_watts",
                profile_builder=unchanged_profile,
                hardware_builder=lambda value, channel=resource, ratio=fixed_ratio: _hardware_with_unit_power_scale(baseline_hardware, channel, value, ratio),
                carbon_variability_builder=unchanged_variability,
                control_kind="negative_fixed_idle_ratio" if fixed_ratio else "fixed_idle_watts_mechanism",
                experiment_note=("Budget and source trace are fixed. Native model capacity/resource-hours scale inversely with unit watts. "
                                 + ("Idle watts scale too, preserving idle/full ratio; normalized utilization, component energy and matched carbon must be invariant."
                                    if fixed_ratio else "Idle watts stay fixed, changing idle/full ratio; energy invariance is not imposed.")),
            )
        exponent = f"{resource}_dynamic_exponent"
        add_experiment(
            sensitivity_group=f"{resource.upper()} dynamic power response",
            parameter_name=exponent, values=[getattr(baseline_hardware, exponent) * scale for scale in sensitivity["dynamic_exponent_scales"]],
            baseline_parameter_value=getattr(baseline_hardware, exponent), parameter_unit="response_exponent",
            profile_builder=unchanged_profile,
            hardware_builder=lambda value, name=exponent: replace(baseline_hardware, **{name: value}),
            carbon_variability_builder=unchanged_variability, control_kind="positive_response_curve",
            experiment_note="Synthetic response-curve contrast with fixed budgets, idle/full fractions and realized resources; not fitted to MLPerf system power.",
        )
    add_experiment(
        sensitivity_group="CPU idle power", parameter_name="cpu_idle_power_ratio",
        values=[ratio for ratio in (baseline_hardware.cpu_idle_power_w_per_core / baseline_hardware.cpu_full_power_w_per_core * scale for scale in sensitivity["idle_ratio_scales"]) if ratio <= 1.],
        baseline_parameter_value=baseline_hardware.cpu_idle_power_w_per_core / baseline_hardware.cpu_full_power_w_per_core,
        parameter_unit="fraction_of_CPU_full_power", profile_builder=unchanged_profile,
        hardware_builder=lambda value: replace(baseline_hardware, cpu_idle_power_w_per_core=baseline_hardware.cpu_full_power_w_per_core * value),
        carbon_variability_builder=unchanged_variability, control_kind="positive_idle_fraction",
        experiment_note="CPU full watts, capacity denominator and source resources stay fixed; only CPU idle/full ratio changes.",
    )
    add_experiment(
        sensitivity_group="CPU power budget", parameter_name="cpu_power_share",
        values=[share for share in (baseline_hardware.cpu_power_share * scale for scale in sensitivity["component_budget_scales"]) if share < 1.],
        baseline_parameter_value=baseline_hardware.cpu_power_share, parameter_unit="fraction_of_IT_full_power",
        profile_builder=unchanged_profile, hardware_builder=lambda value: _hardware_with_power_share(baseline_hardware, "cpu", value),
        carbon_variability_builder=unchanged_variability, control_kind="positive_component_budget",
        experiment_note="Other component shares are proportionally rescaled to preserve total IT budget; native resource capacities change with shares.",
    )

    add_experiment(
        sensitivity_group="Resource timing before power conversion",
        parameter_name="pre_power_resource_variability_scale",
        values=sensitivity["resource_variability_scales"],
        baseline_parameter_value=1.0,
        parameter_unit="multiple_of_source_deviation_from_task_resource_mean",
        profile_builder=lambda value: _resource_shape_control(baseline_profile, value),
        hardware_builder=unchanged_hardware,
        carbon_variability_builder=unchanged_variability,
        experiment_note=(
            "Synthetic pre-power shape control preserves every source task/resource total and fixed "
            "trace-capacity denominator. Capacity clipping, realized activity and nonlinear power "
            "are recalculated; electricity is not constrained. See W05 resource checks and W06 temporal metrics."
        ),
    )

    add_experiment(
        sensitivity_group="GPU idle power",
        parameter_name="gpu_idle_power_ratio",
        values=[ratio for ratio in (baseline_idle_ratio * scale for scale in sensitivity["idle_ratio_scales"]) if ratio <= 1.],
        baseline_parameter_value=baseline_idle_ratio,
        parameter_unit="fraction_of_gpu_full_power",
        profile_builder=unchanged_profile,
        hardware_builder=lambda value: _hardware_with_gpu_idle_ratio(
            baseline_hardware, value
        ),
        carbon_variability_builder=unchanged_variability,
        experiment_note=(
            "GPU idle watts equal the tested fraction of the unchanged GPU "
            "full-power watts."
        ),
        control_kind="positive_idle_fraction",
    )

    for task_type in ("training", "inference"):
        add_experiment(
            sensitivity_group="GPU utilization",
            parameter_name=f"{task_type}_gpu_load_scale",
            values=sensitivity["gpu_utilization_scales"],
            baseline_parameter_value=1.0,
            parameter_unit="multiple_of_baseline_gpu_load",
            profile_builder=lambda value, task=task_type: (
                _scale_task_gpu_utilization(baseline_profile, task, value)
            ),
            hardware_builder=unchanged_hardware,
            carbon_variability_builder=unchanged_variability,
            experiment_note=(
                f"Only {task_type} GPU load is scaled; other task/resource "
                "channels retain their baseline values."
            ),
        )

    task_mix_values = [
        float(
            np.clip(
                baseline_inference_share + shift,
                1e-6,
                1.0 - _classified_gpu_shares(baseline_profile)[2] - 1e-6,
            )
        )
        for shift in sensitivity["task_mix_inference_shifts"]
    ]
    add_experiment(
        sensitivity_group="Training-inference mix",
        parameter_name="classified_inference_gpu_load_share",
        values=task_mix_values,
        baseline_parameter_value=baseline_inference_share,
        parameter_unit="fraction_of_classified_gpu_load",
        profile_builder=lambda value: _change_inference_share(
            baseline_profile, value - baseline_inference_share
        )[0],
        hardware_builder=unchanged_hardware,
        carbon_variability_builder=unchanged_variability,
        experiment_note=(
            "Other-task share and annual resource totals are held constant; "
            "training absorbs the opposite inference-share change."
        ),
    )

    add_experiment(
        sensitivity_group="GPU power budget",
        parameter_name="gpu_power_share",
        values=[share for share in (baseline_hardware.gpu_power_share * scale for scale in sensitivity["component_budget_scales"]) if share < 1.],
        baseline_parameter_value=baseline_hardware.gpu_power_share,
        parameter_unit="fraction_of_it_full_power",
        profile_builder=unchanged_profile,
        hardware_builder=lambda value: _hardware_with_power_share(
            baseline_hardware, "gpu", value
        ),
        carbon_variability_builder=unchanged_variability,
        experiment_note=(
            "Non-GPU component shares are proportionally normalized so all "
            "IT component shares continue to sum to one. The finite-difference "
            "response therefore does not directly identify the baseline GPU energy fraction."
        ),
        control_kind="positive_component_budget",
    )

    add_experiment(
        sensitivity_group="Hourly carbon structure",
        parameter_name="hourly_carbon_variability_scale",
        values=sensitivity["carbon_variability_scales"],
        baseline_parameter_value=1.0,
        parameter_unit="multiple_of_baseline_deviation_from_annual_mean",
        profile_builder=unchanged_profile,
        hardware_builder=unchanged_hardware,
        carbon_variability_builder=lambda value: value,
        experiment_note=(
            "Hourly deviations from each country's annual mean are scaled; "
            "non-negative intensities and the original annual mean are preserved."
        ),
    )

    output_columns = [
        "experiment_order",
        "sensitivity_group",
        "parameter_name",
        "level",
        "is_baseline_level",
        "parameter_value",
        "parameter_unit",
        "baseline_parameter_value",
        "scenario",
        "policy",
        "year_start",
        "year_end",
        "years_averaged",
        "country_count",
        "experiment_note",
        "control_kind",
        "configured_system_W_per_GPU",
        "facility_energy_mwh_change_vs_baseline_pct",
        "carbon_tco2_change_vs_baseline_pct",
        "annual_mean_facility_energy_mwh",
        "baseline_annual_mean_facility_energy_mwh",
        "annual_mean_carbon_tco2",
        "baseline_annual_mean_carbon_tco2",
        "gpu_share_of_total_it_energy_pct",
        "unclassified_share_of_total_it_energy_pct",
        "idle_share_of_total_it_energy_pct",
        "idle_pool_share_of_total_it_energy_pct",
        "realized_classified_inference_gpu_share",
    ]
    output_frame = (
        pd.DataFrame.from_records(records)[output_columns]
        .sort_values("experiment_order", ignore_index=True)
    )
    resolved_output = Path(output_path)
    sheet_names = {"annual_summary": "W05_Annual_Diagnostics", "component_energy": "W05_Component_Energy",
                   "task_component_energy_all_rules": "W05_Task_Component_All_Rules", "idle_pool": "W05_Idle_Pool",
                   "allocation_audit": "W05_Allocation_Audit", "energy_conservation": "W05_Energy_Conservation",
                   "resource_audit": "W05_Resource_Audit", "global_resource_audit": "W05_Global_Resource_Audit",
                   "power_nonlinearity": "W05_Power_Nonlinearity", "grid_carbon_audit": "W05_Grid_Input_Audit",
                   "temporal_alignment": "W06_Trace_Alignment", "temporal_metrics": "W06_Model_Temporal_Metrics",
                   "trace_seams": "W06_Model_Seams", "capacity_parameter_audit": "W07_Capacity_Parameters",
                   "hardware_parameters": "W07_Hardware_Parameters", "parameter_dependencies": "W07_Parameter_Dependencies",
                   "mechanism_conservation": "W07_Mechanism_Conservation"}
    extra_sheets = {sheet_names[name]: pd.concat(tables, ignore_index=True) for name, tables in diagnostics.items()}
    extra_sheets["W05_Resource_Changes"] = pd.concat(resource_changes, ignore_index=True)
    extra_sheets["W05_Control_Checks"] = pd.DataFrame(control_checks)
    extra_sheets["W06_Metric_Changes"] = pd.concat(temporal_changes, ignore_index=True)
    extra_sheets["W07_Parameter_Changes"] = pd.concat(parameter_changes, ignore_index=True)
    extra_sheets["W07_Control_Evidence"] = pd.DataFrame(mechanism_controls)
    extra_sheets.update(baseline_result["timing_stress"])
    extra_sheets["W05_Proxy_Metadata"] = baseline_result["tables"]["resource_proxy_metadata"]
    extra_sheets["W05_Run_Metadata"] = pd.DataFrame([{
        "idle_allocation_rule": idle_allocation_rule, "trace_intervals": baseline_profile.n_intervals,
        "scenario": scenario, "policy": policy, "year_start": year_start, "years": years,
        "carbon_scope": hourly_carbon_scope,
        "trace_hours": baseline_profile.trace_hours, "max_intervals": max_intervals,
        "trace_truncated_by_max_intervals": max_intervals is not None,
        "share_denominator": "all_four_tasks_all_five_components_physical_IT_energy",
        "separate_pool_rule": "task_energy_plus_unallocated_idle_equals_total_IT",
    }])
    sheets = prepare_figure_sources(4, {
        "Fig4a_Sensitivity": output_frame,
        "Accounting_Boundary": build_boundary_metadata(countries, ai_capacity_factors),
        "AI_Capacity_Factors": capacity_factor_table(year_start, years, ai_capacity_factors),
        **extra_sheets,
    }, carbon_scope=hourly_carbon_scope)
    sheets["Shared_Sensitivity_Ranges"] = pd.DataFrame([
        {"range": name, "value": float(value)} for name, values in sensitivity.items() for value in values])
    if experiment_context is not None:
        sheets["Shared_Experiment_Configuration"] = experiment_context.metadata_table()
    write_workbook(resolved_output, sheets)
    if verbose:
        print(f"[figure4a] Excel workbook saved to {resolved_output.resolve()}.")
    return resolved_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Excel calculation results required by Figure 4a."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario", default="Base")
    parser.add_argument("--policy", choices=("CP", "NDC", "NZ"), default="CP")
    parser.add_argument("--year-start", type=int, default=2026)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--countries", nargs="+", default=list(DEFAULT_COUNTRIES))
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
    parser.add_argument("--idle-allocation-rule", choices=IDLE_ALLOCATION_RULES, default="hourly_driver")
    parser.add_argument("--capacity-quantile", type=float, default=0.96)
    parser.add_argument("--max-resource-utilization", type=float, default=1.0)
    parser.add_argument(
        "--max-intervals",
        type=int,
        default=None,
        help="Debugging only; omit for formal paper results.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--hardware-config-file", type=Path, default=None,
                        help="JSON HardwarePowerConfig overrides; all experiments record their actual configuration.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_figure4a_data(
        hardware_config=(load_hardware_config(args.hardware_config_file, HardwarePowerConfig())
                         if args.hardware_config_file else None),
        idle_allocation_rule=args.idle_allocation_rule,
        output_path=args.output,
        scenario=args.scenario,
        policy=args.policy,
        year_start=args.year_start,
        years=args.years,
        countries=args.countries,
        ai_capacity_factors=(load_ai_capacity_factors(args.ai_capacity_factors_file, args.countries)
                             if args.ai_capacity_factors_file else None),
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
