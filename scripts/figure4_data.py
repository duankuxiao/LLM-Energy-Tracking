"""Generate the M4 sensitivity results required by Figure 4a.

The script performs one-at-a-time sensitivity experiments for five core
dimensions that affect the paper's M4 energy and carbon conclusions:

1. GPU idle-power ratio;
2. training and inference GPU utilization;
3. training/inference workload mix;
4. GPU share of the IT full-power budget; and
5. hourly carbon-intensity variability.

Only calculation results are produced.  The output workbook contains one
worksheet, ``Fig4a_M4_Sensitivity``, and no plotting code is included.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import core.m4_hourly_gpu_model as m4_model
from core.m4_hourly_gpu_model import (
    RESOURCES,
    Alibaba2026TraceConfig,
    HardwarePowerConfig,
    WorkloadProfile,
    build_workload_profile,
    run_workload_component_footprint,
)
from core.task_model import TASK_TYPES
from dataset.Installed_capacity_data import DEFAULT_COUNTRIES


DEFAULT_OUTPUT = ROOT_DIR / "results" / "figure4_data.xlsx"

GPU_IDLE_POWER_RATIOS = (0.05, 0.10, 0.15)
GPU_UTILIZATION_SCALES = (0.80, 1.00, 1.20)
TASK_MIX_INFERENCE_SHIFTS = (-0.30, 0.00, 0.30)
GPU_POWER_SHARES = (0.40, 0.50, 0.60)
CARBON_VARIABILITY_SCALES = (0.50, 1.00, 1.50)


def _copy_profile_with_load(
    profile: WorkloadProfile,
    load: np.ndarray,
) -> WorkloadProfile:
    """Return a workload profile that shares metadata but owns its load array."""
    return WorkloadProfile(
        interval_index=profile.interval_index.copy(),
        interval_hours=profile.interval_hours,
        load=np.asarray(load, dtype=np.float64),
        trace_capacity=np.array(profile.trace_capacity, copy=True),
        task_counts=np.array(profile.task_counts, copy=True),
        task_type_summary=profile.task_type_summary.copy(),
        relative_time=profile.relative_time,
        trace_capacity_source=profile.trace_capacity_source,
    )


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
    return _copy_profile_with_load(profile, load)


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

    return _copy_profile_with_load(profile, load), target_inference_share


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


def _hardware_with_gpu_power_share(
    baseline: HardwarePowerConfig,
    gpu_power_share: float,
) -> HardwarePowerConfig:
    """Change the GPU power share and proportionally normalize other shares."""
    if not 0 < gpu_power_share < 1:
        raise ValueError("GPU power share must be in (0, 1).")

    baseline_other_total = 1.0 - baseline.gpu_power_share
    target_other_total = 1.0 - float(gpu_power_share)
    other_scale = target_other_total / baseline_other_total
    config = replace(
        baseline,
        cpu_power_share=baseline.cpu_power_share * other_scale,
        gpu_power_share=float(gpu_power_share),
        memory_power_share=baseline.memory_power_share * other_scale,
        storage_power_share=baseline.storage_power_share * other_scale,
        it_fan_power_share=baseline.it_fan_power_share * other_scale,
    )
    config.validate()
    return config


def _preserve_row_means_after_clipping(
    values: np.ndarray,
    target_means: np.ndarray,
) -> np.ndarray:
    """Clip carbon intensities at zero and restore each country's mean."""
    clipped = np.maximum(values, 0.0)
    clipped_means = clipped.mean(axis=1, keepdims=True)
    return np.divide(
        clipped * target_means,
        clipped_means,
        out=np.zeros_like(clipped),
        where=clipped_means > 0,
    )


@contextmanager
def _hourly_carbon_transformation(
    variability_scale: float = 1.0,
) -> Iterator[None]:
    """Temporarily transform M4 hourly factors, then restore the core loader."""
    if variability_scale < 0:
        raise ValueError("Carbon-variability scale must be non-negative.")

    original_loader = m4_model._load_hourly_carbon_factors

    def transformed_loader(*args, **kwargs):
        timestamps, factors, sources = original_loader(*args, **kwargs)
        transformed = np.asarray(factors, dtype=np.float64).copy()
        original_means = transformed.mean(axis=1, keepdims=True)
        transformed = original_means + float(variability_scale) * (
            transformed - original_means
        )
        transformed = _preserve_row_means_after_clipping(
            transformed, original_means
        )
        return timestamps, transformed, sources

    m4_model._load_hourly_carbon_factors = transformed_loader
    try:
        yield
    finally:
        m4_model._load_hourly_carbon_factors = original_loader


def _run_m4_case(
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
) -> dict[str, float]:
    with _hourly_carbon_transformation(
        variability_scale=carbon_variability_scale,
    ):
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
            hardware_config=hardware_config,
            capacity_quantile=capacity_quantile,
            max_resource_utilization=max_resource_utilization,
            hourly_carbon_factors_dir=hourly_carbon_factors_dir,
            hourly_carbon_scope=hourly_carbon_scope,
            hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
            save_hourly_outputs=False,
            max_intervals=max_intervals,
            workload_profile=workload_profile,
        )

    annual = result["annual_summary"]
    annual_global = annual.groupby("year", as_index=False)[
        ["facility_energy_mwh", "carbon_tco2"]
    ].sum()
    if len(annual_global) != years:
        raise RuntimeError(
            "Unexpected number of annual M4 sensitivity results: "
            f"expected {years}, received {len(annual_global)}."
        )
    facility_energy_mwh = float(annual_global["facility_energy_mwh"].mean())
    carbon_tco2 = float(annual_global["carbon_tco2"].mean())
    return {
        "m4_facility_energy_mwh": facility_energy_mwh,
        "m4_carbon_tco2": carbon_tco2,
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
    result: dict[str, float],
    baseline_result: dict[str, float],
    experiment_note: str,
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
    }
    for metric in (
        "m4_facility_energy_mwh",
        "m4_carbon_tco2",
    ):
        baseline_value = baseline_result[metric]
        difference = result[metric] - baseline_value
        metric_prefix = metric.removeprefix("m4_")
        record[f"{metric_prefix}_change_vs_baseline_pct"] = (
            difference / baseline_value * 100.0
            if baseline_value != 0
            else np.nan
        )
    return record


def _level_label(value: float, baseline: float) -> str:
    if np.isclose(value, baseline):
        return "Base"
    return "Low" if value < baseline else "High"


def _write_workbook(output_path: Path, frame: pd.DataFrame) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Fig4a_M4_Sensitivity", index=False)
        worksheet = writer.sheets["Fig4a_M4_Sensitivity"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions


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
) -> Path:
    """Run the five core M4 sensitivity groups and write Figure 4a data."""
    countries = list(countries)
    if years <= 0:
        raise ValueError("years must be a positive integer.")
    year_end = year_start + years - 1
    baseline_hardware = HardwarePowerConfig()
    baseline_idle_ratio = (
        baseline_hardware.gpu_idle_power_w
        / baseline_hardware.gpu_full_power_w
    )

    if verbose:
        print("[figure4a] Building the shared GPU workload profile.", flush=True)
    baseline_profile = build_workload_profile(
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        capacity_quantile=capacity_quantile,
        trace_config=Alibaba2026TraceConfig(),
        max_intervals=max_intervals,
        verbose=verbose,
    )
    baseline_inference_share = float(_classified_gpu_shares(baseline_profile)[1])

    common_run = {
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
    baseline_result = _run_m4_case(
        **common_run,
        workload_profile=baseline_profile,
        hardware_config=baseline_hardware,
        carbon_variability_scale=1.0,
    )

    records: list[dict[str, object]] = []
    experiment_order = 0

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
                result = _run_m4_case(
                    **common_run,
                    workload_profile=profile_builder(float(value)),
                    hardware_config=hardware_builder(float(value)),
                    carbon_variability_scale=carbon_variability_builder(
                        float(value)
                    ),
                )
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
                )
            )

    unchanged_profile = lambda _value: baseline_profile
    unchanged_hardware = lambda _value: baseline_hardware
    unchanged_variability = lambda _value: 1.0

    add_experiment(
        sensitivity_group="GPU idle power",
        parameter_name="gpu_idle_power_ratio",
        values=GPU_IDLE_POWER_RATIOS,
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
    )

    for task_type in ("training", "inference"):
        add_experiment(
            sensitivity_group="GPU utilization",
            parameter_name=f"{task_type}_gpu_load_scale",
            values=GPU_UTILIZATION_SCALES,
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
        for shift in TASK_MIX_INFERENCE_SHIFTS
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
        values=GPU_POWER_SHARES,
        baseline_parameter_value=baseline_hardware.gpu_power_share,
        parameter_unit="fraction_of_it_full_power",
        profile_builder=unchanged_profile,
        hardware_builder=lambda value: _hardware_with_gpu_power_share(
            baseline_hardware, value
        ),
        carbon_variability_builder=unchanged_variability,
        experiment_note=(
            "Non-GPU component shares are proportionally normalized so all "
            "IT component shares continue to sum to one."
        ),
    )

    add_experiment(
        sensitivity_group="Hourly carbon structure",
        parameter_name="hourly_carbon_variability_scale",
        values=CARBON_VARIABILITY_SCALES,
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
        "facility_energy_mwh_change_vs_baseline_pct",
        "carbon_tco2_change_vs_baseline_pct",
    ]
    output_frame = (
        pd.DataFrame.from_records(records)[output_columns]
        .sort_values("experiment_order", ignore_index=True)
    )
    resolved_output = Path(output_path)
    _write_workbook(resolved_output, output_frame)
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
    generate_figure4a_data(
        output_path=args.output,
        scenario=args.scenario,
        policy=args.policy,
        year_start=args.year_start,
        years=args.years,
        countries=args.countries,
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
