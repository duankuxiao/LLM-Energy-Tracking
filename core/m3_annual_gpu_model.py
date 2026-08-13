"""M3: workload-driven GPU energy model with annual grid carbon factors."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import pandas as pd

from core.m4_hourly_gpu_model import (
    Alibaba2026TraceConfig,
    HardwarePowerConfig,
    WorkloadProfile,
    run_workload_component_footprint,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_NAME = "M3"


def run_m3_annual_gpu_model(
    renewable_energy_policy: str,
    scenarios: Sequence[str],
    years: int = 5,
    countries: Optional[Sequence[str]] = None,
    year_start: int = 2026,
    workload_profile_path: Union[str, Path] = ROOT_DIR / "dataset",
    server_profile_path: Optional[Union[str, Path]] = None,
    hardware_config: Optional[HardwarePowerConfig] = None,
    trace_config: Optional[Alibaba2026TraceConfig] = None,
    task_origin_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
    task_execution_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
    execution_policy: str = "capacity",
    inference_origin_fraction: float = 0.75,
    other_origin_fraction: float = 0.50,
    capacity_quantile: float = 0.96,
    max_resource_utilization: float = 1.0,
    pue_scale: float = 1.0,
    ai_capacity_factors: Optional[Mapping[int, float]] = None,
    max_intervals: Optional[int] = None,
    output_dir: Union[str, Path] = ROOT_DIR / "results" / "m3_annual_gpu",
    save_outputs: bool = True,
    verbose: bool = True,
    workload_profile: Optional[WorkloadProfile] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run M3 using ``m4_hourly_gpu_model.py`` for annual energy.

    Hourly grid matching is explicitly disabled. Carbon is calculated inside
    the workload-component model from the policy-specific annual factors in
    ``dataset/Factors.py``. Only country and global annual tables are returned
    and saved by this wrapper. Pass a prebuilt ``workload_profile`` when M3
    and M4 are run together to avoid reading the large trace dataset twice.
    """
    raw = run_workload_component_footprint(
        renewable_energy_policy=renewable_energy_policy,
        scenarios=scenarios,
        years=years,
        countries=countries,
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        year_start=year_start,
        save_outputs=False,
        verbose=verbose,
        hardware_config=hardware_config,
        trace_config=trace_config,
        task_origin_weights=task_origin_weights,
        task_execution_weights=task_execution_weights,
        execution_policy=execution_policy,
        inference_origin_fraction=inference_origin_fraction,
        other_origin_fraction=other_origin_fraction,
        capacity_quantile=capacity_quantile,
        max_resource_utilization=max_resource_utilization,
        pue_scale=pue_scale,
        ai_capacity_factors=ai_capacity_factors,
        hourly_carbon_factors_dir=None,
        save_hourly_outputs=False,
        max_intervals=max_intervals,
        workload_profile=workload_profile,
    )

    annual_summary = raw["annual_summary"].copy()
    annual_summary.insert(0, "model", MODEL_NAME)
    annual_summary.insert(2, "policy", renewable_energy_policy)
    annual_summary["carbon_factor_kg_per_mwh"] = (
        annual_summary["carbon_tco2"] * 1000.0 / annual_summary["facility_energy_mwh"]
    ).where(annual_summary["facility_energy_mwh"] > 0, 0.0)

    global_summary = (
        annual_summary.groupby(["model", "scenario", "policy", "year"], as_index=False)[
            ["facility_energy_mwh", "power_twh", "carbon_tco2", "carbon_mtco2"]
        ].sum()
    )

    if save_outputs:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        annual_summary.to_csv(output_path / "M3_Country_Annual.csv", index=False)
        global_summary.to_csv(output_path / "M3_Global_Annual.csv", index=False)

    if verbose:
        print(global_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        if save_outputs:
            print(f"Saved M3 annual results to: {Path(output_dir).resolve()}")

    return {"annual_summary": annual_summary, "global_summary": global_summary}


if __name__ == "__main__":
    run_m3_annual_gpu_model(
        renewable_energy_policy="CP",
        scenarios=["Base"],
    )
