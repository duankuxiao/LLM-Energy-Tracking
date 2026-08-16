"""Generate the calculation results required by Figure 1.

The script writes one Excel workbook.  Each worksheet corresponds to one
Figure 1 panel; no plotting code is included.
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

import core.m4_hourly_gpu_model as m4_model
from core.m4_hourly_gpu_model import (
    COMPONENTS,
    TASK_TYPES,
    Alibaba2026TraceConfig,
    HardwarePowerConfig,
    WorkloadProfile,
    build_workload_profile,
    run_workload_component_footprint,
)
from dataset.Installed_capacity_data import (
    DEFAULT_AI_CAPACITY_FACTORS,
    DEFAULT_COUNTRIES,
    IT_CAPACITY,
    IT_RATIO,
    TOTAL_CAPACITY,
)


DEFAULT_OUTPUT = ROOT_DIR / "results" / "figure1_data.xlsx"
DEFAULT_SCENARIOS = ("Base", "Lift-Off", "High Efficiency", "Headwinds")
DATA_YEAR_START = 2025
DISPLAY_COMPONENTS = ("cpu", "gpu", "memory", "it_fan")
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


def _capacity_table(
    scenarios: Sequence[str],
    year_start: int,
    years: int,
) -> pd.DataFrame:
    """Build total and AI capacity results for Figure 1a."""
    records = []
    for scenario in scenarios:
        scenario_col = m4_model.SCENARIO_COL_MAP[scenario]
        for year in range(year_start, year_start + years):
            year_row = year - DATA_YEAR_START
            total_capacity_gw = float(TOTAL_CAPACITY[year_row, scenario_col])
            it_capacity_gw = float(IT_CAPACITY[year_row, scenario_col])
            ai_capacity_factor = float(DEFAULT_AI_CAPACITY_FACTORS[year])
            ai_gpu_it_capacity_gw = it_capacity_gw * ai_capacity_factor
            records.append(
                {
                    "year": year,
                    "scenario": scenario,
                    "total_capacity_gw": total_capacity_gw,
                    "ai_gpu_it_capacity_gw": ai_gpu_it_capacity_gw,
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        ["year", "scenario"], ignore_index=True
    )


def _country_share_table(countries: Sequence[str]) -> pd.DataFrame:
    """Build the descending country IT-capacity share table for Figure 1b."""
    frame = pd.DataFrame(
        {
            "country": [COUNTRY_ISO3[country] for country in countries],
            "it_capacity_share": [float(IT_RATIO[country]) for country in countries],
        }
    )
    frame["it_capacity_share_pct"] = frame["it_capacity_share"] * 100.0
    frame = frame.sort_values(
        "it_capacity_share", ascending=False, ignore_index=True
    )
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def _base_task_component_average(
    task_components: pd.DataFrame,
    task_type: str,
) -> pd.DataFrame:
    """Average one task's component results across the Base-scenario years."""
    base_rows = task_components.loc[
        (task_components["scenario"] == "Base")
        & (task_components["task_type"] == task_type)
    ].copy()
    if base_rows.empty:
        raise ValueError(
            "Figure 1c-e require the Base scenario, but no Base results were "
            "calculated. Include 'Base' in scenarios."
        )

    averaged_years = sorted(base_rows["year"].unique())
    averaged = (
        base_rows.groupby(
            ["scenario", "policy", "task_type", "component"],
            as_index=False,
            sort=False,
        )
        .agg(
            annual_avg_it_energy_mwh=("it_energy_mwh", "mean"),
            annual_avg_displayed_components_it_energy_mwh=(
                "displayed_components_it_energy_mwh",
                "mean",
            ),
        )
    )

    displayed_energy = averaged[
        "annual_avg_displayed_components_it_energy_mwh"
    ]
    averaged["display_share_pct"] = np.where(
        displayed_energy > 0,
        averaged["annual_avg_it_energy_mwh"] / displayed_energy * 100.0,
        np.nan,
    )

    averaged.insert(2, "year_start", int(averaged_years[0]))
    averaged.insert(3, "year_end", int(averaged_years[-1]))
    averaged.insert(4, "years_averaged", len(averaged_years))
    return averaged[
        [
            "scenario",
            "policy",
            "year_start",
            "year_end",
            "years_averaged",
            "task_type",
            "component",
            "annual_avg_displayed_components_it_energy_mwh",
            "display_share_pct",
        ]
    ]


def _capture_task_component_energy(
    *,
    policy: str,
    scenario: str,
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
    """Run one scenario and retain M4's exact task-by-component allocation.

    M4 already calculates the task-by-component matrix internally for its
    task-energy result.  The wrapper records that matrix during this process
    and immediately restores the original function, so no core file is
    modified and the allocation remains identical to M4's own calculation.
    """
    captured_allocations: list[np.ndarray] = []
    original_allocator = m4_model._allocate_energy_to_task_types

    def recording_allocator(*args, **kwargs) -> np.ndarray:
        allocation = original_allocator(*args, **kwargs)
        captured_allocations.append(np.array(allocation, copy=True))
        return allocation

    m4_model._allocate_energy_to_task_types = recording_allocator
    try:
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
    finally:
        m4_model._allocate_energy_to_task_types = original_allocator

    if len(captured_allocations) != years:
        raise RuntimeError(
            "Unexpected number of M4 task-component allocations: "
            f"expected {years}, received {len(captured_allocations)}."
        )

    task_component_records = []
    for year_offset, allocation in enumerate(captured_allocations):
        year = year_start + year_offset
        # Shape: task type x country x IT component.
        global_allocation = allocation.sum(axis=1)
        for task_id, task_type in enumerate(TASK_TYPES):
            all_component_energy = float(global_allocation[task_id].sum())
            displayed_energy = float(
                sum(
                    global_allocation[task_id, COMPONENTS.index(component)]
                    for component in DISPLAY_COMPONENTS
                )
            )
            excluded_storage = float(
                global_allocation[task_id, COMPONENTS.index("storage")]
            )
            for component in DISPLAY_COMPONENTS:
                energy = float(
                    global_allocation[task_id, COMPONENTS.index(component)]
                )
                task_component_records.append(
                    {
                        "scenario": scenario,
                        "policy": policy,
                        "year": year,
                        "task_type": task_type,
                        "component": component,
                        "it_energy_mwh": energy,
                        "display_share_pct": (
                            energy / displayed_energy * 100.0
                            if displayed_energy > 0
                            else np.nan
                        ),
                        "share_of_all_task_it_energy_pct": (
                            energy / all_component_energy * 100.0
                            if all_component_energy > 0
                            else np.nan
                        ),
                        "displayed_components_it_energy_mwh": displayed_energy,
                        "all_components_it_energy_mwh": all_component_energy,
                        "excluded_storage_it_energy_mwh": excluded_storage,
                    }
                )

    return result["annual_summary"].copy(), pd.DataFrame(task_component_records)


def _write_workbook(output_path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions


def generate_figure1_data(
    output_path: Union[str, Path] = DEFAULT_OUTPUT,
    policy: str = "CP",
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    countries: Sequence[str] = DEFAULT_COUNTRIES,
    year_start: int = 2026,
    years: int = 5,
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
    """Calculate all Figure 1 panel data and write one Excel workbook."""
    scenarios = list(scenarios)
    countries = list(countries)

    if verbose:
        print("[figure1] Building the shared GPU workload profile.", flush=True)
    profile = build_workload_profile(
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        capacity_quantile=capacity_quantile,
        max_intervals=max_intervals,
        verbose=verbose,
    )

    annual_frames = []
    task_component_frames = []
    for scenario in scenarios:
        if verbose:
            print(f"[figure1] Calculating M4 results for {scenario}.", flush=True)
        annual, task_components = _capture_task_component_energy(
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
            verbose=verbose,
        )
        annual_frames.append(annual)
        task_component_frames.append(task_components)

    annual_summary = pd.concat(annual_frames, ignore_index=True)
    annual_summary.insert(1, "policy", policy)
    task_components = pd.concat(task_component_frames, ignore_index=True)

    global_annual = (
        annual_summary.groupby(
            ["scenario", "policy", "year"], as_index=False
        )[["facility_energy_mwh", "carbon_tco2"]]
        .sum()
        .sort_values(["year", "scenario"], ignore_index=True)
    )
    global_annual["facility_energy_twh"] = (
        global_annual["facility_energy_mwh"] / 1e6
    )
    global_annual["carbon_mtco2"] = global_annual["carbon_tco2"] / 1e6

    task_sheets = {}
    for task_type, sheet_name in (
        ("training", "Fig1c_Training"),
        ("inference", "Fig1d_Inference"),
        ("other", "Fig1e_Other"),
    ):
        task_sheets[sheet_name] = _base_task_component_average(
            task_components,
            task_type,
        )

    sheets = {
        "Fig1a_Capacity": _capacity_table(scenarios, year_start, years),
        "Fig1b_Country_Share": _country_share_table(countries),
        **task_sheets,
        "Fig1f_Energy": global_annual[
            [
                "scenario",
                "policy",
                "year",
                "facility_energy_twh",
            ]
        ],
        "Fig1g_Carbon": global_annual[
            ["scenario", "policy", "year", "carbon_mtco2"]
        ],
    }

    resolved_output = Path(output_path)
    _write_workbook(resolved_output, sheets)
    if verbose:
        print(f"[figure1] Excel workbook saved to {resolved_output.resolve()}.")
    return resolved_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Excel calculation results required by Figure 1."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", choices=("CP", "NDC", "NZ"), default="CP")
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--countries", nargs="+", default=list(DEFAULT_COUNTRIES))
    parser.add_argument("--year-start", type=int, default=2026)
    parser.add_argument("--years", type=int, default=5)
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
    generate_figure1_data(
        output_path=args.output,
        policy=args.policy,
        scenarios=args.scenarios,
        countries=args.countries,
        year_start=args.year_start,
        years=args.years,
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
