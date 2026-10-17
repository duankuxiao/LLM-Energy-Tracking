"""Generate the calculation results required by Figure 1.

The workbook retains legacy panel tables and adds W08 display, attribution
and provenance tables. Render it separately with scripts/plot_figures.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.accounting_boundary import build_boundary_metadata, capacity_factor_table, load_ai_capacity_factors, resolve_countries, resolve_ai_capacity_factors
from core.energy_attribution import IDLE_ALLOCATION_RULES, allocation_share_tables
from core.research_io import write_workbook
from scripts.figure_contract import SCENARIO_COLUMNS, prepare_figure_sources
from core.hourly_workload_resolved_model import (
    COMPONENTS,
    WorkloadProfile,
    build_workload_profile,
    run_workload_component_footprint,
)
from dataset.Installed_capacity_data import (
    DEFAULT_COUNTRIES,
    COUNTRY_ISO3,
    IT_CAPACITY,
    IT_RATIO,
    TOTAL_CAPACITY,
    TOTAL_RATIO,
)


DEFAULT_OUTPUT = ROOT_DIR / "results" / "figure1_data.xlsx"
DEFAULT_SCENARIOS = ("Base", "Lift-Off", "High Efficiency", "Headwinds")
DATA_YEAR_START = 2025
DISPLAY_COMPONENTS = COMPONENTS

def _capacity_table(
    scenarios: Sequence[str],
    year_start: int,
    years: int,
    countries: Sequence[str] = DEFAULT_COUNTRIES,
    ai_capacity_factors: Optional[Mapping[int, float]] = None,
) -> pd.DataFrame:
    """Build selected-country capacities, retaining explicit global references."""
    countries = resolve_countries(countries)
    it_coverage = sum(IT_RATIO[country] for country in countries)
    total_coverage = sum(TOTAL_RATIO[country] for country in countries)
    factors = resolve_ai_capacity_factors(range(year_start, year_start + years), ai_capacity_factors)
    if year_start < DATA_YEAR_START or year_start + years > DATA_YEAR_START + len(IT_CAPACITY):
        raise ValueError("Figure 1 capacity years must lie within the source matrix.")
    records = []
    for scenario in scenarios:
        scenario_col = SCENARIO_COLUMNS[scenario]
        for year in range(year_start, year_start + years):
            year_row = year - DATA_YEAR_START
            total_capacity_gw = float(TOTAL_CAPACITY[year_row, scenario_col])
            it_capacity_gw = float(IT_CAPACITY[year_row, scenario_col])
            ai_capacity_factor = factors[year]
            ai_gpu_it_capacity_gw = it_capacity_gw * ai_capacity_factor
            records.append(
                {
                    "year": year,
                    "scenario": scenario,
                    "source_scenario_column": scenario_col,
                    "source_year_row": year_row,
                    "total_capacity_gw": total_capacity_gw * total_coverage,
                    "it_capacity_gw": it_capacity_gw * it_coverage,
                    "ai_gpu_it_capacity_gw": ai_gpu_it_capacity_gw * it_coverage,
                    "ai_capacity_factor": ai_capacity_factor,
                    "source_global_total_capacity_gw": total_capacity_gw,
                    "source_global_it_capacity_gw": it_capacity_gw,
                    "source_global_ai_it_capacity_gw": ai_gpu_it_capacity_gw,
                    "accounting_scope": "selected_countries",
                    "total_capacity_share_sum": total_coverage,
                    "it_capacity_share_sum": it_coverage,
                    "display_rule": "parallel_bars_never_total_plus_AI_stack",
                    "ai_capacity_definition": "AI_focused_IT_capacity_includes_all_five_IT_components_not_GPU_only",
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
    frame["share_within_selected_countries_pct"] = frame["it_capacity_share"] / frame["it_capacity_share"].sum() * 100
    frame["input_share_denominator"] = "original_global_capacity_allocation"
    frame["source_base_year"] = "not_documented_in_local_inputs"
    frame["temporal_basis"] = "fixed_input_shares_not_a_multiyear_average"
    frame = frame.sort_values(
        "it_capacity_share", ascending=False, ignore_index=True
    )
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def _base_task_component_average(
    task_components: pd.DataFrame,
    task_type: str,
) -> pd.DataFrame:
    """Ratio of summed energy across years, with all five components included."""
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
            annual_avg_idle_it_energy_mwh=("idle_it_energy_mwh", "mean"),
            annual_avg_dynamic_it_energy_mwh=("dynamic_it_energy_mwh", "mean"),
            annual_avg_facility_energy_mwh=("facility_energy_mwh", "mean"),
            annual_avg_task_it_energy_mwh=("task_it_energy_mwh", "mean"),
            annual_avg_system_it_energy_mwh=("system_it_energy_mwh", "mean"),
        )
    )

    displayed_energy = averaged["annual_avg_task_it_energy_mwh"]
    averaged["display_share_pct"] = np.divide(
        averaged["annual_avg_it_energy_mwh"].to_numpy() * 100., displayed_energy.to_numpy(),
        out=np.full(len(averaged), np.nan), where=displayed_energy.to_numpy() > 0)
    averaged["share_of_system_it_energy_pct"] = np.divide(
        averaged["annual_avg_it_energy_mwh"].to_numpy() * 100.,
        averaged["annual_avg_system_it_energy_mwh"].to_numpy(),
        out=np.full(len(averaged), np.nan), where=averaged["annual_avg_system_it_energy_mwh"].to_numpy() > 0)
    averaged["denominator"] = "assigned_task_IT_energy_including_all_five_components"
    averaged["idle_allocation_rule"] = str(base_rows["idle_allocation_rule"].iloc[0])

    averaged.insert(2, "year_start", int(averaged_years[0]))
    averaged.insert(3, "year_end", int(averaged_years[-1]))
    averaged.insert(4, "years_averaged", len(averaged_years))
    return averaged


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
    ai_capacity_factors: Optional[Mapping[int, float]] = None,
    idle_allocation_rule: str = "hourly_driver",
    model_runner=None,
) -> dict[str, pd.DataFrame]:
    """Use the model's public matrix and diagnostics directly."""
    result = (model_runner or run_workload_component_footprint)(
        renewable_energy_policy=policy, scenarios=[scenario], years=years, countries=countries,
        workload_profile_path=workload_profile_path, server_profile_path=server_profile_path,
        year_start=year_start, save_outputs=False, verbose=verbose,
        capacity_quantile=capacity_quantile, max_resource_utilization=max_resource_utilization,
        hourly_carbon_factors_dir=hourly_carbon_factors_dir, hourly_carbon_scope=hourly_carbon_scope,
        hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
        save_hourly_outputs=False, max_intervals=max_intervals, workload_profile=profile,
        ai_capacity_factors=ai_capacity_factors, idle_allocation_rule=idle_allocation_rule)
    return {name: result[name] for name in (
        "annual_summary", "component_energy", "task_component_energy", "task_component_energy_all_rules",
        "idle_pool", "allocation_audit", "energy_conservation", "resource_audit", "global_resource_audit",
        "power_nonlinearity", "resource_proxy_metadata", "grid_carbon_audit")}


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
    ai_capacity_factors: Optional[Mapping[int, float]] = None,
    idle_allocation_rule: str = "hourly_driver",
    experiment_context=None,
) -> Path:
    """Calculate all Figure 1 panel data and write one Excel workbook."""
    scenarios = list(scenarios)
    countries = resolve_countries(countries)
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("Figure 1 scenarios must be unique.")
    if "Base" not in scenarios:
        raise ValueError("Figure 1 task panels require the Base scenario.")
    if idle_allocation_rule not in IDLE_ALLOCATION_RULES:
        raise ValueError(f"idle_allocation_rule must be one of {IDLE_ALLOCATION_RULES}.")

    if verbose:
        print("[figure1] Reusing the prepared workload profile." if experiment_context is not None else "[figure1] Building the workload profile.", flush=True)
    profile = experiment_context.profile if experiment_context is not None else build_workload_profile(
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        capacity_quantile=capacity_quantile,
        max_intervals=max_intervals,
        verbose=verbose,
    )

    annual_frames = []
    scenario_results = []
    for scenario in scenarios:
        if verbose:
            print(
                f"[figure1] Calculating hourly workload-resolved results for {scenario}.",
                flush=True,
            )
        result = _capture_task_component_energy(
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
            ai_capacity_factors=ai_capacity_factors,
            verbose=verbose,
            idle_allocation_rule=idle_allocation_rule,
            model_runner=experiment_context.run_hourly if experiment_context is not None else None,
        )
        annual_frames.append(result["annual_summary"].copy())
        scenario_results.append(result)

    annual_summary = pd.concat(annual_frames, ignore_index=True)
    annual_summary.insert(1, "policy", policy)
    def combined(table):
        return pd.concat([result[table] for result in scenario_results], ignore_index=True)
    all_matrix = combined("task_component_energy_all_rules")
    idle_pools = combined("idle_pool")
    annual_cells, task_shares, component_shares = allocation_share_tables(
        all_matrix, idle_pools, ["scenario", "policy", "year", "idle_allocation_rule"])
    task_components = annual_cells.loc[annual_cells["idle_allocation_rule"] == idle_allocation_rule]

    global_annual = (
        annual_summary.groupby(
            ["scenario", "policy", "year"], as_index=False
        )[["facility_energy_mwh", "carbon_tco2", "it_energy_mwh", "idle_it_energy_mwh", "dynamic_it_energy_mwh"]]
        .sum()
        .sort_values(["year", "scenario"], ignore_index=True)
    )
    global_annual["facility_energy_twh"] = (
        global_annual["facility_energy_mwh"] / 1e6
    )
    global_annual["carbon_mtco2"] = global_annual["carbon_tco2"] / 1e6
    system_keys = ["scenario", "policy", "year"]
    system_closure = annual_cells[system_keys + ["idle_allocation_rule", "system_it_energy_mwh", "system_facility_energy_mwh"]].drop_duplicates()
    system_closure = system_closure.merge(
        global_annual[system_keys + ["it_energy_mwh", "facility_energy_mwh"]], on=system_keys, validate="many_to_one")
    system_closure["it_residual_mwh"] = system_closure["system_it_energy_mwh"] - system_closure["it_energy_mwh"]
    system_closure["facility_residual_mwh"] = system_closure["system_facility_energy_mwh"] - system_closure["facility_energy_mwh"]
    if not np.allclose(system_closure["system_it_energy_mwh"], system_closure["it_energy_mwh"], rtol=1e-10, atol=1e-6) or not np.allclose(
        system_closure["system_facility_energy_mwh"], system_closure["facility_energy_mwh"], rtol=1e-10, atol=1e-6):
        raise ValueError("Figure 1 task/component/pool totals do not reproduce model IT/facility energy.")
    system_closure["status"] = "PASS"

    task_sheets = {}
    for task_type, sheet_name in (
        ("training", "Fig1c_Training"),
        ("inference", "Fig1d_Inference"),
        ("other", "Fig1e_Other"),
        ("unclassified", "Fig1_Unclassified"),
    ):
        task_sheets[sheet_name] = _base_task_component_average(
            task_components,
            task_type,
        )

    sheets = {
        "W05_Run_Metadata": pd.DataFrame([{
            "idle_allocation_rule": idle_allocation_rule, "idle_comparison_rules": ";".join(IDLE_ALLOCATION_RULES),
            "trace_intervals": profile.n_intervals, "trace_hours": profile.trace_hours,
            "max_intervals": max_intervals, "trace_truncated_by_max_intervals": max_intervals is not None,
            "scope": hourly_carbon_scope, "grid_inputs": str(Path(hourly_carbon_factors_dir).resolve()),
            "attribution_fallback": "capacity_normalized_resource_proxy_then_explicit_equal_reference",
            "share_rule": "sum_absolute_energy_before_division_all_four_tasks_all_five_components",
        }]),
        "W05_Task_Component_Country": combined("task_component_energy"),
        "W05_Physical_Component_Energy": combined("component_energy"),
        "W05_Task_Component_All_Rules": all_matrix,
        "W05_Task_Component_Annual": annual_cells,
        "W05_Task_System_Shares": task_shares,
        "W05_Component_System_Shares": component_shares,
        "W05_Idle_Pool": idle_pools,
        "W05_Allocation_Audit": combined("allocation_audit"),
        "W05_Energy_Conservation": combined("energy_conservation"),
        "W05_System_Closure": system_closure,
        "W05_Resource_Audit": combined("resource_audit"),
        "W05_Global_Resource_Audit": combined("global_resource_audit"),
        "W05_Power_Nonlinearity": combined("power_nonlinearity"),
        "W05_Proxy_Metadata": scenario_results[0]["resource_proxy_metadata"],
        "W05_Grid_Input_Audit": combined("grid_carbon_audit"),
        "Fig1a_Capacity": _capacity_table(scenarios, year_start, years, countries, ai_capacity_factors),
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
        "Accounting_Boundary": build_boundary_metadata(countries, ai_capacity_factors),
        "AI_Capacity_Factors": capacity_factor_table(year_start, years, ai_capacity_factors),
    }

    resolved_output = Path(output_path)
    if experiment_context is not None:
        sheets["Shared_Experiment_Configuration"] = experiment_context.metadata_table()
    sheets = prepare_figure_sources(1, sheets, carbon_scope=hourly_carbon_scope)
    write_workbook(resolved_output, sheets)
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_figure1_data(
        idle_allocation_rule=args.idle_allocation_rule,
        output_path=args.output,
        policy=args.policy,
        scenarios=args.scenarios,
        countries=args.countries,
        ai_capacity_factors=(load_ai_capacity_factors(args.ai_capacity_factors_file, args.countries)
                             if args.ai_capacity_factors_file else None),
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
