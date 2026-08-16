"""Generate the calculation results required by Figure 2.

The script fixes the demand scenario while comparing CP, NDC and NZ.  It
writes one Excel workbook with one worksheet per Figure 2 panel and contains
no plotting code.
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

from core.m4_hourly_gpu_model import (
    WorkloadProfile,
    build_workload_profile,
    run_workload_component_footprint,
)
from dataset.Factors import get_carbon_factor
from dataset.Installed_capacity_data import DEFAULT_COUNTRIES


DEFAULT_OUTPUT = ROOT_DIR / "results" / "figure2_data.xlsx"
POLICIES = ("CP", "NDC", "NZ")
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


def _annual_carbon_factor(
    policy: str,
    country: str,
    year: int,
) -> float:
    return get_carbon_factor(policy, country, year)


def _carbon_factor_map_table(
    countries: Sequence[str],
    year_start: int,
    years: int,
) -> pd.DataFrame:
    records = []
    averaged_years = list(range(year_start, year_start + years))
    for country in countries:
        records.append(
            {
                "country": COUNTRY_ISO3[country],
                "policy": "CP",
                "year_start": averaged_years[0],
                "year_end": averaged_years[-1],
                "years_averaged": len(averaged_years),
                "annual_avg_carbon_factor_kg_per_mwh": float(
                    np.mean(
                        [
                            _annual_carbon_factor("CP", country, year)
                            for year in averaged_years
                        ]
                    )
                ),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        "country", ignore_index=True
    )


def _carbon_factor_change_table(
    countries: Sequence[str],
    year_start: int,
    years: int,
) -> pd.DataFrame:
    records = []
    averaged_years = list(range(year_start, year_start + years))
    for comparison_policy in ("NDC", "NZ"):
        for country in countries:
            cp_factor = float(
                np.mean(
                    [
                        _annual_carbon_factor("CP", country, year)
                        for year in averaged_years
                    ]
                )
            )
            comparison_factor = float(
                np.mean(
                    [
                        _annual_carbon_factor(
                            comparison_policy,
                            country,
                            year,
                        )
                        for year in averaged_years
                    ]
                )
            )
            records.append(
                {
                    "country": COUNTRY_ISO3[country],
                    "baseline_policy": "CP",
                    "comparison_policy": comparison_policy,
                    "year_start": averaged_years[0],
                    "year_end": averaged_years[-1],
                    "years_averaged": len(averaged_years),
                    "annual_avg_cp_carbon_factor_kg_per_mwh": cp_factor,
                    "reduction_vs_cp_pct": (
                        (cp_factor - comparison_factor) / cp_factor * 100.0
                        if cp_factor != 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        ["comparison_policy", "country"], ignore_index=True
    )


def _run_policy(
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
    annual = result["annual_summary"].copy()
    annual.insert(1, "policy", policy)
    annual["load_weighted_carbon_factor_kg_per_mwh"] = np.divide(
        annual["carbon_tco2"].to_numpy(dtype=float) * 1000.0,
        annual["facility_energy_mwh"].to_numpy(dtype=float),
        out=np.full(len(annual), np.nan),
        where=annual["facility_energy_mwh"].to_numpy(dtype=float) != 0,
    )
    return annual


def _write_workbook(output_path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions


def generate_figure2_data(
    output_path: Union[str, Path] = DEFAULT_OUTPUT,
    scenario: str = "Base",
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
    """Calculate all Figure 2 panel data and write one Excel workbook."""
    countries = list(countries)
    if verbose:
        print("[figure2] Building the shared GPU workload profile.", flush=True)
    profile = build_workload_profile(
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        capacity_quantile=capacity_quantile,
        max_intervals=max_intervals,
        verbose=verbose,
    )

    annual_frames = []
    for policy in POLICIES:
        if verbose:
            print(f"[figure2] Calculating M4 results for {policy}.", flush=True)
        annual_frames.append(
            _run_policy(
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
        )
    annual_summary = pd.concat(annual_frames, ignore_index=True)

    global_emissions = (
        annual_summary.groupby(
            ["scenario", "policy", "year"], as_index=False
        )[["facility_energy_mwh", "carbon_tco2"]]
        .sum()
        .sort_values(["year", "policy"], ignore_index=True)
    )
    global_emissions["facility_energy_twh"] = (
        global_emissions["facility_energy_mwh"] / 1e6
    )
    global_emissions["carbon_mtco2"] = global_emissions["carbon_tco2"] / 1e6
    global_emissions["load_weighted_carbon_factor_kg_per_mwh"] = np.divide(
        global_emissions["carbon_tco2"].to_numpy(dtype=float) * 1000.0,
        global_emissions["facility_energy_mwh"].to_numpy(dtype=float),
        out=np.full(len(global_emissions), np.nan),
        where=global_emissions["facility_energy_mwh"].to_numpy(dtype=float) != 0,
    )

    sheets = {
        "Fig2a_CF_Map": _carbon_factor_map_table(countries, year_start, years),
        "Fig2b_CF_Change": _carbon_factor_change_table(
            countries, year_start, years
        ),
        "Fig2c_Global_Carbon": global_emissions[
            ["scenario", "policy", "year", "carbon_mtco2"]
        ],
    }
    resolved_output = Path(output_path)
    _write_workbook(resolved_output, sheets)
    if verbose:
        print(f"[figure2] Excel workbook saved to {resolved_output.resolve()}.")
    return resolved_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Excel calculation results required by Figure 2."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario", default="Base")
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
    generate_figure2_data(
        output_path=args.output,
        scenario=args.scenario,
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
