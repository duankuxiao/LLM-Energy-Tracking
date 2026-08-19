"""Generate the calculation results required by Figure 2.

The script fixes the demand scenario while comparing CP, NDC and NZ. It
writes one Excel workbook for four Figure 2 panels: the CP carbon-intensity
map, 2030 country emissions and avoided emissions, aggregate emissions paths,
and load-weighted carbon-intensity paths. The workbook also retains the full
country-level source table used to aggregate the display data. This module
contains no plotting code.
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


def _country_abatement_tables(
    annual_summary: pd.DataFrame,
    *,
    display_year: int,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Figure 2b display data and its full country-level source table."""
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    selected = annual_summary.loc[annual_summary["year"] == display_year].copy()
    if selected.empty:
        raise ValueError(f"No annual model results found for {display_year}.")

    key_columns = ["scenario", "year", "country", "policy"]
    duplicate_rows = selected.duplicated(key_columns, keep=False)
    if duplicate_rows.any():
        duplicates = selected.loc[duplicate_rows, key_columns]
        raise ValueError(
            "Country-policy results must be unique before Figure 2b pivoting; "
            f"found duplicates: {duplicates.to_dict(orient='records')}"
        )

    carbon = selected.pivot(
        index=["scenario", "year", "country"],
        columns="policy",
        values="carbon_tco2",
    ).reindex(columns=POLICIES)
    energy = selected.pivot(
        index=["scenario", "year", "country"],
        columns="policy",
        values="facility_energy_mwh",
    ).reindex(columns=POLICIES)
    if carbon.isna().any().any() or energy.isna().any().any():
        raise ValueError(
            f"Every country must have complete {POLICIES} results for Figure 2b."
        )

    energy_values = energy.to_numpy(dtype=float)
    if not np.allclose(
        energy_values,
        energy_values[:, [0]],
        rtol=1e-10,
        atol=1e-6,
    ):
        raise ValueError(
            "Facility energy must be identical across CP, NDC and NZ for each country."
        )

    source = carbon.index.to_frame(index=False)
    unmapped = sorted(set(source["country"]) - set(COUNTRY_ISO3))
    if unmapped:
        raise ValueError(f"Missing ISO3 labels for countries: {unmapped}")
    source["country_iso3"] = source["country"].map(COUNTRY_ISO3)
    source["facility_energy_twh"] = energy["CP"].to_numpy(dtype=float) / 1e6
    source["cp_carbon_mtco2"] = carbon["CP"].to_numpy(dtype=float) / 1e6
    source["ndc_carbon_mtco2"] = carbon["NDC"].to_numpy(dtype=float) / 1e6
    source["nz_carbon_mtco2"] = carbon["NZ"].to_numpy(dtype=float) / 1e6
    source["ndc_avoided_vs_cp_mtco2"] = (
        source["cp_carbon_mtco2"] - source["ndc_carbon_mtco2"]
    )
    source["nz_avoided_vs_cp_mtco2"] = (
        source["cp_carbon_mtco2"] - source["nz_carbon_mtco2"]
    )
    if (
        source[["ndc_avoided_vs_cp_mtco2", "nz_avoided_vs_cp_mtco2"]]
        .lt(-1e-12)
        .any()
        .any()
    ):
        raise ValueError("NDC and NZ must not exceed CP country emissions.")

    group_keys = ["scenario", "year"]
    cp_totals = source.groupby(group_keys)["cp_carbon_mtco2"].transform("sum")
    source["cp_emissions_share_pct"] = np.divide(
        source["cp_carbon_mtco2"].to_numpy(dtype=float) * 100.0,
        cp_totals.to_numpy(dtype=float),
        out=np.zeros(len(source), dtype=float),
        where=cp_totals.to_numpy(dtype=float) != 0,
    )
    source = source.sort_values(
        group_keys + ["cp_carbon_mtco2"],
        ascending=[True, True, False],
        ignore_index=True,
    )
    source.insert(
        0,
        "country_rank",
        source.groupby(group_keys, sort=False).cumcount() + 1,
    )

    metric_columns = [
        "facility_energy_twh",
        "cp_carbon_mtco2",
        "ndc_carbon_mtco2",
        "nz_carbon_mtco2",
        "ndc_avoided_vs_cp_mtco2",
        "nz_avoided_vs_cp_mtco2",
        "cp_emissions_share_pct",
    ]
    display_frames = []
    for (scenario, year), group in source.groupby(group_keys, sort=False):
        group = group.sort_values("country_rank").reset_index(drop=True)
        top = group.head(top_n).copy()
        top.insert(0, "display_rank", np.arange(1, len(top) + 1))
        top.insert(1, "display_label", top["country_iso3"])
        top.insert(2, "country_count", 1)
        display_columns = [
            "display_rank",
            "display_label",
            "country_count",
            "scenario",
            "year",
        ] + metric_columns
        display_frames.append(top[display_columns])

        other = group.iloc[top_n:]
        if not other.empty:
            other_record = {
                "display_rank": len(top) + 1,
                "display_label": "Other modelled countries",
                "country_count": len(other),
                "scenario": scenario,
                "year": year,
            }
            other_record.update(
                {column: float(other[column].sum()) for column in metric_columns}
            )
            display_frames.append(pd.DataFrame([other_record]))

    display = pd.concat(display_frames, ignore_index=True).sort_values(
        group_keys + ["display_rank"], ignore_index=True
    )
    for (scenario, year), group in source.groupby(group_keys, sort=False):
        display_group = display.loc[
            (display["scenario"] == scenario) & (display["year"] == year)
        ]
        for column in metric_columns:
            if not np.isclose(
                group[column].sum(),
                display_group[column].sum(),
                rtol=1e-10,
                atol=1e-10,
            ):
                raise ValueError(
                    f"Figure 2b display aggregation does not preserve {column}."
                )

    return display, source


def _validate_country_global_totals(
    country_source: pd.DataFrame,
    global_emissions: pd.DataFrame,
) -> None:
    """Ensure the Figure 2b country totals reproduce Figure 2c exactly."""
    policy_columns = {
        "CP": "cp_carbon_mtco2",
        "NDC": "ndc_carbon_mtco2",
        "NZ": "nz_carbon_mtco2",
    }
    for (scenario, year), countries in country_source.groupby(
        ["scenario", "year"], sort=False
    ):
        for policy, country_column in policy_columns.items():
            match = global_emissions.loc[
                (global_emissions["scenario"] == scenario)
                & (global_emissions["year"] == year)
                & (global_emissions["policy"] == policy),
                "carbon_mtco2",
            ]
            if len(match) != 1:
                raise ValueError(
                    f"Expected one aggregate result for {scenario}, {year}, {policy}."
                )
            if not np.isclose(
                countries[country_column].sum(),
                float(match.iloc[0]),
                rtol=1e-10,
                atol=1e-10,
            ):
                raise ValueError(
                    "Country emissions do not reproduce the aggregate result for "
                    f"{scenario}, {year}, {policy}."
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
    country_display_top_n: int = 8,
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

    display_year = year_start + years - 1
    country_abatement, country_source = _country_abatement_tables(
        annual_summary,
        display_year=display_year,
        top_n=country_display_top_n,
    )
    _validate_country_global_totals(country_source, global_emissions)

    global_carbon = global_emissions[
        ["scenario", "policy", "year", "carbon_mtco2"]
    ].copy()
    cp_by_year = global_carbon.loc[
        global_carbon["policy"] == "CP",
        ["scenario", "year", "carbon_mtco2"],
    ].rename(columns={"carbon_mtco2": "cp_carbon_mtco2"})
    global_carbon = global_carbon.merge(
        cp_by_year,
        on=["scenario", "year"],
        how="left",
        validate="many_to_one",
    )
    global_carbon["avoided_vs_cp_mtco2"] = (
        global_carbon["cp_carbon_mtco2"] - global_carbon["carbon_mtco2"]
    )
    global_carbon = global_carbon.drop(columns="cp_carbon_mtco2")

    weighted_carbon_intensity = global_emissions[
        [
            "scenario",
            "policy",
            "year",
            "facility_energy_twh",
            "carbon_mtco2",
            "load_weighted_carbon_factor_kg_per_mwh",
        ]
    ].rename(
        columns={
            "load_weighted_carbon_factor_kg_per_mwh": (
                "load_weighted_carbon_intensity_kg_per_mwh"
            )
        }
    )

    sheets = {
        "Fig2a_CF_Map": _carbon_factor_map_table(countries, year_start, years),
        "Fig2b_Country_Abatement": country_abatement,
        "Fig2c_Global_Carbon": global_carbon,
        "Fig2d_Weighted_CI": weighted_carbon_intensity,
        "Source_Fig2b_Countries": country_source,
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
    parser.add_argument(
        "--country-display-top-n",
        type=int,
        default=8,
        help=(
            "Number of leading CP-emission countries shown separately in "
            "Figure 2b; remaining modelled countries are aggregated."
        ),
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
        country_display_top_n=args.country_display_top_n,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
