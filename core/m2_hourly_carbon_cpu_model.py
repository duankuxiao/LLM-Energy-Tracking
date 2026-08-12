"""M2: traditional CPU-style energy model with hourly grid carbon factors."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from core.past_research_data_center_energy_carbon_model import (
    calculate_past_research_energy_carbon,
)
from dataset.Factors import CF_CP, CF_NDC, CF_NZ
from dataset.Installed_capacity_data import DEFAULT_AI_CAPACITY_FACTORS, DEFAULT_COUNTRIES


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_NAME = "M2"
DATA_YEAR_START = 2025
DATA_YEAR_END = 2030
HOURLY_CARBON_COUNTRY_DIRS = {"United_Kingdom": "Great Britain"}


def _annual_factors(policy: str):
    if policy == "CP":
        return CF_CP
    if policy == "NDC":
        return CF_NDC
    if policy == "NZ":
        return CF_NZ
    raise ValueError("renewable_energy_policy must be one of: CP, NDC, NZ")


def _resolve_ai_capacity_factors(
    requested_years: Sequence[int],
    overrides: Optional[Mapping[int, float]],
) -> Dict[int, float]:
    source = DEFAULT_AI_CAPACITY_FACTORS if overrides is None else overrides
    missing = [year for year in requested_years if year not in source]
    if missing:
        raise ValueError(f"Missing AI capacity factors for years: {missing}")
    factors = {year: float(source[year]) for year in requested_years}
    invalid = {year: value for year, value in factors.items() if not 0 < value <= 1}
    if invalid:
        raise ValueError(f"AI capacity factors must be in (0, 1], got: {invalid}")
    return factors


def _read_hourly_carbon_factors(
    country: str,
    policy: str,
    year: int,
    hourly_carbon_factors_dir: Union[str, Path],
    scope: str,
    fallback_to_annual: bool,
) -> tuple[pd.DatetimeIndex, np.ndarray, str]:
    country_dir = Path(hourly_carbon_factors_dir) / HOURLY_CARBON_COUNTRY_DIRS.get(country, country)
    matches = sorted(country_dir.glob(f"*-{policy}-{year}-hourly.csv")) if country_dir.exists() else []

    if len(matches) > 1:
        raise ValueError(f"Expected one hourly carbon CSV for {country} {year}, found {len(matches)}.")
    if matches:
        hourly_df = pd.read_csv(matches[0])
        timestamp_columns = [column for column in hourly_df.columns if "datetime" in column.lower()]
        scope_token = "direct" if scope == "direct" else "life cycle" if scope == "life_cycle" else None
        if scope_token is None:
            raise ValueError("hourly_carbon_scope must be one of: direct, life_cycle.")
        intensity_columns = [
            column
            for column in hourly_df.columns
            if "carbon intensity" in column.lower() and scope_token in column.lower()
        ]
        if not timestamp_columns or not intensity_columns:
            raise ValueError(f"Hourly carbon CSV lacks required columns: {matches[0]}")

        timestamps = pd.to_datetime(hourly_df[timestamp_columns[0]], utc=True, errors="coerce")
        factors = pd.to_numeric(hourly_df[intensity_columns[0]], errors="coerce").to_numpy(dtype=float)
        if timestamps.isna().any() or len(factors) == 0 or not np.all(np.isfinite(factors)):
            raise ValueError(f"Hourly carbon CSV contains invalid values: {matches[0]}")
        return pd.DatetimeIndex(timestamps), factors, "hourly"

    if not fallback_to_annual:
        raise FileNotFoundError(f"No hourly carbon factors found for {country} {policy} {year}.")
    annual_factor = float(_annual_factors(policy)[country][year - DATA_YEAR_START])
    timestamps = pd.date_range(f"{year}-01-01", periods=8760, freq="h", tz="UTC")
    return timestamps, np.full(8760, annual_factor, dtype=float), "annual_fallback"


def run_m2_hourly_carbon_cpu_model(
    renewable_energy_policy: str,
    scenarios: Sequence[str],
    years: int = 5,
    countries: Optional[Sequence[str]] = None,
    year_start: int = 2026,
    infer_ratio_by_country: Optional[Dict[str, float]] = None,
    default_p_infer: float = 0.7,
    u_train: float = 0.8,
    u_infer: float = 0.5,
    idle_power_rate: float = 0.23,
    max_power_rate: float = 0.88,
    pue_scale: float = 1.0,
    ai_capacity_factors: Optional[Mapping[int, float]] = None,
    hourly_carbon_factors_dir: Union[str, Path] = ROOT_DIR / "dataset" / "EM-estimate",
    hourly_carbon_scope: str = "direct",
    hourly_carbon_fallback_to_annual: bool = True,
    include_hourly_results: bool = False,
    output_dir: Union[str, Path] = ROOT_DIR / "results" / "m2_hourly_carbon_cpu",
    save_outputs: bool = True,
    save_hourly_outputs: bool = False,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run M2 with past-research annual energy and hourly carbon intensity.

    M2 deliberately introduces neither task profiles nor component-level power.
    Each country-year's M1 annual facility energy is represented as constant
    hourly power, and those hourly MWh are matched to hourly grid factors.
    Therefore M1 and M2 have exactly the same annual energy by construction.
    """
    if years <= 0:
        raise ValueError("years must be positive.")
    year_end = year_start + years - 1
    if year_start < DATA_YEAR_START or year_end > DATA_YEAR_END:
        raise ValueError(f"Requested years must be within {DATA_YEAR_START}-{DATA_YEAR_END}.")

    countries = list(countries or DEFAULT_COUNTRIES)
    requested_years = list(range(year_start, year_end + 1))
    capacity_factors = _resolve_ai_capacity_factors(requested_years, ai_capacity_factors)
    native_years = year_end - DATA_YEAR_START + 1
    annual_records = []
    hourly_frames = []

    for scenario in scenarios:
        raw = calculate_past_research_energy_carbon(
            renewable_energy_policy=renewable_energy_policy,
            scenario=scenario,
            years=native_years,
            countries=countries,
            infer_ratio_by_country=infer_ratio_by_country,
            default_p_infer=default_p_infer,
            u_train=u_train,
            u_infer=u_infer,
            idle_power_rate=idle_power_rate,
            max_power_rate=max_power_rate,
            pue_scale=pue_scale,
        )
        country_power = raw["country_power"].loc[requested_years, countries]

        for year in requested_years:
            for country in countries:
                facility_energy_mwh = (
                    float(country_power.at[year, country]) * 1e6 * capacity_factors[year]
                )
                timestamps, carbon_factors, factor_source = _read_hourly_carbon_factors(
                    country=country,
                    policy=renewable_energy_policy,
                    year=year,
                    hourly_carbon_factors_dir=hourly_carbon_factors_dir,
                    scope=hourly_carbon_scope,
                    fallback_to_annual=hourly_carbon_fallback_to_annual,
                )
                hourly_energy_mwh = np.full(
                    len(carbon_factors), facility_energy_mwh / len(carbon_factors), dtype=float
                )
                hourly_carbon_tco2 = hourly_energy_mwh * carbon_factors / 1000.0
                carbon_tco2 = float(hourly_carbon_tco2.sum())

                annual_records.append(
                    {
                        "model": MODEL_NAME,
                        "scenario": scenario,
                        "policy": renewable_energy_policy,
                        "year": year,
                        "country": country,
                        "ai_capacity_factor": capacity_factors[year],
                        "facility_energy_mwh": facility_energy_mwh,
                        "power_twh": facility_energy_mwh / 1e6,
                        "load_weighted_carbon_factor_kg_per_mwh": (
                            carbon_tco2 * 1000.0 / facility_energy_mwh
                            if facility_energy_mwh > 0
                            else 0.0
                        ),
                        "carbon_tco2": carbon_tco2,
                        "carbon_mtco2": carbon_tco2 / 1e6,
                        "carbon_factor_source": factor_source,
                    }
                )

                if include_hourly_results or save_hourly_outputs:
                    hourly_frames.append(
                        pd.DataFrame(
                            {
                                "model": MODEL_NAME,
                                "scenario": scenario,
                                "policy": renewable_energy_policy,
                                "year": year,
                                "country": country,
                                "hour_index": np.arange(len(carbon_factors)),
                                "timestamp_utc": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "facility_energy_mwh": hourly_energy_mwh,
                                "carbon_factor_kg_per_mwh": carbon_factors,
                                "carbon_tco2": hourly_carbon_tco2,
                                "carbon_factor_source": factor_source,
                            }
                        )
                    )

    annual_summary = pd.DataFrame.from_records(annual_records)
    hourly_carbon = (
        pd.concat(hourly_frames, ignore_index=True) if hourly_frames else pd.DataFrame()
    )
    global_summary = (
        annual_summary.groupby(["model", "scenario", "policy", "year"], as_index=False)[
            ["facility_energy_mwh", "power_twh", "carbon_tco2", "carbon_mtco2"]
        ].sum()
    )

    if save_outputs:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        annual_summary.to_csv(output_path / "M2_Country_Annual.csv", index=False)
        global_summary.to_csv(output_path / "M2_Global_Annual.csv", index=False)
        if save_hourly_outputs:
            hourly_carbon.to_csv(output_path / "M2_Country_Hourly.csv", index=False)

    if verbose:
        print(global_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        if save_outputs:
            print(f"Saved M2 results to: {Path(output_dir).resolve()}")

    return {
        "annual_summary": annual_summary,
        "global_summary": global_summary,
        "hourly_carbon": hourly_carbon if include_hourly_results else pd.DataFrame(),
    }


if __name__ == "__main__":
    run_m2_hourly_carbon_cpu_model(
        renewable_energy_policy="CP",
        scenarios=["Base"],
    )
