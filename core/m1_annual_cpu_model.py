"""M1: traditional CPU-style energy model with annual grid carbon factors."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import pandas as pd

from core.past_research_data_center_energy_carbon_model import (
    calculate_past_research_energy_carbon,
)
from dataset.Installed_capacity_data import DEFAULT_AI_CAPACITY_FACTORS, DEFAULT_COUNTRIES


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_NAME = "M1"
DATA_YEAR_START = 2025
DATA_YEAR_END = 2030


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


def run_m1_annual_cpu_model(
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
    output_dir: Union[str, Path] = ROOT_DIR / "results" / "m1_annual_cpu",
    save_outputs: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run M1 using the past-research data-centre energy and carbon method.

    M1 uses the traditional CPU-style utilization--power relationship and the
    annual country carbon factors applied by the baseline method. No
    task-level time series or IT-component breakdown is introduced.

    The baseline method is evaluated from its native 2025 data origin and then
    sliced to the requested years. Its linear annual results are multiplied by
    the AI-capacity factors so M1 uses the same AI capacity boundary as M3.
    Pass factors equal to 1.0 to model all data-centre IT capacity instead.
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
    annual_frames = []

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
        power_twh = raw["country_power"].loc[requested_years, countries]
        carbon_mtco2 = raw["country_carbon"].loc[requested_years, countries]

        records = []
        for year in requested_years:
            scale = capacity_factors[year]
            for country in countries:
                country_power_twh = float(power_twh.at[year, country]) * scale
                country_carbon_mtco2 = float(carbon_mtco2.at[year, country]) * scale
                energy_mwh = country_power_twh * 1e6
                carbon_tco2 = country_carbon_mtco2 * 1e6
                records.append(
                    {
                        "model": MODEL_NAME,
                        "scenario": scenario,
                        "policy": renewable_energy_policy,
                        "year": year,
                        "country": country,
                        "ai_capacity_factor": scale,
                        "facility_energy_mwh": energy_mwh,
                        "power_twh": country_power_twh,
                        "carbon_factor_kg_per_mwh": (
                            carbon_tco2 * 1000.0 / energy_mwh if energy_mwh > 0 else 0.0
                        ),
                        "carbon_tco2": carbon_tco2,
                        "carbon_mtco2": country_carbon_mtco2,
                    }
                )
        annual_frames.append(pd.DataFrame.from_records(records))

    columns = [
        "model",
        "scenario",
        "policy",
        "year",
        "country",
        "ai_capacity_factor",
        "facility_energy_mwh",
        "power_twh",
        "carbon_factor_kg_per_mwh",
        "carbon_tco2",
        "carbon_mtco2",
    ]
    annual_summary = (
        pd.concat(annual_frames, ignore_index=True)
        if annual_frames
        else pd.DataFrame(columns=columns)
    )
    global_summary = (
        annual_summary.groupby(["model", "scenario", "policy", "year"], as_index=False)[
            ["facility_energy_mwh", "power_twh", "carbon_tco2", "carbon_mtco2"]
        ].sum()
    )

    if save_outputs:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        annual_summary.to_csv(output_path / "M1_Country_Annual.csv", index=False)
        global_summary.to_csv(output_path / "M1_Global_Annual.csv", index=False)

    if verbose:
        print(global_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        if save_outputs:
            print(f"Saved M1 annual results to: {Path(output_dir).resolve()}")

    return {"annual_summary": annual_summary, "global_summary": global_summary}


if __name__ == "__main__":
    run_m1_annual_cpu_model(
        renewable_energy_policy="CP",
        scenarios=["Base"],
    )
