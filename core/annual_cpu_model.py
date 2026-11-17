"""Conventional annual CPU energy model with annual grid carbon factors."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import pandas as pd

from core.accounting_boundary import (
    annotate_selected_aggregate, build_boundary_metadata, load_ai_capacity_factors, resolve_countries, resolve_ai_capacity_factors,
)
from core.annual_activity import ACTIVITY_MODES, AnnualActivityConfig
from core.past_research_data_center_energy_carbon_model import (
    calculate_past_research_energy_carbon,
    SCENARIO_COLUMN,
)
from dataset.Installed_capacity_data import IT_CAPACITY, IT_RATIO


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_NAME = "annual_cpu_model"
DATA_YEAR_START = 2025
DATA_YEAR_END = 2030


def run_annual_cpu_model(
    renewable_energy_policy: str,
    scenarios: Sequence[str],
    years: int = 5,
    countries: Optional[Sequence[str]] = None,
    year_start: int = 2026,
    infer_ratio_by_country: Optional[Dict[str, float]] = None,
    task_ratio_by_country: Optional[Mapping[str, Mapping[str, float]]] = None,
    default_p_infer: float = 0.75,
    default_p_other: float = 0.05,
    u_train: float = 0.8,
    u_infer: float = 0.5,
    u_other: float = 0.5,
    idle_power_rate: float = 0.23,
    max_power_rate: float = 0.88,
    pue_scale: float = 1.0,
    ai_capacity_factors: Optional[Mapping[int, float]] = None,
    output_dir: Union[str, Path] = ROOT_DIR / "results" / "annual_cpu_model",
    save_outputs: bool = True,
    verbose: bool = True,
    activity_config: Optional[AnnualActivityConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run the conventional annual CPU data-centre energy and carbon model.

    The model uses the traditional CPU-style utilization--power relationship
    and the annual country carbon factors applied by the baseline method. No
    task-level time series or IT-component breakdown is introduced. Annual
    energy is nevertheless allocated to the shared training, inference, other,
    and unclassified taxonomy using the same simplified coefficients.

    The baseline method is evaluated from its native 2025 data origin and then
    sliced to the requested years. Its linear annual results are multiplied by
    the AI-capacity factors so both endpoint models use the same AI capacity
    boundary.
    Pass factors equal to 1.0 to model all data-centre IT capacity instead.
    Activity coefficients use fixed calendar anchors; changing the requested
    output window cannot change coefficients for an overlapping year.
    """
    if years <= 0:
        raise ValueError("years must be positive.")
    year_end = year_start + years - 1
    if year_start < DATA_YEAR_START or year_end > DATA_YEAR_END:
        raise ValueError(f"Requested years must be within {DATA_YEAR_START}-{DATA_YEAR_END}.")

    countries = resolve_countries(countries)
    activity_config = activity_config or AnnualActivityConfig()
    activity_config.validate()
    boundary = build_boundary_metadata(countries, ai_capacity_factors)
    requested_years = list(range(year_start, year_end + 1))
    capacity_factors = resolve_ai_capacity_factors(requested_years, ai_capacity_factors)
    native_years = year_end - DATA_YEAR_START + 1
    annual_frames = []
    task_frames = []

    for scenario in scenarios:
        raw = calculate_past_research_energy_carbon(
            renewable_energy_policy=renewable_energy_policy,
            scenario=scenario,
            years=native_years,
            countries=countries,
            infer_ratio_by_country=infer_ratio_by_country,
            task_ratio_by_country=task_ratio_by_country,
            default_p_infer=default_p_infer,
            default_p_other=default_p_other,
            u_train=u_train,
            u_infer=u_infer,
            u_other=u_other,
            idle_power_rate=idle_power_rate,
            max_power_rate=max_power_rate,
            pue_scale=pue_scale,
            activity_config=activity_config,
        )
        power_twh = raw["country_power"].loc[requested_years, countries]
        carbon_mtco2 = raw["country_carbon"].loc[requested_years, countries]
        raw_it_energy = raw["task_type_energy"].groupby(["year", "country"])["it_energy_mwh"].sum()
        raw_utilization = raw["task_type_energy"].groupby(["year", "country"])["utilization_contribution"].sum()

        records = []
        for year in requested_years:
            scale = capacity_factors[year]
            for country in countries:
                country_power_twh = float(power_twh.at[year, country]) * scale
                country_carbon_mtco2 = float(carbon_mtco2.at[year, country]) * scale
                energy_mwh = country_power_twh * 1e6
                carbon_tco2 = country_carbon_mtco2 * 1e6
                it_energy_mwh = float(raw_it_energy.loc[(year, country)]) * scale
                unadjusted_it_mw = float(IT_CAPACITY[year - DATA_YEAR_START, SCENARIO_COLUMN[scenario]]) * 1e3 * IT_RATIO[country]
                records.append(
                    {
                        "model": MODEL_NAME,
                        "scenario": scenario,
                        "policy": renewable_energy_policy,
                        "year": year,
                        "country": country,
                        "ai_capacity_factor": scale,
                        "activity_mode": activity_config.mode,
                        "training_activity": float(activity_config.vector(year)[0]),
                        "inference_other_activity": float(activity_config.vector(year)[1]),
                        "composite_utilization": float(raw_utilization.loc[(year, country)]),
                        "total_data_center_it_mw": unadjusted_it_mw,
                        "installed_it_mw": unadjusted_it_mw * scale,
                        "it_energy_mwh": it_energy_mwh,
                        "facility_energy_mwh": energy_mwh,
                        "power_twh": country_power_twh,
                        "carbon_factor_kg_per_mwh": (
                            carbon_tco2 * 1000.0 / energy_mwh if energy_mwh > 0 else 0.0
                        ),
                        "carbon_tco2": carbon_tco2,
                        "carbon_mtco2": country_carbon_mtco2,
                        "it_mwh_per_installed_it_mw": it_energy_mwh / (unadjusted_it_mw * scale),
                        "facility_mwh_per_installed_it_mw": energy_mwh / (unadjusted_it_mw * scale),
                        "carbon_tco2_per_installed_it_mw": carbon_tco2 / (unadjusted_it_mw * scale),
                    }
                )
        annual_frames.append(pd.DataFrame.from_records(records))

        task_type_energy = raw["task_type_energy"]
        task_type_energy = task_type_energy[
            task_type_energy["year"].isin(requested_years)
        ].copy()
        task_type_energy.insert(0, "policy", renewable_energy_policy)
        task_type_energy.insert(0, "scenario", scenario)
        task_type_energy.insert(0, "model", MODEL_NAME)
        task_type_energy["ai_capacity_factor"] = task_type_energy["year"].map(
            capacity_factors
        )
        for column in ("it_energy_mwh", "facility_energy_mwh", "carbon_tco2"):
            task_type_energy[column] *= task_type_energy["ai_capacity_factor"]
        task_type_energy["facility_energy_twh"] = (
            task_type_energy["facility_energy_mwh"] / 1e6
        )
        task_type_energy["carbon_mtco2"] = task_type_energy["carbon_tco2"] / 1e6
        task_frames.append(task_type_energy)

    columns = [
        "model",
        "scenario",
        "policy",
        "year",
        "country",
        "ai_capacity_factor",
        "total_data_center_it_mw",
        "installed_it_mw",
        "it_energy_mwh",
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
    selected_countries_summary = (
        annual_summary.groupby(["model", "scenario", "policy", "year"], as_index=False)[
            ["total_data_center_it_mw", "installed_it_mw", "it_energy_mwh",
             "facility_energy_mwh", "power_twh", "carbon_tco2", "carbon_mtco2"]
        ].sum()
    )
    selected_countries_summary = annotate_selected_aggregate(selected_countries_summary, boundary)
    task_type_energy = (
        pd.concat(task_frames, ignore_index=True) if task_frames else pd.DataFrame()
    )
    activity_factors = activity_config.table(requested_years)
    annual_parameters = pd.DataFrame([
        {"parameter": name, "value": value, "activity_mode": activity_config.mode,
         "parameter_status": "explicit_reference_configuration"}
        for name, value in {
            "default_p_infer": default_p_infer, "default_p_other": default_p_other,
            "u_train": u_train, "u_infer": u_infer, "u_other": u_other,
            "idle_power_rate": idle_power_rate, "max_power_rate": max_power_rate,
            "pue_scale": pue_scale,
            "training_activity_start": activity_config.training_start,
            "training_activity_target": activity_config.training_target,
            "inference_activity_start": activity_config.inference_start,
            "inference_activity_target": activity_config.inference_target,
        }.items()
    ])

    if save_outputs:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        annual_summary.to_csv(output_path / "Annual_CPU_Model_Country_Annual.csv", index=False)
        selected_countries_summary.to_csv(output_path / "Annual_CPU_Model_Selected_Countries_Annual.csv", index=False)
        boundary.to_csv(output_path / "Accounting_Boundary.csv", index=False)
        activity_factors.to_csv(output_path / "Annual_Activity_Factors.csv", index=False)
        annual_parameters.to_csv(output_path / "Annual_Reference_Parameters.csv", index=False)
        task_type_energy.to_csv(
            output_path / "Annual_CPU_Model_Country_TaskType_Energy.csv", index=False
        )

    if verbose:
        print("Annual CPU model: selected-country aggregate.")
        print(selected_countries_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        if save_outputs:
            print(f"Saved annual CPU model results to: {Path(output_dir).resolve()}")

    return {
        "annual_summary": annual_summary,
        "selected_countries_summary": selected_countries_summary,
        "global_summary": selected_countries_summary,  # Compatibility alias; never a world total.
        "accounting_boundary": boundary,
        "task_type_energy": task_type_energy,
        "activity_factors": activity_factors,
        "annual_parameters": annual_parameters,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the annual endpoint without reading workload traces.")
    parser.add_argument("--policy", choices=("CP", "NDC", "NZ"), default="CP")
    parser.add_argument("--scenarios", nargs="+", default=["Base"])
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--year-start", type=int, default=2026)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--activity-mode", choices=ACTIVITY_MODES, default="calendar")
    parser.add_argument("--ai-capacity-factors-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results" / "annual_cpu_model")
    args = parser.parse_args()
    run_annual_cpu_model(
        renewable_energy_policy=args.policy, scenarios=args.scenarios, countries=args.countries,
        year_start=args.year_start, years=args.years, output_dir=args.output_dir,
        activity_config=AnnualActivityConfig(mode=args.activity_mode),
        ai_capacity_factors=(load_ai_capacity_factors(args.ai_capacity_factors_file, args.countries)
                             if args.ai_capacity_factors_file else None),
    )
