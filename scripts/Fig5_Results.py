import os
import sys
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.Carbon_water_footprint import AIFootprint


def build_fig5_results():
    years = 6
    countries = [
        "USA", "China", "Japan", "France", "India", "Singapore",
        "Canada", "Germany", "United_Kingdom", "Australia", "Italy", "South_Korea",
        "South_Africa", "Ireland", "UAE", "Brazil", "Israel",
        "Netherlands", "Spain", "Sweden", "Belgium", "Norway",
        "Poland", "Switzerland"
    ]
    scenarios = ["Base"]
    renewable_energy_policy = "CP"
    output_dir = str(ROOT_DIR / "results" / "fig5")
    os.makedirs(output_dir, exist_ok=True)

    base_kwargs = dict(
        renewable_energy_policy=renewable_energy_policy,
        scenarios=scenarios,
        years=years,
        countries=countries,
        infer_ratio_by_country=None,
        default_p_infer=0.7,
        save_outputs=False,
        verbose=False,
        return_results=True,
    )

    sensitivity_cases = {
        "training_utilization": {
            "parameter_name": "Training utilization",
            "argument_name": "u_train",
            "low": 0.6,
            "applied": 0.8,
            "high": 0.9,
        },
        "inference_utilization": {
            "parameter_name": "Inference utilization",
            "argument_name": "u_infer",
            "low": 0.4,
            "applied": 0.5,
            "high": 0.6,
        },
        "idle_power_rate": {
            "parameter_name": "Server idle power",
            "argument_name": "idle_power_rate",
            "low": 0.10,
            "applied": 0.23,
            "high": 0.35,
        },
        "max_power_rate": {
            "parameter_name": "Server maximum power",
            "argument_name": "max_power_rate",
            "low": 0.70,
            "applied": 0.88,
            "high": 0.98,
        },
        "pue_scale": {
            "parameter_name": "Country-level PUE",
            "argument_name": "pue_scale",
            "low": 0.9,
            "applied": 1.0,
            "high": 1.1,
        },
    }

    baseline = AIFootprint(**base_kwargs)["aggregate"]["Base"]
    case_records = []
    summary_records = []

    for parameter_key, config in sensitivity_cases.items():
        per_case = {}
        for level in ["low", "applied", "high"]:
            run_kwargs = dict(base_kwargs)
            run_kwargs[config["argument_name"]] = config[level]
            aggregate = AIFootprint(**run_kwargs)["aggregate"]["Base"]

            record = {
                "parameter_key": parameter_key,
                "parameter_name": config["parameter_name"],
                "argument_name": config["argument_name"],
                "case": level,
                "setting_value": config[level],
                "power_twh_2025_2030": aggregate["power_twh_2025_2030"],
                "carbon_mtco2_2025_2030": aggregate["carbon_mtco2_2025_2030"],
                "water_million_m3_2025_2030": aggregate["water_million_m3_2025_2030"],
                "power_change_pct_vs_applied": (aggregate["power_twh_2025_2030"] - baseline["power_twh_2025_2030"]) / baseline["power_twh_2025_2030"] * 100.0,
                "carbon_change_pct_vs_applied": (aggregate["carbon_mtco2_2025_2030"] - baseline["carbon_mtco2_2025_2030"]) / baseline["carbon_mtco2_2025_2030"] * 100.0,
                "water_change_pct_vs_applied": (aggregate["water_million_m3_2025_2030"] - baseline["water_million_m3_2025_2030"]) / baseline["water_million_m3_2025_2030"] * 100.0,
            }
            case_records.append(record)
            per_case[level] = record

        summary_records.append({
            "parameter_key": parameter_key,
            "parameter_name": config["parameter_name"],
            "argument_name": config["argument_name"],
            "low_value": config["low"],
            "applied_value": config["applied"],
            "high_value": config["high"],
            "low_power_change_pct": per_case["low"]["power_change_pct_vs_applied"],
            "high_power_change_pct": per_case["high"]["power_change_pct_vs_applied"],
            "low_carbon_change_pct": per_case["low"]["carbon_change_pct_vs_applied"],
            "high_carbon_change_pct": per_case["high"]["carbon_change_pct_vs_applied"],
            "low_water_change_pct": per_case["low"]["water_change_pct_vs_applied"],
            "high_water_change_pct": per_case["high"]["water_change_pct_vs_applied"],
        })

    df_cases = pd.DataFrame(case_records)
    df_summary = pd.DataFrame(summary_records)

    df_cases.to_csv(os.path.join(output_dir, "Fig5_sensitivity_cases.csv"), index=False)
    df_summary.to_csv(os.path.join(output_dir, "Fig5_sensitivity_summary.csv"), index=False)

    print("Done. Saved Fig. 5 sensitivity results to:", os.path.abspath(output_dir))
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    build_fig5_results()
