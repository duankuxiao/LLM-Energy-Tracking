import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.Factors import *
from dataset.Installed_capacity_data import *


def AIFootprint(
    renewable_energy_policy: str,
    scenarios: list,
    years: int,
    countries: list,
    infer_ratio_by_country: Optional[Dict[str, float]] = None,
    default_p_infer: float = 0.7,
    output_dir: str = "../results",
    year_start: int = 2025,
    u_train: float = 0.8,
    u_infer: float = 0.5,
    idle_power_rate: float = 0.23,
    max_power_rate: float = 0.88,
    pue_scale: float = 1.0,
    save_outputs: bool = True,
    verbose: bool = True,
    return_results: bool = False,
):
    """
    Compute country-level and aggregate AI data-center footprints.

    The function keeps the original project workflow intact:
    1. Allocate global IT capacity to countries.
    2. Convert capacity to annual electricity use with the utilization-based model.
    3. Apply country-level PUE, grid carbon factors, WUE, and grid water factors.
    4. Export per-country and aggregate CSV outputs when requested.

    Output units are scaled by 1e6:
    - Power: TWh
    - Carbon: MtCO2
    - Water: million m3
    """
    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

    # Map scenario names to the corresponding columns in the capacity and PUE tables.
    scenario_col_map = {
        "Base": 0,
        "Lift-Off": 1,
        "High Efficiency": 2,
        "Headwinds": 3,
    }
    for s in scenarios:
        if s not in scenario_col_map:
            raise ValueError(f"Unknown scenario '{s}'. Allowed: {list(scenario_col_map.keys())}")
    col_idx_list = [scenario_col_map[s] for s in scenarios]
    n_countries = len(countries)
    year_labels = [year_start + i for i in range(years)]

    # Allow scenario scripts to override country-level training/inference splits.
    if not (0.0 <= float(default_p_infer) <= 1.0):
        raise ValueError(f"default_p_infer must be in [0,1], got {default_p_infer}")

    p_infer_vec = np.full((n_countries,), float(default_p_infer), dtype=float)
    if infer_ratio_by_country is not None:
        country_to_idx = {c: i for i, c in enumerate(countries)}
        for c, v in infer_ratio_by_country.items():
            if c not in country_to_idx:
                raise ValueError(
                    f"infer_ratio_by_country contains unknown country '{c}'. "
                    f"Allowed countries: {countries}"
                )
            v = float(v)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"p_infer for '{c}' must be in [0,1], got {v}")
            p_infer_vec[country_to_idx[c]] = v
    p_train_vec = 1.0 - p_infer_vec

    # Build the time-varying utilization inputs used by the server power model.
    r_train_0, r_infer_0 = 0.9, 0.5
    r_train_2030, r_infer_2030 = 0.925, 0.7
    year_idx = np.arange(years, dtype=float)
    r_train_year = r_train_0 + (r_train_2030 - r_train_0) / years * year_idx
    r_infer_year = r_infer_0 + (r_infer_2030 - r_infer_0) / years * year_idx
    utilization_level = (
        p_train_vec[None, :] * u_train * r_train_year[:, None]
        + p_infer_vec[None, :] * u_infer * r_infer_year[:, None]
    )

    # Model the gradual improvement in direct liquid cooling adoption.
    dlc_rate_0 = 0.05
    dlc_increase = 0.2
    dlc_rate = np.zeros((years,), dtype=float)
    dlc_rate[0] = dlc_rate_0
    for y in range(1, years):
        dlc_rate[y] = dlc_rate[y - 1] * (1 + dlc_increase)

    # Select the policy-dependent grid carbon and grid water factors.
    if renewable_energy_policy == "CP":
        e_data = carbon_emissions_factors_CP
        w_data = grid_water_factors_CP
    elif renewable_energy_policy == "NDC":
        e_data = carbon_emissions_factors_NDC
        w_data = grid_water_factors_NDC
    elif renewable_energy_policy == "NZ":
        e_data = carbon_emissions_factors_NZ
        w_data = grid_water_factors_NZ
    else:
        raise ValueError("renewable_energy_policy must be one of: CP, NDC, NZ")

    # Preassemble country-year factor matrices shared across scenarios.
    emission_data = np.stack([np.array(e_data[c][:years]) for c in countries], axis=1) / 1000.0
    grid_water = np.stack([np.array(w_data[c][:years]) for c in countries], axis=1)

    wue_vec = np.array([WUE[c] for c in countries], dtype=float)
    wue_year_country = (
        wue_vec[None, :] * (1 - dlc_rate[:, None])
        + (wue_vec[None, :] - 0.137) * dlc_rate[:, None]
    )
    country_share = np.array([it_ratio[c] for c in countries], dtype=float)

    # Collect outputs in data frames so all result scripts share the same export format.
    df_power = pd.DataFrame(index=year_labels)
    df_water = pd.DataFrame(index=year_labels)
    df_dwater = pd.DataFrame(index=year_labels)
    df_carbon = pd.DataFrame(index=year_labels)
    df_power_train = pd.DataFrame(index=year_labels)
    df_power_infer = pd.DataFrame(index=year_labels)
    df_total = pd.DataFrame(index=year_labels)

    for s, col_idx in zip(scenarios, col_idx_list):
        # Step 1: allocate global IT capacity to each country.
        installed_capacity_global = np.array(it_capacity[:years, col_idx], dtype=float) * 1e3
        installed_capacity_country = installed_capacity_global[:, None] * country_share[None, :]

        # Step 2: convert installed capacity to effective IT power.
        pmax_country = installed_capacity_country * max_power_rate
        pmin_country = installed_capacity_country * idle_power_rate
        capacity_eff_country = (pmax_country - pmin_country) * utilization_level + pmin_country
        power_it = capacity_eff_country * 8760

        # Keep separate training and inference electricity series for Fig. 3 reporting.
        util_train_component = p_train_vec[None, :] * u_train * r_train_year[:, None]
        util_infer_component = p_infer_vec[None, :] * u_infer * r_infer_year[:, None]
        capacity_eff_train_country = (
            (pmax_country - pmin_country) * util_train_component
            + pmin_country * p_train_vec[None, :]
        )
        capacity_eff_infer_country = (
            (pmax_country - pmin_country) * util_infer_component
            + pmin_country * p_infer_vec[None, :]
        )
        power_it_train = capacity_eff_train_country * 8760
        power_it_infer = capacity_eff_infer_country * 8760

        # Step 3: apply facility and grid factors.
        pue = np.stack([np.array(PUE[c][:years, col_idx]) for c in countries], axis=1) * pue_scale
        power_usage = pue * power_it
        carbon = emission_data * power_usage
        direct_water = wue_year_country * power_usage
        water = direct_water + grid_water * power_usage
        power_usage_train = pue * power_it_train
        power_usage_infer = pue * power_it_infer

        # Step 4: write per-country outputs for the current scenario.
        for j, c in enumerate(countries):
            colname = f"{c}"
            df_power[colname] = power_usage[:, j] / 1e6
            df_water[colname] = water[:, j] / 1e6
            df_dwater[colname] = direct_water[:, j] / 1e6
            df_carbon[colname] = carbon[:, j] / 1e6
            df_power_train[colname] = power_usage_train[:, j] / 1e6
            df_power_infer[colname] = power_usage_infer[:, j] / 1e6

        # Store scenario-level totals in the same units used by the manuscript figures.
        df_total[f"Power_{s}"] = np.sum(power_usage / 1e6, axis=1)
        df_total[f"Water_{s}"] = np.sum(water / 1e6, axis=1)
        df_total[f"DirectWater_{s}"] = np.sum(direct_water / 1e6, axis=1)
        df_total[f"Carbon_{s}"] = np.sum(carbon / 1e6, axis=1)

    tag = "-".join([s.replace(" ", "") for s in scenarios])
    tag = tag if tag else "None"

    if save_outputs:
        df_power.to_csv(os.path.join(output_dir, f"Country_PowerUsage_{renewable_energy_policy}_{tag}.csv"))
        df_water.to_csv(os.path.join(output_dir, f"Country_WaterUsage_{renewable_energy_policy}_{tag}.csv"))
        df_dwater.to_csv(
            os.path.join(output_dir, f"Country_DirectWaterUsage_{renewable_energy_policy}_{tag}.csv")
        )
        df_carbon.to_csv(os.path.join(output_dir, f"Country_CarbonEmission_{renewable_energy_policy}_{tag}.csv"))
        df_total.to_csv(os.path.join(output_dir, f"Total_Summary_Results_{renewable_energy_policy}_{tag}.csv"))
        df_power_train.to_csv(
            os.path.join(output_dir, f"Country_PowerUsage_Train_{renewable_energy_policy}_{tag}.csv")
        )
        df_power_infer.to_csv(
            os.path.join(output_dir, f"Country_PowerUsage_Infer_{renewable_energy_policy}_{tag}.csv")
        )

    if verbose:
        if save_outputs:
            print("Done. Saved to:", os.path.abspath(output_dir))
        print("Inference ratios (p_infer) by country:")
        for c, pi in zip(countries, p_infer_vec):
            print(f"  {c}: p_infer={pi:.3f}, p_train={1 - pi:.3f}")

    if return_results:
        aggregate = {}
        for s in scenarios:
            aggregate[s] = {
                "power_twh_2025_2030": float(df_total[f"Power_{s}"].sum()),
                "carbon_mtco2_2025_2030": float(df_total[f"Carbon_{s}"].sum()),
                "water_million_m3_2025_2030": float(df_total[f"Water_{s}"].sum()),
                "direct_water_million_m3_2025_2030": float(df_total[f"DirectWater_{s}"].sum()),
            }

        return {
            "country_power": df_power,
            "country_water": df_water,
            "country_direct_water": df_dwater,
            "country_carbon": df_carbon,
            "country_power_train": df_power_train,
            "country_power_infer": df_power_infer,
            "total_summary": df_total,
            "aggregate": aggregate,
        }
