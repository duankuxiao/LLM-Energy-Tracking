"""过去研究中的数据中心能耗碳排放计算方法。"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from dataset.Factors import CF_CP, CF_NDC, CF_NZ, PUE
from dataset.Installed_capacity_data import IT_CAPACITY, IT_RATIO
from core.task_model import TASK_TYPES, build_country_task_ratio_table


DATA_YEAR_START = 2025
SCENARIO_COLUMN = {
    "Base": 0,
    "Lift-Off": 1,
    "High Efficiency": 2,
    "Headwinds": 3,
}
POLICY_CARBON_FACTORS = {
    "CP": CF_CP,
    "NDC": CF_NDC,
    "NZ": CF_NZ,
}


def calculate_past_research_energy_carbon(
    renewable_energy_policy: str,
    scenario: str,
    years: int,
    countries: Sequence[str],
    infer_ratio_by_country: Optional[Mapping[str, float]] = None,
    task_ratio_by_country: Optional[Mapping[str, Mapping[str, float]]] = None,
    default_p_infer: float = 0.75,
    default_p_other: float = 0.05,
    u_train: float = 0.8,
    u_infer: float = 0.5,
    u_other: float = 0.5,
    idle_power_rate: float = 0.23,
    max_power_rate: float = 0.88,
    pue_scale: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate annual data-centre energy use and carbon emissions.

    This is the traditional method used as the M1/M2 hardware baseline. It
    allocates global IT capacity to countries, applies a linear
    utilization--power relationship, converts IT energy to facility energy
    with PUE, and applies annual country carbon factors.

    The method always starts from the native 2025 input year. Returned units:
    ``country_power`` is TWh and ``country_carbon`` is MtCO2.
    """
    if not 1 <= years <= IT_CAPACITY.shape[0]:
        raise ValueError(f"years must be within 1-{IT_CAPACITY.shape[0]}.")
    if scenario not in SCENARIO_COLUMN:
        raise ValueError(f"Unknown scenario '{scenario}'. Allowed: {list(SCENARIO_COLUMN)}")
    if renewable_energy_policy not in POLICY_CARBON_FACTORS:
        raise ValueError("renewable_energy_policy must be one of: CP, NDC, NZ")
    if not countries:
        raise ValueError("countries must not be empty.")
    for name, value in {
        "u_train": u_train,
        "u_infer": u_infer,
        "u_other": u_other,
    }.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if not 0.0 <= idle_power_rate <= max_power_rate:
        raise ValueError("Power rates must satisfy 0 <= idle_power_rate <= max_power_rate.")
    if pue_scale <= 0:
        raise ValueError("pue_scale must be positive.")

    countries = list(countries)
    missing_countries = [country for country in countries if country not in IT_RATIO]
    if missing_countries:
        raise ValueError(f"Unknown countries: {missing_countries}")

    task_ratios = build_country_task_ratio_table(
        countries=countries,
        task_ratio_by_country=task_ratio_by_country,
        infer_ratio_by_country=infer_ratio_by_country,
        default_p_infer=default_p_infer,
        default_p_other=default_p_other,
    )

    year_index = np.arange(years, dtype=float)
    training_activity = 0.9 + (0.925 - 0.9) / years * year_index
    inference_activity = 0.5 + (0.7 - 0.5) / years * year_index
    activity = np.stack(
        [
            training_activity,
            inference_activity,
            inference_activity,
            inference_activity,
        ],
        axis=1,
    )
    utilization_rates = np.array([u_train, u_infer, u_other, u_other], dtype=float)
    task_utilization = (
        task_ratios[None, :, :]
        * activity[:, None, :]
        * utilization_rates[None, None, :]
    )
    utilization = task_utilization.sum(axis=2)

    scenario_column = SCENARIO_COLUMN[scenario]
    global_it_capacity_mw = IT_CAPACITY[:years, scenario_column] * 1e3
    country_shares = np.array([IT_RATIO[country] for country in countries], dtype=float)
    installed_it_capacity_mw = global_it_capacity_mw[:, None] * country_shares[None, :]

    idle_power_mw = installed_it_capacity_mw * idle_power_rate
    maximum_power_mw = installed_it_capacity_mw * max_power_rate
    it_power_mw = idle_power_mw + (maximum_power_mw - idle_power_mw) * utilization
    it_energy_mwh = it_power_mw * 8760.0
    task_it_power_mw = installed_it_capacity_mw[:, :, None] * (
        idle_power_rate * task_ratios[None, :, :]
        + (max_power_rate - idle_power_rate) * task_utilization
    )
    task_it_energy_mwh = task_it_power_mw * 8760.0

    pue = np.stack(
        [PUE[country][:years, scenario_column] for country in countries], axis=1
    ) * pue_scale
    facility_energy_mwh = it_energy_mwh * pue
    task_facility_energy_mwh = task_it_energy_mwh * pue[:, :, None]

    carbon_factors = POLICY_CARBON_FACTORS[renewable_energy_policy]
    annual_carbon_tco2_per_mwh = np.stack(
        [np.asarray(carbon_factors[country][:years], dtype=float) for country in countries],
        axis=1,
    ) / 1000.0
    carbon_tco2 = facility_energy_mwh * annual_carbon_tco2_per_mwh
    task_carbon_tco2 = (
        task_facility_energy_mwh * annual_carbon_tco2_per_mwh[:, :, None]
    )

    years_index = pd.Index(range(DATA_YEAR_START, DATA_YEAR_START + years), name="year")
    task_records = []
    for year_id, year in enumerate(years_index):
        for country_id, country in enumerate(countries):
            for task_type_id, task_type in enumerate(TASK_TYPES):
                task_records.append(
                    {
                        "year": int(year),
                        "country": country,
                        "task_type": task_type,
                        "task_ratio": task_ratios[country_id, task_type_id],
                        "utilization_contribution": task_utilization[
                            year_id, country_id, task_type_id
                        ],
                        "it_energy_mwh": task_it_energy_mwh[
                            year_id, country_id, task_type_id
                        ],
                        "facility_energy_mwh": task_facility_energy_mwh[
                            year_id, country_id, task_type_id
                        ],
                        "carbon_tco2": task_carbon_tco2[
                            year_id, country_id, task_type_id
                        ],
                    }
                )
    return {
        "country_power": pd.DataFrame(
            facility_energy_mwh / 1e6,
            index=years_index,
            columns=countries,
        ),
        "country_carbon": pd.DataFrame(
            carbon_tco2 / 1e6,
            index=years_index,
            columns=countries,
        ),
        "task_type_energy": pd.DataFrame.from_records(task_records),
    }
