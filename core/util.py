import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.Factors import *
from dataset.Installed_capacity_data import *

years = [2025, 2026, 2027, 2028, 2029, 2030]
scenario_names = ["Base", "Lift-Off", "High Efficiency", "Headwinds"]

_countries = [
    "USA", "China", "Japan", "France",
    "India", "Singapore", "Canada", "Germany",
    "United_Kingdom", "Australia", "Italy", "South_Korea",
    "South_Africa", "Ireland", "UAE", "Brazil",
    "Israel", "Netherlands", "Spain", "Sweden",
    "Belgium", "Norway", "Poland", "Switzerland",
]

_region = [
    0, 2, 2, 1,
    2, 2, 2, 1,
    2, 2, 1, 2,
    2, 1, 2, 2,
    2, 1, 1, 1,
    1, 2, 1, 2,
]


def get_property(scene="Base", renewable_energy_policy="CP"):
    """
    Assemble the core country-year tensors used by the optimization models.

    The returned arrays follow the same ordering across all modules so the
    optimization layer can work entirely on precomputed matrices.
    """
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

    n_countries, n_years = len(_countries), len(years)
    r_train_0, r_inference_0 = 0.9, 0.5
    r_train_2030, r_inference_2030 = 0.925, 0.7
    dlc_rate_0, dlc_increase = 0.05, 0.2
    scenario_index = scenario_names.index(scene)

    # Sort countries by region so the constrained deployment model can
    # operate on contiguous region blocks.
    countries = [_countries[i] for i in np.argsort(_region)]
    region = np.sort(_region)
    gw_base = it_capacity[:, scenario_index]
    gw_country = np.array([gw_base * it_ratio[c] for c in countries], dtype=float).T

    # Build the year-dependent training and inference activity assumptions.
    r_train = np.zeros(n_years)
    r_infer = np.zeros(n_years)
    dlc_rate = np.zeros(n_years)
    for i in range(n_years):
        r_train[i] = r_train_0 + (r_train_2030 - r_train_0) / n_years * i
        r_infer[i] = r_inference_0 + (r_inference_2030 - r_inference_0) / n_years * i
        dlc_rate[i] = dlc_rate_0 if i == 0 else dlc_rate[i - 1] * (1 + dlc_increase)

    # Precompute environmental coefficients for each country and year.
    emission_data = np.zeros((n_years, n_countries))
    water_data = np.zeros((n_years, n_countries))
    pue = np.zeros((n_years, n_countries))
    wue = np.zeros((n_years, n_countries))
    for y in range(n_years):
        for j, c in enumerate(countries):
            emission_data[y, j] = e_data[c][y] / 1000.0
            water_data[y, j] = w_data[c][y]
            pue[y, j] = PUE[c][y, scenario_index]
            wue[y, j] = WUE[c] * (1 - dlc_rate[y]) + (WUE[c] - 0.137) * dlc_rate[y]

    return region, gw_country, r_train, r_infer, emission_data, water_data, pue, wue, countries
