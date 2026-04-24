import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.Carbon_water_footprint import AIFootprint

if __name__ == '__main__':
    years = 6
    countries = ["USA", "China", "Japan", "France", "India", "Singapore",
                 "Canada", "Germany", "United_Kingdom", "Australia", "Italy", "South_Korea",
                 "South_Africa", "Ireland", "UAE", "Brazil", "Israel",
                 "Netherlands", "Spain", "Sweden", "Belgium", "Norway",
                 "Poland", "Switzerland"]
    scenarios = ["Base"]
    renewable_energy_policy = 'CP'
    LLM_policy = 'SDS'
    if LLM_policy == 'SDS':  # Strict Data Sovereignty
        infer_ratio_by_country = {'USA': 0.7, 'Belgium': 0.7}
    elif LLM_policy == 'UGD':  # Unconstrained Global Deployment
        infer_ratio_by_country = {'USA': 0.6106, 'Belgium': 0.0000, 'France': 0.0000, 'Sweden': 0.0000, 'Spain': 0.0000, 'Netherlands': 0.0000, 'Germany': 0.0000, 'Italy': 0.0000,
                                  'Poland': 1.0000, 'Ireland': 0.0000, 'Norway': 1.0000, 'Israel': 1.0000, 'Brazil': 1.0000, 'UAE': 1.0000, 'South_Korea': 0.0000,
                                  'Australia': 1.0000,'United_Kingdom': 0.0000, 'Canada': 1.0000, 'Singapore': 0.0000, 'India': 1.0000, 'Japan': 1.0000, 'China': 1.0000,
                                  'South_Africa': 1.0000, 'Switzerland': 1.0000}
    elif LLM_policy == 'BCD':  # Bloc-Constrained Deployment
        infer_ratio_by_country = {"USA": 0.7000, "Belgium": 0.0000, "France": 0.3575, "Sweden": 0.0000, "Spain": 0.0000, "Netherlands": 1.0000, "Germany": 1.0000, "Italy": 1.0000,
                                  "Poland": 1.0000, "Ireland": 1.0000, "Norway": 1.0000, "Israel": 0.0000, "Brazil": 1.0000, "UAE": 1.0000, "South_Korea": 0.0000,
                                  "Australia": 1.0000, "United_Kingdom": 0.0000, "Canada": 1.0000, "Singapore": 0.0000, "India": 1.0000, "Japan": 1.0000, "China": 0.6852,
                                  "South_Africa": 1.0000, "Switzerland": 0.0000, }

    scenarios = ["Base"]

    AIFootprint(renewable_energy_policy,
                scenarios,
                years,
                countries,
                infer_ratio_by_country=infer_ratio_by_country,
                default_p_infer=0.7,
                output_dir=str(ROOT_DIR / "results" / "fig3" / LLM_policy),
                year_start=2025)
