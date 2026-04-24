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
    scenarios = ["Base", "Lift-Off", "High Efficiency", "Headwinds"]
    renewable_energy_policy = 'CP'
    infer_ratio_by_country = {'USA': 0.7, 'Belgium': 0.7}
    for s in scenarios:
        AIFootprint(renewable_energy_policy,
                    scenarios,
                    years,
                    countries,
                    infer_ratio_by_country=infer_ratio_by_country,
                    default_p_infer=0.7,
                    output_dir=str(ROOT_DIR / "results" / "fig1" / s),
                    year_start=2025)
