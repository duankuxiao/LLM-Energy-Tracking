from pathlib import Path

from core import AIFootprint


def main():
    countries = [
        "USA", "China", "Japan", "France", "India", "Singapore",
        "Canada", "Germany", "United_Kingdom", "Australia", "Italy", "South_Korea",
        "South_Africa", "Ireland", "UAE", "Brazil", "Israel",
        "Netherlands", "Spain", "Sweden", "Belgium", "Norway",
        "Poland", "Switzerland",
    ]

    output_dir = Path(__file__).resolve().parent / "results" / "example_run"

    result = AIFootprint(
        renewable_energy_policy="CP",
        scenarios=["Base"],
        years=6,
        countries=countries,
        default_p_infer=0.7,
        output_dir=str(output_dir),
        save_outputs=True,
        verbose=False,
        return_results=True,
    )

    print("Example run completed.")
    print("Output directory:", output_dir)
    print()
    print("Aggregate 2025-2030 results for the Base scenario:")
    print(result["aggregate"]["Base"])
    print()
    print("First rows of the annual summary table:")
    print(result["total_summary"].head())


if __name__ == "__main__":
    main()
