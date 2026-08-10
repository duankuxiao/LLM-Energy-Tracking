import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.workload_component_model import run_workload_component_footprint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the workload-driven component energy model."
    )
    parser.add_argument("--policy", choices=["CP", "NDC", "NZ"], default="CP")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["Base"],
        choices=["Base", "Lift-Off", "High Efficiency", "Headwinds"],
    )
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--year-start", type=int, default=2026)
    parser.add_argument(
        "--workload-path",
        default=str(ROOT_DIR / "dataset" / "result_df_full_year_2020.pkl"),
    )
    parser.add_argument("--workload-year", type=int, default=2020)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "results" / "workload_component_model"),
    )
    parser.add_argument(
        "--execution-policy",
        choices=["capacity", "origin", "hybrid"],
        default="capacity",
        help="capacity allocates execution by data-center capacity; origin keeps execution near demand; hybrid mixes both.",
    )
    parser.add_argument(
        "--inference-origin-fraction",
        type=float,
        default=0.75,
        help="Hybrid-policy fraction of inference execution tied to origin demand weights.",
    )
    parser.add_argument(
        "--cpu-data-origin-fraction",
        type=float,
        default=0.50,
        help="Hybrid-policy fraction of CPU/data execution tied to origin demand weights.",
    )
    parser.add_argument(
        "--capacity-quantile",
        type=float,
        default=0.97,
        help="Trace load quantile treated as the reference provisioned capacity.",
    )
    parser.add_argument(
        "--max-resource-utilization",
        type=float,
        default=1.0,
        help="Upper bound applied to trace-derived utilization before scaling to future capacity.",
    )
    parser.add_argument(
        "--hourly-carbon-dir",
        default=str(ROOT_DIR / "dataset" / "EM-estimate"),
        help="Directory containing country-level hourly carbon factor CSV files.",
    )
    parser.add_argument(
        "--hourly-carbon-scope",
        choices=["direct", "life_cycle"],
        default="direct",
        help="Hourly carbon intensity column to use.",
    )
    parser.add_argument(
        "--disable-hourly-carbon",
        action="store_true",
        help="Use the original annual carbon factors instead of hourly carbon CSV files.",
    )
    parser.add_argument(
        "--strict-hourly-carbon",
        action="store_true",
        help="Fail when an hourly carbon CSV is missing instead of falling back to annual factors.",
    )
    parser.add_argument(
        "--save-hourly-carbon",
        action="store_true",
        help="Also save the large country-hour carbon output CSV.",
    )
    parser.add_argument(
        "--max-intervals",
        type=int,
        default=None,
        help="Optional smoke-test limit for the number of 15-minute workload intervals.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run calculations without writing CSV outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_workload_component_footprint(
        renewable_energy_policy=args.policy,
        scenarios=args.scenarios,
        years=args.years,
        workload_profile_path=args.workload_path,
        workload_year=args.workload_year,
        year_start=args.year_start,
        output_dir=args.output_dir,
        save_outputs=not args.no_save,
        execution_policy=args.execution_policy,
        inference_origin_fraction=args.inference_origin_fraction,
        cpu_data_origin_fraction=args.cpu_data_origin_fraction,
        capacity_quantile=args.capacity_quantile,
        max_resource_utilization=args.max_resource_utilization,
        hourly_carbon_factors_dir=None if args.disable_hourly_carbon else args.hourly_carbon_dir,
        hourly_carbon_scope=args.hourly_carbon_scope,
        hourly_carbon_fallback_to_annual=not args.strict_hourly_carbon,
        save_hourly_outputs=args.save_hourly_carbon,
        max_intervals=args.max_intervals,
        verbose=True,
    )

    summary = results["annual_summary"].groupby(["scenario", "year"], as_index=False)[
        ["power_twh", "carbon_mtco2", "water_million_m3"]
    ].sum()
    print()
    print("Aggregate summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
