import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.workload_training_migration_optimization import (  # noqa: E402
    MIGRATION_CONSTRAINTS,
    SOLVE_MODES,
    SOLVERS,
    run_training_migration_optimization,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize training workload migration for minimum global CO2 emissions."
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
        "--constraint",
        choices=MIGRATION_CONSTRAINTS,
        default="europe_only",
        help="global allows all modeled countries; europe_only only moves European training within Europe.",
    )
    parser.add_argument(
        "--delay-hours",
        type=int,
        default=72,
        help="Maximum allowed training delay in hours.",
    )
    parser.add_argument(
        "--solve-mode",
        choices=SOLVE_MODES,
        default="auto",
        help="auto uses a single LP for small runs and rolling windows for large full-year runs.",
    )
    parser.add_argument(
        "--solver",
        choices=SOLVERS,
        default="scipy",
        help="Linear programming backend used inside each optimization window.",
    )
    parser.add_argument(
        "--commit-hours",
        type=int,
        default=72,
        help="Number of source hours committed per rolling-window LP.",
    )
    parser.add_argument(
        "--monolithic-variable-limit",
        type=int,
        default=1_000_000,
        help="In auto mode, switch to windowed solving above this estimated variable count.",
    )
    parser.add_argument(
        "--workload-path",
        default=str(ROOT_DIR / "dataset" / "result_df_full_year_2020.pkl"),
    )
    parser.add_argument("--workload-year", type=int, default=2020)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "results" / "workload_training_migration_optimization"),
    )
    parser.add_argument(
        "--execution-policy",
        choices=["capacity", "origin", "hybrid"],
        default="capacity",
        help="Execution policy for fixed inference and CPU/data workloads.",
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
        default=0.96,
        help="Trace load quantile treated as the reference provisioned capacity.",
    )
    parser.add_argument(
        "--max-resource-utilization",
        type=float,
        default=1.0,
        help="Maximum country-level utilization allowed by the migration optimizer.",
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
        help="Use annual carbon factors expanded to hourly values.",
    )
    parser.add_argument(
        "--strict-hourly-carbon",
        action="store_true",
        help="Fail when an hourly carbon CSV is missing instead of falling back to annual factors.",
    )
    parser.add_argument(
        "--save-schedule",
        action="store_true",
        help="Save non-zero migration decision variables. This can be large for full-year runs.",
    )
    parser.add_argument(
        "--max-intervals",
        type=int,
        default=None,
        help="Optional smoke-test limit for the number of 15-minute workload intervals.",
    )
    parser.add_argument(
        "--linprog-time-limit",
        type=float,
        default=None,
        help="Optional HiGHS time limit in seconds.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run calculations without writing CSV outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    linprog_options = {}
    if args.linprog_time_limit is not None:
        linprog_options["time_limit"] = args.linprog_time_limit

    results = run_training_migration_optimization(
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
        delay_hours=args.delay_hours,
        migration_constraint=args.constraint,
        save_schedule=args.save_schedule,
        max_intervals=args.max_intervals,
        solve_mode=args.solve_mode,
        commit_hours=args.commit_hours,
        monolithic_variable_limit=args.monolithic_variable_limit,
        solver=args.solver,
        linprog_options=linprog_options or None,
        verbose=True,
    )

    print()
    print("Aggregate optimization summary:")
    print(
        results["annual_summary"].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


if __name__ == "__main__":
    main()
