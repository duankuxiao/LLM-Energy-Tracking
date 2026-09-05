"""Run the two endpoint models and create comparable annual result tables."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from core.annual_cpu_model import run_annual_cpu_model
from core.hourly_workload_resolved_model import (
    Alibaba2026TraceConfig,
    HardwarePowerConfig,
    build_workload_profile,
    run_workload_component_footprint,
)


ROOT_DIR = Path(__file__).resolve().parent
MODEL_ORDER = ("annual_cpu_model", "hourly_workload_resolved_model")
MODEL_PREFIXES = {
    "annual_cpu_model": "annual_cpu",
    "hourly_workload_resolved_model": "hourly_workload_resolved",
}
SUMMARY_METRICS = (
    "facility_energy_mwh",
    "power_twh",
    "load_weighted_carbon_factor_kg_per_mwh",
    "carbon_tco2",
    "carbon_mtco2",
)


def _print_progress(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[model-comparison] {message}", flush=True)


def _standardize_annual_summary(
    frame: pd.DataFrame,
    model: str,
    policy: str,
    include_country: bool,
) -> pd.DataFrame:
    result = frame.copy()
    if "model" not in result:
        result.insert(0, "model", model)
    else:
        result["model"] = model
    if "policy" not in result:
        insert_at = min(2, len(result.columns))
        result.insert(insert_at, "policy", policy)
    else:
        result["policy"] = policy

    if "power_twh" not in result:
        result["power_twh"] = result["facility_energy_mwh"] / 1e6
    if "carbon_mtco2" not in result:
        result["carbon_mtco2"] = result["carbon_tco2"] / 1e6
    result["load_weighted_carbon_factor_kg_per_mwh"] = np.divide(
        result["carbon_tco2"].to_numpy(dtype=float) * 1000.0,
        result["facility_energy_mwh"].to_numpy(dtype=float),
        out=np.zeros(len(result), dtype=float),
        where=result["facility_energy_mwh"].to_numpy(dtype=float) > 0,
    )

    keys = ["model", "scenario", "policy", "year"]
    if include_country:
        keys.append("country")
    return result[keys + list(SUMMARY_METRICS)]


def _build_wide_comparison(
    summary: pd.DataFrame,
    index_columns: Sequence[str],
) -> pd.DataFrame:
    available_models = set(summary["model"].unique())
    missing_models = [model for model in MODEL_ORDER if model not in available_models]
    if missing_models:
        raise ValueError(f"Cannot build endpoint comparison; missing results: {missing_models}")

    comparison = summary[list(index_columns)].drop_duplicates().copy()
    for model in MODEL_ORDER:
        prefix = MODEL_PREFIXES[model]
        model_columns = {metric: f"{prefix}_{metric}" for metric in SUMMARY_METRICS}
        model_result = summary.loc[
            summary["model"] == model,
            list(index_columns) + list(SUMMARY_METRICS),
        ].rename(columns=model_columns)
        comparison = comparison.merge(
            model_result,
            how="inner",
            on=list(index_columns),
            validate="one_to_one",
        )

    annual_energy = comparison["annual_cpu_facility_energy_mwh"]
    hourly_energy = comparison["hourly_workload_resolved_facility_energy_mwh"]
    annual_carbon = comparison["annual_cpu_carbon_tco2"]
    hourly_carbon = comparison["hourly_workload_resolved_carbon_tco2"]
    comparison["annual_cpu_energy_difference_vs_hourly_mwh"] = (
        annual_energy - hourly_energy
    )
    comparison["annual_cpu_energy_difference_vs_hourly_pct"] = np.divide(
        (annual_energy - hourly_energy).to_numpy(dtype=float) * 100.0,
        hourly_energy.to_numpy(dtype=float),
        out=np.full(len(comparison), np.nan),
        where=hourly_energy.to_numpy(dtype=float) != 0,
    )
    comparison["annual_cpu_carbon_difference_vs_hourly_tco2"] = (
        annual_carbon - hourly_carbon
    )
    comparison["annual_cpu_carbon_difference_vs_hourly_pct"] = np.divide(
        (annual_carbon - hourly_carbon).to_numpy(dtype=float) * 100.0,
        hourly_carbon.to_numpy(dtype=float),
        out=np.full(len(comparison), np.nan),
        where=hourly_carbon.to_numpy(dtype=float) != 0,
    )

    return comparison.sort_values(list(index_columns)).reset_index(drop=True)


def run_model_comparison(
    renewable_energy_policy: str = "CP",
    scenarios: Sequence[str] = ("Base",),
    years: int = 5,
    countries: Optional[Sequence[str]] = None,
    year_start: int = 2026,
    workload_profile_path: Union[str, Path] = ROOT_DIR / "dataset",
    server_profile_path: Optional[Union[str, Path]] = None,
    hourly_carbon_factors_dir: Union[str, Path] = ROOT_DIR / "dataset" / "EM-CPNDCNZ",
    hourly_carbon_scope: str = "direct",
    hourly_carbon_fallback_to_annual: bool = True,
    hardware_config: Optional[HardwarePowerConfig] = None,
    trace_config: Optional[Alibaba2026TraceConfig] = None,
    capacity_quantile: float = 0.96,
    max_resource_utilization: float = 1.0,
    pue_scale: float = 1.0,
    ai_capacity_factors: Optional[Mapping[int, float]] = None,
    max_intervals: Optional[int] = None,
    output_dir: Union[str, Path] = ROOT_DIR / "results" / "model_comparison",
    save_outputs: bool = True,
    save_hourly_outputs: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run the annual and hourly endpoint models and return comparison tables."""
    scenarios = list(scenarios)
    countries = list(countries) if countries is not None else None
    output_path = Path(output_dir)
    run_start = time.perf_counter()

    common = {
        "renewable_energy_policy": renewable_energy_policy,
        "scenarios": scenarios,
        "years": years,
        "countries": countries,
        "year_start": year_start,
        "pue_scale": pue_scale,
        "ai_capacity_factors": ai_capacity_factors,
        "save_outputs": save_outputs,
        "verbose": verbose,
    }

    _print_progress(verbose, "Running the annual CPU model.")
    annual_cpu = run_annual_cpu_model(
        **common,
        output_dir=output_path / "annual_cpu_model",
    )

    _print_progress(verbose, "Reading the GPU trace for the hourly workload-resolved model.")
    shared_workload_profile = build_workload_profile(
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        capacity_quantile=capacity_quantile,
        trace_config=trace_config,
        max_intervals=max_intervals,
        verbose=verbose,
    )

    _print_progress(verbose, "Running the hourly workload-resolved model.")
    hourly_workload_resolved = run_workload_component_footprint(
        **common,
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        hardware_config=hardware_config,
        trace_config=trace_config,
        capacity_quantile=capacity_quantile,
        max_resource_utilization=max_resource_utilization,
        max_intervals=max_intervals,
        workload_profile=shared_workload_profile,
        hourly_carbon_factors_dir=hourly_carbon_factors_dir,
        hourly_carbon_scope=hourly_carbon_scope,
        hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
        output_dir=output_path / "hourly_workload_resolved_model",
        save_hourly_outputs=save_hourly_outputs,
    )

    country_summary = pd.concat(
        [
            _standardize_annual_summary(
                annual_cpu["annual_summary"],
                "annual_cpu_model",
                renewable_energy_policy,
                True,
            ),
            _standardize_annual_summary(
                hourly_workload_resolved["annual_summary"],
                "hourly_workload_resolved_model",
                renewable_energy_policy,
                True,
            ),
        ],
        ignore_index=True,
    ).sort_values(["scenario", "policy", "year", "country", "model"]).reset_index(drop=True)

    hourly_global_raw = hourly_workload_resolved["annual_summary"].groupby(
        ["scenario", "year"], as_index=False
    )[["facility_energy_mwh", "power_twh", "carbon_tco2", "carbon_mtco2"]].sum()
    global_summary = pd.concat(
        [
            _standardize_annual_summary(
                annual_cpu["global_summary"],
                "annual_cpu_model",
                renewable_energy_policy,
                False,
            ),
            _standardize_annual_summary(
                hourly_global_raw,
                "hourly_workload_resolved_model",
                renewable_energy_policy,
                False,
            ),
        ],
        ignore_index=True,
    ).sort_values(["scenario", "policy", "year", "model"]).reset_index(drop=True)

    country_comparison = _build_wide_comparison(
        country_summary,
        ["scenario", "policy", "year", "country"],
    )
    global_comparison = _build_wide_comparison(
        global_summary,
        ["scenario", "policy", "year"],
    )

    if save_outputs:
        summary_dir = output_path / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        country_summary.to_csv(summary_dir / "Endpoint_Models_Country_Annual.csv", index=False)
        global_summary.to_csv(summary_dir / "Endpoint_Models_Global_Annual.csv", index=False)
        country_comparison.to_csv(
            summary_dir / "Endpoint_Comparison_Country_Annual.csv", index=False
        )
        global_comparison.to_csv(
            summary_dir / "Endpoint_Comparison_Global_Annual.csv", index=False
        )

    _print_progress(
        verbose,
        f"Both endpoint models completed in {time.perf_counter() - run_start:.1f}s"
        + (f"; results saved under {output_path.resolve()}." if save_outputs else "."),
    )
    if verbose:
        display_columns = [
            "model",
            "scenario",
            "policy",
            "year",
            "power_twh",
            "carbon_mtco2",
        ]
        print(global_summary[display_columns].to_string(index=False, float_format="%.6f"))

    return {
        "annual_cpu": annual_cpu,
        "hourly_workload_resolved": hourly_workload_resolved,
        "shared_workload_profile": shared_workload_profile,
        "country_summary": country_summary,
        "global_summary": global_summary,
        "country_comparison": country_comparison,
        "global_comparison": global_comparison,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the annual CPU and hourly workload-resolved endpoint models "
            "and generate annual comparison CSV files."
        )
    )
    parser.add_argument("--policy", choices=("CP", "NDC", "NZ"), default="CP")
    parser.add_argument("--scenarios", nargs="+", default=["Base"])
    parser.add_argument("--year-start", type=int, default=2026)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--workload-profile-path", type=Path, default=ROOT_DIR / "dataset")
    parser.add_argument("--server-profile-path", type=Path, default=None)
    parser.add_argument(
        "--hourly-carbon-factors-dir",
        type=Path,
        default=ROOT_DIR / "dataset" / "EM-CPNDCNZ",
    )
    parser.add_argument(
        "--hourly-carbon-scope", choices=("direct", "life_cycle"), default="direct"
    )
    parser.add_argument(
        "--strict-hourly-carbon",
        action="store_true",
        help="Fail instead of using annual factors when an hourly factor file is missing.",
    )
    parser.add_argument("--capacity-quantile", type=float, default=0.96)
    parser.add_argument("--max-resource-utilization", type=float, default=1.0)
    parser.add_argument("--pue-scale", type=float, default=1.0)
    parser.add_argument(
        "--max-intervals",
        type=int,
        default=None,
        help="Trace-hour limit for smoke tests only; omit for formal results.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT_DIR / "results" / "model_comparison"
    )
    parser.add_argument("--save-hourly-outputs", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_model_comparison(
        renewable_energy_policy=args.policy,
        scenarios=args.scenarios,
        years=args.years,
        countries=args.countries,
        year_start=args.year_start,
        workload_profile_path=args.workload_profile_path,
        server_profile_path=args.server_profile_path,
        hourly_carbon_factors_dir=args.hourly_carbon_factors_dir,
        hourly_carbon_scope=args.hourly_carbon_scope,
        hourly_carbon_fallback_to_annual=not args.strict_hourly_carbon,
        capacity_quantile=args.capacity_quantile,
        max_resource_utilization=args.max_resource_utilization,
        pue_scale=args.pue_scale,
        max_intervals=args.max_intervals,
        output_dir=args.output_dir,
        save_hourly_outputs=args.save_hourly_outputs,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
