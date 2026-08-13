"""Run M1--M4 together and create comparable annual result tables."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from core.m1_annual_cpu_model import run_m1_annual_cpu_model
from core.m2_hourly_carbon_cpu_model import run_m2_hourly_carbon_cpu_model
from core.m3_annual_gpu_model import run_m3_annual_gpu_model
from core.m4_hourly_gpu_model import (
    Alibaba2026TraceConfig,
    HardwarePowerConfig,
    build_workload_profile,
    run_workload_component_footprint,
)


ROOT_DIR = Path(__file__).resolve().parent
MODEL_ORDER = ("M1", "M2", "M3", "M4")
SUMMARY_METRICS = (
    "facility_energy_mwh",
    "power_twh",
    "load_weighted_carbon_factor_kg_per_mwh",
    "carbon_tco2",
    "carbon_mtco2",
)


def _print_progress(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[run-m1-m4] {message}", flush=True)


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
        raise ValueError(f"Cannot build comparison; missing model results: {missing_models}")

    comparison = summary[list(index_columns)].drop_duplicates().copy()
    for model in MODEL_ORDER:
        model_columns = {
            metric: f"{model.lower()}_{metric}" for metric in SUMMARY_METRICS
        }
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

    for model in MODEL_ORDER[:-1]:
        prefix = model.lower()
        energy = comparison[f"{prefix}_facility_energy_mwh"]
        reference_energy = comparison["m4_facility_energy_mwh"]
        carbon = comparison[f"{prefix}_carbon_tco2"]
        reference_carbon = comparison["m4_carbon_tco2"]
        comparison[f"{prefix}_energy_difference_vs_m4_mwh"] = energy - reference_energy
        comparison[f"{prefix}_energy_difference_vs_m4_pct"] = np.divide(
            (energy - reference_energy).to_numpy(dtype=float) * 100.0,
            reference_energy.to_numpy(dtype=float),
            out=np.full(len(comparison), np.nan),
            where=reference_energy.to_numpy(dtype=float) != 0,
        )
        comparison[f"{prefix}_carbon_difference_vs_m4_tco2"] = carbon - reference_carbon
        comparison[f"{prefix}_carbon_difference_vs_m4_pct"] = np.divide(
            (carbon - reference_carbon).to_numpy(dtype=float) * 100.0,
            reference_carbon.to_numpy(dtype=float),
            out=np.full(len(comparison), np.nan),
            where=reference_carbon.to_numpy(dtype=float) != 0,
        )

    return comparison.sort_values(list(index_columns)).reset_index(drop=True)


def run_all_models(
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
    output_dir: Union[str, Path] = ROOT_DIR / "results" / "m1_m4_comparison",
    save_outputs: bool = True,
    save_hourly_outputs: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run all four cases and return model results plus comparison tables."""
    scenarios = list(scenarios)
    countries = list(countries) if countries is not None else None
    output_path = Path(output_dir)
    run_start = time.perf_counter()

    common_annual = {
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

    _print_progress(verbose, "Running M1 annual CPU case.")
    m1 = run_m1_annual_cpu_model(
        **common_annual,
        output_dir=output_path / "m1_annual_cpu",
    )

    _print_progress(verbose, "Running M2 hourly-carbon CPU case.")
    m2 = run_m2_hourly_carbon_cpu_model(
        **common_annual,
        hourly_carbon_factors_dir=hourly_carbon_factors_dir,
        hourly_carbon_scope=hourly_carbon_scope,
        hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
        output_dir=output_path / "m2_hourly_carbon_cpu",
        save_hourly_outputs=save_hourly_outputs,
    )

    _print_progress(verbose, "Reading the GPU trace once for both M3 and M4.")
    shared_workload_profile = build_workload_profile(
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        capacity_quantile=capacity_quantile,
        trace_config=trace_config,
        max_intervals=max_intervals,
        verbose=verbose,
    )

    common_gpu = {
        **common_annual,
        "workload_profile_path": workload_profile_path,
        "server_profile_path": server_profile_path,
        "hardware_config": hardware_config,
        "trace_config": trace_config,
        "capacity_quantile": capacity_quantile,
        "max_resource_utilization": max_resource_utilization,
        "max_intervals": max_intervals,
        "workload_profile": shared_workload_profile,
    }

    _print_progress(verbose, "Running M3 annual-carbon GPU case with the shared trace profile.")
    m3 = run_m3_annual_gpu_model(
        **common_gpu,
        output_dir=output_path / "m3_annual_gpu",
    )

    _print_progress(verbose, "Running M4 hourly-carbon GPU case with the shared trace profile.")
    m4 = run_workload_component_footprint(
        **common_gpu,
        hourly_carbon_factors_dir=hourly_carbon_factors_dir,
        hourly_carbon_scope=hourly_carbon_scope,
        hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
        output_dir=output_path / "m4_hourly_gpu",
        save_hourly_outputs=save_hourly_outputs,
    )

    country_frames = [
        _standardize_annual_summary(m1["annual_summary"], "M1", renewable_energy_policy, True),
        _standardize_annual_summary(m2["annual_summary"], "M2", renewable_energy_policy, True),
        _standardize_annual_summary(m3["annual_summary"], "M3", renewable_energy_policy, True),
        _standardize_annual_summary(m4["annual_summary"], "M4", renewable_energy_policy, True),
    ]
    country_summary = pd.concat(country_frames, ignore_index=True).sort_values(
        ["scenario", "policy", "year", "country", "model"]
    ).reset_index(drop=True)

    global_frames = [
        _standardize_annual_summary(m1["global_summary"], "M1", renewable_energy_policy, False),
        _standardize_annual_summary(m2["global_summary"], "M2", renewable_energy_policy, False),
        _standardize_annual_summary(m3["global_summary"], "M3", renewable_energy_policy, False),
    ]
    m4_global_raw = m4["annual_summary"].groupby(
        ["scenario", "year"], as_index=False
    )[["facility_energy_mwh", "power_twh", "carbon_tco2", "carbon_mtco2"]].sum()
    global_frames.append(
        _standardize_annual_summary(m4_global_raw, "M4", renewable_energy_policy, False)
    )
    global_summary = pd.concat(global_frames, ignore_index=True).sort_values(
        ["scenario", "policy", "year", "model"]
    ).reset_index(drop=True)

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
        country_summary.to_csv(summary_dir / "All_Models_Country_Annual.csv", index=False)
        global_summary.to_csv(summary_dir / "All_Models_Global_Annual.csv", index=False)
        country_comparison.to_csv(
            summary_dir / "Model_Comparison_Country_Annual.csv", index=False
        )
        global_comparison.to_csv(
            summary_dir / "Model_Comparison_Global_Annual.csv", index=False
        )

    _print_progress(
        verbose,
        f"All four cases completed in {time.perf_counter() - run_start:.1f}s"
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
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "m4": m4,
        "shared_workload_profile": shared_workload_profile,
        "country_summary": country_summary,
        "global_summary": global_summary,
        "country_comparison": country_comparison,
        "global_comparison": global_comparison,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M1--M4 once and generate annual model-comparison CSV files."
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
        "--output-dir", type=Path, default=ROOT_DIR / "results" / "m1_m4_comparison"
    )
    parser.add_argument("--save-hourly-outputs", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_all_models(
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
