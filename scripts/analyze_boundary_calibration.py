"""Compare calibration conventions for the 24 major AI study countries.

The homogeneous-intensity case assumes identical facility electricity per unit
of IT capacity inside and outside the study boundary. It is a conditional
boundary comparison, not observed electricity coverage. One common factor set,
constructed with the hourly reference, is applied to BOTH endpoint models.
Existing default factors and raw inputs are never overwritten.

Run: python -B scripts/analyze_boundary_calibration.py --strict-hourly-carbon
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import faulthandler
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.accounting_boundary import (
    AGGREGATE_SCOPE, annotate_selected_aggregate, build_boundary_metadata, resolve_countries,
)
from core.annual_cpu_model import run_annual_cpu_model
from core.research_io import sha256_file
from core.run_provenance import runtime_environment as runtime_versions
from core.hourly_workload_resolved_model import (
    Alibaba2026TraceConfig, HardwarePowerConfig, WorkloadProfile,
    DEFAULT_PARQUET_BATCH_SIZE, TraceReadCallback,
    build_workload_profile, run_workload_component_footprint,
)
from dataset.Installed_capacity_data import (
    BASE_ELECTRICITY_REFERENCE_ROLE, BASE_GLOBAL_ELECTRICITY_REFERENCE_TWH,
    DEFAULT_AI_CAPACITY_FACTORS,
)


CASES = ("legacy_factors", "recalibrated_global_reference", "homogeneous_intensity_boundary")
SUM_METRICS = ("installed_it_mw", "it_energy_mwh", "facility_energy_mwh", "carbon_tco2")


def _workspace_output(directory: Path) -> Path:
    output = (directory if directory.is_absolute() else ROOT_DIR / directory).resolve()
    if not output.is_relative_to(ROOT_DIR):
        raise ValueError("Output directory must be inside the workspace.")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory; this path already exists: {output}")
    return output


def analyze_boundary_calibration(
    countries: Optional[Sequence[str]] = None,
    year_start: int = 2026,
    years: int = 5,
    policy: str = "CP",
    workload_profile_path: Path = ROOT_DIR / "dataset",
    server_profile_path: Optional[Path] = None,
    hourly_carbon_factors_dir: Path = ROOT_DIR / "dataset" / "EM-CPNDCNZ",
    hourly_carbon_scope: str = "direct",
    strict_hourly_carbon: bool = True,
    max_intervals: Optional[int] = None,
    workload_profile: Optional[WorkloadProfile] = None,
    verbose: bool = True,
    parquet_batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
    read_callback: Optional[TraceReadCallback] = None,
    experiment_context=None,
) -> dict:
    """Calculate comparisons in memory, reusing a single workload profile."""
    selected = resolve_countries(countries)
    requested_years = list(range(year_start, year_start + years))
    if years <= 0 or any(year not in BASE_GLOBAL_ELECTRICITY_REFERENCE_TWH for year in requested_years):
        raise ValueError("Requested years must be within 2025-2030.")
    if max_intervals is not None and max_intervals <= 0:
        raise ValueError("max_intervals must be positive.")
    boundary = build_boundary_metadata(selected)
    coverage = float(boundary.iloc[0]["it_capacity_share_sum"])
    profile = experiment_context.profile if experiment_context is not None else workload_profile
    if profile is None:
        profile = build_workload_profile(
            workload_profile_path=workload_profile_path,
            server_profile_path=server_profile_path,
            max_intervals=max_intervals,
            verbose=verbose,
            parquet_batch_size=parquet_batch_size,
            read_callback=read_callback,
        )
    common = dict(renewable_energy_policy=policy, scenarios=["Base"], years=years,
                  countries=selected, year_start=year_start, save_outputs=False, verbose=False)
    hourly_options = dict(workload_profile=profile, hourly_carbon_scope=hourly_carbon_scope,
                          hourly_carbon_fallback_to_annual=not strict_hourly_carbon)
    raw_result = (experiment_context.run_hourly if experiment_context is not None else run_workload_component_footprint)(
        **common, **hourly_options, ai_capacity_factors={year: 1.0 for year in requested_years},
        hourly_carbon_factors_dir=None,
    )
    raw_energy = raw_result["selected_countries_summary"].set_index("year")["facility_energy_mwh"] / 1e6
    if not np.isfinite(raw_energy).all() or (raw_energy <= 0).any():
        raise ValueError("Raw selected-country electricity must be finite and positive.")
    factor_sets = {
        "legacy_factors": {year: DEFAULT_AI_CAPACITY_FACTORS[year] for year in requested_years},
        "recalibrated_global_reference": {
            year: BASE_GLOBAL_ELECTRICITY_REFERENCE_TWH[year] / float(raw_energy.loc[year])
            for year in requested_years
        },
        "homogeneous_intensity_boundary": {
            year: BASE_GLOBAL_ELECTRICITY_REFERENCE_TWH[year] * coverage / float(raw_energy.loc[year])
            for year in requested_years
        },
    }
    del raw_result
    country_frames, selected_frames, calibration_records, carbon_records = [], [], [], []
    for case in CASES:
        if verbose:
            print(f"[boundary] {case}: both endpoints; countries={len(selected)}, years={year_start}-{requested_years[-1]}", flush=True)
        factors = factor_sets[case]
        if any(not np.isfinite(value) or not 0 < value <= 1 for value in factors.values()):
            raise ValueError(f"{case} requires a factor outside (0, 1]; review the calibration assumptions.")
        annual = (experiment_context.run_annual if experiment_context is not None else run_annual_cpu_model)(**common, ai_capacity_factors=factors)
        hourly = (experiment_context.run_hourly if experiment_context is not None else run_workload_component_footprint)(
            **common, **hourly_options, ai_capacity_factors=factors,
            hourly_carbon_factors_dir=hourly_carbon_factors_dir,
        )
        for model, result in (("annual_cpu_model", annual), ("hourly_workload_resolved_model", hourly)):
            frame = result["annual_summary"][["scenario", "year", "country", *SUM_METRICS]].copy()
            frame.insert(0, "model", model)
            frame.insert(0, "calibration_case", case)
            frame["policy"] = policy
            frame["ai_capacity_factor"] = frame["year"].map(factors)
            frame["power_twh"] = frame["facility_energy_mwh"] / 1e6
            frame["carbon_mtco2"] = frame["carbon_tco2"] / 1e6
            frame["facility_mwh_per_installed_it_mw"] = frame["facility_energy_mwh"] / frame["installed_it_mw"]
            country_frames.append(frame)
            aggregate = frame.groupby(["calibration_case", "model", "scenario", "policy", "year"], as_index=False)[list(SUM_METRICS)].sum()
            aggregate["power_twh"] = aggregate["facility_energy_mwh"] / 1e6
            aggregate["carbon_mtco2"] = aggregate["carbon_tco2"] / 1e6
            aggregate["effective_pue"] = aggregate["facility_energy_mwh"] / aggregate["it_energy_mwh"]
            aggregate["facility_mwh_per_installed_it_mw"] = aggregate["facility_energy_mwh"] / aggregate["installed_it_mw"]
            selected_frames.append(annotate_selected_aggregate(aggregate, boundary))
        hourly_rows = hourly["hourly_carbon"]
        if not hourly_rows.empty:
            sources = hourly_rows.groupby(["year", "country", "carbon_factor_source"], as_index=False).size()
            sources.insert(0, "calibration_case", case)
            carbon_records.append(sources)
        measured = hourly["selected_countries_summary"].set_index("year")["power_twh"]
        for year in requested_years:
            reference = BASE_GLOBAL_ELECTRICITY_REFERENCE_TWH[year]
            case_target = reference * coverage if case == "homogeneous_intensity_boundary" else reference
            residual = float(measured.loc[year]) - case_target
            if case != "legacy_factors" and not np.isclose(residual, 0.0, rtol=0, atol=1e-6):
                raise AssertionError(f"Target closure failed for {case}/{year}: {residual} TWh")
            calibration_records.append({
                "calibration_case": case, "year": year,
                "accounting_scope": AGGREGATE_SCOPE,
                "reference_scope": "global", "reference_role": BASE_ELECTRICITY_REFERENCE_ROLE[year],
                "global_reference_twh": reference, "it_capacity_share_sum": coverage,
                "assumed_energy_coverage_fraction": coverage if case == "homogeneous_intensity_boundary" else None,
                "target_twh": case_target,
                "target_basis": ("homogeneous_facility_electricity_per_IT_capacity_assumption" if case == "homogeneous_intensity_boundary"
                                 else "global_reference_applied_to_selected_country_sum"),
                "uncalibrated_selected_electricity_twh": float(raw_energy.loc[year]),
                "common_ai_capacity_factor": factors[year], "inverse_factor": 1.0 / factors[year],
                "calibrated_hourly_selected_electricity_twh": float(measured.loc[year]),
                "target_residual_twh": residual,
                "calibration_reference_model": "hourly_workload_resolved_model_not_measured_truth",
            })
        del annual, hourly

    selected_annual = pd.concat(selected_frames, ignore_index=True)
    country_annual = pd.concat(country_frames, ignore_index=True)
    comparison = selected_annual.pivot(
        index=["calibration_case", "scenario", "policy", "year"], columns="model", values=list(SUM_METRICS),
    )
    comparison.columns = [f"{model}_{metric}" for metric, model in comparison.columns]
    comparison = comparison.reset_index()
    checks = []
    for metric, stem in (("facility_energy_mwh", "energy"), ("carbon_tco2", "carbon")):
        hourly_column, annual_column = f"hourly_workload_resolved_model_{metric}", f"annual_cpu_model_{metric}"
        comparison[f"annual_minus_hourly_{stem}_absolute"] = comparison[annual_column] - comparison[hourly_column]
        comparison[f"annual_minus_hourly_{stem}_pct_of_hourly"] = (
            100 * (comparison[annual_column] - comparison[hourly_column]) / comparison[hourly_column]
        )
        reference = comparison.loc[comparison["calibration_case"] == "legacy_factors"].set_index("year")
        delta_column = f"annual_minus_hourly_{stem}_pct_of_hourly"
        delta = comparison[delta_column] - comparison["year"].map(reference[delta_column])
        comparison[f"{stem}_gap_change_percentage_points_vs_legacy"] = delta
        if not np.allclose(delta, 0, rtol=0, atol=1e-9):
            raise AssertionError("A common linear calibration changed the endpoint relative gap.")
        checks.append({"check": f"common_scaling_preserves_{stem}_relative_gap", "status": "PASS",
                       "max_absolute_residual": float(delta.abs().max()), "unit": "percentage_points"})
    base = country_annual.loc[country_annual["calibration_case"] == "legacy_factors"].set_index(["model", "year", "country"])
    for case in CASES[1:]:
        candidate = country_annual.loc[country_annual["calibration_case"] == case].set_index(["model", "year", "country"])
        base_aligned = base.reindex(candidate.index)
        ratios = np.array([factor_sets[case][year] / factor_sets["legacy_factors"][year]
                           for _, year, _ in candidate.index])
        for metric in SUM_METRICS:
            expected = base_aligned[metric].to_numpy() * ratios
            actual = candidate[metric].to_numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-6)
        checks.append({"check": f"country_capacity_energy_carbon_scale_together/{case}", "status": "PASS",
                       "max_absolute_residual": "", "unit": ""})
    comparison = annotate_selected_aggregate(comparison, boundary)
    metadata = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "accounting_scope": AGGREGATE_SCOPE, "countries": selected, "years": requested_years,
        "policy": policy, "it_capacity_share_sum": coverage, "capacity_shares_renormalized": False,
        "cases": list(CASES), "smoke_test_only": max_intervals is not None,
        "trace_intervals": profile.n_intervals, "trace_hours": profile.trace_hours,
        "profile_origin": "provided_profile" if workload_profile is not None else "local_parquet_inputs",
        "runtime_versions": runtime_versions(),
        "parquet_reader": ({"batch_size": parquet_batch_size, "use_threads": False,
                            "memory_map": False, "pre_buffer": False, "split_blocks": True}
                           if workload_profile is None else None),
        "trace_capacity": profile.trace_capacity.tolist(),
        "trace_capacity_source": list(profile.trace_capacity_source),
        "hardware_configuration": asdict(experiment_context.config.hardware if experiment_context is not None else HardwarePowerConfig()),
        "trace_processing_defaults": asdict(experiment_context.config.trace if experiment_context is not None else Alibaba2026TraceConfig()),
        "capacity_quantile": experiment_context.config.common["capacity_quantile"] if experiment_context is not None else 0.96,
        "max_resource_utilization": experiment_context.config.common["max_resource_utilization"] if experiment_context is not None else 1.0,
        "pue_scale": experiment_context.config.common["pue_scale"] if experiment_context is not None else 1.0,
        "shared_suite_run_id": experiment_context.record.run_id if experiment_context is not None else None,
        "hourly_carbon_scope": hourly_carbon_scope, "strict_hourly_carbon": strict_hourly_carbon,
        "workload_profile_path": str(workload_profile_path), "server_profile_path": str(server_profile_path) if server_profile_path else None,
        "hourly_carbon_factors_dir": str(hourly_carbon_factors_dir),
        "assumption": "Homogeneous case assumes equal facility electricity per unit IT capacity in covered and omitted countries.",
        "interpretation": "Conditional boundary comparison. No new observed energy coverage or validation accuracy is established.",
        "reference_source": "IEA (2026), Key Questions on Energy and AI, Figure 1.5/p.26; intermediate years are author estimates.",
        "reference_url": "https://www.iea.org/reports/key-questions-on-energy-and-ai",
        "source_code_sha256": {
            relative: sha256_file(ROOT_DIR / relative)
            for relative in ("dataset/Installed_capacity_data.py", "dataset/Factors.py",
                             "core/accounting_boundary.py", "core/annual_cpu_model.py",
                             "core/hourly_workload_resolved_model.py",
                             "core/past_research_data_center_energy_carbon_model.py",
                             "scripts/analyze_boundary_calibration.py")
        },
    }
    return {
        "country_annual": country_annual, "selected_countries_annual": selected_annual,
        "endpoint_comparison": comparison, "calibration_audit": pd.DataFrame(calibration_records),
        "accounting_boundary": boundary, "invariant_checks": pd.DataFrame(checks),
        "carbon_input_coverage": pd.concat(carbon_records, ignore_index=True) if carbon_records else pd.DataFrame(),
        "metadata": metadata, "factor_sets": factor_sets,
    }


def write_analysis(result: dict, directory: Path) -> Path:
    output = _workspace_output(directory)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("country_annual", "selected_countries_annual", "endpoint_comparison", "calibration_audit",
                 "accounting_boundary", "invariant_checks", "carbon_input_coverage"):
        result[name].to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    with (output / "analysis_metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(result["metadata"], stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    for case, factors in result["factor_sets"].items():
        record = {**result["metadata"], "calibration_case": case, "ai_capacity_factors": factors}
        with (output / f"ai_capacity_factors_{case}.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return output


def main() -> None:
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/revision_audit/boundary"))
    parser.add_argument("--year-start", type=int, default=2026)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--policy", choices=("CP", "NDC", "NZ"), default="CP")
    parser.add_argument("--workload-profile-path", type=Path, default=ROOT_DIR / "dataset")
    parser.add_argument("--server-profile-path", type=Path, default=None)
    parser.add_argument("--hourly-carbon-factors-dir", type=Path, default=ROOT_DIR / "dataset" / "EM-CPNDCNZ")
    parser.add_argument("--hourly-carbon-scope", choices=("direct", "life_cycle"), default="direct")
    parser.add_argument("--strict-hourly-carbon", action="store_true", help="Require hourly inputs for every country-year.")
    parser.add_argument("--max-intervals", type=int, default=None, help="Smoke tests only; omit for research analysis.")
    parser.add_argument("--parquet-batch-size", type=int, default=DEFAULT_PARQUET_BATCH_SIZE,
                        help="Maximum rows per single-threaded parquet batch (default: 65536).")
    parser.add_argument("--trace-read-log", type=Path, default=None,
                        help="New workspace log path. Default: <output-dir>_trace_read.log. Refuses overwrite.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.parquet_batch_size <= 0:
        parser.error("--parquet-batch-size must be positive.")
    output = _workspace_output(args.output_dir)
    log_path = _workspace_output(args.trace_read_log or output.with_name(output.name + "_trace_read.log"))
    if log_path == output or output in log_path.parents or log_path in output.parents:
        parser.error("The read log and output directory must be separate paths.")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[boundary] Runtime:", json.dumps(runtime_versions()), flush=True)
    print("[boundary] Trace read log:", log_path, flush=True)
    with log_path.open("x", encoding="utf-8", buffering=1) as read_log:
        def record_read(event: dict) -> None:
            record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
            read_log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            read_log.flush()

        # Keep this stream open until the native handler is reset. A fatal dump
        # may follow the JSON records, so the diagnostic file uses .log.
        faulthandler.enable(file=read_log, all_threads=True)
        try:
            record_read({"event": "run_start", "runtime_versions": runtime_versions(),
                         "parquet_batch_size": args.parquet_batch_size})
            result = analyze_boundary_calibration(
                countries=args.countries, year_start=args.year_start, years=args.years, policy=args.policy,
                workload_profile_path=args.workload_profile_path, server_profile_path=args.server_profile_path,
                hourly_carbon_factors_dir=args.hourly_carbon_factors_dir, hourly_carbon_scope=args.hourly_carbon_scope,
                strict_hourly_carbon=args.strict_hourly_carbon, max_intervals=args.max_intervals, verbose=not args.quiet,
                parquet_batch_size=args.parquet_batch_size, read_callback=record_read,
            )
            record_read({"event": "analysis_complete"})
        except BaseException as exc:
            record_read({"event": "run_error", "error_type": type(exc).__name__, "error": str(exc)})
            raise
        finally:
            faulthandler.enable(all_threads=True)
    result["metadata"]["trace_read_log"] = str(log_path)
    print(result["calibration_audit"].to_string(index=False))
    print(result["invariant_checks"].to_string(index=False))
    print("Reports written to:", write_analysis(result, output))
    if args.max_intervals is not None:
        print("SMOKE TEST ONLY: exported factors are blocked from use as research inputs.")


if __name__ == "__main__":
    main()
