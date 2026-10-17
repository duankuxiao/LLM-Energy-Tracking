"""Run the two endpoint models and create comparable annual result tables."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from core.annual_cpu_model import run_annual_cpu_model
from core.annual_activity import ACTIVITY_MODES, AnnualActivityConfig
from core.energy_attribution import IDLE_ALLOCATION_RULES
from core.accounting_boundary import (
    annotate_selected_aggregate, load_ai_capacity_factors, resolve_countries, resolve_ai_capacity_factors,
)
from core.hourly_workload_resolved_model import (
    Alibaba2026TraceConfig,
    HardwarePowerConfig,
    DEFAULT_PARQUET_BATCH_SIZE,
    SCENARIO_COL_MAP,
    build_workload_profile,
    run_workload_component_footprint,
)
from core.grid_carbon import find_hourly_file
from core.power_parameter_audit import load_hardware_config
from core.run_provenance import RunProvenance, json_value
from dataset.Factors import PUE, get_carbon_factor
from dataset.Installed_capacity_data import IT_CAPACITY, IT_RATIO


ROOT_DIR = Path(__file__).resolve().parent
MODEL_ORDER = ("annual_cpu_model", "hourly_workload_resolved_model")
MODEL_PREFIXES = {
    "annual_cpu_model": "annual_cpu",
    "hourly_workload_resolved_model": "hourly_workload_resolved",
}
SUMMARY_METRICS = (
    "installed_it_mw",
    "it_energy_mwh",
    "facility_energy_mwh",
    "power_twh",
    "effective_pue",
    "facility_mwh_per_installed_it_mw",
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
    for name, denominator in (("effective_pue", "it_energy_mwh"),
                              ("facility_mwh_per_installed_it_mw", "installed_it_mw")):
        result[name] = np.divide(
            result["facility_energy_mwh"].to_numpy(dtype=float),
            result[denominator].to_numpy(dtype=float),
            out=np.full(len(result), np.nan),
            where=result[denominator].to_numpy(dtype=float) > 0,
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
    expected_rows = len(comparison)
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
        if len(comparison) != expected_rows:
            raise ValueError(f"Endpoint key coverage differs for {model}; refusing a partial comparison.")

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
    activity_config: Optional[AnnualActivityConfig] = None,
    idle_allocation_rule: str = "hourly_driver",
    hardware_config_file: Optional[Union[str, Path]] = None,
    trace_config_file: Optional[Union[str, Path]] = None,
    activity_config_file: Optional[Union[str, Path]] = None,
    ai_capacity_factors_file: Optional[Union[str, Path]] = None,
    parquet_batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
    trace_read_log: Optional[Union[str, Path]] = None,
    hash_inputs: bool = False,
    invocation: Optional[Mapping[str, object]] = None,
    experiment_context=None,
) -> Dict[str, object]:
    """Run both endpoints and record resolved inputs, versions and fallbacks.

    Existing positional parameters and result keys are retained. New arguments
    are appended. With save_outputs=False all provenance stays in memory and an
    explicit trace_read_log is rejected. JSON files and configuration objects
    are mutually exclusive for each setting; defaults retain the legacy
    calibration label instead of silently becoming 'provided' factors.
    """
    scenarios = list(scenarios)
    countries = list(countries) if countries is not None else None
    requested = dict(
        shared_suite_run_id=experiment_context.record.run_id if experiment_context is not None else None,
        policy=renewable_energy_policy, scenarios=scenarios, countries=countries,
        year_start=year_start, years=years, workload_profile_path=Path(workload_profile_path),
        server_profile_path=Path(server_profile_path) if server_profile_path is not None else None,
        hourly_carbon_factors_dir=Path(hourly_carbon_factors_dir) if hourly_carbon_factors_dir is not None else None,
        hourly_carbon_scope=hourly_carbon_scope, hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
        hardware_config=hardware_config, trace_config=trace_config, activity_config=activity_config,
        ai_capacity_factors=ai_capacity_factors, capacity_quantile=capacity_quantile,
        max_resource_utilization=max_resource_utilization, pue_scale=pue_scale,
        max_intervals=max_intervals, save_hourly_outputs=save_hourly_outputs,
        idle_allocation_rule=idle_allocation_rule, parquet_batch_size=parquet_batch_size,
        hardware_config_file=Path(hardware_config_file) if hardware_config_file is not None else None,
        trace_config_file=Path(trace_config_file) if trace_config_file is not None else None,
        activity_config_file=Path(activity_config_file) if activity_config_file is not None else None,
        ai_capacity_factors_file=Path(ai_capacity_factors_file) if ai_capacity_factors_file is not None else None,
        verbose=verbose,
    )
    record = experiment_context.record if experiment_context is not None else RunProvenance(
        ROOT_DIR, Path(output_dir), save_outputs=save_outputs, hash_inputs=hash_inputs,
        requested=requested, trace_read_log=trace_read_log, invocation=invocation)
    with (nullcontext(record) if experiment_context is not None else record):
        _print_progress(verbose, f"Recording run under {record.directory}." if record.directory else "Recording run metadata in memory.")
        countries = resolve_countries(countries)
        for label, config, file in (("hardware", hardware_config, hardware_config_file),
                                    ("trace", trace_config, trace_config_file),
                                    ("activity", activity_config, activity_config_file),
                                    ("capacity factors", ai_capacity_factors, ai_capacity_factors_file)):
            if file is not None and config is not None:
                raise ValueError(f"Supply a {label} configuration object or file, not both.")
            if file is not None:
                record.add_input(file, "configuration_json", hash_content=True, usage="configuration_selected")
        hardware_config = (load_hardware_config(hardware_config_file, HardwarePowerConfig())
                           if hardware_config_file is not None else hardware_config or HardwarePowerConfig())
        trace_config = (_load_config(trace_config_file, Alibaba2026TraceConfig())
                        if trace_config_file is not None else trace_config or Alibaba2026TraceConfig())
        activity_config = (_load_config(activity_config_file, AnnualActivityConfig())
                           if activity_config_file is not None else activity_config or AnnualActivityConfig())
        hardware_config.validate()
        activity_config.validate()
        if ai_capacity_factors_file is not None:
            ai_capacity_factors = load_ai_capacity_factors(ai_capacity_factors_file, countries)
        selected_years = list(range(year_start, year_start + years))
        factors = resolve_ai_capacity_factors(selected_years, ai_capacity_factors)
        for file in (hardware_config_file, trace_config_file, activity_config_file, ai_capacity_factors_file):
            if file is not None:
                record.add_input(file, "configuration_json", hash_content=True, usage="configuration_loaded")
        effective = {**requested, "countries": countries, "hardware_config": asdict(hardware_config),
                     "trace_config": asdict(trace_config), "activity_config": asdict(activity_config),
                     "ai_capacity_factors": factors,
                     "ai_capacity_factor_basis": "repository_legacy_defaults" if ai_capacity_factors is None else "provided_common_factors",
                     "trace_truncated_by_max_intervals": max_intervals is not None,
                     "save_outputs": save_outputs, "hash_inputs": hash_inputs}
        if experiment_context is None:
            record.configure(effective)
        else:
            record.record["comparison_configuration"] = json_value(effective)
        record.table("Resolved_Model_Inputs", pd.DataFrame([
            {"scenario": scenario, "source_scenario_column": SCENARIO_COL_MAP[scenario],
             "policy": renewable_energy_policy, "country": country, "year": year,
             "source_pue": float(PUE[country][year - 2025, SCENARIO_COL_MAP[scenario]]),
             "pue_scale": pue_scale, "applied_pue": float(PUE[country][year - 2025, SCENARIO_COL_MAP[scenario]]) * pue_scale,
             "ai_capacity_factor": factors[year], "source_it_capacity_gw": float(IT_CAPACITY[year - 2025, SCENARIO_COL_MAP[scenario]]),
             "original_it_allocation_share": float(IT_RATIO[country]),
             "applied_ai_it_capacity_mw": float(IT_CAPACITY[year - 2025, SCENARIO_COL_MAP[scenario]]) * IT_RATIO[country] * factors[year] * 1000,
             "annual_direct_ci_kg_per_mwh": get_carbon_factor(renewable_energy_policy, country, year),
             "input_table_scope": "requested_output_years",
             "source_status": "local_numeric_inputs_not_independent_source_or_license_verification"}
            for scenario in scenarios for year in selected_years for country in countries]))
        record.table("AI_Capacity_Factors", pd.DataFrame([
            {"year": year, "ai_capacity_factor": value, "basis": effective["ai_capacity_factor_basis"]}
            for year, value in factors.items()]))
        record.stage("grid_input_selection")
        grid_selection = []
        for country in countries:
            for year in selected_years:
                path = find_hourly_file(hourly_carbon_factors_dir, country, renewable_energy_policy, year) if hourly_carbon_factors_dir is not None else None
                source = "annual_configured" if hourly_carbon_factors_dir is None else "hourly_file_selected" if path else "hourly_file_missing"
                grid_selection.append({"country": country, "year": year, "policy": renewable_energy_policy,
                                       "selection": source, "path": str(path.resolve()) if path else None})
                if path is not None:
                    if experiment_context is not None:
                        experiment_context.grid.series(country=country, renewable_energy_policy=renewable_energy_policy,
                                                       year=year, hourly_carbon_factors_dir=hourly_carbon_factors_dir,
                                                       hourly_carbon_scope=hourly_carbon_scope)
                    else:
                        record.add_input(path, "hourly_grid_csv", hash_content=True, usage="selected_before_execution")
        record.table("Grid_Input_Selection", pd.DataFrame(grid_selection))
        if not hourly_carbon_fallback_to_annual and any(row["selection"] == "hourly_file_missing" for row in grid_selection):
            raise FileNotFoundError("Strict hourly comparison is missing grid files; see Grid_Input_Selection.csv in this run's provenance directory.")
        result = _execute_model_comparison(
            renewable_energy_policy=renewable_energy_policy, scenarios=scenarios, years=years, countries=countries,
            year_start=year_start, workload_profile_path=workload_profile_path, server_profile_path=server_profile_path,
            hourly_carbon_factors_dir=hourly_carbon_factors_dir, hourly_carbon_scope=hourly_carbon_scope,
            hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
            hardware_config=hardware_config, trace_config=trace_config, capacity_quantile=capacity_quantile,
            max_resource_utilization=max_resource_utilization, pue_scale=pue_scale,
            ai_capacity_factors=ai_capacity_factors, max_intervals=max_intervals, output_dir=output_dir,
            save_outputs=save_outputs, save_hourly_outputs=save_hourly_outputs, verbose=verbose,
            activity_config=activity_config, idle_allocation_rule=idle_allocation_rule,
            parquet_batch_size=parquet_batch_size, provenance=record, experiment_context=experiment_context)
        record.stage("record_diagnostics")
        _record_diagnostics(record, result)
    result["run_metadata"] = record.record
    result["provenance_tables"] = record.tables
    result["provenance_directory"] = record.directory
    _print_progress(verbose, f"Run record: {record.directory}" if record.directory else "Run provenance returned in memory; no files written.")
    return result


def _load_config(path, baseline):
    overrides = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(overrides, dict):
        raise ValueError("Configuration JSON must contain an object.")
    if isinstance(baseline, Alibaba2026TraceConfig) and "active_states" in overrides:
        if not isinstance(overrides["active_states"], list):
            raise ValueError("active_states must be a JSON list of state names.")
        overrides["active_states"] = tuple(overrides["active_states"])
    return replace(baseline, **overrides)


def _record_diagnostics(record, result):
    annual, hourly = result["annual_cpu"], result["hourly_workload_resolved"]
    for name, table in (("Accounting_Boundary", result["accounting_boundary"]),
                        ("Annual_Activity_Factors", annual["activity_factors"]),
                        ("Annual_Reference_Parameters", annual["annual_parameters"]),
                        ("Hardware_Parameters", hourly["hardware_parameters"]),
                        ("Trace_Resource_Capacity", hourly["trace_resource_capacity"]),
                        ("Workload_Trace_Metadata", hourly["workload_trace_metadata"]),
                        ("Grid_Input_Audit", hourly["grid_carbon_audit"]),
                        ("Allocation_Audit", hourly["allocation_audit"]),
                        ("Temporal_Metrics", hourly["temporal_metrics"])):
        record.table(name, table)
    grid = hourly["grid_carbon_audit"]
    for row in grid.loc[grid["carbon_factor_source"] == "hourly"].to_dict("records"):
        entry = record.add_input(row["input_file"], "hourly_grid_csv", hash_content=True, usage="used_by_hourly_model")
        entry["model_reported_sha256"] = row["input_sha256"]
        entry["model_hash_mismatch"] = bool(entry.get("model_hash_mismatch", False) or entry["sha256"] != row["input_sha256"])
    keys = ["policy", "country", "year", "scope", "carbon_factor_source", "fallback_reason", "calendar_status"]
    grid_inputs = grid[keys].drop_duplicates()
    rows = []
    for row in grid_inputs.to_dict("records"):
        if row["carbon_factor_source"] == "annual_fallback":
            rows.append({"kind": "grid_annual_fallback", "resource": None, **row})
        elif row["carbon_factor_source"] == "annual_configured":
            rows.append({"kind": "grid_annual_configured", "resource": None, **row})
        if row["calendar_status"] != "full_calendar":
            rows.append({"kind": "grid_calendar_limitation", "resource": None, **row})
    for row in hourly["trace_resource_capacity"].to_dict("records"):
        if row["capacity_source"] != "server_inventory_quantile":
            rows.append({"kind": "trace_capacity_fallback" if row["resource"] in ("cpu", "gpu") else "trace_resource_proxy",
                         "resource": row["resource"], "detail": row["capacity_source"]})
    allocation = hourly["allocation_audit"]
    selected = allocation.loc[allocation["idle_fallback_applied_in_selected_rule"]]
    for row in selected.to_dict("records"):
        rows.append({"kind": "idle_attribution_fallback", "scenario": row.get("scenario"), "country": row.get("country"),
                     "year": row.get("year"), "component": row.get("component"), "detail": row.get("fallback_basis"),
                     "zero_driver_intervals": row.get("zero_driver_intervals")})
    fallback = pd.DataFrame(rows, columns=["kind", "policy", "scenario", "country", "year", "scope", "resource", "component",
                                           "carbon_factor_source", "fallback_reason", "calendar_status", "detail", "zero_driver_intervals"])
    record.table("Fallback_Summary", fallback)
    record.record["fallback_counts"] = fallback["kind"].value_counts().to_dict()
    record.record["grid_coverage_counting_unit"] = "distinct_policy_country_year_scope_source_calendar_not_scenario_rows"
    record.record["grid_coverage"] = {"unique_inputs": len(grid_inputs),
                                    "hourly_inputs": int(grid_inputs["carbon_factor_source"].eq("hourly").sum()),
                                    "full_calendar_inputs": int(grid_inputs["calendar_status"].eq("full_calendar").sum())}
    record.record["trace_metadata"] = hourly["workload_trace_metadata"].to_dict("records")


def _execute_model_comparison(
    renewable_energy_policy, scenarios, years, countries, year_start,
    workload_profile_path, server_profile_path, hourly_carbon_factors_dir,
    hourly_carbon_scope, hourly_carbon_fallback_to_annual, hardware_config, trace_config,
    capacity_quantile, max_resource_utilization, pue_scale, ai_capacity_factors,
    max_intervals, output_dir, save_outputs, save_hourly_outputs, verbose,
    activity_config, idle_allocation_rule, parquet_batch_size, provenance, experiment_context=None,
) -> Dict[str, object]:
    """Compare the endpoints within the selected major-AI-country boundary."""
    if hourly_carbon_scope != "direct":
        raise ValueError("Endpoint comparisons require direct factors in both models; no comparable annual life-cycle table is supplied.")
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
    provenance.stage("annual_cpu_model")
    annual_cpu = run_annual_cpu_model(
        **common,
        activity_config=activity_config,
        output_dir=output_path / "annual_cpu_model",
    )

    _print_progress(verbose, "Reusing the prepared workload profile." if experiment_context is not None else "Reading the GPU trace for the hourly workload-resolved model.")
    provenance.stage("workload_trace")
    shared_workload_profile = experiment_context.profile if experiment_context is not None else build_workload_profile(
        workload_profile_path=workload_profile_path,
        server_profile_path=server_profile_path,
        capacity_quantile=capacity_quantile,
        trace_config=trace_config,
        max_intervals=max_intervals,
        verbose=verbose,
        parquet_batch_size=parquet_batch_size,
        read_callback=provenance.trace_event,
    )

    _print_progress(verbose, "Running the hourly workload-resolved model.")
    provenance.stage("hourly_workload_resolved_model")
    hourly_workload_resolved = (experiment_context.run_hourly if experiment_context is not None else run_workload_component_footprint)(
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
        idle_allocation_rule=idle_allocation_rule,
    )

    provenance.stage("comparison_tables")
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

    boundary = hourly_workload_resolved["accounting_boundary"]
    selected_countries_summary = pd.concat(
        [
            _standardize_annual_summary(
                annual_cpu["selected_countries_summary"],
                "annual_cpu_model",
                renewable_energy_policy,
                False,
            ),
            _standardize_annual_summary(
                hourly_workload_resolved["selected_countries_summary"],
                "hourly_workload_resolved_model",
                renewable_energy_policy,
                False,
            ),
        ],
        ignore_index=True,
    ).sort_values(["scenario", "policy", "year", "model"]).reset_index(drop=True)
    selected_countries_summary = annotate_selected_aggregate(selected_countries_summary, boundary)

    country_comparison = _build_wide_comparison(
        country_summary,
        ["scenario", "policy", "year", "country"],
    )
    selected_countries_comparison = _build_wide_comparison(
        selected_countries_summary,
        ["scenario", "policy", "year"],
    )
    selected_countries_comparison = annotate_selected_aggregate(selected_countries_comparison, boundary)

    if save_outputs:
        summary_dir = output_path / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        country_summary.to_csv(summary_dir / "Endpoint_Models_Country_Annual.csv", index=False)
        selected_countries_summary.to_csv(summary_dir / "Endpoint_Models_Selected_Countries_Annual.csv", index=False)
        country_comparison.to_csv(
            summary_dir / "Endpoint_Comparison_Country_Annual.csv", index=False
        )
        selected_countries_comparison.to_csv(
            summary_dir / "Endpoint_Comparison_Selected_Countries_Annual.csv", index=False
        )
        boundary.to_csv(summary_dir / "Accounting_Boundary.csv", index=False)
        annual_cpu["activity_factors"].to_csv(summary_dir / "Annual_Activity_Factors.csv", index=False)
        annual_cpu["annual_parameters"].to_csv(summary_dir / "Annual_Reference_Parameters.csv", index=False)
        pd.DataFrame([{
            "annual_activity_mode": (activity_config or AnnualActivityConfig()).mode,
            "hourly_activity": "unchanged_trace",
            "relative_gap_denominator": "hourly_workload_resolved_endpoint",
            "estimand": "conditional_representation_difference_not_observed_prediction_error",
            "max_intervals": max_intervals,
            "idle_allocation_rule": idle_allocation_rule,
            "run_id": provenance.run_id,
            "parquet_batch_size": parquet_batch_size,
            "provenance_directory": str(provenance.directory),
        }]).to_csv(summary_dir / "Comparison_Configuration.csv", index=False)

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
        print("Selected-country aggregate (not a global estimate):")
        print(selected_countries_summary[display_columns].to_string(index=False, float_format="%.6f"))

    return {
        "annual_cpu": annual_cpu,
        "hourly_workload_resolved": hourly_workload_resolved,
        "shared_workload_profile": shared_workload_profile,
        "country_summary": country_summary,
        "selected_countries_summary": selected_countries_summary,
        "global_summary": selected_countries_summary,  # Compatibility alias for existing callers.
        "accounting_boundary": boundary,
        "country_comparison": country_comparison,
        "selected_countries_comparison": selected_countries_comparison,
        "global_comparison": selected_countries_comparison,  # Compatibility alias, same selected-country scope.
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the annual CPU and hourly workload-resolved endpoint models "
            "and generate annual comparison CSV files."
        )
    )
    parser.add_argument("--policy", choices=("CP", "NDC", "NZ"), default="CP")
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIO_COL_MAP), default=["Base"])
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
    parser.add_argument("--ai-capacity-factors-file", type=Path, default=None,
                        help="JSON with common factors, optionally exported by the boundary analysis.")
    activity_group = parser.add_mutually_exclusive_group()
    activity_group.add_argument("--annual-activity-mode", choices=ACTIVITY_MODES, default=None,
                                help="Default calendar: 2025/2030 anchors; fixed: hold 2025; legacy_six_year: original six-year denominator.")
    activity_group.add_argument("--annual-activity-config-file", type=Path, default=None,
                                help="JSON overrides for AnnualActivityConfig, including its mode and activity endpoints.")
    parser.add_argument("--hardware-config-file", type=Path, default=None,
                        help="JSON HardwarePowerConfig overrides, using the same schema as Figure 4 and validation.")
    parser.add_argument("--trace-config-file", type=Path, default=None,
                        help="JSON Alibaba2026TraceConfig overrides; active_states is a list of strings.")
    parser.add_argument("--parquet-batch-size", type=int, default=DEFAULT_PARQUET_BATCH_SIZE,
                        help="Rows per projected Parquet batch; recorded without changing the scientific trace window.")
    parser.add_argument("--trace-read-log", type=Path, default=None,
                        help="Optional new JSONL path; default is the unique run provenance directory. Existing logs are never overwritten.")
    parser.add_argument("--hash-inputs", action="store_true",
                        help="Also hash each Parquet file actually opened; source, JSON and selected grid CSV hashes are always recorded.")
    parser.add_argument("--idle-allocation-rule", choices=IDLE_ALLOCATION_RULES, default="hourly_driver")
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
        idle_allocation_rule=args.idle_allocation_rule,
        activity_config=AnnualActivityConfig(mode=args.annual_activity_mode) if args.annual_activity_mode else None,
        activity_config_file=args.annual_activity_config_file,
        hardware_config_file=args.hardware_config_file,
        trace_config_file=args.trace_config_file,
        parquet_batch_size=args.parquet_batch_size,
        trace_read_log=args.trace_read_log,
        hash_inputs=args.hash_inputs,
        invocation={"kind": "cli", "executable": sys.executable, "argv": list(sys.argv)},
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
        ai_capacity_factors_file=args.ai_capacity_factors_file,
        max_intervals=args.max_intervals,
        output_dir=args.output_dir,
        save_hourly_outputs=args.save_hourly_outputs,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
