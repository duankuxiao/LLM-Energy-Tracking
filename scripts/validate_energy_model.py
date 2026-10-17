"""External benchmark and boundary diagnostics for the data-centre energy model.

The script deliberately separates three questions:

1. MLPerf Power compares a configured full-load reference with reported
   whole-system inference power. This is an external configuration diagnostic,
   not validation of utilization curves, GPU-only power or future energy.
2. Official national electricity series audit the reporting scope only; their
   all-data-centre historical totals are not treated as prediction targets for
   the model's future AI-only estimates.
3. The first and last 24 h of the Alibaba trace diagnose the discontinuity at
   the repeat boundary using a resource-load proxy, not measured power.

Outputs are deterministic except for non-parametric bootstrap confidence
intervals, which use a fixed random seed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.hourly_workload_resolved_model import HardwarePowerConfig, _alibaba_2026_partition_files as _partition_files
from core.temporal_robustness import power_metrics
from core.power_parameter_audit import hardware_parameter_table, load_hardware_config, parameter_dependency_table
from core.research_io import sha256_file


DEFAULT_MLPERF = ROOT_DIR / "dataset" / "validation" / "raw" / "mlperf_power_raw_data.csv"
DEFAULT_NATIONAL = ROOT_DIR / "dataset" / "validation" / "national_data_center_energy.csv"
DEFAULT_TRACE_ROOT = ROOT_DIR / "dataset" / "asi_opensource_pod_hourly"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "model_validation"
DEFAULT_FIGURE_STEM = ROOT_DIR / "figures" / "supplementary_energy_validation"
BOOTSTRAP_SEED = 20260819
BOOTSTRAP_REPLICATES = 10_000
TRACE_COLUMNS = (
    "state_public",
    "gpu_request",
    "used_gpu_hours",
    "avg_gpu_sm_util",
    "cpu_request_cores",
    "avg_cpu_request_util",
    "avg_memory_util",
)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _accelerator_family(value: object) -> str:
    text = str(value).upper()
    families = [family for family in ("GH200", "GB200", "B200", "H200", "H100", "A100", "L40S", "L40", "L4", "A40", "A30", "A10", "A2", "V100", "P100", "T4")
                if re.search(rf"(?<![A-Z0-9]){family}(?![A-Z0-9])", text)]
    return families[0] if len(families) == 1 else "Mixed NVIDIA" if families else "Other NVIDIA"


def _metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    return {
        "normalized_mean_bias_error_pct": 100.0 * float(residual.sum() / observed.sum()),
        "median_absolute_percentage_error_pct": 100.0
        * float(np.median(np.abs(residual) / observed)),
        "root_mean_squared_error_w": float(np.sqrt(np.mean(residual**2))),
        "median_prediction_observation_ratio": float(np.median(predicted / observed)),
    }


def _bootstrap_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    accelerator_count: np.ndarray,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n = len(observed)
    observed, predicted, accelerator_count = (np.asarray(values, dtype=float) for values in (observed, predicted, accelerator_count))
    if n == 0 or observed.shape != (n,) or predicted.shape != (n,) or accelerator_count.shape != (n,):
        raise ValueError("Bootstrap requires aligned nonempty system-level vectors.")
    if replicates <= 0 or not all(np.isfinite(values).all() for values in (observed, predicted, accelerator_count)):
        raise ValueError("Bootstrap inputs must be finite and replicates positive.")
    if np.any(observed <= 0) or np.any(accelerator_count <= 0) or np.any(predicted < 0):
        raise ValueError("Observed power and accelerator counts must be positive.")
    metric_names = (
        "normalized_mean_bias_error_pct",
        "median_absolute_percentage_error_pct",
        "root_mean_squared_error_w",
        "median_prediction_observation_ratio",
        "median_observed_power_per_accelerator_w",
    )

    point = _metric_values(observed, predicted)
    point["median_observed_power_per_accelerator_w"] = float(
        np.median(observed / accelerator_count)
    )
    samples = {name: np.full(replicates, np.nan, dtype=float) for name in metric_names}
    if n >= 2:
        # Batch vectorization bounds memory while retaining Public ID resampling.
        for start in range(0, replicates, 256):
            stop = min(start + 256, replicates)
            idx = rng.randint(0, n, size=(stop - start, n))
            obs, pred = observed[idx], predicted[idx]
            residual = pred - obs
            samples["normalized_mean_bias_error_pct"][start:stop] = 100. * residual.sum(axis=1) / obs.sum(axis=1)
            samples["median_absolute_percentage_error_pct"][start:stop] = 100. * np.median(np.abs(residual) / obs, axis=1)
            samples["root_mean_squared_error_w"][start:stop] = np.sqrt(np.mean(residual**2, axis=1))
            samples["median_prediction_observation_ratio"][start:stop] = np.median(pred / obs, axis=1)
            samples["median_observed_power_per_accelerator_w"][start:stop] = np.median(obs / accelerator_count[idx], axis=1)

    records = []
    for name in metric_names:
        lower, upper = np.quantile(samples[name], [0.025, 0.975]) if n >= 2 else (np.nan, np.nan)
        records.append(
            {
                "metric": name,
                "estimate": point[name],
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "independent_unit": "MLPerf Public ID system submission",
                "n": n,
                "bootstrap_replicates": replicates if n >= 2 else 0,
                "bootstrap_seed": seed,
                "uncertainty_status": "percentile_bootstrap" if n >= 2 else "not_estimable_single_Public_ID",
                "validation_role": "external_configuration_diagnostic",
                "generalizes_to_future_fleet": False,
            }
        )
    return pd.DataFrame.from_records(records)


def _stratified_metrics(workload, columns, replicates, seed, kind):
    records = []
    group_key = columns[0] if len(columns) == 1 else columns
    for ordinal, (values, group) in enumerate(workload.groupby(group_key, sort=True, dropna=False)):
        values = (values,) if len(columns) == 1 else values
        # One vote per Public ID within each stratum, irrespective of workloads.
        systems = group.groupby("public_id", as_index=False).agg(
            observed=("observed_system_power_w", "max"),
            predicted=("predicted_full_load_reference_w", "first"),
            accelerators=("accelerator_count", "first"))
        table = _bootstrap_metrics(systems["observed"].to_numpy(), systems["predicted"].to_numpy(),
                                   systems["accelerators"].to_numpy(), replicates, seed + ordinal)
        for name, value in zip(columns, values):
            table[name] = value
        table["stratum_kind"] = kind
        table["n_workload_rows"] = len(group)
        table["within_ID_aggregation"] = "maximum_reported_workload_power_in_this_stratum"
        table["strata_may_share_Public_IDs"] = True
        records.append(table)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _validation_claim_table():
    return pd.DataFrame([
        {"claim": "whole_system_inference_configuration_comparison", "evidence_status": "available_conditionally",
         "evidence": "reported system power; maximum across retained workload rows per Public ID",
         "limitation": "configured reference versus reported benchmark power; utilization/full-load equivalence is not observed; unit watts can cancel from normalized energy at fixed idle/full"},
        {"claim": "GPU_component_power_or_idle_dynamic_curve", "evidence_status": "not_identifiable",
         "evidence": "no independent component-resolved power versus utilization series in this input",
         "limitation": "whole-system full-power summaries cannot identify GPU idle/full ratio or curve exponent"},
        {"claim": "training_power", "evidence_status": "not_validated",
         "evidence": "retained rows are inference only", "limitation": "no training-power validation claim"},
        {"claim": "facility_energy_or_PUE", "evidence_status": "not_validated",
         "evidence": "reported system boundary; no independent cooling/facility energy series",
         "limitation": "system power cannot validate facility electricity or PUE"},
        {"claim": "future_national_fleet_accuracy_or_conservativeness", "evidence_status": "not_validated",
         "evidence": "hardware/workload mix is submission-selected, not a future fleet sample",
         "limitation": "bootstrap quantifies retained-sample uncertainty, not external validity"},
        {"claim": "Public_ID_independence", "evidence_status": "analysis_unit_assumption",
         "evidence": "repeated workload rows aggregate inside each Public ID",
         "limitation": "different IDs may reuse physical hardware; physical-system independence is not established"},
        {"claim": "power_curve_fit_and_holdout", "evidence_status": "not_performed",
         "evidence": "no curve fitted to these system-power rows",
         "limitation": "future fitting requires independent-system train/validation separation and utilization measurements"},
    ])


def _a100_h100_comparison(systems, replicates, seed):
    a = systems.loc[systems["accelerator_family"].eq("A100"), "observed_max_power_per_accelerator_w"].to_numpy(dtype=float)
    h = systems.loc[systems["accelerator_family"].eq("H100"), "observed_max_power_per_accelerator_w"].to_numpy(dtype=float)
    median_a = float(np.median(a)) if len(a) else np.nan
    median_h = float(np.median(h)) if len(h) else np.nan
    lower = upper = np.nan
    if len(a) >= 2 and len(h) >= 2:
        rng = np.random.RandomState(seed)
        samples = np.empty(replicates)
        for start in range(0, replicates, 256):
            stop = min(start + 256, replicates)
            sa = a[rng.randint(0, len(a), size=(stop - start, len(a)))]
            sh = h[rng.randint(0, len(h), size=(stop - start, len(h)))]
            samples[start:stop] = np.median(sh, axis=1) - np.median(sa, axis=1)
        lower, upper = np.quantile(samples, [.025, .975])
    return pd.DataFrame([{"n_A100_Public_IDs": len(a), "n_H100_Public_IDs": len(h),
                          "A100_median_reported_system_W_per_accelerator": median_a,
                          "H100_median_reported_system_W_per_accelerator": median_h,
                          "H100_minus_A100_median_W_per_accelerator": median_h - median_a,
                          "ci_95_lower": lower, "ci_95_upper": upper,
                          "bootstrap_seed": seed, "bootstrap_replicates": replicates if len(a) >= 2 and len(h) >= 2 else 0,
                          "interpretation": "descriptive_hardware_mix_comparison_not_future_fleet_accuracy",
                          "status": "both_groups_have_multiple_IDs" if len(a) >= 2 and len(h) >= 2 else "insufficient_independent_units_for_interval"}])


def validate_mlperf(
    input_path: Path,
    output_dir: Path,
    replicates: int,
    seed: int,
    hardware_config: HardwarePowerConfig | None = None,
    input_frame: pd.DataFrame | None = None,
    input_sha256: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(input_path, low_memory=False) if input_frame is None else input_frame.copy()
    required = {"Benchmark", "SystemType", "Division", "Scenario", "Units", "has_power",
                "accelerator_model_name", "Result", "Total Accelerators", "Public ID",
                "Organization", "SystemName", "Model", "version", "date"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"MLPerf input is missing columns: {sorted(missing)}")
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit: list[dict[str, object]] = []

    def retain(stage: str, mask: pd.Series) -> None:
        nonlocal raw
        before = len(raw)
        raw = raw.loc[mask].copy()
        audit.append({"stage": stage, "rows_before": before, "rows_after": len(raw)})

    audit.append({"stage": "raw_rows", "rows_before": len(raw), "rows_after": len(raw)})
    before_dedup = len(raw)
    raw = raw.drop_duplicates().copy()
    audit.append({"stage": "remove_exact_duplicate_input_rows", "rows_before": before_dedup, "rows_after": len(raw)})
    retain("benchmark_inference", raw["Benchmark"].astype(str).str.casefold().eq("inference"))
    retain("system_type_datacenter", raw["SystemType"].astype(str).str.casefold().eq("datacenter"))
    retain("division_closed", raw["Division"].astype(str).str.casefold().eq("closed"))
    retain("scenario_server_or_offline", raw["Scenario"].astype(str).isin({"Server", "Offline"}))
    normalized_units = raw["Units"].astype(str).str.strip().str.casefold()
    retain(
        "whole_system_power_units",
        normalized_units.isin({"system power (w)", "system power", "watts", "power (w)"}),
    )
    retain("has_power_true", _bool_series(raw["has_power"]))
    retain(
        "nvidia_accelerator",
        raw["accelerator_model_name"].astype(str).str.contains("NVIDIA", case=False, na=False),
    )

    raw["observed_system_power_w"] = _numeric(raw["Result"])
    raw["accelerator_count"] = _numeric(raw["Total Accelerators"])
    valid_numeric = (
        np.isfinite(raw["observed_system_power_w"])
        & (raw["observed_system_power_w"] > 0)
        & np.isfinite(raw["accelerator_count"])
        & (raw["accelerator_count"] > 0)
        & np.isclose(raw["accelerator_count"], np.round(raw["accelerator_count"]), rtol=0., atol=1e-9)
        & raw["Public ID"].notna()
        & raw["Public ID"].astype(str).str.strip().ne("")
    )
    retain("positive_numeric_power_accelerators_and_public_id", valid_numeric)

    power_config = hardware_config or HardwarePowerConfig()
    power_config.validate()
    if power_config.gpu_power_share <= 0:
        raise ValueError("MLPerf configured reference requires a positive GPU power-budget share.")
    reference_w_per_accelerator = power_config.gpu_full_power_w / power_config.gpu_power_share
    raw["accelerator_family"] = raw["accelerator_model_name"].map(_accelerator_family)
    raw["predicted_full_load_reference_w"] = raw["accelerator_count"] * reference_w_per_accelerator
    raw["public_id"] = raw["Public ID"].astype(str).str.strip()
    raw["node_count"] = _numeric(raw["number_of_nodes"]) if "number_of_nodes" in raw else np.nan
    raw["node_boundary"] = np.where(raw["node_count"].eq(1), "single_node", np.where(raw["node_count"].gt(1), "multi_node", "node_count_unknown"))
    raw["measurement_boundary"] = "reported_whole_system_SUT_power_components_unresolved"
    raw["scenario"] = raw["Scenario"].astype(str).str.strip()
    raw["task_model"] = raw["Model"].fillna("unknown_model").astype(str).str.strip().replace("", "unknown_model")
    accelerator_labels = raw["accelerator_model_name"].fillna("").astype(str)
    raw["accelerator_form_factor"] = np.where(accelerator_labels.str.contains("SXM", case=False, na=False), "SXM_label",
                                             np.where(accelerator_labels.str.contains("PCI", case=False, na=False), "PCI_label", "unspecified"))
    # Never silently take median accelerator counts from inconsistent IDs.
    id_audit = []
    invariant_columns = ["Organization", "SystemName", "accelerator_model_name", "accelerator_family", "accelerator_count", "node_count", "version", "measurement_boundary"]
    for public_id, group in raw.groupby("public_id", sort=True):
        counts = {column: int(group[column].fillna("unknown").astype(str).str.strip().str.casefold().nunique()) for column in invariant_columns}
        conflicts = [column for column, count in counts.items() if count > 1]
        id_audit.append({"public_id": public_id, "n_workload_rows": len(group),
                         "metadata_consistent": not conflicts, "conflicting_fields": ";".join(conflicts),
                         **{f"unique_{column}": count for column, count in counts.items()},
                         "independence_assumption": "Public_ID_not_verified_unique_physical_hardware"})
    id_audit = pd.DataFrame(id_audit, columns=["public_id", "n_workload_rows", "metadata_consistent", "conflicting_fields"]
                            + [f"unique_{column}" for column in invariant_columns] + ["independence_assumption"])
    id_audit.to_csv(output_dir / "mlperf_public_id_audit.csv", index=False)
    valid_ids = id_audit.loc[id_audit["metadata_consistent"].eq(True), "public_id"]
    retain("consistent_metadata_within_Public_ID", raw["public_id"].isin(valid_ids))
    _validation_claim_table().to_csv(output_dir / "validation_claim_scope.csv", index=False)
    hardware_parameter_table(power_config).to_csv(output_dir / "validation_hardware_parameters.csv", index=False)
    parameter_dependency_table(power_config).to_csv(output_dir / "validation_parameter_dependencies.csv", index=False)
    if raw.empty:
        pd.DataFrame(audit).to_csv(output_dir / "validation_filter_audit.csv", index=False)
        raise ValueError("No eligible consistent MLPerf Public IDs remain; inspect filter and Public ID audits.")

    workload_columns = [
        "public_id",
        "Organization",
        "SystemName",
        "Scenario",
        "Model",
        "accelerator_model_name",
        "accelerator_family",
        "accelerator_count",
        "observed_system_power_w",
        "predicted_full_load_reference_w",
        "version",
        "date",
        "node_count", "node_boundary", "measurement_boundary", "scenario", "task_model", "accelerator_form_factor", "Units",
    ]
    workload = raw[workload_columns].copy()
    workload["reference_role"] = "configured_full_load_reference_not_observed_utilization"

    def first_nonempty(values: Iterable[object]) -> str:
        for value in values:
            if pd.notna(value) and str(value).strip():
                return str(value)
        return ""

    systems = (
        workload.groupby("public_id", as_index=False)
        .agg(
            organization=("Organization", first_nonempty),
            system_name=("SystemName", first_nonempty),
            accelerator_model=("accelerator_model_name", first_nonempty),
            accelerator_family=("accelerator_family", first_nonempty),
            accelerator_count=("accelerator_count", "first"),
            accelerator_form_factor=("accelerator_form_factor", first_nonempty),
            node_count=("node_count", "first"), node_boundary=("node_boundary", first_nonempty),
            measurement_boundary=("measurement_boundary", first_nonempty),
            benchmark_version=("version", first_nonempty),
            n_task_models=("task_model", "nunique"), n_scenarios=("scenario", "nunique"),
            observed_power_min_w=("observed_system_power_w", "min"),
            observed_power_median_w=("observed_system_power_w", "median"),
            observed_power_max_w=("observed_system_power_w", "max"),
            n_workload_measurements=("observed_system_power_w", "size"),
        )
        .sort_values(["accelerator_family", "accelerator_count", "public_id"], ignore_index=True)
    )
    systems["predicted_full_load_reference_w"] = (
        systems["accelerator_count"] * reference_w_per_accelerator
    )
    systems["observed_max_power_per_accelerator_w"] = (
        systems["observed_power_max_w"] / systems["accelerator_count"]
    )
    systems["prediction_observation_ratio"] = (
        systems["predicted_full_load_reference_w"] / systems["observed_power_max_w"]
    )
    systems["absolute_percentage_error_pct"] = (
        100.0
        * np.abs(systems["predicted_full_load_reference_w"] - systems["observed_power_max_w"])
        / systems["observed_power_max_w"]
    )
    systems["within_ID_aggregation"] = "maximum_across_retained_reported_workload_power_rows_not_observed_time_peak"
    systems["validation_role"] = "external_configuration_diagnostic"
    systems["independent_unit_assumption"] = "Public_ID_system_submission_physical_reuse_across_IDs_unknown"

    observed = systems["observed_power_max_w"].to_numpy(dtype=float)
    predicted = systems["predicted_full_load_reference_w"].to_numpy(dtype=float)
    accelerators = systems["accelerator_count"].to_numpy(dtype=float)
    summary = _bootstrap_metrics(observed, predicted, accelerators, replicates, seed)
    reference_percentile = 100.0 * float(
        np.mean(systems["observed_max_power_per_accelerator_w"] <= reference_w_per_accelerator)
    )
    summary = pd.concat(
        [
            summary,
            pd.DataFrame.from_records(
                [
                    {
                        "metric": "model_implied_full_load_power_per_accelerator_w",
                        "estimate": reference_w_per_accelerator,
                        "ci_95_lower": np.nan,
                        "ci_95_upper": np.nan,
                        "independent_unit": "model configuration",
                        "n": 1,
                        "bootstrap_replicates": 0,
                        "bootstrap_seed": seed,
                        "validation_role": "external_configuration_diagnostic",
                    },
                    {
                        "metric": "empirical_percentile_of_model_reference_pct",
                        "estimate": reference_percentile,
                        "ci_95_lower": np.nan,
                        "ci_95_upper": np.nan,
                        "independent_unit": "MLPerf Public ID system submission",
                        "n": len(systems),
                        "bootstrap_replicates": 0,
                        "bootstrap_seed": seed,
                        "validation_role": "descriptive_retained_sample_percentile_not_conservativeness",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    family = (
        systems.groupby("accelerator_family", as_index=False)
        .agg(
            n_systems=("public_id", "size"),
            median_observed_max_power_per_accelerator_w=(
                "observed_max_power_per_accelerator_w",
                "median",
            ),
            q1_observed_max_power_per_accelerator_w=(
                "observed_max_power_per_accelerator_w",
                lambda x: float(np.quantile(x, 0.25)),
            ),
            q3_observed_max_power_per_accelerator_w=(
                "observed_max_power_per_accelerator_w",
                lambda x: float(np.quantile(x, 0.75)),
            ),
        )
        .sort_values("median_observed_max_power_per_accelerator_w", ignore_index=True)
    )
    family["share_of_retained_Public_IDs_pct"] = family["n_systems"] * 100. / len(systems)
    family["validation_role"] = "hardware_stratified_configuration_diagnostic"
    stratified = pd.concat([
        _stratified_metrics(workload, ["accelerator_family"], replicates, seed + 100, "hardware_family"),
        _stratified_metrics(workload, ["measurement_boundary", "node_boundary"], replicates, seed + 200, "measurement_boundary"),
        _stratified_metrics(workload, ["scenario"], replicates, seed + 300, "inference_scenario"),
        _stratified_metrics(workload, ["task_model"], replicates, seed + 400, "inference_task"),
        _stratified_metrics(workload, ["accelerator_family", "accelerator_form_factor", "node_boundary", "scenario", "task_model"],
                            replicates, seed + 500, "hardware_boundary_task"),
    ], ignore_index=True)
    for metric in ("normalized_mean_bias_error_pct", "median_absolute_percentage_error_pct", "median_prediction_observation_ratio"):
        selected = stratified.loc[stratified["stratum_kind"].eq("hardware_family") & stratified["metric"].eq(metric),
                                  ["accelerator_family", "estimate", "ci_95_lower", "ci_95_upper"]]
        selected = selected.rename(columns={name: f"{metric}_{name}" for name in ("estimate", "ci_95_lower", "ci_95_upper")})
        family = family.merge(selected, on="accelerator_family", validate="one_to_one")
    coverage_rows = []
    for family_name in sorted(set(systems["accelerator_family"]) | {"A100", "H100"}):
        subset = systems.loc[systems["accelerator_family"].eq(family_name)]
        measurement_subset = workload.loc[workload["accelerator_family"].eq(family_name)]
        coverage_rows.append({"accelerator_family": family_name, "n_Public_IDs": len(subset),
                              "n_workload_rows": len(measurement_subset), "n_task_models": measurement_subset["task_model"].nunique(),
                              "n_scenarios": measurement_subset["scenario"].nunique(),
                              "share_of_retained_Public_IDs_pct": len(subset) * 100. / len(systems),
                              "status": "absent" if subset.empty else "single_ID_no_interval" if len(subset) == 1 else "multiple_IDs",
                              "future_fleet_representativeness_established": False})
    _a100_h100_comparison(systems, replicates, seed + 600).to_csv(output_dir / "mlperf_a100_h100_comparison.csv", index=False)
    stratified.to_csv(output_dir / "mlperf_stratified_metrics.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output_dir / "mlperf_hardware_coverage.csv", index=False)
    pd.DataFrame([{"input_file": str(input_path.resolve()), "input_sha256": input_sha256 or sha256_file(input_path),
                   "reference_formula_W_per_accelerator": "gpu_full_power_w/gpu_power_share",
                   "configured_reference_W_per_accelerator": reference_w_per_accelerator,
                   "reference_source": "author_hardware_configuration_not_fitted",
                   "observed_statistic": "maximum_of_reported_workload_powers_per_Public_ID_not_assumed_full_utilization",
                   "measurement_boundary": "reported_whole_system_SUT_power_components_unresolved",
                   "CI_method": "Public_ID_percentile_bootstrap_within_each_stratum",
                   "bootstrap_seed": seed, "requested_bootstrap_replicates": replicates,
                   "power_utilization_curve_fitted": False,
                   "training_or_facility_validated": False,
                   "single_ID_interval_policy": "missing_not_degenerate_95pct_interval"}]).to_csv(output_dir / "validation_run_metadata.csv", index=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    workload.to_csv(output_dir / "mlperf_workload_measurements.csv", index=False)
    systems.to_csv(output_dir / "mlperf_system_validation.csv", index=False)
    family.to_csv(output_dir / "mlperf_family_summary.csv", index=False)
    summary.to_csv(output_dir / "validation_metric_summary.csv", index=False)
    pd.DataFrame.from_records(audit).to_csv(output_dir / "validation_filter_audit.csv", index=False)
    return workload, systems, family, summary


def audit_national_scope(input_path: Path, output_dir: Path, input_frame=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    national = pd.read_csv(input_path) if input_frame is None else input_frame.copy()
    latest = (
        national.sort_values(["country", "year"])
        .groupby("country", as_index=False)
        .tail(1)
        .copy()
    )
    latest["comparison_role"] = (
        "scope and order-of-magnitude audit only; no prediction error because "
        "the observation is historical all-data-centre electricity whereas the model is future AI-only"
    )
    latest["compatible_as_direct_prediction_target"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    national.to_csv(output_dir / "national_official_energy_series.csv", index=False)
    latest.to_csv(output_dir / "national_scope_audit.csv", index=False)
    return national, latest


def aggregate_trace_batch(frame: pd.DataFrame) -> dict[str, float]:
    """The validation-specific activity filter, callable during the first read."""
    state = frame["state_public"].fillna("").astype(str)
    def values(column):
        array = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(array), np.maximum(array, 0.), 0.)
    gpu_request, used_gpu_hours, gpu_sm = (values(name) for name in ("gpu_request", "used_gpu_hours", "avg_gpu_sm_util"))
    cpu_request, cpu_util, memory_util = (values(name) for name in ("cpu_request_cores", "avg_cpu_request_util", "avg_memory_util"))
    active = state.eq("Running").to_numpy() | (used_gpu_hours > 0) | (gpu_sm > 0) | (cpu_util > 0)
    active &= ~state.eq("Standby").to_numpy()
    gpu = gpu_sm[active] / 100.
    gpu = np.minimum(gpu, np.where(gpu_request[active] > 0, gpu_request[active], gpu))
    return {"cpu_used_cores": float((cpu_request[active] * cpu_util[active]).sum()),
            "gpu_sm_equivalent": float(gpu.sum()),
            "memory_activity_proxy": float((cpu_request[active] * memory_util[active]).sum()),
            "active_pod_rows": int(active.sum()), "all_pod_rows": len(frame)}


def _aggregate_trace_hour(paths: Sequence[Path]) -> dict[str, float]:
    totals = {"cpu_used_cores": 0.0, "gpu_sm_equivalent": 0.0, "memory_activity_proxy": 0.0}
    active_rows = 0
    all_rows = 0
    for path in paths:
        frame = pd.read_parquet(path, columns=list(TRACE_COLUMNS))
        batch = aggregate_trace_batch(frame)
        all_rows += batch["all_pod_rows"]
        active_rows += batch["active_pod_rows"]
        for name in totals:
            totals[name] += batch[name]
    return {**totals, "active_pod_rows": active_rows, "all_pod_rows": all_rows}


def diagnose_trace_boundary(
    trace_root: Path,
    output_dir: Path,
    window_hours: int = 24,
    hour_aggregates: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Endpoint resource proxy only; full-year model ramps are exported by W06.

    Standalone calls read two small endpoint windows. Supplied hour aggregates
    reuse the validation-specific filter collected during the suite's first pass.
    """
    if not isinstance(window_hours, int) or window_hours < 1:
        raise ValueError("window_hours must be a positive integer.")
    grouped: dict[tuple[int, int], list[Path]] = {}
    aggregates = None
    if hour_aggregates is None:
        partitions = _partition_files(trace_root)
        if not partitions:
            raise FileNotFoundError(f"No pod-hour parquet partitions found under {trace_root}")
        for day, hour, path in partitions:
            grouped.setdefault((day, hour), []).append(path)
        hours = sorted(grouped)
    else:
        if hour_aggregates.empty or hour_aggregates.duplicated(["trace_day", "trace_hour"]).any():
            raise ValueError("Shared validation aggregates require unique observed trace hours.")
        aggregates = hour_aggregates.set_index(["trace_day", "trace_hour"])
        hours = sorted(aggregates.index.tolist())
    if len(hours) < 2 * window_hours:
        raise ValueError(f"Trace has {len(hours)} hours; at least {2 * window_hours} are required.")

    chosen = hours[-window_hours:] + hours[:window_hours]
    relative = list(range(-window_hours, 0)) + list(range(0, window_hours))
    records = []
    for rel_hour, (day, hour) in zip(relative, chosen):
        aggregate = (_aggregate_trace_hour(grouped[(day, hour)]) if aggregates is None else aggregates.loc[(day, hour)].to_dict())
        records.append(
            {"relative_hour": rel_hour, "trace_day": day, "trace_hour": hour, **aggregate}
        )
        print(
            f"[trace-boundary] relative hour {rel_hour:+d}: day={day}, hour={hour}",
            flush=True,
        )
    window = pd.DataFrame.from_records(records).sort_values("relative_hour", ignore_index=True)

    resource_columns = ("cpu_used_cores", "gpu_sm_equivalent", "memory_activity_proxy")
    configured_weights = np.array([0.30, 0.50, 0.12], dtype=float)
    active_weights = configured_weights.copy()
    normalized_columns = []
    for column in resource_columns:
        mean = float(window[column].mean())
        normalized = f"normalized_{column}"
        normalized_columns.append(normalized)
        if not np.isfinite(mean) or mean <= 0:
            window[normalized] = 0.0
            active_weights[len(normalized_columns) - 1] = 0.0
        else:
            window[normalized] = window[column] / mean
    active_weights = (active_weights / active_weights.sum()
                      if active_weights.sum() > 0 else np.full(len(resource_columns), 1. / len(resource_columns)))
    window["normalized_composite_resource_load"] = np.average(
        window[normalized_columns].to_numpy(dtype=float), axis=1, weights=active_weights
    )

    composite = window["normalized_composite_resource_load"].to_numpy(dtype=float)
    seam_index = window.index[window["relative_hour"].eq(0)][0]
    seam_flags = np.zeros(len(window), dtype=bool)
    seam_flags[seam_index] = True
    hour_positions = window["trace_day"].to_numpy() * 24 + window["trace_hour"].to_numpy()
    discontinuities = np.r_[False, np.diff(hour_positions) != 1]
    metrics = power_metrics(composite, seam_flags, discontinuities)
    scale_to_pct = 100. / composite.mean() if composite.mean() > 0 else np.nan
    seam_ramp_pct_mean = abs(composite[seam_index] - composite[seam_index - 1]) * scale_to_pct
    max_internal_ramp = metrics["max_abs_ramp_excluding_seams_mw_per_h"] * scale_to_pct
    max_window_ramp = metrics["max_abs_ramp_mw_per_h"] * scale_to_pct
    window["trace_repeat_seam_before"] = seam_flags
    window["alignment_discontinuity_before"] = discontinuities
    window["absolute_proxy_ramp_pct_of_window_mean"] = np.r_[np.nan, np.abs(np.diff(composite)) * scale_to_pct]

    diagnostics: list[dict[str, object]] = [
        {
            "diagnostic": "trace_hours_total",
            "value": len(hours),
            "unit": "h",
            "interpretation": "number of unique hourly partitions in the public trace",
        },
        {
            "diagnostic": "composite_seam_ramp_pct_of_window_mean",
            "value": seam_ramp_pct_mean,
            "unit": "%",
            "interpretation": "last-to-first-hour discontinuity of normalized resource-load proxy",
        },
        {
            "diagnostic": "max_nonseam_ramp_pct_of_window_mean",
            "value": max_internal_ramp,
            "unit": "%",
            "interpretation": f"largest adjacent-hour proxy ramp within the two {window_hours}-h endpoint windows, excluding the repeat seam",
        },
        {
            "diagnostic": "seam_is_largest_ramp_in_endpoint_window",
            "value": bool(np.isclose(seam_ramp_pct_mean, max_window_ramp)),
            "unit": "boolean",
            "interpretation": "local endpoint-window diagnostic; not a full-trace maximum claim",
        },
    ]
    for metric, name in (
        ("max_abs_ramp_mw_per_h", "raw_max_proxy_ramp_pct_of_window_mean"),
        ("max_abs_ramp_excluding_alignment_mw_per_h", "max_unbroken_proxy_ramp_pct_of_window_mean"),
        ("p95_abs_ramp_mw_per_h", "p95_proxy_ramp_pct_of_window_mean"),
        ("p99_abs_ramp_mw_per_h", "p99_proxy_ramp_pct_of_window_mean"),
        ("p95_abs_ramp_excluding_seams_mw_per_h", "p95_nonseam_proxy_ramp_pct_of_window_mean"),
        ("p99_abs_ramp_excluding_seams_mw_per_h", "p99_nonseam_proxy_ramp_pct_of_window_mean"),
    ):
        diagnostics.append({"diagnostic": name, "value": metrics[metric] * scale_to_pct,
                            "unit": "% of endpoint-window proxy mean per hour",
                            "interpretation": "local normalized resource proxy; not full-trace power or a national ramp"})
    endpoint_before = window.loc[window["relative_hour"].eq(-1)].iloc[0]
    endpoint_after = window.loc[window["relative_hour"].eq(0)].iloc[0]
    for column in resource_columns:
        denominator = 0.5 * (abs(float(endpoint_before[column])) + abs(float(endpoint_after[column])))
        seam = (
            100.0 * abs(float(endpoint_after[column]) - float(endpoint_before[column])) / denominator
            if denominator > 0
            else np.nan
        )
        diagnostics.append(
            {
                "diagnostic": f"{column}_seam_difference_pct_of_endpoint_mean",
                "value": seam,
                "unit": "%",
                "interpretation": "resource-specific last-to-first-hour discontinuity",
            }
        )

    diagnostics_frame = pd.DataFrame.from_records(diagnostics)
    diagnostics_frame["diagnostic_scope"] = "two_endpoint_windows_resource_proxy_only"
    diagnostics_frame["window_hours_each_endpoint"] = window_hours
    diagnostics_frame["is_model_power_validation"] = False
    diagnostics_frame["full_calendar_power_diagnostics"] = "hourly_model temporal_metrics / trace_seams; Figure3 W06_Baseline_Peak_Ramp"
    output_dir.mkdir(parents=True, exist_ok=True)
    window.to_csv(output_dir / "trace_seam_window.csv", index=False)
    diagnostics_frame.to_csv(output_dir / "trace_boundary_diagnostics.csv", index=False)
    return window, diagnostics_frame


def _metric(summary: pd.DataFrame, name: str) -> pd.Series:
    matches = summary.loc[summary["metric"].eq(name)]
    if len(matches) != 1:
        raise ValueError(f"Expected one summary row for {name!r}, found {len(matches)}")
    return matches.iloc[0]


def create_validation_figure(
    systems: pd.DataFrame,
    family: pd.DataFrame,
    summary: pd.DataFrame,
    trace_window: pd.DataFrame | None,
    trace_diagnostics: pd.DataFrame | None,
    output_stem: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    colors = {
        "T4": "#4E79A7",
        "A2": "#76B7B2",
        "A10": "#59A14F",
        "A30": "#EDC948",
        "A100": "#F28E2B",
        "L40S": "#B07AA1",
        "H100": "#E15759",
        "Other NVIDIA": "#9C9C9C",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.55), gridspec_kw={"wspace": 0.42})

    ax = axes[0]
    for family_name, group in systems.groupby("accelerator_family", sort=False):
        ax.scatter(
            group["observed_power_max_w"] / 1000.0,
            group["predicted_full_load_reference_w"] / 1000.0,
            s=16,
            color=colors.get(family_name, "#9C9C9C"),
            edgecolors="white",
            linewidths=0.35,
            alpha=0.88,
            label=family_name,
            zorder=3,
        )
    max_axis = 1.05 * max(
        float((systems["observed_power_max_w"] / 1000.0).max()),
        float((systems["predicted_full_load_reference_w"] / 1000.0).max()),
    )
    ax.plot([0, max_axis], [0, max_axis], color="#555555", lw=0.8, ls="--", zorder=1)
    ax.set_xlim(0, max_axis)
    ax.set_ylim(0, max_axis)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Largest reported system\nworkload power (kW)")
    ax.set_ylabel("Configured full-load\nsystem reference (kW)")
    mdape = _metric(summary, "median_absolute_percentage_error_pct")
    nmbe = _metric(summary, "normalized_mean_bias_error_pct")
    ax.text(
        0.04,
        0.96,
        f"n = {len(systems)} Public IDs\nMdAPE = {mdape['estimate']:.1f}%\nNMBE = {nmbe['estimate']:+.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
    )

    ax = axes[1]
    ordered = family["accelerator_family"].tolist()
    for position, family_name in enumerate(ordered):
        values = systems.loc[
            systems["accelerator_family"].eq(family_name),
            "observed_max_power_per_accelerator_w",
        ].to_numpy(dtype=float)
        jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=15,
            color=colors.get(family_name, "#9C9C9C"),
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
            label="_nolegend_",
        )
        ax.plot(
            [position - 0.20, position + 0.20],
            [np.median(values), np.median(values)],
            color="#222222",
            lw=1.1,
            zorder=4,
        )
    reference = _metric(summary, "model_implied_full_load_power_per_accelerator_w")["estimate"]
    ax.axhline(reference, color="#D62728", lw=0.9, ls="--")
    overall = _metric(summary, "median_observed_power_per_accelerator_w")
    overall_position = len(ordered)
    interval = (np.array([[overall["estimate"] - overall["ci_95_lower"]],
                          [overall["ci_95_upper"] - overall["estimate"]]])
                if np.isfinite(overall["ci_95_lower"]) and np.isfinite(overall["ci_95_upper"]) else None)
    ax.errorbar(
        overall_position, overall["estimate"], yerr=interval,
        fmt="D",
        ms=3.5,
        color="#222222",
        ecolor="#222222",
        elinewidth=0.8,
        capsize=2,
        zorder=5,
    )
    ax.set_xticks(
        range(len(ordered) + 1),
        [*ordered, "Overall"],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Largest reported system power\nper accelerator (W)")
    percentile = _metric(summary, "empirical_percentile_of_model_reference_pct")["estimate"]
    ax.text(
        0.98,
        0.04,
        f"{reference:g} W configured reference\n(retained-sample P{percentile:.1f})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )

    ax = axes[2]
    if trace_window is not None and not trace_window.empty:
        ax.plot(
            trace_window["relative_hour"],
            trace_window["normalized_composite_resource_load"],
            color="#3B6FB6",
            lw=1.0,
        )
        ax.scatter(
            trace_window["relative_hour"],
            trace_window["normalized_composite_resource_load"],
            color="#3B6FB6",
            s=6,
            linewidths=0,
        )
        ax.axvline(-0.5, color="#D62728", lw=0.8, ls="--")
        ax.set_xlabel("Hour relative to repeat boundary")
        ax.set_ylabel("Normalized composite\nresource-load proxy")
        if trace_diagnostics is not None:
            row = trace_diagnostics.loc[
                trace_diagnostics["diagnostic"].eq("composite_seam_ramp_pct_of_window_mean")
            ]
            if len(row) == 1:
                ax.text(
                    0.98,
                    0.96,
                    f"Seam change\n{float(row.iloc[0]['value']):.1f}% of\nwindow mean",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6,
                )
    else:
        ax.text(0.5, 0.5, "Trace diagnostic not run", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    for label, ax in zip("abc", axes):
        ax.text(-0.18, 1.13 if label == "a" else 1.06, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        if ax.axison:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(width=0.6, length=2.5)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlperf", type=Path, default=DEFAULT_MLPERF)
    parser.add_argument("--national", type=Path, default=DEFAULT_NATIONAL)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE_STEM)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--skip-trace", action="store_true")
    parser.add_argument("--trace-window-hours", type=int, default=24)
    parser.add_argument("--hardware-config-file", type=Path, default=None,
                        help="JSON overrides of HardwarePowerConfig; parameters are recorded, not fitted.")
    parser.add_argument("--skip-figure", action="store_true", help="Write diagnostic tables without rendering a figure.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")

    hardware = load_hardware_config(args.hardware_config_file, HardwarePowerConfig()) if args.hardware_config_file else HardwarePowerConfig()
    _, systems, family, summary = validate_mlperf(
        args.mlperf, args.output_dir, args.bootstrap_replicates, args.seed, hardware_config=hardware
    )
    audit_national_scope(args.national, args.output_dir)

    trace_window = None
    trace_diagnostics = None
    if not args.skip_trace:
        trace_window, trace_diagnostics = diagnose_trace_boundary(
            args.trace_root, args.output_dir, args.trace_window_hours
        )
    if not args.skip_figure:
        create_validation_figure(
            systems, family, summary, trace_window, trace_diagnostics, args.figure_stem
        )

    mdape = _metric(summary, "median_absolute_percentage_error_pct")
    nmbe = _metric(summary, "normalized_mean_bias_error_pct")
    ratio = _metric(summary, "median_prediction_observation_ratio")
    print(
        "[external configuration diagnostic; not future energy accuracy] "
        f"n={len(systems)} system submissions; "
        f"MdAPE={mdape['estimate']:.2f}% "
        f"(95% bootstrap CI {mdape['ci_95_lower']:.2f}–{mdape['ci_95_upper']:.2f}%); "
        f"NMBE={nmbe['estimate']:+.2f}% "
        f"(95% bootstrap CI {nmbe['ci_95_lower']:+.2f}–{nmbe['ci_95_upper']:+.2f}%); "
        f"median ratio={ratio['estimate']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
