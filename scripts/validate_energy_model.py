"""External benchmark and boundary diagnostics for the data-centre energy model.

The script deliberately separates three questions:

1. MLPerf Power benchmarks the model's reference full-load IT power per
   accelerator against independent whole-system inference measurements.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.m4_hourly_gpu_model import HardwarePowerConfig


DEFAULT_MLPERF = ROOT_DIR / "dataset" / "validation" / "raw" / "mlperf_power_raw_data.csv"
DEFAULT_NATIONAL = ROOT_DIR / "dataset" / "validation" / "national_data_center_energy.csv"
DEFAULT_TRACE_ROOT = ROOT_DIR / "dataset" / "asi_opensource_pod_hourly"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "model_validation"
DEFAULT_FIGURE_STEM = ROOT_DIR / "figures" / "supplementary_energy_validation"
BOOTSTRAP_SEED = 20260819
BOOTSTRAP_REPLICATES = 10_000
PARTITION_RE = re.compile(r"day=(\d+)/hour=(\d+)/part-[^/]+\.parquet$")
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
    for family in ("H100", "A100", "L40S", "A30", "A10", "A2", "T4"):
        if family in text:
            return family
    return "Other NVIDIA"


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
    samples = {name: np.empty(replicates, dtype=float) for name in metric_names}
    for replicate in range(replicates):
        idx = rng.randint(0, n, size=n)
        values = _metric_values(observed[idx], predicted[idx])
        values["median_observed_power_per_accelerator_w"] = float(
            np.median(observed[idx] / accelerator_count[idx])
        )
        for name in metric_names:
            samples[name][replicate] = values[name]

    records = []
    for name in metric_names:
        lower, upper = np.quantile(samples[name], [0.025, 0.975])
        records.append(
            {
                "metric": name,
                "estimate": point[name],
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "independent_unit": "MLPerf Public ID system submission",
                "n": n,
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
            }
        )
    return pd.DataFrame.from_records(records)


def validate_mlperf(
    input_path: Path,
    output_dir: Path,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(input_path, low_memory=False)
    audit: list[dict[str, object]] = []

    def retain(stage: str, mask: pd.Series) -> None:
        nonlocal raw
        before = len(raw)
        raw = raw.loc[mask].copy()
        audit.append({"stage": stage, "rows_before": before, "rows_after": len(raw)})

    audit.append({"stage": "raw_rows", "rows_before": len(raw), "rows_after": len(raw)})
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
        & raw["Public ID"].notna()
    )
    retain("positive_numeric_power_accelerators_and_public_id", valid_numeric)

    power_config = HardwarePowerConfig()
    power_config.validate()
    reference_w_per_accelerator = power_config.gpu_full_power_w / power_config.gpu_power_share
    raw["accelerator_family"] = raw["accelerator_model_name"].map(_accelerator_family)
    raw["predicted_full_load_reference_w"] = raw["accelerator_count"] * reference_w_per_accelerator

    workload_columns = [
        "Public ID",
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
    ]
    workload = raw[workload_columns].rename(columns={"Public ID": "public_id"})

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
            accelerator_count=("accelerator_count", "median"),
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

    output_dir.mkdir(parents=True, exist_ok=True)
    workload.to_csv(output_dir / "mlperf_workload_measurements.csv", index=False)
    systems.to_csv(output_dir / "mlperf_system_validation.csv", index=False)
    family.to_csv(output_dir / "mlperf_family_summary.csv", index=False)
    summary.to_csv(output_dir / "validation_metric_summary.csv", index=False)
    pd.DataFrame.from_records(audit).to_csv(output_dir / "validation_filter_audit.csv", index=False)
    return workload, systems, family, summary


def audit_national_scope(input_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    national = pd.read_csv(input_path)
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


def _partition_files(trace_root: Path) -> list[tuple[int, int, Path]]:
    records = []
    for path in trace_root.glob("day=*/hour=*/part-*.parquet"):
        match = PARTITION_RE.search(path.as_posix())
        if match:
            records.append((int(match.group(1)), int(match.group(2)), path))
    return sorted(records, key=lambda row: (row[0], row[1], row[2].name))


def _aggregate_trace_hour(paths: Sequence[Path]) -> dict[str, float]:
    totals = {"cpu_used_cores": 0.0, "gpu_sm_equivalent": 0.0, "memory_activity_proxy": 0.0}
    active_rows = 0
    all_rows = 0
    for path in paths:
        frame = pd.read_parquet(path, columns=list(TRACE_COLUMNS))
        all_rows += len(frame)
        state = frame["state_public"].fillna("").astype(str)

        def values(column: str) -> np.ndarray:
            array = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            return np.where(np.isfinite(array), np.maximum(array, 0.0), 0.0)

        gpu_request = values("gpu_request")
        used_gpu_hours = values("used_gpu_hours")
        gpu_sm = values("avg_gpu_sm_util")
        cpu_request = values("cpu_request_cores")
        cpu_util = values("avg_cpu_request_util")
        memory_util = values("avg_memory_util")

        active = state.eq("Running").to_numpy()
        active |= used_gpu_hours > 0
        active |= gpu_sm > 0
        active |= cpu_util > 0
        active &= ~state.eq("Standby").to_numpy()
        active_rows += int(active.sum())
        if not np.any(active):
            continue

        cpu_used = cpu_request[active] * cpu_util[active]
        gpu_equivalent = gpu_sm[active] / 100.0
        gpu_equivalent = np.minimum(
            gpu_equivalent,
            np.where(gpu_request[active] > 0, gpu_request[active], gpu_equivalent),
        )
        memory_proxy = cpu_request[active] * memory_util[active]
        totals["cpu_used_cores"] += float(cpu_used.sum())
        totals["gpu_sm_equivalent"] += float(gpu_equivalent.sum())
        totals["memory_activity_proxy"] += float(memory_proxy.sum())
    return {**totals, "active_pod_rows": active_rows, "all_pod_rows": all_rows}


def diagnose_trace_boundary(
    trace_root: Path,
    output_dir: Path,
    window_hours: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    partitions = _partition_files(trace_root)
    if not partitions:
        raise FileNotFoundError(f"No pod-hour parquet partitions found under {trace_root}")

    grouped: dict[tuple[int, int], list[Path]] = {}
    for day, hour, path in partitions:
        grouped.setdefault((day, hour), []).append(path)
    hours = sorted(grouped)
    if len(hours) < 2 * window_hours:
        raise ValueError(f"Trace has {len(hours)} hours; at least {2 * window_hours} are required.")

    chosen = hours[-window_hours:] + hours[:window_hours]
    relative = list(range(-window_hours, 0)) + list(range(0, window_hours))
    records = []
    for rel_hour, (day, hour) in zip(relative, chosen):
        aggregate = _aggregate_trace_hour(grouped[(day, hour)])
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
    active_weights = active_weights / active_weights.sum()
    window["normalized_composite_resource_load"] = np.average(
        window[normalized_columns].to_numpy(dtype=float), axis=1, weights=active_weights
    )

    composite = window["normalized_composite_resource_load"].to_numpy(dtype=float)
    seam_index = window.index[window["relative_hour"].eq(0)][0]
    seam_ramp_pct_mean = 100.0 * abs(composite[seam_index] - composite[seam_index - 1]) / composite.mean()
    all_ramps = 100.0 * np.abs(np.diff(composite)) / composite.mean()
    internal_mask = np.ones_like(all_ramps, dtype=bool)
    internal_mask[seam_index - 1] = False
    max_internal_ramp = float(all_ramps[internal_mask].max())
    max_window_ramp = float(all_ramps.max())

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
            "interpretation": "largest adjacent-hour ramp within the two 24-h endpoint windows, excluding the repeat seam",
        },
        {
            "diagnostic": "seam_is_largest_ramp_in_endpoint_window",
            "value": bool(np.isclose(seam_ramp_pct_mean, max_window_ramp)),
            "unit": "boolean",
            "interpretation": "local endpoint-window diagnostic; not a full-trace maximum claim",
        },
    ]
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
    ax.set_xlabel("Observed maximum system power (kW)")
    ax.set_ylabel("Full-load reference (kW)")
    mdape = _metric(summary, "median_absolute_percentage_error_pct")
    nmbe = _metric(summary, "normalized_mean_bias_error_pct")
    ax.text(
        0.04,
        0.96,
        f"n = {len(systems)} systems\nMdAPE = {mdape['estimate']:.1f}%\nNMBE = {nmbe['estimate']:+.1f}%",
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
    ax.errorbar(
        overall_position,
        overall["estimate"],
        yerr=np.array(
            [
                [overall["estimate"] - overall["ci_95_lower"]],
                [overall["ci_95_upper"] - overall["estimate"]],
            ]
        ),
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
    ax.set_ylabel("Observed maximum power\nper accelerator (W)")
    percentile = _metric(summary, "empirical_percentile_of_model_reference_pct")["estimate"]
    ax.text(
        0.02,
        0.96,
        f"500 W reference\n(empirical P{percentile:.1f})",
        transform=ax.transAxes,
        ha="left",
        va="top",
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
                    0.04,
                    0.96,
                    f"Seam ramp = {float(row.iloc[0]['value']):.1f}% of mean",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6,
                )
    else:
        ax.text(0.5, 0.5, "Trace diagnostic not run", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    for label, ax in zip("abc", axes):
        ax.text(-0.18, 1.06, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")

    _, systems, family, summary = validate_mlperf(
        args.mlperf, args.output_dir, args.bootstrap_replicates, args.seed
    )
    audit_national_scope(args.national, args.output_dir)

    trace_window = None
    trace_diagnostics = None
    if not args.skip_trace:
        trace_window, trace_diagnostics = diagnose_trace_boundary(
            args.trace_root, args.output_dir, args.trace_window_hours
        )
    create_validation_figure(
        systems, family, summary, trace_window, trace_diagnostics, args.figure_stem
    )

    mdape = _metric(summary, "median_absolute_percentage_error_pct")
    nmbe = _metric(summary, "normalized_mean_bias_error_pct")
    ratio = _metric(summary, "median_prediction_observation_ratio")
    print(
        "[validation] "
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
