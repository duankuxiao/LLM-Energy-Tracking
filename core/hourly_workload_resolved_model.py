"""Trace-driven, component-level model with hourly grid carbon matching."""

from __future__ import annotations

import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterator, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.Factors import (  # noqa: E402
    CARBON_FACTORS,
    PUE,
    get_carbon_factor,
)
from dataset.Installed_capacity_data import (  # noqa: E402
    IT_CAPACITY,
    IT_RATIO,
    TOTAL_RATIO,
    SCENARIO_COLUMNS as SCENARIO_COL_MAP,
)
from core.task_model import TASK_TYPES, task_type_ids  # noqa: E402
from core.accounting_boundary import (  # noqa: E402
    annotate_selected_aggregate, build_boundary_metadata, resolve_countries, resolve_ai_capacity_factors,
)
from core.grid_carbon import (  # noqa: E402
    SCOPES, calendar_hours, calendar_status, find_hourly_file,
    matching_diagnostics, read_hourly_file, validate_direct_mean,
)
from core.research_io import sha256_file
from core.energy_attribution import (  # noqa: E402
    IDLE_ALLOCATION_RULES, allocate_component_energy, allocation_records,
    allocation_share_tables, resource_proxy_metadata, split_component_power,
)
from core.temporal_robustness import (  # noqa: E402
    power_metrics, seam_event_records, shared_shape_diagnostics, trace_hour_mapping,
)
from core.power_parameter_audit import (  # noqa: E402
    capacity_parameter_records, hardware_parameter_table, mechanism_conservation_table,
    parameter_dependency_table, power_curve_metadata,
)


ALIBABA_2026_POD_TABLE = "asi_opensource_pod_hourly"
ALIBABA_2026_SERVER_TABLE = "asi_opensource_server_hourly"
ALIBABA_2026_PARTITION_RE = re.compile(r"day=(?P<day>\d+)/hour=(?P<hour>\d+)")
ALIBABA_2026_JOB_TYPE_START_DAY = 109
ALIBABA_2026_JOB_TYPE_END_DAY = 184

RESOURCES = ("cpu", "gpu", "memory", "storage")
COMPONENTS = ("cpu", "gpu", "memory", "storage", "it_fan")
DATA_YEAR_START = 2025

POD_PROFILE_COLUMNS = (
    "pod_id",
    "state_public",
    "job_type_public",
    "gpu_request",
    "gpu_mem_request",
    "used_gpu_hours",
    "avg_gpu_sm_util",
    "avg_gpu_mem_gib",
    "cpu_request_cores",
    "avg_cpu_request_util",
    "avg_memory_util",
)
SERVER_PROFILE_COLUMNS = ("gpu_count", "cpu_capacity_cores")
DEFAULT_PARQUET_BATCH_SIZE = 65536
TraceReadCallback = Callable[[dict], None]

HOURLY_CARBON_COUNTRY_DIRS = {
    "United_Kingdom": "Great Britain",
}
HOURLY_CARBON_COLUMNS = (
    "scenario",
    "year",
    "country",
    "hour_index",
    "timestamp_utc",
    "facility_energy_mwh",
    "carbon_factor_kg_per_mwh",
    "carbon_tco2",
    "carbon_factor_source",
    "source_hour_position",
    "source_repeat_cycle",
    "trace_repeat_seam_before",
    "alignment_discontinuity_before",
    "alignment_method",
)


@dataclass(frozen=True)
class Alibaba2026TraceConfig:
    """Controls conversion of the public Alibaba GPU v2026 trace into model loads."""

    trace_anchor_year: int = 2001
    # CPU-only pods still consume CPU and memory energy inside the AI cluster.
    # The paper-validation task mix applies its own GPU-only filter separately.
    include_zero_gpu_pods: bool = True
    active_states: tuple[str, ...] = ("Running",)
    cap_gpu_sm_by_request: bool = True


@dataclass(frozen=True)
class HardwarePowerConfig:
    """
    Power and capacity assumptions used to map installed IT MW to resources.

    The component shares should sum to 1.0. They define the full-load IT power
    budget allocated to each component before PUE is applied.
    """

    cpu_power_share: float = 0.30
    gpu_power_share: float = 0.50
    memory_power_share: float = 0.12
    storage_power_share: float = 0.03
    it_fan_power_share: float = 0.05

    cpu_full_power_w_per_core: float = 12.0
    cpu_idle_power_w_per_core: float = 2.5
    gpu_full_power_w: float = 250.0
    gpu_idle_power_w: float = 25.0
    memory_power_w_per_gb: float = 0.07
    memory_idle_fraction: float = 0.80
    storage_power_w_per_tb: float = 6.5
    storage_idle_fraction: float = 0.60
    it_fan_idle_fraction: float = 0.20

    fan_cpu_weight: float = 0.35
    fan_gpu_weight: float = 0.50
    fan_memory_weight: float = 0.15
    # Shape stress parameters; one reproduces the original CPU/GPU formulas.
    cpu_dynamic_exponent: float = 1.0
    gpu_dynamic_exponent: float = 1.0

    def validate(self) -> None:
        shares = np.array(
            [
                self.cpu_power_share,
                self.gpu_power_share,
                self.memory_power_share,
                self.storage_power_share,
                self.it_fan_power_share,
            ],
            dtype=float,
        )
        if not np.isfinite(shares).all() or np.any(shares < 0):
            raise ValueError("Hardware component power shares must be non-negative.")
        if not np.isclose(shares.sum(), 1.0, atol=1e-8):
            raise ValueError(f"Hardware component power shares must sum to 1.0, got {shares.sum():.6f}.")
        if self.cpu_full_power_w_per_core <= 0 or self.gpu_full_power_w <= 0:
            raise ValueError("CPU and GPU full-load unit power values must be positive.")
        if self.memory_power_w_per_gb <= 0 or self.storage_power_w_per_tb <= 0:
            raise ValueError("Memory and storage unit power values must be positive.")
        full = (self.cpu_full_power_w_per_core, self.gpu_full_power_w,
                self.memory_power_w_per_gb, self.storage_power_w_per_tb)
        if not np.isfinite(full).all():
            raise ValueError("Full-load unit powers must be finite.")
        idle_fractions = (self.cpu_idle_power_w_per_core / self.cpu_full_power_w_per_core,
                          self.gpu_idle_power_w / self.gpu_full_power_w, self.memory_idle_fraction,
                          self.storage_idle_fraction, self.it_fan_idle_fraction)
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in idle_fractions):
            raise ValueError("Idle fractions must be finite and within [0, 1].")
        fan_weights = (self.fan_cpu_weight, self.fan_gpu_weight, self.fan_memory_weight)
        if not np.isfinite(fan_weights).all() or min(fan_weights) < 0 or sum(fan_weights) <= 0:
            raise ValueError("Fan driver weights must be finite, non-negative and have a positive sum.")
        exponents = (self.cpu_dynamic_exponent, self.gpu_dynamic_exponent)
        if not np.isfinite(exponents).all() or min(exponents) <= 0:
            raise ValueError("CPU/GPU dynamic response exponents must be finite and positive.")


@dataclass
class WorkloadProfile:
    interval_index: pd.DatetimeIndex
    interval_hours: float
    load: np.ndarray
    trace_capacity: np.ndarray
    task_counts: np.ndarray
    task_type_summary: pd.DataFrame
    relative_time: bool
    trace_capacity_source: tuple[str, ...]

    def with_load(self, load):
        """Replace the load array while retaining the trace metadata."""
        return replace(self, load=np.array(load, dtype=float, copy=True))

    @property
    def n_intervals(self) -> int:
        return len(self.interval_index)

    @property
    def trace_hours(self) -> float:
        return self.n_intervals * self.interval_hours

    @property
    def annualization_factor_8760(self) -> float:
        if self.trace_hours <= 0:
            raise ValueError("Workload trace must contain at least one positive-duration interval.")
        return 8760.0 / self.trace_hours


def _print_progress(verbose: bool, message: str) -> None:
    """Print an immediately visible progress message when verbose mode is enabled."""
    if verbose:
        print(f"[workload-component] {message}", flush=True)


def _progress_interval(total: int) -> int:
    """Return an interval that reports partition progress about ten times."""
    return max(1, (total + 9) // 10)


def _standard_hourly_index(year: int, hours: Optional[int] = None) -> pd.DatetimeIndex:
    if hours is None:
        return calendar_hours(year)
    return pd.date_range(start=f"{year}-01-01", periods=hours, freq="h", tz="UTC")


def _read_hourly_carbon_series(
    country: str,
    renewable_energy_policy: str,
    year: int,
    hourly_carbon_factors_dir: Union[str, Path],
    hourly_carbon_scope: str,
):
    path = find_hourly_file(hourly_carbon_factors_dir, country, renewable_energy_policy, year)
    if path is None:
        return None
    timestamps, factors = read_hourly_file(path, year, hourly_carbon_scope, allow_legacy=True)
    if hourly_carbon_scope == "direct":
        validate_direct_mean(factors, country, renewable_energy_policy, year)
    return timestamps, factors


def _load_hourly_carbon_factors(
    countries: Sequence[str],
    renewable_energy_policy: str,
    year: int,
    hourly_carbon_factors_dir: Union[str, Path],
    hourly_carbon_scope: str,
    hourly_carbon_fallback_to_annual: bool,
    grid_data=None,
):
    if hourly_carbon_scope not in SCOPES:
        raise ValueError(f"hourly_carbon_scope must be one of {SCOPES}.")
    loaded = {}
    reference_timestamps = None
    legacy_countries = []

    for country in countries:
        series_reader = grid_data.series if grid_data is not None else _read_hourly_carbon_series
        series = series_reader(
            country=country,
            renewable_energy_policy=renewable_energy_policy,
            year=year,
            hourly_carbon_factors_dir=hourly_carbon_factors_dir,
            hourly_carbon_scope=hourly_carbon_scope,
        )
        if series is None:
            continue

        timestamps, factors = series
        if calendar_status(timestamps, year) != "full_calendar":
            if not hourly_carbon_fallback_to_annual:
                raise ValueError(f"Strict hourly mode requires a full UTC calendar for {country}/{year}. "
                                 "Rebuild legacy leap-year inputs with scripts/build_hourly_grid_scenarios.py.")
            legacy_countries.append(country)
        if reference_timestamps is None:
            reference_timestamps = timestamps
        elif not timestamps.equals(reference_timestamps):
            raise ValueError(f"Hourly carbon calendars differ for {country}/{year}; refusing to drop hours.")
        loaded[country] = factors

    missing = [country for country in countries if country not in loaded]
    if missing and hourly_carbon_scope != "direct":
        raise ValueError(f"No life-cycle annual fallback is available for {missing}/{year}; "
                         "the annual table contains direct factors only.")
    if legacy_countries:
        warnings.warn(f"{renewable_energy_policy}/{year}: legacy 8760-hour inputs truncate the leap year "
                      f"for {legacy_countries}; recorded in grid_carbon_audit, unsuitable for full-calendar conclusions.",
                      RuntimeWarning, stacklevel=2)
    if missing and hourly_carbon_fallback_to_annual:
        warnings.warn(f"{renewable_energy_policy}/{year}: annual direct-factor fallback for {missing}; "
                      "recorded in grid_carbon_audit.", RuntimeWarning, stacklevel=2)

    if reference_timestamps is None:
        if not hourly_carbon_fallback_to_annual:
            raise FileNotFoundError(f"No hourly carbon factors found for {renewable_energy_policy} {year}.")
        reference_timestamps = _standard_hourly_index(year)

    factors = np.zeros((len(countries), len(reference_timestamps)), dtype=np.float64)
    used_hourly = np.zeros((len(countries),), dtype=bool)
    for country_id, country in enumerate(countries):
        if country in loaded:
            factors[country_id] = loaded[country]
            used_hourly[country_id] = True
            continue
        if not hourly_carbon_fallback_to_annual:
            raise FileNotFoundError(f"No hourly carbon factors found for {country} {year}.")
        factors[country_id] = get_carbon_factor(
            renewable_energy_policy,
            country,
            year,
        )

    return reference_timestamps, factors, used_hourly


def _resize_hourly_energy_by_position(values: np.ndarray, target_hours: int) -> np.ndarray:
    source_hours = values.shape[1]
    if source_hours == target_hours:
        return values
    if source_hours <= 0 or target_hours <= 0:
        raise ValueError("Hourly energy arrays must have positive length.")

    positions = np.rint(np.linspace(0, source_hours - 1, target_hours)).astype(int)
    resized = values[:, positions]
    source_total = values.sum(axis=1)
    resized_total = resized.sum(axis=1)
    positive = resized_total > 0
    resized[positive] *= (source_total[positive] / resized_total[positive])[:, None]
    return resized


def _align_interval_energy_to_target_hours(
    interval_energy_mwh: np.ndarray,
    interval_index: pd.DatetimeIndex,
    target_timestamps: pd.DatetimeIndex,
    relative_trace: bool = False,
    return_mapping: bool = False,
    preserve_source_total: bool = True,
):
    original_shape = interval_energy_mwh.shape
    flat = interval_energy_mwh.reshape(-1, original_shape[-1])
    source_hours = pd.DatetimeIndex(interval_index).floor("h")
    hourly = pd.DataFrame(flat.T, index=source_hours).groupby(level=0, sort=True).sum()
    target = pd.DatetimeIndex(target_timestamps)
    hourly_values = hourly.to_numpy(dtype=np.float64).T

    mapping = trace_hour_mapping(hourly.index, target, relative_trace)
    if mapping["alignment_method"].iloc[0] == "position_resample" and preserve_source_total:
        aligned = _resize_hourly_energy_by_position(hourly_values, len(target))
    else:
        aligned = hourly_values[:, mapping["source_hour_position"].to_numpy(dtype=int)]
    result = aligned.reshape(original_shape[:-1] + (len(target),))
    return (result, mapping) if return_mapping else result


def _hourly_facility_energy(
    component_power_mw: np.ndarray,
    pue: np.ndarray,
    annual_facility_energy_mwh: np.ndarray,
    interval_index: pd.DatetimeIndex,
    interval_hours: float,
    target_timestamps: pd.DatetimeIndex,
    relative_trace: bool = False,
    return_alignment: bool = False,
):
    interval_facility_energy_mwh = component_power_mw.sum(axis=0) * interval_hours * pue[:, None]
    hourly_facility_energy_mwh, mapping = _align_interval_energy_to_target_hours(
        interval_energy_mwh=interval_facility_energy_mwh,
        interval_index=interval_index,
        target_timestamps=target_timestamps,
        relative_trace=relative_trace,
        return_mapping=True,
        preserve_source_total=False,
    )

    # Keep annual PUE-based energy unchanged; use the trace only for the hourly shape.
    hourly_total = hourly_facility_energy_mwh.sum(axis=1)
    positive = hourly_total > 0
    result = np.zeros_like(hourly_facility_energy_mwh)
    result[positive] = hourly_facility_energy_mwh[positive] * (
        annual_facility_energy_mwh[positive] / hourly_total[positive]
    )[:, None]
    if np.any(~positive):
        result[~positive] = annual_facility_energy_mwh[~positive, None] / len(target_timestamps)
    if not np.allclose(result.sum(axis=1), annual_facility_energy_mwh, rtol=1e-10, atol=1e-6):
        raise ValueError("Calendar alignment failed to preserve annual facility electricity.")
    if return_alignment:
        scale = np.divide(annual_facility_energy_mwh, hourly_total,
                          out=np.full_like(hourly_total, np.nan), where=positive)
        return result, mapping, hourly_total, scale
    return result



def _hourly_carbon_frame(
    scenario: str,
    year: int,
    countries: Sequence[str],
    timestamps: pd.DatetimeIndex,
    hourly_facility_energy_mwh: np.ndarray,
    hourly_emission_kg_per_mwh: np.ndarray,
    used_hourly_factors: np.ndarray,
    alignment: pd.DataFrame,
) -> pd.DataFrame:
    n_countries = len(countries)
    n_hours = len(timestamps)
    hourly_carbon_tco2 = hourly_facility_energy_mwh * hourly_emission_kg_per_mwh / 1000.0
    source_labels = np.where(used_hourly_factors, "hourly", "annual_fallback")

    return pd.DataFrame(
        {
            "scenario": scenario,
            "year": year,
            "country": np.repeat(np.array(countries, dtype=object), n_hours),
            "hour_index": np.tile(np.arange(n_hours), n_countries),
            "timestamp_utc": np.tile(timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"), n_countries),
            "facility_energy_mwh": hourly_facility_energy_mwh.reshape(-1),
            "carbon_factor_kg_per_mwh": hourly_emission_kg_per_mwh.reshape(-1),
            "carbon_tco2": hourly_carbon_tco2.reshape(-1),
            "carbon_factor_source": np.repeat(source_labels, n_hours),
            **{column: np.tile(alignment[column].to_numpy(), n_countries) for column in (
                "source_hour_position", "source_repeat_cycle", "trace_repeat_seam_before",
                "alignment_discontinuity_before", "alignment_method")},
        },
        columns=HOURLY_CARBON_COLUMNS,
    )


def _normalize_weights(countries: Sequence[str], weights: Mapping[str, float]) -> np.ndarray:
    values = np.array([float(weights.get(country, 0.0)) for country in countries], dtype=float)
    if np.any(values < 0):
        raise ValueError("Country weights must be non-negative.")
    total = values.sum()
    if total <= 0:
        raise ValueError("At least one country weight must be positive.")
    return values / total


def _as_task_weight_table(
    countries: Sequence[str],
    task_weights: Optional[Mapping[str, Mapping[str, float]]],
) -> np.ndarray:
    if task_weights is None:
        defaults = {
            "training": IT_RATIO,
            "inference": TOTAL_RATIO,
            "other": TOTAL_RATIO,
            "unclassified": TOTAL_RATIO,
        }
        return np.stack([_normalize_weights(countries, defaults[task_type]) for task_type in TASK_TYPES])

    table = []
    for task_type in TASK_TYPES:
        if task_type not in task_weights:
            if task_type == "unclassified":
                table.append(_normalize_weights(countries, TOTAL_RATIO))
                continue
            raise ValueError(f"Missing weights for task type '{task_type}'.")
        table.append(_normalize_weights(countries, task_weights[task_type]))
    return np.stack(table)


def _resolve_alibaba_2026_table_root(base_path: Union[str, Path], table_name: str) -> Path:
    base = Path(base_path)
    candidates = []
    if base.name in {ALIBABA_2026_POD_TABLE, ALIBABA_2026_SERVER_TABLE}:
        candidates.append(base.parent / table_name)
        if base.name == table_name:
            candidates.insert(0, base)
    candidates.extend(
        [
            base / table_name,
            base / "data" / table_name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else base / table_name


def _alibaba_2026_partition_files(table_root: Path) -> list[tuple[int, int, Path]]:
    rows = []
    for path in sorted(table_root.glob("day=*/hour=*/part-*.parquet")):
        match = ALIBABA_2026_PARTITION_RE.search(path.as_posix())
        if match:
            rows.append((int(match.group("day")), int(match.group("hour")), path))
    return rows


def _iter_parquet_frames(
    path: Path,
    columns: Sequence[str],
    batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
    read_callback: Optional[TraceReadCallback] = None,
) -> Iterator[pd.DataFrame]:
    """Read projected columns in bounded, single-threaded Arrow batches.

    The callback runs before opening each file and after every converted batch.
    Native crashes cannot be caught here; a flushed callback log identifies the
    last file and completed batch even when Python cannot raise an exception.
    """
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("parquet_batch_size must be a positive integer.")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Reading Alibaba GPU v2026 parquet files requires pyarrow in the active environment.") from exc

    def report(event: str, **details) -> None:
        if read_callback is not None:
            read_callback({"event": event, "path": str(path), **details})

    rows_read = 0
    batches_read = 0
    parquet = None
    try:
        report("file_start", batch_size=int(batch_size), columns=list(columns))
        parquet = pq.ParquetFile(path, memory_map=False, pre_buffer=False)
        expected_rows = parquet.metadata.num_rows
        report("file_metadata", expected_rows=expected_rows)
        for batch in parquet.iter_batches(batch_size=int(batch_size), columns=list(columns), use_threads=False):
            frame = batch.to_pandas(use_threads=False, split_blocks=True)
            rows_read += len(frame)
            batches_read += 1
            report("batch_read", batch_index=batches_read, batch_rows=len(frame), rows_read=rows_read)
            yield frame
            del frame, batch
        if rows_read != expected_rows:
            raise ValueError(f"Read {rows_read} rows; parquet metadata declares {expected_rows}.")
        report("file_complete", rows_read=rows_read, batches_read=batches_read)
    except Exception as exc:
        report("file_error", rows_read=rows_read, batches_read=batches_read,
               error_type=type(exc).__name__, error=str(exc))
        raise RuntimeError(f"Parquet read failed for {path}: {exc}") from exc
    finally:
        if parquet is not None:
            parquet.close()


def _to_numeric_array(df: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)


@dataclass
class _PodBatchTotals:
    load: np.ndarray
    pod_hour_counts: np.ndarray
    source_used_gpu_hours: np.ndarray
    source_gpu_memory_gib_hours: np.ndarray
    paper_validation_used_gpu_hours: np.ndarray


def _aggregate_pod_batch(df: pd.DataFrame, day: int, trace_config: Alibaba2026TraceConfig) -> _PodBatchTotals:
    """Apply the existing task mapping, activity filters and resource formulas."""
    load = np.zeros((len(TASK_TYPES), len(RESOURCES)), dtype=np.float64)
    pod_hour_counts = np.zeros(len(TASK_TYPES), dtype=np.int64)
    source_used_gpu_hours = np.zeros(len(TASK_TYPES), dtype=np.float64)
    source_gpu_memory_gib_hours = np.zeros(len(TASK_TYPES), dtype=np.float64)
    paper_validation_used_gpu_hours = np.zeros(len(TASK_TYPES), dtype=np.float64)
    totals = _PodBatchTotals(load, pod_hour_counts, source_used_gpu_hours,
                             source_gpu_memory_gib_hours, paper_validation_used_gpu_hours)
    if df.empty:
        return totals
    mapped_task_type_ids = task_type_ids(df["job_type_public"])
    state = df["state_public"].fillna("").astype(str)
    gpu_request = _to_numeric_array(df, "gpu_request")
    gpu_mem_request = _to_numeric_array(df, "gpu_mem_request")
    used_gpu_hours = _to_numeric_array(df, "used_gpu_hours")
    gpu_sm = _to_numeric_array(df, "avg_gpu_sm_util")
    gpu_mem = _to_numeric_array(df, "avg_gpu_mem_gib")
    cpu_request = _to_numeric_array(df, "cpu_request_cores")
    cpu_request_util = _to_numeric_array(df, "avg_cpu_request_util")
    memory_util = _to_numeric_array(df, "avg_memory_util")

    if ALIBABA_2026_JOB_TYPE_START_DAY <= day <= ALIBABA_2026_JOB_TYPE_END_DAY:
        paper_valid = (
            np.isfinite(gpu_mem_request)
            & (gpu_mem_request > 0)
            & np.isfinite(used_gpu_hours)
            & (mapped_task_type_ids != TASK_TYPES.index("unclassified"))
        )
        if np.any(paper_valid):
            paper_validation_used_gpu_hours += np.bincount(
                mapped_task_type_ids[paper_valid],
                weights=np.maximum(used_gpu_hours[paper_valid], 0.0),
                minlength=len(TASK_TYPES),
            )

    # NaN utilization means no measured activity for the corresponding driver.
    gpu_request = np.where(
        np.isfinite(gpu_request), np.maximum(gpu_request, 0.0), 0.0
    )
    cpu_request = np.where(
        np.isfinite(cpu_request), np.maximum(cpu_request, 0.0), 0.0
    )
    gpu_sm = np.where(np.isfinite(gpu_sm), np.maximum(gpu_sm, 0.0), 0.0)
    gpu_mem = np.where(np.isfinite(gpu_mem), np.maximum(gpu_mem, 0.0), 0.0)
    cpu_request_util = np.where(
        np.isfinite(cpu_request_util), np.maximum(cpu_request_util, 0.0), 0.0
    )
    memory_util = np.where(np.isfinite(memory_util), np.maximum(memory_util, 0.0), 0.0)
    used_gpu_hours = np.where(
        np.isfinite(used_gpu_hours), np.maximum(used_gpu_hours, 0.0), 0.0
    )

    active = state.isin(trace_config.active_states).to_numpy()
    active |= used_gpu_hours > 0
    active |= gpu_sm > 0
    active |= cpu_request_util > 0
    # Standby GPU capacity is already represented by the component idle-power term.
    active &= ~state.eq("Standby").to_numpy()
    # This optional sensitivity boundary excludes CPU-only pods. It must not
    # affect the separate paper-validation task-mix sample constructed above.
    if not trace_config.include_zero_gpu_pods:
        active &= gpu_request > 0
    if not np.any(active):
        return totals

    mapped_task_type_ids = mapped_task_type_ids[active]
    gpu_request = gpu_request[active]
    used_gpu_hours = used_gpu_hours[active]
    gpu_sm = gpu_sm[active]
    gpu_mem = gpu_mem[active]
    cpu_request = cpu_request[active]
    cpu_request_util = cpu_request_util[active]
    memory_util = memory_util[active]

    cpu_used_cores = cpu_request * cpu_request_util
    gpu_sm_equivalent = gpu_sm / 100.0
    if trace_config.cap_gpu_sm_by_request:
        gpu_sm_equivalent = np.minimum(
            gpu_sm_equivalent,
            np.where(gpu_request > 0, gpu_request, gpu_sm_equivalent),
        )
    # Server memory capacity is not released; CPU request acts only as a pod-size weight.
    memory_activity_proxy = cpu_request * memory_util
    storage_activity = np.zeros_like(cpu_used_cores)
    resource_values = (
        cpu_used_cores,
        gpu_sm_equivalent,
        memory_activity_proxy,
        storage_activity,
    )

    pod_hour_counts += np.bincount(mapped_task_type_ids, minlength=len(TASK_TYPES))
    source_used_gpu_hours += np.bincount(
        mapped_task_type_ids, weights=used_gpu_hours, minlength=len(TASK_TYPES)
    )
    source_gpu_memory_gib_hours += np.bincount(
        mapped_task_type_ids, weights=gpu_mem, minlength=len(TASK_TYPES)
    )
    for resource_id, values in enumerate(resource_values):
        load[:, resource_id] += np.bincount(
            mapped_task_type_ids, weights=values, minlength=len(TASK_TYPES)
        )
    return totals


def _server_trace_capacity(
    server_root: Path,
    n_intervals: int,
    capacity_quantile: float,
    verbose: bool = False,
    parquet_batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
    read_callback: Optional[TraceReadCallback] = None,
) -> Optional[np.ndarray]:
    files = _alibaba_2026_partition_files(server_root)
    if not files:
        return None

    files_to_process = [
        (day, hour, path) for day, hour, path in files if day * 24 + hour < n_intervals
    ]
    progress_interval = _progress_interval(len(files_to_process))
    capacity = np.full((2, n_intervals), np.nan, dtype=np.float64)
    for file_index, (day, hour, path) in enumerate(files_to_process, start=1):
        if (
            file_index == 1
            or file_index == len(files_to_process)
            or file_index % progress_interval == 0
        ):
            _print_progress(
                verbose,
                f"Reading server partitions: {file_index}/{len(files_to_process)} "
                f"({file_index / len(files_to_process):.0%})",
            )
        pos = day * 24 + hour
        partition_capacity = np.zeros(2, dtype=np.float64)
        for df in _iter_parquet_frames(path, SERVER_PROFILE_COLUMNS, parquet_batch_size, read_callback):
            gpu_count = pd.to_numeric(df["gpu_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
            cpu_cores = pd.to_numeric(df["cpu_capacity_cores"], errors="coerce").fillna(0.0).clip(lower=0.0)
            partition_capacity[0] += float(cpu_cores.sum())
            partition_capacity[1] += float(gpu_count.sum())
            del df, gpu_count, cpu_cores
        capacity[:, pos] = partition_capacity

    result = np.full((2,), np.nan, dtype=np.float64)
    for idx in range(2):
        valid = capacity[idx, np.isfinite(capacity[idx]) & (capacity[idx] > 0)]
        if valid.size:
            result[idx] = float(np.quantile(valid, capacity_quantile))
    if not np.all(np.isfinite(result)):
        return None
    return result


def build_workload_profile(
    workload_profile_path: Union[str, Path],
    server_profile_path: Optional[Union[str, Path]] = None,
    capacity_quantile: float = 0.96,
    trace_config: Optional[Alibaba2026TraceConfig] = None,
    max_intervals: Optional[int] = None,
    verbose: bool = False,
    parquet_batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
    read_callback: Optional[TraceReadCallback] = None,
    batch_observer=None,
) -> WorkloadProfile:
    """
    Convert Alibaba Cluster Trace GPU v2026 pod-hour parquet partitions into
    hourly resource-load time series used by the downstream footprint model.

    The public trace has relative `day/hour` partitions rather than calendar
    timestamps. CPU load is request cores multiplied by CPU-request utilization.
    GPU load is GPU-SM-equivalent count (`avg_gpu_sm_util / 100`), consistent
    with the release's own utilization processing. Host-memory capacity is not
    public, so `cpu_request_cores * avg_memory_util` is used only as a relative
    memory-activity proxy. No pod-level storage/I/O metric exists in the public
    2026 trace; the storage workload channel is therefore zero and the hardware
    model retains only its configured idle storage power.
    """
    if not isinstance(parquet_batch_size, (int, np.integer)) or parquet_batch_size <= 0:
        raise ValueError("parquet_batch_size must be a positive integer.")
    if not (0 < capacity_quantile <= 1):
        raise ValueError("capacity_quantile must be in (0, 1].")
    if max_intervals is not None and max_intervals <= 0:
        raise ValueError("max_intervals must be positive when provided.")

    trace_config = trace_config or Alibaba2026TraceConfig()
    pod_root = _resolve_alibaba_2026_table_root(workload_profile_path, ALIBABA_2026_POD_TABLE)
    pod_files = _alibaba_2026_partition_files(pod_root)
    if not pod_files:
        raise FileNotFoundError(
            f"No Alibaba GPU v2026 pod-hour parquet partitions found under {pod_root}. "
            "Expected day=<day>/hour=<hour>/part-*.parquet."
        )

    max_trace_pos = max(day * 24 + hour for day, hour, _ in pod_files) + 1
    n_intervals = min(max_trace_pos, max_intervals) if max_intervals is not None else max_trace_pos
    pod_files_to_process = [
        (day, hour, path) for day, hour, path in pod_files if day * 24 + hour < n_intervals
    ]
    _print_progress(
        verbose,
        f"Found {len(pod_files_to_process)} pod partitions under {pod_root}; "
        f"building {n_intervals} hourly intervals.",
    )
    interval_hours = 1.0
    interval_index = pd.date_range(
        start=f"{trace_config.trace_anchor_year}-01-01",
        periods=n_intervals,
        freq="h",
        tz="UTC",
    )

    load = np.zeros((len(TASK_TYPES), len(RESOURCES), n_intervals), dtype=np.float64)
    pod_hour_counts = np.zeros((len(TASK_TYPES),), dtype=np.int64)
    source_used_gpu_hours = np.zeros((len(TASK_TYPES),), dtype=np.float64)
    source_gpu_memory_gib_hours = np.zeros((len(TASK_TYPES),), dtype=np.float64)
    paper_validation_used_gpu_hours = np.zeros((len(TASK_TYPES),), dtype=np.float64)

    progress_interval = _progress_interval(len(pod_files_to_process))
    for file_index, (day, hour, path) in enumerate(pod_files_to_process, start=1):
        if (
            file_index == 1
            or file_index == len(pod_files_to_process)
            or file_index % progress_interval == 0
        ):
            _print_progress(
                verbose,
                f"Reading pod partitions: {file_index}/{len(pod_files_to_process)} "
                f"({file_index / len(pod_files_to_process):.0%})",
            )
        pos = day * 24 + hour

        for df in _iter_parquet_frames(path, POD_PROFILE_COLUMNS, parquet_batch_size, read_callback):
            if batch_observer is not None:
                batch_observer(df, day=day, hour=hour, path=path)
            totals = _aggregate_pod_batch(df, day, trace_config)
            load[:, :, pos] += totals.load
            pod_hour_counts += totals.pod_hour_counts
            source_used_gpu_hours += totals.source_used_gpu_hours
            source_gpu_memory_gib_hours += totals.source_gpu_memory_gib_hours
            paper_validation_used_gpu_hours += totals.paper_validation_used_gpu_hours
            del df, totals

    total_load = load.sum(axis=0)
    trace_capacity = np.ones((len(RESOURCES),), dtype=np.float64)
    capacity_source = ["pod_load_quantile"] * len(RESOURCES)

    server_root = (
        Path(server_profile_path)
        if server_profile_path is not None
        else _resolve_alibaba_2026_table_root(workload_profile_path, ALIBABA_2026_SERVER_TABLE)
    )
    if server_root.exists():
        _print_progress(verbose, f"Reading server inventory from {server_root}.")
        server_capacity = _server_trace_capacity(
            server_root,
            n_intervals,
            capacity_quantile,
            verbose=verbose,
            parquet_batch_size=parquet_batch_size,
            read_callback=read_callback,
        )
    else:
        _print_progress(
            verbose,
            f"Server inventory not found under {server_root}; using pod-load capacity estimates.",
        )
        server_capacity = None
    if server_capacity is not None:
        trace_capacity[RESOURCES.index("cpu")] = server_capacity[0]
        trace_capacity[RESOURCES.index("gpu")] = server_capacity[1]
        capacity_source[RESOURCES.index("cpu")] = "server_inventory_quantile"
        capacity_source[RESOURCES.index("gpu")] = "server_inventory_quantile"

    for resource in ("cpu", "gpu", "memory"):
        resource_id = RESOURCES.index(resource)
        if capacity_source[resource_id] == "server_inventory_quantile":
            continue
        values = total_load[resource_id]
        positive = values[np.isfinite(values) & (values > 0)]
        if positive.size:
            trace_capacity[resource_id] = float(np.quantile(positive, capacity_quantile))
        else:
            trace_capacity[resource_id] = 1.0

    # Storage activity is not exposed by the public 2026 pod trace.
    trace_capacity[RESOURCES.index("storage")] = 1.0
    capacity_source[RESOURCES.index("storage")] = "not_observed_idle_only"
    capacity_source[RESOURCES.index("memory")] = "cpu_weighted_memory_util_proxy"

    trace_capacity = np.maximum(trace_capacity, 1e-12)
    resource_hours = load.sum(axis=2) * interval_hours
    annualization = 8760.0 / (n_intervals * interval_hours)
    total_pod_hour_rows = int(pod_hour_counts.sum())
    total_source_used_gpu_hours = float(source_used_gpu_hours.sum())
    classified_source_used_gpu_hours = float(
        source_used_gpu_hours[: TASK_TYPES.index("unclassified")].sum()
    )
    paper_validation_total = float(paper_validation_used_gpu_hours.sum())
    summary_records = []
    for task_type_id, task_type in enumerate(TASK_TYPES):
        summary_records.append(
            {
                "task_type": task_type,
                "is_classified": task_type != "unclassified",
                "pod_hour_rows": int(pod_hour_counts[task_type_id]),
                "pod_hour_rows_share_all": (
                    float(pod_hour_counts[task_type_id]) / total_pod_hour_rows
                    if total_pod_hour_rows > 0
                    else 0.0
                ),
                "source_used_gpu_hours": float(source_used_gpu_hours[task_type_id]),
                "source_used_gpu_hours_share_all": (
                    float(source_used_gpu_hours[task_type_id]) / total_source_used_gpu_hours
                    if total_source_used_gpu_hours > 0
                    else 0.0
                ),
                "source_used_gpu_hours_share_classified": (
                    float(source_used_gpu_hours[task_type_id])
                    / classified_source_used_gpu_hours
                    if task_type != "unclassified" and classified_source_used_gpu_hours > 0
                    else np.nan
                ),
                "paper_validation_used_gpu_hours": float(
                    paper_validation_used_gpu_hours[task_type_id]
                ),
                "paper_validation_used_gpu_hours_share_classified": (
                    float(paper_validation_used_gpu_hours[task_type_id])
                    / paper_validation_total
                    if task_type != "unclassified" and paper_validation_total > 0
                    else np.nan
                ),
                "source_gpu_memory_gib_hours": float(source_gpu_memory_gib_hours[task_type_id]),
                "cpu_core_hours": float(resource_hours[task_type_id, RESOURCES.index("cpu")]),
                "gpu_sm_equivalent_hours": float(resource_hours[task_type_id, RESOURCES.index("gpu")]),
                "memory_proxy_hours": float(resource_hours[task_type_id, RESOURCES.index("memory")]),
                "annualized_cpu_core_hours": float(
                    resource_hours[task_type_id, RESOURCES.index("cpu")] * annualization
                ),
                "annualized_gpu_sm_equivalent_hours": float(
                    resource_hours[task_type_id, RESOURCES.index("gpu")] * annualization
                ),
            }
        )

    profile = WorkloadProfile(
        interval_index=interval_index,
        interval_hours=interval_hours,
        load=load,
        trace_capacity=trace_capacity,
        task_counts=pod_hour_counts,
        task_type_summary=pd.DataFrame(summary_records),
        relative_time=True,
        trace_capacity_source=tuple(capacity_source),
    )
    _print_progress(
        verbose,
        f"Workload profile ready: {profile.n_intervals} intervals, "
        f"{int(pod_hour_counts.sum()):,} active pod-hour rows.",
    )
    return profile



def _component_full_power(country_it_mw: np.ndarray, config: HardwarePowerConfig) -> np.ndarray:
    shares = np.array(
        [
            config.cpu_power_share,
            config.gpu_power_share,
            config.memory_power_share,
            config.storage_power_share,
            config.it_fan_power_share,
        ],
        dtype=float,
    )
    return country_it_mw[:, None] * shares[None, :]


def _resource_capacity(country_it_mw: np.ndarray, config: HardwarePowerConfig) -> np.ndarray:
    component_mw = _component_full_power(country_it_mw, config)
    capacities = np.zeros((len(country_it_mw), len(RESOURCES)), dtype=np.float64)
    capacities[:, RESOURCES.index("cpu")] = (
        component_mw[:, COMPONENTS.index("cpu")] * 1e6 / config.cpu_full_power_w_per_core
    )
    capacities[:, RESOURCES.index("gpu")] = (
        component_mw[:, COMPONENTS.index("gpu")] * 1e6 / config.gpu_full_power_w
    )
    capacities[:, RESOURCES.index("memory")] = (
        component_mw[:, COMPONENTS.index("memory")] * 1e6 / config.memory_power_w_per_gb
    )
    capacities[:, RESOURCES.index("storage")] = (
        component_mw[:, COMPONENTS.index("storage")] * 1e6 / config.storage_power_w_per_tb
    )
    return np.maximum(capacities, 1e-12)


def _build_execution_weights(
    countries: Sequence[str],
    country_it_mw: np.ndarray,
    origin_weights: np.ndarray,
    execution_policy: str,
    inference_origin_fraction: float,
    other_origin_fraction: float,
    task_execution_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> np.ndarray:
    if task_execution_weights is not None:
        return _as_task_weight_table(countries, task_execution_weights)

    capacity_weight = country_it_mw / country_it_mw.sum()
    if execution_policy == "capacity":
        return np.repeat(capacity_weight[None, :], len(TASK_TYPES), axis=0)
    if execution_policy == "origin":
        return origin_weights.copy()
    if execution_policy != "hybrid":
        raise ValueError("execution_policy must be one of: capacity, origin, hybrid.")

    if not (0 <= inference_origin_fraction <= 1):
        raise ValueError("inference_origin_fraction must be in [0, 1].")
    if not (0 <= other_origin_fraction <= 1):
        raise ValueError("other_origin_fraction must be in [0, 1].")

    weights = np.zeros_like(origin_weights)
    weights[TASK_TYPES.index("training")] = capacity_weight
    weights[TASK_TYPES.index("inference")] = (
        inference_origin_fraction * origin_weights[TASK_TYPES.index("inference")]
        + (1 - inference_origin_fraction) * capacity_weight
    )
    weights[TASK_TYPES.index("other")] = (
        other_origin_fraction * origin_weights[TASK_TYPES.index("other")]
        + (1 - other_origin_fraction) * capacity_weight
    )
    weights[TASK_TYPES.index("unclassified")] = capacity_weight
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights


def _scale_workload_to_capacity(
    profile: WorkloadProfile,
    global_resource_capacity: np.ndarray,
    max_resource_utilization: float,
) -> np.ndarray:
    normalized = profile.load / profile.trace_capacity[None, :, None]
    total_normalized = normalized.sum(axis=0)
    scale = np.ones_like(total_normalized)
    positive = total_normalized > 0
    scale[positive] = np.minimum(1.0, max_resource_utilization / total_normalized[positive])
    normalized = normalized * scale[None, :, :]
    return normalized * global_resource_capacity[None, :, None]


def _component_power_timeseries(
    country_it_mw: np.ndarray,
    resource_utilization: np.ndarray,
    config: HardwarePowerConfig,
) -> np.ndarray:
    component_full_mw = _component_full_power(country_it_mw, config)
    power = np.zeros((len(COMPONENTS), len(country_it_mw), resource_utilization.shape[2]), dtype=np.float64)

    cpu_util = resource_utilization[:, RESOURCES.index("cpu"), :]
    gpu_util = resource_utilization[:, RESOURCES.index("gpu"), :]
    memory_util = resource_utilization[:, RESOURCES.index("memory"), :]
    storage_util = resource_utilization[:, RESOURCES.index("storage"), :]

    cpu_idle_fraction = config.cpu_idle_power_w_per_core / config.cpu_full_power_w_per_core
    gpu_idle_fraction = config.gpu_idle_power_w / config.gpu_full_power_w

    power[COMPONENTS.index("cpu")] = component_full_mw[:, COMPONENTS.index("cpu"), None] * (
        cpu_idle_fraction + (1 - cpu_idle_fraction) * cpu_util**config.cpu_dynamic_exponent
    )
    power[COMPONENTS.index("gpu")] = component_full_mw[:, COMPONENTS.index("gpu"), None] * (
        gpu_idle_fraction + (1 - gpu_idle_fraction) * np.log2(1 + gpu_util**config.gpu_dynamic_exponent)
    )
    power[COMPONENTS.index("memory")] = component_full_mw[:, COMPONENTS.index("memory"), None] * (
        config.memory_idle_fraction + (1 - config.memory_idle_fraction) * memory_util
    )
    power[COMPONENTS.index("storage")] = component_full_mw[:, COMPONENTS.index("storage"), None] * (
        config.storage_idle_fraction + (1 - config.storage_idle_fraction) * storage_util
    )

    fan_weight_sum = config.fan_cpu_weight + config.fan_gpu_weight + config.fan_memory_weight
    effective_heat_load = (
        config.fan_cpu_weight * cpu_util
        + config.fan_gpu_weight * gpu_util
        + config.fan_memory_weight * memory_util
    ) / fan_weight_sum
    power[COMPONENTS.index("it_fan")] = component_full_mw[:, COMPONENTS.index("it_fan"), None] * (
        config.it_fan_idle_fraction + (1 - config.it_fan_idle_fraction) * effective_heat_load**3
    )
    return power


def _allocate_energy_to_task_types(
    component_power_mw: np.ndarray,
    country_type_resource_load: np.ndarray,
    resource_capacities: np.ndarray,
    interval_hours: float,
    config: HardwarePowerConfig,
    idle_power_mw: Optional[np.ndarray] = None,
    idle_allocation_rule: str = "hourly_driver",
    return_details: bool = False,
) -> Union[np.ndarray, dict]:
    """Attribute power using normalized drivers and explicit idle rules.

    Legacy callers without a split still receive a total-energy matrix. Detailed
    output requires the physical idle floor supplied by the model.
    """
    if idle_allocation_rule not in IDLE_ALLOCATION_RULES:
        raise ValueError(f"idle_allocation_rule must be one of {IDLE_ALLOCATION_RULES}.")
    if return_details and idle_power_mw is None:
        raise ValueError("Detailed energy attribution requires an explicit idle-power split.")
    if idle_power_mw is None and idle_allocation_rule != "hourly_driver":
        raise ValueError("Alternative idle rules require an explicit idle-power split.")
    idle = component_power_mw if idle_power_mw is None else idle_power_mw
    dynamic = np.maximum(component_power_mw - idle, 0.)
    details = allocate_component_energy(
        idle, dynamic, country_type_resource_load, resource_capacities, interval_hours,
        COMPONENTS, RESOURCES,
        {"cpu": config.fan_cpu_weight, "gpu": config.fan_gpu_weight, "memory": config.fan_memory_weight})
    details["physical_total"] = component_power_mw.sum(axis=2).T * interval_hours
    return details if return_details else details["rules"][idle_allocation_rule]["total"]


def run_workload_component_footprint(
    renewable_energy_policy: str,
    scenarios: Sequence[str],
    years: int = 5,
    countries: Optional[Sequence[str]] = None,
    workload_profile_path: Union[str, Path] = (
        ROOT_DIR / "dataset"
    ),
    server_profile_path: Optional[Union[str, Path]] = None,
    year_start: int = 2026,
    output_dir: Union[str, Path] = ROOT_DIR / "results" / "hourly_workload_resolved_model",
    save_outputs: bool = True,
    verbose: bool = True,
    hardware_config: Optional[HardwarePowerConfig] = None,
    trace_config: Optional[Alibaba2026TraceConfig] = None,
    task_origin_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
    task_execution_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
    execution_policy: str = "capacity",
    inference_origin_fraction: float = 0.75,
    other_origin_fraction: float = 0.50,
    capacity_quantile: float = 0.96,
    max_resource_utilization: float = 1.0,
    pue_scale: float = 1.0,
    ai_capacity_factors: Optional[Mapping[int, float]] = None,
    hourly_carbon_factors_dir: Optional[Union[str, Path]] = ROOT_DIR / "dataset" / "EM-CPNDCNZ",
    hourly_carbon_scope: str = "direct",
    hourly_carbon_fallback_to_annual: bool = True,
    save_hourly_outputs: bool = False,
    max_intervals: Optional[int] = None,
    workload_profile: Optional[WorkloadProfile] = None,
    idle_allocation_rule: str = "hourly_driver",
    grid_data=None,
    carbon_variability_scale: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """
    Compute country-level AI footprint with a workload-driven component model.

    The workload layer uses Alibaba Cluster Trace GPU v2026 pod-hour parquet
    facts as the source of temporal CPU/GPU utilization and public workload mix,
    then scales the relative production-cluster trace to scenario IT capacities.
    Because the public trace covers about six months and has no calendar dates,
    annual energy/resource-hour outputs are annualized to 8760 hours and the
    relative hourly profile is repeated when matching hourly carbon factors.
    A prebuilt ``workload_profile`` can be supplied so callers can reuse a
    single read of the large trace dataset across endpoint and bridge runs.
    """
    if years <= 0:
        raise ValueError("years must be positive.")
    if not np.isfinite(carbon_variability_scale) or carbon_variability_scale < 0:
        raise ValueError("carbon_variability_scale must be finite and non-negative.")
    if idle_allocation_rule not in IDLE_ALLOCATION_RULES:
        raise ValueError(f"idle_allocation_rule must be one of {IDLE_ALLOCATION_RULES}.")
    if not np.isfinite(pue_scale) or pue_scale <= 0:
        raise ValueError("pue_scale must be finite and positive.")
    if hourly_carbon_scope not in SCOPES:
        raise ValueError(f"hourly_carbon_scope must be one of {SCOPES}.")
    if hourly_carbon_factors_dir is None and hourly_carbon_scope != "direct":
        raise ValueError("Annual-only mode has direct factors only; life_cycle requires hourly files.")
    data_year_end = DATA_YEAR_START + IT_CAPACITY.shape[0] - 1
    if year_start < DATA_YEAR_START or year_start + years - 1 > data_year_end:
        raise ValueError(f"Requested years must be within {DATA_YEAR_START}-{data_year_end}.")
    if not (0 < max_resource_utilization <= 1):
        raise ValueError("max_resource_utilization must be in (0, 1].")
    resolved_ai_capacity_factors = resolve_ai_capacity_factors(range(year_start, year_start + years), ai_capacity_factors)

    hardware_config = hardware_config or HardwarePowerConfig()
    hardware_config.validate()
    countries = resolve_countries(countries)
    boundary = build_boundary_metadata(countries, ai_capacity_factors)

    scenarios = list(scenarios)
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be nonempty and unique.")
    for scenario in scenarios:
        if scenario not in SCENARIO_COL_MAP:
            raise ValueError(f"Unknown scenario '{scenario}'. Allowed: {list(SCENARIO_COL_MAP.keys())}")
    unknown_countries = [country for country in countries if country not in IT_RATIO]
    if unknown_countries:
        raise ValueError(f"Unknown countries: {unknown_countries}")

    if renewable_energy_policy not in CARBON_FACTORS:
        raise ValueError("renewable_energy_policy must be one of: CP, NDC, NZ")
    origin_weights = _as_task_weight_table(countries, task_origin_weights)
    country_share = np.array([float(IT_RATIO[country]) for country in countries], dtype=float)

    run_start = time.perf_counter()
    _print_progress(
        verbose,
        f"Starting model: policy={renewable_energy_policy}, scenarios={list(scenarios)}, "
        f"years={year_start}-{year_start + years - 1}, countries={len(countries)}.",
    )
    profile_start = time.perf_counter()
    if workload_profile is None:
        _print_progress(verbose, "Building workload profile from trace partitions.")
        profile = build_workload_profile(
            workload_profile_path=workload_profile_path,
            server_profile_path=server_profile_path,
            capacity_quantile=capacity_quantile,
            trace_config=trace_config,
            max_intervals=max_intervals,
            verbose=verbose,
        )
        profile_stage = "built"
    else:
        if not isinstance(workload_profile, WorkloadProfile):
            raise TypeError("workload_profile must be a WorkloadProfile instance.")
        profile = workload_profile
        profile_stage = "reused"
        _print_progress(verbose, "Reusing prebuilt workload profile; trace files are not read again.")
    annualization_factor = profile.annualization_factor_8760
    _print_progress(
        verbose,
        f"Workload profile {profile_stage} in {time.perf_counter() - profile_start:.1f}s; "
        f"annualization factor={annualization_factor:.4f}.",
    )

    annual_records = []
    component_records = []
    task_demand_records = []
    task_execution_records = []
    task_energy_records = []
    overflow_records = []
    hourly_carbon_frames = []
    grid_audit_records = []
    grid_file_hashes = {}
    task_component_records = []
    idle_pool_records = []
    attribution_records = []
    conservation_records = []
    resource_audit_records = []
    global_resource_audit_records = []
    nonlinearity_records = []
    temporal_alignment_records = []
    temporal_metric_records = []
    trace_seam_records = []
    capacity_parameter_audit_records = []

    calculation_count = len(scenarios) * years
    calculation_index = 0
    for scenario in scenarios:
        scenario_col = SCENARIO_COL_MAP[scenario]
        for output_year_idx in range(years):
            calculation_index += 1
            calculation_start = time.perf_counter()
            year = year_start + output_year_idx
            _print_progress(
                verbose,
                f"Calculation {calculation_index}/{calculation_count} started: "
                f"scenario={scenario}, year={year}.",
            )
            data_year_idx = year - DATA_YEAR_START
            total_data_center_global_it_mw = float(IT_CAPACITY[data_year_idx, scenario_col]) * 1e3
            ai_capacity_factor = resolved_ai_capacity_factors[year]
            global_it_mw = total_data_center_global_it_mw * ai_capacity_factor
            total_data_center_country_it_mw = total_data_center_global_it_mw * country_share
            country_it_mw = global_it_mw * country_share
            _print_progress(
                verbose,
                f"Calculation {calculation_index}/{calculation_count}: IT capacity "
                f"{total_data_center_global_it_mw / 1e3:.3f} GW x "
                f"AI factor {ai_capacity_factor:.6f} = {global_it_mw / 1e3:.3f} GW.",
            )
            resource_capacities = _resource_capacity(country_it_mw, hardware_config)
            global_resource_capacity = resource_capacities.sum(axis=0)
            component_full_mw = _component_full_power(country_it_mw, hardware_config)
            execution_weights = _build_execution_weights(
                countries=countries,
                country_it_mw=country_it_mw,
                origin_weights=origin_weights,
                execution_policy=execution_policy,
                inference_origin_fraction=inference_origin_fraction,
                other_origin_fraction=other_origin_fraction,
                task_execution_weights=task_execution_weights,
            )

            global_type_resource_load = _scale_workload_to_capacity(
                profile=profile,
                global_resource_capacity=global_resource_capacity,
                max_resource_utilization=max_resource_utilization,
            )
            country_type_resource_load = (
                global_type_resource_load[:, None, :, :] * execution_weights[:, :, None, None]
            )
            country_resource_load = country_type_resource_load.sum(axis=0)
            overflow = np.maximum(country_resource_load - resource_capacities[:, :, None], 0.0)
            resource_utilization = np.clip(
                country_resource_load / resource_capacities[:, :, None],
                0.0,
                max_resource_utilization,
            )
            realized_country_resource_load = resource_utilization * resource_capacities[:, :, None]
            country_load_scale = np.divide(
                realized_country_resource_load, country_resource_load,
                out=np.zeros_like(country_resource_load), where=country_resource_load > 0)
            realized_type_resource_load = country_type_resource_load * country_load_scale[None, :, :, :]
            capacity_parameter_audit_records.extend(capacity_parameter_records(
                {"scenario": scenario, "policy": renewable_energy_policy, "year": year},
                countries, country_it_mw, resource_capacities, resource_utilization, hardware_config, profile.trace_capacity))

            component_power_mw = _component_power_timeseries(
                country_it_mw=country_it_mw,
                resource_utilization=resource_utilization,
                config=hardware_config,
            )
            component_it_energy_mwh = (
                component_power_mw.sum(axis=2).T * profile.interval_hours * annualization_factor
            )
            idle_fractions = (
                hardware_config.cpu_idle_power_w_per_core / hardware_config.cpu_full_power_w_per_core,
                hardware_config.gpu_idle_power_w / hardware_config.gpu_full_power_w,
                hardware_config.memory_idle_fraction, hardware_config.storage_idle_fraction,
                hardware_config.it_fan_idle_fraction)
            idle_power_mw = split_component_power(
                component_power_mw, component_full_mw, idle_fractions)[0]
            allocation_details = _allocate_energy_to_task_types(
                component_power_mw=component_power_mw,
                country_type_resource_load=realized_type_resource_load,
                resource_capacities=resource_capacities,
                interval_hours=profile.interval_hours * annualization_factor,
                config=hardware_config,
                idle_power_mw=idle_power_mw, idle_allocation_rule=idle_allocation_rule,
                return_details=True,
            )
            selected_allocation = allocation_details["rules"][idle_allocation_rule]
            task_type_component_it_mwh = selected_allocation["total"]
            component_idle_mwh = allocation_details["physical_idle"]
            component_dynamic_mwh = allocation_details["physical_dynamic"]
            if not np.allclose(component_idle_mwh + component_dynamic_mwh, component_it_energy_mwh,
                               rtol=1e-10, atol=1e-6):
                raise ValueError("Idle and dynamic component energy do not close to total power.")

            pue = np.array([PUE[country][data_year_idx, scenario_col] for country in countries], dtype=float) * pue_scale
            annual_emission_kg_per_mwh = np.array(
                [
                    get_carbon_factor(
                        renewable_energy_policy,
                        country,
                        year,
                    )
                    for country in countries
                ],
                dtype=float,
            )

            country_it_energy_mwh = component_it_energy_mwh.sum(axis=1)
            facility_energy_mwh = country_it_energy_mwh * pue
            context = {"scenario": scenario, "policy": renewable_energy_policy, "year": year}
            matrix_rows, pool_rows, check_rows = allocation_records(
                allocation_details, countries, COMPONENTS, pue, context, idle_allocation_rule)
            task_component_records.extend(matrix_rows)
            idle_pool_records.extend(pool_rows)
            conservation_records.extend(check_rows)
            for row in allocation_details["metadata"]:
                metadata = {**context, **row, "country": countries[row["country_id"]]}
                metadata["selected_idle_allocation_rule"] = idle_allocation_rule
                metadata["idle_fallback_applied_in_selected_rule"] = (
                    idle_allocation_rule != "separate_pool" and row["zero_driver_intervals"] > 0)
                metadata.pop("country_id")
                attribution_records.append(metadata)
            mean_power_mw = _component_power_timeseries(
                country_it_mw, resource_utilization.mean(axis=2, keepdims=True), hardware_config)
            mean_energy_mwh = mean_power_mw[:, :, 0].T * 8760.
            if hourly_carbon_factors_dir is None:
                _print_progress(
                    verbose,
                    f"Calculation {calculation_index}/{calculation_count}: using annual carbon factors.",
                )
                carbon_tco2 = facility_energy_mwh * annual_emission_kg_per_mwh / 1000.0
                hourly_timestamps = calendar_hours(year)
                hourly_emission_kg_per_mwh = np.repeat(
                    annual_emission_kg_per_mwh[:, None], len(hourly_timestamps), axis=1)
                used_hourly_factors = np.zeros(len(countries), dtype=bool)
            else:
                _print_progress(
                    verbose,
                    f"Calculation {calculation_index}/{calculation_count}: "
                    "loading and aligning hourly carbon factors.",
                )
                factor_loader = grid_data.load if grid_data is not None else _load_hourly_carbon_factors
                hourly_timestamps, hourly_emission_kg_per_mwh, used_hourly_factors = factor_loader(
                    countries=countries,
                    renewable_energy_policy=renewable_energy_policy,
                    year=year,
                    hourly_carbon_factors_dir=hourly_carbon_factors_dir,
                    hourly_carbon_scope=hourly_carbon_scope,
                    hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
                )

            if carbon_variability_scale != 1.0:
                means = hourly_emission_kg_per_mwh.mean(axis=1, keepdims=True)
                transformed = np.maximum(means + carbon_variability_scale * (hourly_emission_kg_per_mwh - means), 0.)
                hourly_emission_kg_per_mwh = np.divide(transformed * means, transformed.mean(axis=1, keepdims=True),
                                                      out=np.zeros_like(transformed), where=transformed.mean(axis=1, keepdims=True) > 0)

            hourly_facility_energy_mwh, alignment, aligned_total, annual_shape_scale = _hourly_facility_energy(
                component_power_mw=component_power_mw, pue=pue,
                annual_facility_energy_mwh=facility_energy_mwh,
                interval_index=profile.interval_index, interval_hours=profile.interval_hours,
                target_timestamps=hourly_timestamps, relative_trace=profile.relative_time,
                return_alignment=True,
            )
            hourly_carbon_tco2 = hourly_facility_energy_mwh * hourly_emission_kg_per_mwh / 1000.0
            carbon_tco2 = hourly_carbon_tco2.sum(axis=1)
            if hourly_carbon_factors_dir is not None:
                hourly_carbon_frames.append(
                    _hourly_carbon_frame(
                        scenario=scenario,
                        year=year,
                        countries=countries,
                        timestamps=hourly_timestamps,
                        hourly_facility_energy_mwh=hourly_facility_energy_mwh,
                        hourly_emission_kg_per_mwh=hourly_emission_kg_per_mwh,
                        used_hourly_factors=used_hourly_factors,
                        alignment=alignment,
                    )
                )

            seam_flags = alignment["trace_repeat_seam_before"].to_numpy(dtype=bool)
            alignment_flags = alignment["alignment_discontinuity_before"].to_numpy(dtype=bool)
            shape = shared_shape_diagnostics(hourly_facility_energy_mwh)
            timing_context = {**context, "hours": len(hourly_timestamps),
                              "calendar_status": calendar_status(hourly_timestamps, year),
                              "power_unit": "MW", "ramp_unit": "MW h^-1",
                              "ramp_definition": "Maximum absolute hourly ramp",
                              "execution_policy": execution_policy, **shape}
            # Align the configured idle power using the same map and TOTAL
            # power normalization, rather than normalizing idle independently.
            aligned_idle = _align_interval_energy_to_target_hours(
                idle_power_mw.sum(axis=0) * pue[:, None] * profile.interval_hours,
                profile.interval_index, hourly_timestamps, relative_trace=profile.relative_time,
                preserve_source_total=False)
            aligned_idle *= np.nan_to_num(annual_shape_scale, nan=0.)[:, None]
            for country_id, country in enumerate(countries):
                values = hourly_facility_energy_mwh[country_id]
                country_context = {**timing_context, "scope": "country", "country": country}
                temporal_metric_records.append({
                    **country_context, **power_metrics(values, seam_flags, alignment_flags),
                    "configured_facility_idle_floor_mw": float(idle_power_mw[:, country_id, 0].sum() * pue[country_id]),
                    "aligned_mean_idle_power_mw": float(aligned_idle[country_id].mean()),
                    "aligned_idle_energy_share": float(aligned_idle[country_id].sum() / facility_energy_mwh[country_id]) if facility_energy_mwh[country_id] else np.nan,
                    "minimum_aligned_dynamic_power_mw": float((values - aligned_idle[country_id]).min()),
                    "shape_interpretation": "shared_trace_scaled_and_capacity_clipped_not_independent_country_prediction",
                })
                temporal_alignment_records.append({
                    **context, "country": country, "trace_hours": profile.trace_hours,
                    "trace_intervals": profile.n_intervals, "relative_trace": profile.relative_time,
                    "target_calendar_hours": len(hourly_timestamps),
                    "expected_calendar_hours": len(calendar_hours(year)),
                    "calendar_status": calendar_status(hourly_timestamps, year),
                    "alignment_method": alignment["alignment_method"].iloc[0],
                    "source_hour_count": int(alignment["source_hour_count"].iloc[0]),
                    "source_missing_hour_edges": int(alignment["source_missing_hour_edges"].iloc[0]),
                    "repeat_count_including_partial": int(alignment["source_repeat_cycle"].max() + 1),
                    "repeat_seam_count": int(seam_flags.sum()),
                    "alignment_discontinuity_count": int(alignment_flags.sum()),
                    "leap_day_copied_hours": int(alignment["leap_day_copied"].sum()),
                    "aligned_energy_before_normalization_mwh": float(aligned_total[country_id]),
                    "annual_shape_normalization_factor": float(annual_shape_scale[country_id]),
                    "annual_facility_energy_mwh": float(facility_energy_mwh[country_id]),
                    "energy_closure_residual_mwh": float(values.sum() - facility_energy_mwh[country_id]),
                    "energy_convention": "8760_hour_annualized_trace_energy_preserved_over_CI_calendar",
                })
                trace_seam_records.extend(seam_event_records(
                    values, hourly_timestamps, seam_flags, alignment_flags, country_context, alignment=alignment))
            aggregate_values = hourly_facility_energy_mwh.sum(axis=0)
            sum_country_peaks = float(hourly_facility_energy_mwh.max(axis=1).sum())
            aggregate_context = {**timing_context, "scope": "aggregate", "country": "ALL"}
            temporal_metric_records.append({
                **aggregate_context, **power_metrics(aggregate_values, seam_flags, alignment_flags),
                "peak_construction": "sum_country_power_at_same_UTC_hour_then_max",
                "sum_of_individual_country_peaks_mw": sum_country_peaks,
                "peak_coincidence_ratio": float(aggregate_values.max() / sum_country_peaks) if sum_country_peaks else np.nan,
                "configured_facility_idle_floor_mw": float((idle_power_mw[:, :, 0].sum(axis=0) * pue).sum()),
                "aligned_mean_idle_power_mw": float(aligned_idle.sum(axis=0).mean()),
                "aligned_idle_energy_share": float(aligned_idle.sum() / facility_energy_mwh.sum()) if facility_energy_mwh.sum() else np.nan,
                "minimum_aligned_dynamic_power_mw": float((aggregate_values - aligned_idle.sum(axis=0)).min()),
            })
            trace_seam_records.extend(seam_event_records(
                aggregate_values, hourly_timestamps, seam_flags, alignment_flags, aggregate_context, alignment=alignment))

            for country_id, country in enumerate(countries):
                source_path = (find_hourly_file(hourly_carbon_factors_dir, country, renewable_energy_policy, year)
                               if hourly_carbon_factors_dir is not None else None)
                if source_path is not None and source_path not in grid_file_hashes:
                    grid_file_hashes[source_path] = grid_data.fingerprint(source_path) if grid_data is not None else sha256_file(source_path)
                source = ("annual_configured" if hourly_carbon_factors_dir is None else
                          "hourly" if used_hourly_factors[country_id] else "annual_fallback")
                diagnostics = matching_diagnostics(hourly_facility_energy_mwh[country_id],
                                                   hourly_emission_kg_per_mwh[country_id],
                                                   facility_energy_mwh[country_id])
                comparable = hourly_carbon_scope == "direct"
                target = annual_emission_kg_per_mwh[country_id] if comparable else np.nan
                status = calendar_status(hourly_timestamps, year)
                grid_audit_records.append({
                    "scenario": scenario, "policy": renewable_energy_policy,
                    "year": year, "country": country, "scope": hourly_carbon_scope,
                    "carbon_factor_source": source,
                    "carbon_variability_scale": carbon_variability_scale,
                    "fallback_reason": "missing_hourly_file" if source == "annual_fallback" else "",
                    "input_file": str(source_path.resolve()) if source_path else "",
                    "input_sha256": grid_file_hashes.get(source_path, ""),
                    "timezone": "UTC", "hours": len(hourly_timestamps),
                    "expected_calendar_hours": len(calendar_hours(year)), "calendar_status": status,
                    "eligible_for_full_calendar_hourly_claim": source == "hourly" and status == "full_calendar",
                    "annual_reference_scope": "direct",
                    "annual_mean_comparison_available": comparable,
                    "annual_target_ci_kg_per_mwh": target,
                    "annual_mean_residual_kg_per_mwh": diagnostics["hourly_mean_ci_kg_per_mwh"] - target,
                    "annual_energy_convention": "8760_hour_annualized_trace_energy_preserved",
                    **diagnostics,
                })

            origin_resource_hours = (
                global_type_resource_load.sum(axis=2)[:, None, :]
                * origin_weights[:, :, None]
                * profile.interval_hours
                * annualization_factor
            )
            execution_resource_hours = (
                global_type_resource_load.sum(axis=2)[:, None, :]
                * execution_weights[:, :, None]
                * profile.interval_hours
                * annualization_factor
            )
            annual_interval_hours = profile.interval_hours * annualization_factor
            uncapped_global_load = (profile.load / profile.trace_capacity[None, :, None]
                                    * global_resource_capacity[None, :, None])
            if not np.allclose(realized_type_resource_load.sum(axis=0), realized_country_resource_load,
                               rtol=1e-10, atol=1e-8):
                raise ValueError("Realized task loads do not sum to the country resource load.")
            for resource_id, resource in enumerate(RESOURCES):
                before = float(uncapped_global_load[:, resource_id, :].sum() * annual_interval_hours)
                after = float(global_type_resource_load[:, resource_id, :].sum() * annual_interval_hours)
                origin = float(origin_resource_hours[:, :, resource_id].sum())
                executed = float(execution_resource_hours[:, :, resource_id].sum())
                realized = float(realized_country_resource_load[:, resource_id, :].sum() * annual_interval_hours)
                if not np.allclose([origin, executed], after, rtol=1e-10, atol=1e-6):
                    raise ValueError(f"Resource allocation does not conserve {resource} hours.")
                global_resource_audit_records.append({
                    **context, "resource": resource,
                    "source_trace_resource_hours": float(profile.load[:, resource_id, :].sum() * annual_interval_hours),
                    "normalized_source_activity_hours": float((profile.load[:, resource_id, :] / profile.trace_capacity[resource_id]).sum() * annual_interval_hours),
                    "before_global_cap_resource_hours": before, "after_global_cap_resource_hours": after,
                    "global_clipped_resource_hours": max(0., before - after),
                    "origin_resource_hours": origin, "execution_resource_hours": executed,
                    "realized_resource_hours": realized, "country_clipped_resource_hours": max(0., executed - realized),
                    "normalized_realized_activity_hours": realized / global_resource_capacity[resource_id],
                    "origin_execution_residual_resource_hours": origin - executed,
                    "status": "PASS", "source_and_model_units_distinguished": True,
                })

            for country_id, country in enumerate(countries):
                annual_records.append(
                    {
                        "scenario": scenario,
                        "year": year,
                        "country": country,
                        "total_data_center_it_mw": total_data_center_country_it_mw[country_id],
                        "ai_capacity_factor": ai_capacity_factor,
                        "installed_it_mw": country_it_mw[country_id],
                        "it_energy_mwh": country_it_energy_mwh[country_id],
                        "idle_allocation_rule": idle_allocation_rule,
                        "idle_it_energy_mwh": float(component_idle_mwh[country_id].sum()),
                        "dynamic_it_energy_mwh": float(component_dynamic_mwh[country_id].sum()),
                        "idle_facility_energy_mwh": float(component_idle_mwh[country_id].sum() * pue[country_id]),
                        "dynamic_facility_energy_mwh": float(component_dynamic_mwh[country_id].sum() * pue[country_id]),
                        "unallocated_idle_it_energy_mwh": float(selected_allocation["pool"][country_id].sum()),
                        "unallocated_idle_facility_energy_mwh": float(selected_allocation["pool"][country_id].sum() * pue[country_id]),
                        "facility_energy_mwh": facility_energy_mwh[country_id],
                        "power_twh": facility_energy_mwh[country_id] / 1e6,
                        "carbon_tco2": carbon_tco2[country_id],
                        "carbon_mtco2": carbon_tco2[country_id] / 1e6,
                        "avg_cpu_utilization": resource_utilization[country_id, RESOURCES.index("cpu")].mean(),
                        "avg_gpu_utilization": resource_utilization[country_id, RESOURCES.index("gpu")].mean(),
                        "avg_memory_utilization": resource_utilization[country_id, RESOURCES.index("memory")].mean(),
                        "avg_storage_utilization": resource_utilization[country_id, RESOURCES.index("storage")].mean(),
                        "peak_cpu_utilization": resource_utilization[country_id, RESOURCES.index("cpu")].max(),
                        "peak_gpu_utilization": resource_utilization[country_id, RESOURCES.index("gpu")].max(),
                        "peak_memory_utilization": resource_utilization[country_id, RESOURCES.index("memory")].max(),
                        "peak_storage_utilization": resource_utilization[country_id, RESOURCES.index("storage")].max(),
                    }
                )

                for component_id, component in enumerate(COMPONENTS):
                    component_records.append(
                        {
                            "scenario": scenario,
                            "year": year,
                            "country": country,
                            "component": component,
                            "policy": renewable_energy_policy,
                            "idle_allocation_rule": idle_allocation_rule,
                            "full_power_mw": component_full_mw[country_id, component_id],
                            "idle_fraction": idle_fractions[component_id],
                            "it_energy_mwh": component_it_energy_mwh[country_id, component_id],
                            "idle_it_energy_mwh": component_idle_mwh[country_id, component_id],
                            "dynamic_it_energy_mwh": component_dynamic_mwh[country_id, component_id],
                            "idle_facility_energy_mwh": component_idle_mwh[country_id, component_id] * pue[country_id],
                            "dynamic_facility_energy_mwh": component_dynamic_mwh[country_id, component_id] * pue[country_id],
                            "task_assigned_it_energy_mwh": task_type_component_it_mwh[:, country_id, component_id].sum(),
                            "unallocated_idle_it_energy_mwh": selected_allocation["pool"][country_id, component_id],
                            "facility_energy_mwh": component_it_energy_mwh[country_id, component_id] * pue[country_id],
                            "facility_energy_twh": component_it_energy_mwh[country_id, component_id] * pue[country_id] / 1e6,
                        }
                    )
                    nonlinearity_records.append({
                        **context, "country": country, "component": component,
                        "hourly_it_energy_mwh": component_it_energy_mwh[country_id, component_id],
                        "flat_realized_mean_it_energy_mwh": mean_energy_mwh[country_id, component_id],
                        "time_variation_effect_it_energy_mwh": component_it_energy_mwh[country_id, component_id] - mean_energy_mwh[country_id, component_id],
                        "time_variation_effect_facility_energy_mwh": (component_it_energy_mwh[country_id, component_id] - mean_energy_mwh[country_id, component_id]) * pue[country_id],
                        **power_curve_metadata(hardware_config)[component],
                        "preserved_quantity": "post_clipping_mean_resource_utilization",
                        "idle_energy_unchanged_in_flat_control": True,
                    })

                for resource_id, resource in enumerate(RESOURCES):
                    requested_hours = float(country_resource_load[country_id, resource_id].sum() * annual_interval_hours)
                    realized_hours = float(realized_country_resource_load[country_id, resource_id].sum() * annual_interval_hours)
                    resource_audit_records.append({
                        **context, "country": country, "resource": resource,
                        "requested_resource_hours": requested_hours, "realized_resource_hours": realized_hours,
                        "country_clipped_resource_hours": max(0., requested_hours - realized_hours),
                        "mean_realized_utilization": float(resource_utilization[country_id, resource_id].mean()),
                        "task_sum_residual_resource_hours": float((realized_type_resource_load[:, country_id, resource_id].sum() - realized_country_resource_load[country_id, resource_id].sum()) * annual_interval_hours),
                        "status": "PASS",
                    })
                    overflow_records.append(
                        {
                            "scenario": scenario,
                            "year": year,
                            "country": country,
                            "resource": resource,
                            "overflow_resource_hours": overflow[country_id, resource_id].sum()
                            * profile.interval_hours
                            * annualization_factor,
                        }
                    )

                for task_type_id, task_type in enumerate(TASK_TYPES):
                    for resource_id, resource in enumerate(RESOURCES):
                        task_demand_records.append(
                            {
                                "scenario": scenario,
                                "year": year,
                                "country": country,
                                "task_type": task_type,
                                "resource": resource,
                                "resource_hours": origin_resource_hours[task_type_id, country_id, resource_id],
                            }
                        )
                        task_execution_records.append(
                            {
                                "scenario": scenario,
                                "year": year,
                                "country": country,
                                "task_type": task_type,
                                "resource": resource,
                                "resource_hours": execution_resource_hours[task_type_id, country_id, resource_id],
                                "realized_resource_hours": float(realized_type_resource_load[task_type_id, country_id, resource_id].sum() * annual_interval_hours),
                            }
                        )

                    type_it_energy = task_type_component_it_mwh[task_type_id, country_id].sum()
                    task_energy_records.append(
                        {
                            "scenario": scenario,
                            "year": year,
                            "country": country,
                            "task_type": task_type,
                            "idle_allocation_rule": idle_allocation_rule,
                            "it_energy_mwh": type_it_energy,
                            "idle_it_energy_mwh": selected_allocation["idle"][task_type_id, country_id].sum(),
                            "dynamic_it_energy_mwh": selected_allocation["dynamic"][task_type_id, country_id].sum(),
                            "idle_facility_energy_mwh": selected_allocation["idle"][task_type_id, country_id].sum() * pue[country_id],
                            "dynamic_facility_energy_mwh": selected_allocation["dynamic"][task_type_id, country_id].sum() * pue[country_id],
                            "task_share_of_system_it_energy_pct": type_it_energy / country_it_energy_mwh[country_id] * 100. if country_it_energy_mwh[country_id] > 0 else np.nan,
                            "facility_energy_mwh": type_it_energy * pue[country_id],
                            "facility_energy_twh": type_it_energy * pue[country_id] / 1e6,
                        }
                    )

            _print_progress(
                verbose,
                f"Calculation {calculation_index}/{calculation_count} completed in "
                f"{time.perf_counter() - calculation_start:.1f}s: "
                f"scenario={scenario}, year={year}.",
            )

    _print_progress(verbose, "Assembling result tables.")
    task_matrix = pd.DataFrame(task_component_records)
    idle_pools = pd.DataFrame(idle_pool_records)
    matrix_keys = ["scenario", "policy", "year", "country", "idle_allocation_rule"]
    enriched_matrix, _, _ = allocation_share_tables(task_matrix, idle_pools, matrix_keys)
    extra_columns = [column for column in enriched_matrix if column not in task_matrix]
    task_matrix = task_matrix.merge(enriched_matrix[matrix_keys + ["task_type", "component"] + extra_columns],
                                    on=matrix_keys + ["task_type", "component"], validate="one_to_one")
    tag = "-".join([scenario.replace(" ", "") for scenario in scenarios]) or "None"
    results = {
        "annual_summary": pd.DataFrame(annual_records),
        "grid_carbon_audit": pd.DataFrame(grid_audit_records),
        "temporal_alignment": pd.DataFrame(temporal_alignment_records),
        "temporal_metrics": pd.DataFrame(temporal_metric_records),
        "trace_seams": pd.DataFrame(trace_seam_records),
        "capacity_parameter_audit": pd.DataFrame(capacity_parameter_audit_records),
        "hardware_parameters": hardware_parameter_table(hardware_config),
        "parameter_dependencies": parameter_dependency_table(hardware_config),
        "component_energy": pd.DataFrame(component_records),
        "task_demand": pd.DataFrame(task_demand_records),
        "task_execution": pd.DataFrame(task_execution_records),
        "task_type_energy": pd.DataFrame(task_energy_records),
        "task_component_energy": task_matrix.loc[task_matrix["is_selected_rule"]].reset_index(drop=True),
        "task_component_energy_all_rules": task_matrix,
        "idle_pool": idle_pools,
        "allocation_audit": pd.DataFrame(attribution_records),
        "energy_conservation": pd.DataFrame(conservation_records),
        "resource_audit": pd.DataFrame(resource_audit_records),
        "global_resource_audit": pd.DataFrame(global_resource_audit_records),
        "power_nonlinearity": pd.DataFrame(nonlinearity_records),
        "resource_proxy_metadata": resource_proxy_metadata(),
        "capacity_overflow": pd.DataFrame(overflow_records),
        "hourly_carbon": (
            pd.concat(hourly_carbon_frames, ignore_index=True)
            if hourly_carbon_frames
            else pd.DataFrame(columns=HOURLY_CARBON_COLUMNS)
        ),
        "workload_profile_summary": profile.task_type_summary,
        "workload_trace_metadata": pd.DataFrame(
            [
                {
                    "source": "Alibaba Cluster Trace GPU v2026",
                    "interval_hours": profile.interval_hours,
                    "trace_intervals": profile.n_intervals,
                    "trace_hours": profile.trace_hours,
                    "annualization_factor_8760": annualization_factor,
                    "relative_time": profile.relative_time,
                }
            ]
        ),
        "trace_resource_capacity": pd.DataFrame(
            {
                "resource": RESOURCES,
                "trace_capacity_at_quantile": profile.trace_capacity,
                "capacity_quantile": capacity_quantile,
                "capacity_source": profile.trace_capacity_source,
            }
        ),
    }

    selected_summary = results["annual_summary"].groupby(["scenario", "year"], as_index=False)[
        ["total_data_center_it_mw", "installed_it_mw", "it_energy_mwh",
         "facility_energy_mwh", "power_twh", "carbon_tco2", "carbon_mtco2"]
        + ["idle_it_energy_mwh", "dynamic_it_energy_mwh", "idle_facility_energy_mwh",
           "dynamic_facility_energy_mwh", "unallocated_idle_it_energy_mwh", "unallocated_idle_facility_energy_mwh"]
    ].sum()
    selected_summary.insert(1, "policy", renewable_energy_policy)
    results["selected_countries_summary"] = annotate_selected_aggregate(selected_summary, boundary)
    results["accounting_boundary"] = boundary
    results["mechanism_conservation"] = mechanism_conservation_table(results, renewable_energy_policy)

    if save_outputs:
        output_path = Path(output_dir)
        _print_progress(verbose, f"Saving result tables to {output_path.resolve()}.")
        os.makedirs(output_path, exist_ok=True)
        results["selected_countries_summary"].to_csv(
            output_path / f"Selected_Countries_Annual_Summary_{renewable_energy_policy}_{tag}.csv", index=False,
        )
        boundary.to_csv(output_path / f"Accounting_Boundary_{renewable_energy_policy}_{tag}.csv", index=False)
        results["grid_carbon_audit"].to_csv(
            output_path / f"Grid_Carbon_Audit_{renewable_energy_policy}_{tag}.csv", index=False)
        results["annual_summary"].to_csv(
            output_path / f"Country_Annual_Summary_{renewable_energy_policy}_{tag}.csv",
            index=False,
        )
        results["component_energy"].to_csv(
            output_path / f"Country_Component_Energy_{renewable_energy_policy}_{tag}.csv",
            index=False,
        )
        for table_name in ("task_component_energy", "task_component_energy_all_rules", "idle_pool",
                           "allocation_audit", "energy_conservation", "resource_audit",
                           "global_resource_audit", "power_nonlinearity", "resource_proxy_metadata"):
            results[table_name].to_csv(output_path / f"{table_name}_{renewable_energy_policy}_{tag}.csv", index=False)
        for table_name in ("temporal_alignment", "temporal_metrics", "trace_seams"):
            results[table_name].to_csv(output_path / f"{table_name}_{renewable_energy_policy}_{tag}.csv", index=False)
        for table_name in ("capacity_parameter_audit", "hardware_parameters", "parameter_dependencies", "mechanism_conservation"):
            results[table_name].to_csv(output_path / f"{table_name}_{renewable_energy_policy}_{tag}.csv", index=False)
        results["task_demand"].to_csv(
            output_path / f"Country_Task_Demand_{renewable_energy_policy}_{tag}.csv",
            index=False,
        )
        results["task_execution"].to_csv(
            output_path / f"Country_Task_Execution_{renewable_energy_policy}_{tag}.csv",
            index=False,
        )
        results["task_type_energy"].to_csv(
            output_path / f"Country_TaskType_Energy_{renewable_energy_policy}_{tag}.csv",
            index=False,
        )
        results["capacity_overflow"].to_csv(
            output_path / f"Country_Capacity_Overflow_{renewable_energy_policy}_{tag}.csv",
            index=False,
        )
        if save_hourly_outputs:
            _print_progress(
                verbose,
                f"Saving hourly carbon table ({len(results['hourly_carbon']):,} rows).",
            )
            results["hourly_carbon"].to_csv(
                output_path / f"Country_Hourly_Carbon_{renewable_energy_policy}_{tag}.csv",
                index=False,
            )
        results["workload_profile_summary"].to_csv(output_path / "Workload_Profile_Summary.csv", index=False)
        results["trace_resource_capacity"].to_csv(output_path / "Trace_Resource_Capacity.csv", index=False)
        _print_progress(verbose, "All requested output tables have been saved.")

    if verbose:
        totals = results["annual_summary"].groupby(["scenario", "year"], as_index=False)[
            ["power_twh", "carbon_mtco2"]
        ].sum()
        print(totals.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        if save_outputs:
            print("Saved workload component results to:", os.path.abspath(output_dir))
        _print_progress(verbose, f"Model completed in {time.perf_counter() - run_start:.1f}s.")

    return results


if __name__ == "__main__":
    run_workload_component_footprint(
        renewable_energy_policy="CP",
        scenarios=["Base"],
        years=6,
        year_start=2025,
    )
