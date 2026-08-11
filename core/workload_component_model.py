import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.Factors import (  # noqa: E402
    PUE,
    WUE,
    carbon_emissions_factors_CP,
    carbon_emissions_factors_NDC,
    carbon_emissions_factors_NZ,
    grid_water_factors_CP,
    grid_water_factors_NDC,
    grid_water_factors_NZ,
)
from dataset.Installed_capacity_data import (  # noqa: E402
    countries as DEFAULT_COUNTRIES,
    it_capacity,
    it_ratio,
    total_ratio,
)


TASK_COLUMNS = (
    "job_id",
    "start_time",
    "end_time",
    "start_dt",
    "duration_min",
    "cpu_usage",
    "gpu_wrk_util",
    "avg_mem_gb",
    "avg_gpu_wrk_mem_gb",
    "bandwidth_gb",
    "weekday_name",
    "weekday_num",
)
TASK_TYPES = ("training", "inference", "cpu_data")
RESOURCES = ("cpu", "gpu", "memory", "storage")
COMPONENTS = ("cpu", "gpu", "memory", "storage", "it_fan")
DATA_YEAR_START = 2025
SCENARIO_COL_MAP = {
    "Base": 0,
    "Lift-Off": 1,
    "High Efficiency": 2,
    "Headwinds": 3,
}
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
)


@dataclass(frozen=True)
class TaskClassificationConfig:
    """Rule-based classifier for task rows without explicit workload labels."""

    min_gpu_for_gpu_task: float = 0.05
    training_min_duration_min: float = 60.0
    training_min_gpus: float = 0.5
    training_min_gpu_mem_gb: float = 8.0
    long_gpu_training_duration_min: float = 120.0


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
        if np.any(shares < 0):
            raise ValueError("Hardware component power shares must be non-negative.")
        if not np.isclose(shares.sum(), 1.0, atol=1e-8):
            raise ValueError(f"Hardware component power shares must sum to 1.0, got {shares.sum():.6f}.")
        if self.cpu_full_power_w_per_core <= 0 or self.gpu_full_power_w <= 0:
            raise ValueError("CPU and GPU full-load unit power values must be positive.")
        if self.memory_power_w_per_gb <= 0 or self.storage_power_w_per_tb <= 0:
            raise ValueError("Memory and storage unit power values must be positive.")


@dataclass
class WorkloadProfile:
    interval_index: pd.DatetimeIndex
    interval_hours: float
    load: np.ndarray
    trace_capacity: np.ndarray
    task_counts: np.ndarray
    task_type_summary: pd.DataFrame

    @property
    def n_intervals(self) -> int:
        return len(self.interval_index)


def _policy_factors(renewable_energy_policy: str):
    if renewable_energy_policy == "CP":
        return carbon_emissions_factors_CP, grid_water_factors_CP
    if renewable_energy_policy == "NDC":
        return carbon_emissions_factors_NDC, grid_water_factors_NDC
    if renewable_energy_policy == "NZ":
        return carbon_emissions_factors_NZ, grid_water_factors_NZ
    raise ValueError("renewable_energy_policy must be one of: CP, NDC, NZ")


def _standard_hourly_index(year: int, hours: int = 8760) -> pd.DatetimeIndex:
    return pd.date_range(start=f"{year}-01-01", periods=hours, freq="h", tz="UTC")


def _hourly_carbon_dir_name(country: str) -> str:
    return HOURLY_CARBON_COUNTRY_DIRS.get(country, country)


def _hourly_carbon_intensity_column(columns: Sequence[str], scope: str) -> str:
    if scope == "direct":
        token = "direct"
    elif scope == "life_cycle":
        token = "life cycle"
    else:
        raise ValueError("hourly_carbon_scope must be one of: direct, life_cycle.")

    matches = [
        column
        for column in columns
        if "carbon intensity" in column.lower() and token in column.lower()
    ]
    if not matches:
        raise ValueError(f"Could not find hourly carbon intensity column for scope '{scope}'.")
    return matches[0]


def _read_hourly_carbon_series(
    country: str,
    renewable_energy_policy: str,
    year: int,
    hourly_carbon_factors_dir: Union[str, Path],
    hourly_carbon_scope: str,
):
    country_dir = Path(hourly_carbon_factors_dir) / _hourly_carbon_dir_name(country)
    if not country_dir.exists():
        return None

    matches = sorted(country_dir.glob(f"*-{renewable_energy_policy}-{year}-hourly.csv"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Expected one hourly carbon CSV for {country} {year}, found {len(matches)}.")

    hourly_df = pd.read_csv(matches[0])
    timestamp_columns = [column for column in hourly_df.columns if "datetime" in column.lower()]
    if not timestamp_columns:
        raise ValueError(f"Hourly carbon CSV has no datetime column: {matches[0]}")

    intensity_column = _hourly_carbon_intensity_column(hourly_df.columns, hourly_carbon_scope)
    timestamps = pd.to_datetime(hourly_df[timestamp_columns[0]], utc=True, errors="coerce")
    factors = pd.to_numeric(hourly_df[intensity_column], errors="coerce").to_numpy(dtype=np.float64)

    if timestamps.isna().any():
        raise ValueError(f"Hourly carbon CSV contains invalid timestamps: {matches[0]}")
    if len(factors) == 0 or not np.all(np.isfinite(factors)):
        raise ValueError(f"Hourly carbon CSV contains invalid carbon factors: {matches[0]}")

    return pd.DatetimeIndex(timestamps), factors


def _load_hourly_carbon_factors(
    countries: Sequence[str],
    renewable_energy_policy: str,
    year: int,
    year_idx: int,
    annual_emission_factors: Mapping[str, Sequence[float]],
    hourly_carbon_factors_dir: Union[str, Path],
    hourly_carbon_scope: str,
    hourly_carbon_fallback_to_annual: bool,
):
    loaded = {}
    reference_timestamps = None

    for country in countries:
        series = _read_hourly_carbon_series(
            country=country,
            renewable_energy_policy=renewable_energy_policy,
            year=year,
            hourly_carbon_factors_dir=hourly_carbon_factors_dir,
            hourly_carbon_scope=hourly_carbon_scope,
        )
        if series is None:
            continue

        timestamps, factors = series
        if reference_timestamps is None:
            reference_timestamps = timestamps
        elif not timestamps.equals(reference_timestamps):
            reindexed = pd.Series(factors, index=timestamps).reindex(reference_timestamps)
            if reindexed.isna().any():
                raise ValueError(f"Hourly carbon timestamps for {country} do not match the reference year.")
            factors = reindexed.to_numpy(dtype=np.float64)
        loaded[country] = factors

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
        factors[country_id] = float(annual_emission_factors[country][year_idx])

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


def _fill_calendar_indexer(source_keys: pd.MultiIndex, target_keys: pd.MultiIndex, indexer: np.ndarray) -> np.ndarray:
    if np.all(indexer >= 0):
        return indexer

    lookup = {key: pos for pos, key in enumerate(source_keys)}
    filled = indexer.copy()
    for target_pos in np.flatnonzero(filled < 0):
        month, day, hour = target_keys[target_pos]
        fallback_key = (2, 28, hour) if month == 2 and day == 29 else None
        if fallback_key is None or fallback_key not in lookup:
            raise ValueError(f"Cannot align workload hour to carbon factor timestamp {target_keys[target_pos]}.")
        filled[target_pos] = lookup[fallback_key]
    return filled


def _align_interval_energy_to_target_hours(
    interval_energy_mwh: np.ndarray,
    interval_index: pd.DatetimeIndex,
    target_timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    original_shape = interval_energy_mwh.shape
    flat = interval_energy_mwh.reshape(-1, original_shape[-1])
    source_hours = pd.DatetimeIndex(interval_index).floor("h")
    hourly = pd.DataFrame(flat.T, index=source_hours).groupby(level=0, sort=True).sum()
    target = pd.DatetimeIndex(target_timestamps)

    if hourly.index.equals(target):
        aligned = hourly.to_numpy(dtype=np.float64).T
    else:
        source_keys = pd.MultiIndex.from_arrays([hourly.index.month, hourly.index.day, hourly.index.hour])
        target_keys = pd.MultiIndex.from_arrays([target.month, target.day, target.hour])
        if source_keys.is_unique:
            indexer = source_keys.get_indexer(target_keys)
            hourly_values = hourly.to_numpy(dtype=np.float64).T
            try:
                indexer = _fill_calendar_indexer(source_keys, target_keys, indexer)
                aligned = hourly_values[:, indexer]
            except ValueError:
                aligned = _resize_hourly_energy_by_position(hourly_values, len(target))
        else:
            aligned = _resize_hourly_energy_by_position(hourly.to_numpy(dtype=np.float64).T, len(target))

    return aligned.reshape(original_shape[:-1] + (len(target),))


def _hourly_facility_energy(
    component_power_mw: np.ndarray,
    pue: np.ndarray,
    annual_facility_energy_mwh: np.ndarray,
    interval_index: pd.DatetimeIndex,
    interval_hours: float,
    target_timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    interval_facility_energy_mwh = component_power_mw.sum(axis=0) * interval_hours * pue[:, None]
    hourly_facility_energy_mwh = _align_interval_energy_to_target_hours(
        interval_energy_mwh=interval_facility_energy_mwh,
        interval_index=interval_index,
        target_timestamps=target_timestamps,
    )

    # Keep annual PUE-based energy unchanged; use the workload trace only for the hourly shape.
    hourly_total = hourly_facility_energy_mwh.sum(axis=1)
    positive = hourly_total > 0
    result = np.zeros_like(hourly_facility_energy_mwh)
    result[positive] = hourly_facility_energy_mwh[positive] * (
        annual_facility_energy_mwh[positive] / hourly_total[positive]
    )[:, None]
    if np.any(~positive):
        result[~positive] = annual_facility_energy_mwh[~positive, None] / len(target_timestamps)
    return result


def _hourly_carbon_frame(
    scenario: str,
    year: int,
    countries: Sequence[str],
    timestamps: pd.DatetimeIndex,
    hourly_facility_energy_mwh: np.ndarray,
    hourly_emission_kg_per_mwh: np.ndarray,
    used_hourly_factors: np.ndarray,
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
            "training": it_ratio,
            "inference": total_ratio,
            "cpu_data": total_ratio,
        }
        return np.stack([_normalize_weights(countries, defaults[task_type]) for task_type in TASK_TYPES])

    table = []
    for task_type in TASK_TYPES:
        if task_type not in task_weights:
            raise ValueError(f"Missing weights for task type '{task_type}'.")
        table.append(_normalize_weights(countries, task_weights[task_type]))
    return np.stack(table)


def _classify_tasks(
    duration_min: np.ndarray,
    gpu_count: np.ndarray,
    gpu_mem_gb: np.ndarray,
    config: TaskClassificationConfig,
) -> np.ndarray:
    task_type = np.full(duration_min.shape, TASK_TYPES.index("inference"), dtype=np.int64)

    cpu_only = gpu_count < config.min_gpu_for_gpu_task
    training = (~cpu_only) & (
        (
            duration_min >= config.training_min_duration_min
        )
        & (
            (gpu_count >= config.training_min_gpus)
            | (gpu_mem_gb >= config.training_min_gpu_mem_gb)
        )
    )
    training |= (~cpu_only) & (duration_min >= config.long_gpu_training_duration_min)

    task_type[cpu_only] = TASK_TYPES.index("cpu_data")
    task_type[training] = TASK_TYPES.index("training")
    return task_type


def build_workload_profile(
    workload_profile_path: Union[str, Path],
    workload_year: Optional[int] = 2020,
    interval_minutes: int = 15,
    capacity_quantile: float = 0.96,
    classification_config: Optional[TaskClassificationConfig] = None,
    max_intervals: Optional[int] = None,
) -> WorkloadProfile:
    """
    Convert a 15-minute task pickle into resource-load time series by task type.

    The Alibaba-style trace stores CPU and GPU requests as percentages, so
    600.0 means 6 CPU cores and 50.0 means 0.5 GPUs. Storage is represented as
    a TB-equivalent load derived from each task's bandwidth_gb field.
    """
    if not (0 < capacity_quantile <= 1):
        raise ValueError("capacity_quantile must be in (0, 1].")
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive.")

    classification_config = classification_config or TaskClassificationConfig()
    workload_path = Path(workload_profile_path)
    task_df = pd.read_pickle(workload_path)
    if not {"interval_15m", "tasks_matrix"}.issubset(task_df.columns):
        raise ValueError("workload pickle must contain 'interval_15m' and 'tasks_matrix' columns.")

    task_df = task_df.copy()
    task_df["interval_15m"] = pd.to_datetime(task_df["interval_15m"], utc=True)
    if workload_year is not None:
        start = pd.Timestamp(f"{workload_year}-01-01", tz="UTC")
        end = pd.Timestamp(f"{workload_year + 1}-01-01", tz="UTC")
        task_df = task_df[(task_df["interval_15m"] >= start) & (task_df["interval_15m"] < end)]
        interval_index = pd.date_range(
            start=start,
            end=end - pd.Timedelta(minutes=interval_minutes),
            freq=f"{interval_minutes}min",
            tz="UTC",
        )
    else:
        task_df = task_df.sort_values("interval_15m")
        start = task_df["interval_15m"].min()
        end = task_df["interval_15m"].max()
        interval_index = pd.date_range(start=start, end=end, freq=f"{interval_minutes}min", tz="UTC")

    if max_intervals is not None:
        if max_intervals <= 0:
            raise ValueError("max_intervals must be positive when provided.")
        interval_index = interval_index[:max_intervals]
        task_df = task_df[task_df["interval_15m"].isin(interval_index)]

    task_df = task_df.sort_values("interval_15m")
    interval_to_pos = {ts: pos for pos, ts in enumerate(interval_index)}
    n_intervals = len(interval_index)
    interval_hours = interval_minutes / 60.0

    diff = np.zeros((len(TASK_TYPES), len(RESOURCES), n_intervals + 1), dtype=np.float64)
    task_counts = np.zeros((len(TASK_TYPES),), dtype=np.int64)

    for interval, matrix in zip(task_df["interval_15m"], task_df["tasks_matrix"]):
        start_idx = interval_to_pos.get(interval)
        if start_idx is None:
            continue

        arr = np.asarray(matrix, dtype=object)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < len(TASK_COLUMNS):
            raise ValueError(f"Expected task rows with {len(TASK_COLUMNS)} columns, got {arr.shape[1]}.")

        duration_min = arr[:, 4].astype(float)
        cpu_cores = arr[:, 5].astype(float) / 100.0
        gpu_count = arr[:, 6].astype(float) / 100.0
        memory_gb = arr[:, 7].astype(float)
        gpu_memory_gb = arr[:, 8].astype(float)
        storage_tb = arr[:, 9].astype(float) / 1024.0

        valid = (
            np.isfinite(duration_min)
            & (duration_min > 0)
            & np.isfinite(cpu_cores)
            & np.isfinite(gpu_count)
            & np.isfinite(memory_gb)
            & np.isfinite(gpu_memory_gb)
            & np.isfinite(storage_tb)
        )
        if not np.any(valid):
            continue

        duration_min = duration_min[valid]
        cpu_cores = np.maximum(cpu_cores[valid], 0.0)
        gpu_count = np.maximum(gpu_count[valid], 0.0)
        memory_gb = np.maximum(memory_gb[valid], 0.0)
        gpu_memory_gb = np.maximum(gpu_memory_gb[valid], 0.0)
        storage_tb = np.maximum(storage_tb[valid], 0.0)

        task_type_ids = _classify_tasks(duration_min, gpu_count, gpu_memory_gb, classification_config)
        task_counts += np.bincount(task_type_ids, minlength=len(TASK_TYPES))

        steps = np.maximum(1, np.ceil(duration_min / interval_minutes).astype(int))
        end_idx = np.minimum(start_idx + steps, n_intervals)
        active = end_idx > start_idx
        if not np.any(active):
            continue

        task_type_ids = task_type_ids[active]
        end_idx = end_idx[active]
        resource_values = np.stack(
            [
                cpu_cores[active],
                gpu_count[active],
                memory_gb[active],
                storage_tb[active],
            ],
            axis=1,
        )

        for task_type_id in range(len(TASK_TYPES)):
            type_mask = task_type_ids == task_type_id
            if not np.any(type_mask):
                continue
            ends = end_idx[type_mask]
            starts = np.full((ends.shape[0],), start_idx, dtype=np.int64)
            for resource_id in range(len(RESOURCES)):
                values = resource_values[type_mask, resource_id]
                np.add.at(diff[task_type_id, resource_id], starts, values)
                np.add.at(diff[task_type_id, resource_id], ends, -values)

    load = np.cumsum(diff[:, :, :-1], axis=2)
    total_load = load.sum(axis=0)
    trace_capacity = np.nanquantile(total_load, capacity_quantile, axis=1)
    trace_peak = np.nanmax(total_load, axis=1)
    trace_capacity = np.where(trace_capacity > 0, trace_capacity, trace_peak)
    trace_capacity = np.where(trace_capacity > 0, trace_capacity, 1.0)

    resource_hours = load.sum(axis=2) * interval_hours
    summary_records = []
    for task_type_id, task_type in enumerate(TASK_TYPES):
        record = {
            "task_type": task_type,
            "task_count": int(task_counts[task_type_id]),
        }
        for resource_id, resource in enumerate(RESOURCES):
            unit = "core_hours" if resource == "cpu" else "hours"
            if resource == "memory":
                unit = "gb_hours"
            elif resource == "storage":
                unit = "tb_hours"
            record[f"{resource}_{unit}"] = float(resource_hours[task_type_id, resource_id])
        summary_records.append(record)

    return WorkloadProfile(
        interval_index=interval_index,
        interval_hours=interval_hours,
        load=load,
        trace_capacity=trace_capacity,
        task_counts=task_counts,
        task_type_summary=pd.DataFrame(summary_records),
    )


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
    cpu_data_origin_fraction: float,
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
    if not (0 <= cpu_data_origin_fraction <= 1):
        raise ValueError("cpu_data_origin_fraction must be in [0, 1].")

    weights = np.zeros_like(origin_weights)
    weights[TASK_TYPES.index("training")] = capacity_weight
    weights[TASK_TYPES.index("inference")] = (
        inference_origin_fraction * origin_weights[TASK_TYPES.index("inference")]
        + (1 - inference_origin_fraction) * capacity_weight
    )
    weights[TASK_TYPES.index("cpu_data")] = (
        cpu_data_origin_fraction * origin_weights[TASK_TYPES.index("cpu_data")]
        + (1 - cpu_data_origin_fraction) * capacity_weight
    )
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
        cpu_idle_fraction + (1 - cpu_idle_fraction) * cpu_util
    )
    power[COMPONENTS.index("gpu")] = component_full_mw[:, COMPONENTS.index("gpu"), None] * (
        gpu_idle_fraction + (1 - gpu_idle_fraction) * np.log2(1 + gpu_util)
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
) -> np.ndarray:
    allocation = np.zeros((len(TASK_TYPES), component_power_mw.shape[1], len(COMPONENTS)), dtype=np.float64)
    component_driver_resource = {
        "cpu": "cpu",
        "gpu": "gpu",
        "memory": "memory",
        "storage": "storage",
    }

    for component_id, component in enumerate(COMPONENTS):
        if component == "it_fan":
            cpu = country_type_resource_load[:, :, RESOURCES.index("cpu"), :] / resource_capacities[
                None, :, RESOURCES.index("cpu"), None
            ]
            gpu = country_type_resource_load[:, :, RESOURCES.index("gpu"), :] / resource_capacities[
                None, :, RESOURCES.index("gpu"), None
            ]
            memory = country_type_resource_load[:, :, RESOURCES.index("memory"), :] / resource_capacities[
                None, :, RESOURCES.index("memory"), None
            ]
            driver = config.fan_cpu_weight * cpu + config.fan_gpu_weight * gpu + config.fan_memory_weight * memory
        else:
            resource_id = RESOURCES.index(component_driver_resource[component])
            driver = country_type_resource_load[:, :, resource_id, :]

        for country_id in range(component_power_mw.shape[1]):
            local_driver = driver[:, country_id, :]
            total_driver = local_driver.sum(axis=0)
            annual_driver = local_driver.sum(axis=1)
            if annual_driver.sum() > 0:
                fallback = annual_driver / annual_driver.sum()
            else:
                fallback = np.full((len(TASK_TYPES),), 1 / len(TASK_TYPES))

            shares = np.repeat(fallback[:, None], local_driver.shape[1], axis=1)
            active = total_driver > 0
            shares[:, active] = local_driver[:, active] / total_driver[active][None, :]

            allocation[:, country_id, component_id] = (
                shares * component_power_mw[component_id, country_id, :][None, :]
            ).sum(axis=1) * interval_hours

    return allocation


def _dlc_adjusted_wue(countries: Sequence[str], year_idx: int, dlc_rate_0: float, dlc_increase: float) -> np.ndarray:
    dlc_rate = dlc_rate_0 * ((1 + dlc_increase) ** year_idx)
    base_wue = np.array([WUE[country] for country in countries], dtype=float)
    return base_wue * (1 - dlc_rate) + (base_wue - 0.137) * dlc_rate


def run_workload_component_footprint(
    renewable_energy_policy: str,
    scenarios: Sequence[str],
    years: int = 5,
    countries: Optional[Sequence[str]] = None,
    workload_profile_path: Union[str, Path] = ROOT_DIR / "dataset" / "result_df_full_year_2020.pkl",
    workload_year: Optional[int] = 2020,
    year_start: int = 2026,
    output_dir: Union[str, Path] = ROOT_DIR / "results" / "workload_component_model",
    save_outputs: bool = True,
    verbose: bool = True,
    hardware_config: Optional[HardwarePowerConfig] = None,
    classification_config: Optional[TaskClassificationConfig] = None,
    task_origin_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
    task_execution_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
    execution_policy: str = "capacity",
    inference_origin_fraction: float = 0.75,
    cpu_data_origin_fraction: float = 0.50,
    capacity_quantile: float = 0.96,
    max_resource_utilization: float = 1.0,
    pue_scale: float = 1.0,
    dlc_rate_0: float = 0.05,
    dlc_increase: float = 0.20,
    hourly_carbon_factors_dir: Optional[Union[str, Path]] = ROOT_DIR / "dataset" / "EM-estimate",
    hourly_carbon_scope: str = "direct",
    hourly_carbon_fallback_to_annual: bool = True,
    save_hourly_outputs: bool = False,
    max_intervals: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compute country-level AI footprint with a workload-driven component model.

    This is a standalone replacement for the old utilization-only energy model.
    It treats the pkl task trace as the source of temporal utilization shape and
    task-type mix, then scales that shape to scenario IT capacities.
    """
    if years <= 0:
        raise ValueError("years must be positive.")
    data_year_end = DATA_YEAR_START + it_capacity.shape[0] - 1
    if year_start < DATA_YEAR_START or year_start + years - 1 > data_year_end:
        raise ValueError(f"Requested years must be within {DATA_YEAR_START}-{data_year_end}.")
    if not (0 < max_resource_utilization <= 1):
        raise ValueError("max_resource_utilization must be in (0, 1].")

    hardware_config = hardware_config or HardwarePowerConfig()
    hardware_config.validate()
    countries = list(countries or DEFAULT_COUNTRIES)

    for scenario in scenarios:
        if scenario not in SCENARIO_COL_MAP:
            raise ValueError(f"Unknown scenario '{scenario}'. Allowed: {list(SCENARIO_COL_MAP.keys())}")
    unknown_countries = [country for country in countries if country not in it_ratio]
    if unknown_countries:
        raise ValueError(f"Unknown countries: {unknown_countries}")

    emission_factors, grid_water_factors = _policy_factors(renewable_energy_policy)
    origin_weights = _as_task_weight_table(countries, task_origin_weights)
    country_share = np.array([float(it_ratio[country]) for country in countries], dtype=float)

    profile = build_workload_profile(
        workload_profile_path=workload_profile_path,
        workload_year=workload_year,
        interval_minutes=15,
        capacity_quantile=capacity_quantile,
        classification_config=classification_config,
        max_intervals=max_intervals,
    )

    annual_records = []
    component_records = []
    task_demand_records = []
    task_execution_records = []
    task_energy_records = []
    overflow_records = []
    hourly_carbon_frames = []

    for scenario in scenarios:
        scenario_col = SCENARIO_COL_MAP[scenario]
        for output_year_idx in range(years):
            year = year_start + output_year_idx
            data_year_idx = year - DATA_YEAR_START
            global_it_mw = float(it_capacity[data_year_idx, scenario_col]) * 1e3
            country_it_mw = global_it_mw * country_share
            resource_capacities = _resource_capacity(country_it_mw, hardware_config)
            global_resource_capacity = resource_capacities.sum(axis=0)
            component_full_mw = _component_full_power(country_it_mw, hardware_config)
            execution_weights = _build_execution_weights(
                countries=countries,
                country_it_mw=country_it_mw,
                origin_weights=origin_weights,
                execution_policy=execution_policy,
                inference_origin_fraction=inference_origin_fraction,
                cpu_data_origin_fraction=cpu_data_origin_fraction,
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

            component_power_mw = _component_power_timeseries(
                country_it_mw=country_it_mw,
                resource_utilization=resource_utilization,
                config=hardware_config,
            )
            component_it_energy_mwh = component_power_mw.sum(axis=2).T * profile.interval_hours
            task_type_component_it_mwh = _allocate_energy_to_task_types(
                component_power_mw=component_power_mw,
                country_type_resource_load=country_type_resource_load,
                resource_capacities=resource_capacities,
                interval_hours=profile.interval_hours,
                config=hardware_config,
            )

            pue = np.array([PUE[country][data_year_idx, scenario_col] for country in countries], dtype=float) * pue_scale
            wue = _dlc_adjusted_wue(countries, data_year_idx, dlc_rate_0, dlc_increase)
            annual_emission_kg_per_mwh = np.array(
                [emission_factors[country][data_year_idx] for country in countries],
                dtype=float,
            )
            grid_water_m3_per_mwh = np.array([grid_water_factors[country][data_year_idx] for country in countries])

            country_it_energy_mwh = component_it_energy_mwh.sum(axis=1)
            facility_energy_mwh = country_it_energy_mwh * pue
            direct_water_m3 = facility_energy_mwh * wue
            grid_water_m3 = facility_energy_mwh * grid_water_m3_per_mwh
            if hourly_carbon_factors_dir is None:
                carbon_tco2 = facility_energy_mwh * annual_emission_kg_per_mwh / 1000.0
            else:
                hourly_timestamps, hourly_emission_kg_per_mwh, used_hourly_factors = _load_hourly_carbon_factors(
                    countries=countries,
                    renewable_energy_policy=renewable_energy_policy,
                    year=year,
                    year_idx=data_year_idx,
                    annual_emission_factors=emission_factors,
                    hourly_carbon_factors_dir=hourly_carbon_factors_dir,
                    hourly_carbon_scope=hourly_carbon_scope,
                    hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
                )
                hourly_facility_energy_mwh = _hourly_facility_energy(
                    component_power_mw=component_power_mw,
                    pue=pue,
                    annual_facility_energy_mwh=facility_energy_mwh,
                    interval_index=profile.interval_index,
                    interval_hours=profile.interval_hours,
                    target_timestamps=hourly_timestamps,
                )
                hourly_carbon_tco2 = hourly_facility_energy_mwh * hourly_emission_kg_per_mwh / 1000.0
                carbon_tco2 = hourly_carbon_tco2.sum(axis=1)
                hourly_carbon_frames.append(
                    _hourly_carbon_frame(
                        scenario=scenario,
                        year=year,
                        countries=countries,
                        timestamps=hourly_timestamps,
                        hourly_facility_energy_mwh=hourly_facility_energy_mwh,
                        hourly_emission_kg_per_mwh=hourly_emission_kg_per_mwh,
                        used_hourly_factors=used_hourly_factors,
                    )
                )

            origin_resource_hours = (
                global_type_resource_load.sum(axis=2)[:, None, :]
                * origin_weights[:, :, None]
                * profile.interval_hours
            )
            execution_resource_hours = (
                global_type_resource_load.sum(axis=2)[:, None, :]
                * execution_weights[:, :, None]
                * profile.interval_hours
            )

            for country_id, country in enumerate(countries):
                annual_records.append(
                    {
                        "scenario": scenario,
                        "year": year,
                        "country": country,
                        "installed_it_mw": country_it_mw[country_id],
                        "it_energy_mwh": country_it_energy_mwh[country_id],
                        "facility_energy_mwh": facility_energy_mwh[country_id],
                        "power_twh": facility_energy_mwh[country_id] / 1e6,
                        "carbon_tco2": carbon_tco2[country_id],
                        "carbon_mtco2": carbon_tco2[country_id] / 1e6,
                        "water_m3": direct_water_m3[country_id] + grid_water_m3[country_id],
                        "water_million_m3": (direct_water_m3[country_id] + grid_water_m3[country_id]) / 1e6,
                        "direct_water_m3": direct_water_m3[country_id],
                        "grid_water_m3": grid_water_m3[country_id],
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
                            "full_power_mw": component_full_mw[country_id, component_id],
                            "it_energy_mwh": component_it_energy_mwh[country_id, component_id],
                            "facility_energy_mwh": component_it_energy_mwh[country_id, component_id] * pue[country_id],
                            "facility_energy_twh": component_it_energy_mwh[country_id, component_id] * pue[country_id] / 1e6,
                        }
                    )

                for resource_id, resource in enumerate(RESOURCES):
                    overflow_records.append(
                        {
                            "scenario": scenario,
                            "year": year,
                            "country": country,
                            "resource": resource,
                            "overflow_resource_hours": overflow[country_id, resource_id].sum()
                            * profile.interval_hours,
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
                            }
                        )

                    type_it_energy = task_type_component_it_mwh[task_type_id, country_id].sum()
                    task_energy_records.append(
                        {
                            "scenario": scenario,
                            "year": year,
                            "country": country,
                            "task_type": task_type,
                            "it_energy_mwh": type_it_energy,
                            "facility_energy_mwh": type_it_energy * pue[country_id],
                            "facility_energy_twh": type_it_energy * pue[country_id] / 1e6,
                        }
                    )

    tag = "-".join([scenario.replace(" ", "") for scenario in scenarios]) or "None"
    results = {
        "annual_summary": pd.DataFrame(annual_records),
        "component_energy": pd.DataFrame(component_records),
        "task_demand": pd.DataFrame(task_demand_records),
        "task_execution": pd.DataFrame(task_execution_records),
        "task_type_energy": pd.DataFrame(task_energy_records),
        "capacity_overflow": pd.DataFrame(overflow_records),
        "hourly_carbon": (
            pd.concat(hourly_carbon_frames, ignore_index=True)
            if hourly_carbon_frames
            else pd.DataFrame(columns=HOURLY_CARBON_COLUMNS)
        ),
        "workload_profile_summary": profile.task_type_summary,
        "trace_resource_capacity": pd.DataFrame(
            {
                "resource": RESOURCES,
                "trace_capacity_at_quantile": profile.trace_capacity,
                "capacity_quantile": capacity_quantile,
            }
        ),
    }

    if save_outputs:
        output_path = Path(output_dir)
        os.makedirs(output_path, exist_ok=True)
        results["annual_summary"].to_csv(
            output_path / f"Country_Annual_Summary_{renewable_energy_policy}_{tag}.csv",
            index=False,
        )
        results["component_energy"].to_csv(
            output_path / f"Country_Component_Energy_{renewable_energy_policy}_{tag}.csv",
            index=False,
        )
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
            results["hourly_carbon"].to_csv(
                output_path / f"Country_Hourly_Carbon_{renewable_energy_policy}_{tag}.csv",
                index=False,
            )
        results["workload_profile_summary"].to_csv(output_path / "Workload_Profile_Summary.csv", index=False)
        results["trace_resource_capacity"].to_csv(output_path / "Trace_Resource_Capacity.csv", index=False)

    if verbose:
        totals = results["annual_summary"].groupby(["scenario", "year"], as_index=False)[
            ["power_twh", "carbon_mtco2", "water_million_m3"]
        ].sum()
        print(totals.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        if save_outputs:
            print("Saved workload component results to:", os.path.abspath(output_dir))

    return results


if __name__ == "__main__":
    run_workload_component_footprint(
        renewable_energy_policy="CP",
        scenarios=["Base"],
        years=6,
        year_start=2025,
    )