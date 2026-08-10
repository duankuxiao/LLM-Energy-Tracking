import os
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.workload_component_model import (  # noqa: E402
    DATA_YEAR_START,
    RESOURCES,
    SCENARIO_COL_MAP,
    TASK_TYPES,
    HardwarePowerConfig,
    TaskClassificationConfig,
    _align_interval_energy_to_target_hours,
    _as_task_weight_table,
    _build_execution_weights,
    _component_full_power,
    _component_power_timeseries,
    _load_hourly_carbon_factors,
    _normalize_weights,
    _policy_factors,
    _resource_capacity,
    _scale_workload_to_capacity,
    _standard_hourly_index,
    build_workload_profile,
)
from dataset.Factors import PUE  # noqa: E402
from dataset.Installed_capacity_data import (  # noqa: E402
    countries as DEFAULT_COUNTRIES,
    it_capacity,
    it_ratio,
)


EUROPEAN_COUNTRIES = (
    "France",
    "Germany",
    "United_Kingdom",
    "Italy",
    "Ireland",
    "Netherlands",
    "Spain",
    "Sweden",
    "Belgium",
    "Norway",
    "Poland",
    "Switzerland",
)

MIGRATION_CONSTRAINTS = ("global", "europe_only")
SOLVE_MODES = ("auto", "monolithic", "windowed")
SOLVERS = ("scipy", "gurobi")
CAPACITY_TOLERANCE = 1e-3


def _print_progress(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[training-migration] {message}", flush=True)


def _hourly_load_from_intervals(
    interval_load: np.ndarray,
    interval_index: pd.DatetimeIndex,
    interval_hours: float,
    target_timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    resource_hours = interval_load * interval_hours
    return _align_interval_energy_to_target_hours(
        interval_energy_mwh=resource_hours,
        interval_index=interval_index,
        target_timestamps=target_timestamps,
    )


def _hourly_emission_factors(
    countries: Sequence[str],
    renewable_energy_policy: str,
    year: int,
    year_idx: int,
    hourly_carbon_factors_dir: Optional[Union[str, Path]],
    hourly_carbon_scope: str,
    hourly_carbon_fallback_to_annual: bool,
):
    emission_factors, _ = _policy_factors(renewable_energy_policy)
    annual = np.array([emission_factors[country][year_idx] for country in countries], dtype=np.float64)

    if hourly_carbon_factors_dir is None:
        timestamps = _standard_hourly_index(year)
        hourly = np.repeat(annual[:, None], len(timestamps), axis=1)
        used_hourly = np.zeros((len(countries),), dtype=bool)
        return timestamps, hourly, used_hourly

    return _load_hourly_carbon_factors(
        countries=countries,
        renewable_energy_policy=renewable_energy_policy,
        year=year,
        year_idx=year_idx,
        annual_emission_factors=emission_factors,
        hourly_carbon_factors_dir=hourly_carbon_factors_dir,
        hourly_carbon_scope=hourly_carbon_scope,
        hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
    )


def _marginal_it_mwh_per_resource_hour(
    country_it_mw: np.ndarray,
    fixed_resource_load: np.ndarray,
    resource_capacities: np.ndarray,
    config: HardwarePowerConfig,
) -> np.ndarray:
    n_countries, _, n_hours = fixed_resource_load.shape
    coeff = np.zeros((n_countries, len(RESOURCES), n_hours), dtype=np.float64)
    component_full_mw = _component_full_power(country_it_mw, config)
    utilization = np.clip(fixed_resource_load / resource_capacities[:, :, None], 0.0, 1.0)

    cpu_id = RESOURCES.index("cpu")
    gpu_id = RESOURCES.index("gpu")
    memory_id = RESOURCES.index("memory")
    storage_id = RESOURCES.index("storage")

    cpu_idle = config.cpu_idle_power_w_per_core / config.cpu_full_power_w_per_core
    gpu_idle = config.gpu_idle_power_w / config.gpu_full_power_w

    coeff[:, cpu_id, :] += (
        component_full_mw[:, 0, None]
        * (1 - cpu_idle)
        / resource_capacities[:, cpu_id, None]
    )
    coeff[:, gpu_id, :] += (
        component_full_mw[:, 1, None]
        * (1 - gpu_idle)
        / ((1 + utilization[:, gpu_id, :]) * np.log(2.0))
        / resource_capacities[:, gpu_id, None]
    )
    coeff[:, memory_id, :] += (
        component_full_mw[:, 2, None]
        * (1 - config.memory_idle_fraction)
        / resource_capacities[:, memory_id, None]
    )
    coeff[:, storage_id, :] += (
        component_full_mw[:, 3, None]
        * (1 - config.storage_idle_fraction)
        / resource_capacities[:, storage_id, None]
    )

    fan_weight_sum = config.fan_cpu_weight + config.fan_gpu_weight + config.fan_memory_weight
    heat_load = (
        config.fan_cpu_weight * utilization[:, cpu_id, :]
        + config.fan_gpu_weight * utilization[:, gpu_id, :]
        + config.fan_memory_weight * utilization[:, memory_id, :]
    ) / fan_weight_sum
    fan_slope = component_full_mw[:, 4, None] * (1 - config.it_fan_idle_fraction) * 3 * heat_load**2
    coeff[:, cpu_id, :] += fan_slope * (
        config.fan_cpu_weight / fan_weight_sum / resource_capacities[:, cpu_id, None]
    )
    coeff[:, gpu_id, :] += fan_slope * (
        config.fan_gpu_weight / fan_weight_sum / resource_capacities[:, gpu_id, None]
    )
    coeff[:, memory_id, :] += fan_slope * (
        config.fan_memory_weight / fan_weight_sum / resource_capacities[:, memory_id, None]
    )
    return coeff


def _evaluate_country_hourly_carbon(
    country_it_mw: np.ndarray,
    country_resource_load: np.ndarray,
    resource_capacities: np.ndarray,
    pue: np.ndarray,
    hourly_emission_kg_per_mwh: np.ndarray,
    hardware_config: HardwarePowerConfig,
    max_resource_utilization: float,
) -> Dict[str, np.ndarray]:
    utilization = np.clip(
        country_resource_load / resource_capacities[:, :, None],
        0.0,
        max_resource_utilization,
    )
    component_power_mw = _component_power_timeseries(
        country_it_mw=country_it_mw,
        resource_utilization=utilization,
        config=hardware_config,
    )
    it_energy_mwh = component_power_mw.sum(axis=0)
    facility_energy_mwh = it_energy_mwh * pue[:, None]
    carbon_tco2 = facility_energy_mwh * hourly_emission_kg_per_mwh / 1000.0
    capacity_limit = resource_capacities[:, :, None] * max_resource_utilization
    overflow = np.maximum(country_resource_load - capacity_limit, 0.0)
    return {
        "it_energy_mwh": it_energy_mwh,
        "facility_energy_mwh": facility_energy_mwh,
        "carbon_tco2": carbon_tco2,
        "utilization": utilization,
        "overflow": overflow,
    }


def _build_variable_index(
    active_sources: np.ndarray,
    destination_ids: np.ndarray,
    n_hours: int,
    delay_hours: int,
):
    source_parts = []
    destination_parts = []
    execution_parts = []

    for source_hour in active_sources:
        last_hour = min(n_hours - 1, int(source_hour) + delay_hours)
        execution_hours = np.arange(int(source_hour), last_hour + 1, dtype=np.int32)
        local_sources = np.full(
            len(destination_ids) * len(execution_hours),
            int(source_hour),
            dtype=np.int32,
        )
        local_destinations = np.repeat(destination_ids.astype(np.int32), len(execution_hours))
        local_executions = np.tile(execution_hours, len(destination_ids))
        source_parts.append(local_sources)
        destination_parts.append(local_destinations)
        execution_parts.append(local_executions)

    if not source_parts:
        empty = np.array([], dtype=np.int32)
        return empty, empty, empty

    return (
        np.concatenate(source_parts),
        np.concatenate(destination_parts),
        np.concatenate(execution_parts),
    )


def _estimate_variable_count(
    active_sources: np.ndarray,
    n_hours: int,
    n_destinations: int,
    delay_hours: int,
) -> int:
    if len(active_sources) == 0 or n_destinations == 0:
        return 0
    execution_counts = np.minimum(n_hours - 1, active_sources + delay_hours) - active_sources + 1
    return int(execution_counts.sum() * n_destinations)


def _solve_sparse_lp_scipy(
    cost: np.ndarray,
    equality_matrix: sparse.csr_matrix,
    equality_rhs: np.ndarray,
    inequality_matrix: sparse.csr_matrix,
    inequality_rhs: np.ndarray,
    linprog_options: Optional[Mapping[str, Union[bool, float, int]]] = None,
):
    options = {"presolve": True}
    if linprog_options:
        options.update(dict(linprog_options))

    result = linprog(
        c=cost,
        A_ub=inequality_matrix,
        b_ub=inequality_rhs,
        A_eq=equality_matrix,
        b_eq=equality_rhs,
        bounds=(0, None),
        method="highs",
        options=options,
    )
    if not result.success:
        raise RuntimeError(f"Training migration optimization failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64), float(result.fun), int(result.status), str(result.message)


def _solve_sparse_lp_gurobi(
    cost: np.ndarray,
    equality_matrix: sparse.csr_matrix,
    equality_rhs: np.ndarray,
    inequality_matrix: sparse.csr_matrix,
    inequality_rhs: np.ndarray,
    verbose: bool = False,
    gurobi_options: Optional[Mapping[str, Union[bool, float, int, str]]] = None,
):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError(
            "solver='gurobi' requires the gurobipy package and a valid Gurobi license."
        ) from exc

    try:
        model = gp.Model("training_migration_lp")
        params = {
            "OutputFlag": 0,
            "NumericFocus": 3,
            "FeasibilityTol": 1e-9,
            "OptimalityTol": 1e-9,
            "ScaleFlag": 2,
        }
        if gurobi_options:
            params.update(dict(gurobi_options))
        for name, value in params.items():
            model.setParam(name, value)

        variables = model.addMVar(shape=cost.shape[0], lb=0.0, name="x")
        model.setObjective(cost @ variables, GRB.MINIMIZE)
        model.addMConstr(equality_matrix, variables, GRB.EQUAL, equality_rhs, name="source_balance")
        if inequality_matrix.shape[0] > 0:
            model.addMConstr(inequality_matrix, variables, GRB.LESS_EQUAL, inequality_rhs, name="capacity")

        if verbose:
            print("[training-migration] Gurobi model built; optimizing.", flush=True)
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            status_name = _gurobi_status_name(model.Status)
            raise RuntimeError(
                "Training migration optimization failed with Gurobi: "
                f"{status_name} (status {model.Status})."
            )

        solution = np.asarray(variables.X, dtype=np.float64)
        return solution, float(model.ObjVal), int(model.Status), _gurobi_status_name(model.Status)
    except gp.GurobiError as exc:
        raise RuntimeError(f"Training migration optimization failed with Gurobi: {exc}") from exc


def _gurobi_status_name(status: int) -> str:
    status_names = {
        1: "LOADED",
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        6: "CUTOFF",
        7: "ITERATION_LIMIT",
        8: "NODE_LIMIT",
        9: "TIME_LIMIT",
        10: "SOLUTION_LIMIT",
        11: "INTERRUPTED",
        12: "NUMERIC",
        13: "SUBOPTIMAL",
        14: "INPROGRESS",
        15: "USER_OBJ_LIMIT",
        16: "WORK_LIMIT",
        17: "MEM_LIMIT",
    }
    return status_names.get(status, f"UNKNOWN_STATUS_{status}")


def _solve_training_migration_lp(
    source_training_load: np.ndarray,
    movable_share: float,
    destination_ids: np.ndarray,
    residual_capacity: np.ndarray,
    marginal_carbon_cost: np.ndarray,
    delay_hours: int,
    return_schedule: bool,
    verbose: bool = False,
    progress_prefix: str = "",
    solver: str = "scipy",
    linprog_options: Optional[Mapping[str, Union[bool, float, int]]] = None,
    gurobi_options: Optional[Mapping[str, Union[bool, float, int, str]]] = None,
):
    prefix = f"{progress_prefix}: " if progress_prefix else ""
    _print_progress(verbose, f"{prefix}building LP source-hour set.")
    n_resources, n_hours = source_training_load.shape
    n_countries = residual_capacity.shape[0]
    active_sources = np.flatnonzero(np.any(source_training_load > 1e-12, axis=0))

    movable_load = np.zeros((n_countries, n_resources, n_hours), dtype=np.float64)
    empty_schedule = pd.DataFrame(
        columns=[
            "source_hour",
            "execution_hour",
            "execution_country_id",
            "delay_hours",
            "share",
        ]
    )
    if movable_share <= 1e-12 or len(active_sources) == 0:
        return movable_load, empty_schedule, {
            "status": 0,
            "message": "No movable training workload.",
            "objective": 0.0,
            "variables": 0,
        }

    if delay_hours < 0:
        raise ValueError("delay_hours must be non-negative.")
    if len(destination_ids) == 0:
        raise ValueError("At least one destination country is required.")
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of: {SOLVERS}.")

    _print_progress(
        verbose,
        f"{prefix}building LP variables for {len(active_sources)} source hours, "
        f"{len(destination_ids)} destination countries, and a {delay_hours}-hour delay window.",
    )
    source_ids, country_ids, execution_ids = _build_variable_index(
        active_sources=active_sources,
        destination_ids=destination_ids,
        n_hours=n_hours,
        delay_hours=delay_hours,
    )
    n_variables = len(source_ids)
    variable_ids = np.arange(n_variables, dtype=np.int32)

    _print_progress(verbose, f"{prefix}building LP objective vector with {n_variables} variables.")
    cost = np.zeros((n_variables,), dtype=np.float64)
    for resource_id in range(n_resources):
        cost += (
            source_training_load[resource_id, source_ids]
            * marginal_carbon_cost[country_ids, resource_id, execution_ids]
        )

    _print_progress(verbose, f"{prefix}building LP equality constraints.")
    source_row_lookup = {int(hour): row for row, hour in enumerate(active_sources)}
    equality_rows = np.array([source_row_lookup[int(hour)] for hour in source_ids], dtype=np.int32)
    equality_matrix = sparse.coo_matrix(
        (
            np.ones((n_variables,), dtype=np.float64),
            (equality_rows, variable_ids),
        ),
        shape=(len(active_sources), n_variables),
    ).tocsr()
    equality_rhs = np.full((len(active_sources),), float(movable_share), dtype=np.float64)

    _print_progress(verbose, f"{prefix}building LP capacity constraints.")
    capacity_rows = []
    capacity_cols = []
    capacity_data = []
    for resource_id in range(n_resources):
        values = source_training_load[resource_id, source_ids]
        positive = values > 1e-12
        if not np.any(positive):
            continue
        rows = (
            (country_ids[positive] * n_resources + resource_id) * n_hours
            + execution_ids[positive]
        )
        capacity_rows.append(rows.astype(np.int64, copy=False))
        capacity_cols.append(variable_ids[positive].astype(np.int64, copy=False))
        capacity_data.append(values[positive])

    if capacity_rows:
        flat_capacity_rows = np.concatenate(capacity_rows)
        unique_capacity_rows, compact_rows = np.unique(flat_capacity_rows, return_inverse=True)
        inequality_matrix = sparse.coo_matrix(
            (
                np.concatenate(capacity_data),
                (compact_rows, np.concatenate(capacity_cols)),
            ),
            shape=(len(unique_capacity_rows), n_variables),
        ).tocsr()
        inequality_rhs = residual_capacity.reshape(-1)[unique_capacity_rows]
    else:
        inequality_matrix = sparse.csr_matrix((0, n_variables))
        inequality_rhs = np.zeros((0,), dtype=np.float64)
    if not np.all(np.isfinite(cost)):
        raise ValueError("LP objective contains NaN or infinite values.")
    if not np.all(np.isfinite(equality_rhs)) or not np.all(np.isfinite(inequality_rhs)):
        raise ValueError("LP constraint right-hand side contains NaN or infinite values.")
    if inequality_rhs.size and inequality_rhs.min() < -CAPACITY_TOLERANCE:
        raise ValueError(f"LP has negative residual capacity: {inequality_rhs.min():.6g}.")
    inequality_rhs = np.maximum(inequality_rhs, 0.0)

    _print_progress(verbose, f"{prefix}solving LP with {solver}.")
    if solver == "scipy":
        solution, objective, solver_status, solver_message = _solve_sparse_lp_scipy(
            cost=cost,
            equality_matrix=equality_matrix,
            equality_rhs=equality_rhs,
            inequality_matrix=inequality_matrix,
            inequality_rhs=inequality_rhs,
            linprog_options=linprog_options,
        )
    else:
        solution, objective, solver_status, solver_message = _solve_sparse_lp_gurobi(
            cost=cost,
            equality_matrix=equality_matrix,
            equality_rhs=equality_rhs,
            inequality_matrix=inequality_matrix,
            inequality_rhs=inequality_rhs,
            verbose=verbose,
            gurobi_options=gurobi_options,
        )

    _print_progress(verbose, f"{prefix}LP solved: {solver_message}")
    _print_progress(verbose, f"{prefix}decoding optimized training allocation.")
    for resource_id in range(n_resources):
        contribution = solution * source_training_load[resource_id, source_ids]
        np.add.at(movable_load[:, resource_id, :], (country_ids, execution_ids), contribution)

    if return_schedule:
        _print_progress(verbose, f"{prefix}building non-zero migration schedule table.")
        schedule = pd.DataFrame(
            {
                "source_hour": source_ids,
                "execution_hour": execution_ids,
                "execution_country_id": country_ids,
                "delay_hours": execution_ids - source_ids,
                "share": solution,
            }
        )
        schedule = schedule[schedule["share"] > 1e-10].reset_index(drop=True)
    else:
        schedule = empty_schedule

    return movable_load, schedule, {
        "status": int(solver_status),
        "message": str(solver_message),
        "objective": float(objective),
        "variables": int(n_variables),
    }


def _solve_training_migration_windowed(
    source_training_load: np.ndarray,
    movable_share: float,
    destination_ids: np.ndarray,
    capacity_limit: np.ndarray,
    fixed_for_optimization: np.ndarray,
    baseline_movable_load: np.ndarray,
    country_it_mw: np.ndarray,
    resource_capacities: np.ndarray,
    pue: np.ndarray,
    hourly_emission_kg_per_mwh: np.ndarray,
    hardware_config: HardwarePowerConfig,
    delay_hours: int,
    commit_hours: int,
    return_schedule: bool,
    verbose: bool = False,
    progress_prefix: str = "",
    solver: str = "scipy",
    linprog_options: Optional[Mapping[str, Union[bool, float, int]]] = None,
    gurobi_options: Optional[Mapping[str, Union[bool, float, int, str]]] = None,
):
    if commit_hours <= 0:
        raise ValueError("commit_hours must be positive.")
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of: {SOLVERS}.")

    n_resources, n_hours = source_training_load.shape
    n_windows = int(np.ceil(n_hours / commit_hours))
    scheduled_movable_load = np.zeros_like(baseline_movable_load)
    reservation_remaining = baseline_movable_load.copy()
    schedule_frames = []
    total_objective = 0.0
    total_variables = 0

    prefix = f"{progress_prefix}: " if progress_prefix else ""
    _print_progress(
        verbose,
        f"{prefix}using windowed LP solve with {n_windows} windows "
        f"({commit_hours} committed hours per window).",
    )

    for window_id, source_start in enumerate(range(0, n_hours, commit_hours), start=1):
        source_end = min(n_hours, source_start + commit_hours)
        reservation_remaining[:, :, source_start:source_end] = 0.0

        local_source_load = np.zeros_like(source_training_load)
        local_source_load[:, source_start:source_end] = source_training_load[:, source_start:source_end]
        active_sources = np.flatnonzero(np.any(local_source_load > 1e-12, axis=0))
        if len(active_sources) == 0:
            _print_progress(
                verbose,
                f"{prefix}window {window_id}/{n_windows}: source hours "
                f"{source_start}-{source_end - 1} have no movable training load.",
            )
            continue

        committed_load = fixed_for_optimization + scheduled_movable_load
        committed_residual = capacity_limit - committed_load
        min_committed_residual = committed_residual.min()
        if min_committed_residual < -CAPACITY_TOLERANCE:
            idx = np.unravel_index(np.argmin(committed_residual), committed_residual.shape)
            raise ValueError(
                "Windowed committed workload exceeds capacity before LP solve: "
                f"country/resource/hour index={idx}, residual={min_committed_residual:.6g}."
            )

        available_capacity = np.maximum(committed_residual, 0.0)
        effective_reservation = np.minimum(reservation_remaining, available_capacity)
        reservation_excess = reservation_remaining - effective_reservation
        max_reservation_excess = reservation_excess.max()
        if max_reservation_excess > CAPACITY_TOLERANCE:
            idx = np.unravel_index(np.argmax(reservation_excess), reservation_excess.shape)
            _print_progress(
                verbose,
                f"{prefix}window {window_id}/{n_windows}: clipped future reservation "
                f"at country/resource/hour index={idx} by up to {max_reservation_excess:.6g}.",
            )

        base_load_for_window = committed_load + effective_reservation
        residual_capacity = capacity_limit - base_load_for_window
        residual_capacity = np.maximum(residual_capacity, 0.0)

        marginal_it = _marginal_it_mwh_per_resource_hour(
            country_it_mw=country_it_mw,
            fixed_resource_load=base_load_for_window,
            resource_capacities=resource_capacities,
            config=hardware_config,
        )
        marginal_carbon_cost = (
            marginal_it * pue[:, None, None] * hourly_emission_kg_per_mwh[:, None, :] / 1000.0
        )

        estimated_variables = _estimate_variable_count(
            active_sources=active_sources,
            n_hours=n_hours,
            n_destinations=len(destination_ids),
            delay_hours=delay_hours,
        )
        _print_progress(
            verbose,
            f"{prefix}window {window_id}/{n_windows}: solving source hours "
            f"{source_start}-{source_end - 1} with {estimated_variables} variables using {solver}.",
        )
        window_load, window_schedule, solver_info = _solve_training_migration_lp(
            source_training_load=local_source_load,
            movable_share=movable_share,
            destination_ids=destination_ids,
            residual_capacity=residual_capacity,
            marginal_carbon_cost=marginal_carbon_cost,
            delay_hours=delay_hours,
            return_schedule=return_schedule,
            verbose=False,
            progress_prefix=f"{progress_prefix} window {window_id}/{n_windows}",
            solver=solver,
            linprog_options=linprog_options,
            gurobi_options=gurobi_options,
        )
        scheduled_movable_load += window_load
        total_objective += solver_info["objective"]
        total_variables += solver_info["variables"]
        if return_schedule and not window_schedule.empty:
            schedule_frames.append(window_schedule)
        _print_progress(
            verbose,
            f"{prefix}window {window_id}/{n_windows}: solved "
            f"({solver_info['variables']} variables).",
        )

    if return_schedule and schedule_frames:
        schedule = pd.concat(schedule_frames, ignore_index=True)
    else:
        schedule = pd.DataFrame(
            columns=[
                "source_hour",
                "execution_hour",
                "execution_country_id",
                "delay_hours",
                "share",
            ]
        )

    return scheduled_movable_load, schedule, {
        "status": 0,
        "message": "Windowed LP solve completed.",
        "objective": float(total_objective),
        "variables": int(total_variables),
        "windows": int(n_windows),
    }


def _country_records(
    scenario: str,
    year: int,
    countries: Sequence[str],
    baseline: Dict[str, np.ndarray],
    optimized: Dict[str, np.ndarray],
):
    records = []
    baseline_country_carbon = baseline["carbon_tco2"].sum(axis=1)
    optimized_country_carbon = optimized["carbon_tco2"].sum(axis=1)
    baseline_facility = baseline["facility_energy_mwh"].sum(axis=1)
    optimized_facility = optimized["facility_energy_mwh"].sum(axis=1)

    for country_id, country in enumerate(countries):
        reduction = baseline_country_carbon[country_id] - optimized_country_carbon[country_id]
        records.append(
            {
                "scenario": scenario,
                "year": year,
                "country": country,
                "baseline_facility_energy_mwh": baseline_facility[country_id],
                "optimized_facility_energy_mwh": optimized_facility[country_id],
                "baseline_carbon_tco2": baseline_country_carbon[country_id],
                "optimized_carbon_tco2": optimized_country_carbon[country_id],
                "carbon_reduction_tco2": reduction,
                "carbon_reduction_pct": reduction / baseline_country_carbon[country_id]
                if baseline_country_carbon[country_id] > 0
                else 0.0,
                "baseline_peak_gpu_utilization": baseline["utilization"][
                    country_id, RESOURCES.index("gpu")
                ].max(),
                "optimized_peak_gpu_utilization": optimized["utilization"][
                    country_id, RESOURCES.index("gpu")
                ].max(),
                "optimized_overflow_resource_hours": optimized["overflow"][country_id].sum(),
            }
        )
    return records


def _training_execution_records(
    scenario: str,
    year: int,
    countries: Sequence[str],
    training_origin_weights: np.ndarray,
    source_training_load: np.ndarray,
    baseline_training_load: np.ndarray,
    optimized_training_load: np.ndarray,
):
    records = []
    global_training_resource_hours = source_training_load.sum(axis=1)
    baseline_resource_hours = baseline_training_load.sum(axis=2)
    optimized_resource_hours = optimized_training_load.sum(axis=2)

    for country_id, country in enumerate(countries):
        record = {
            "scenario": scenario,
            "year": year,
            "country": country,
            "origin_training_share": training_origin_weights[country_id],
        }
        for resource_id, resource in enumerate(RESOURCES):
            origin_hours = training_origin_weights[country_id] * global_training_resource_hours[resource_id]
            baseline_hours = baseline_resource_hours[country_id, resource_id]
            optimized_hours = optimized_resource_hours[country_id, resource_id]
            record[f"origin_{resource}_hours"] = origin_hours
            record[f"baseline_execution_{resource}_hours"] = baseline_hours
            record[f"optimized_execution_{resource}_hours"] = optimized_hours
            record[f"net_execution_change_{resource}_hours"] = optimized_hours - baseline_hours
        records.append(record)
    return records


def _schedule_records(
    schedule: pd.DataFrame,
    scenario: str,
    year: int,
    countries: Sequence[str],
    timestamps: pd.DatetimeIndex,
):
    if schedule.empty:
        return schedule

    result = schedule.copy()
    result.insert(0, "year", year)
    result.insert(0, "scenario", scenario)
    result["source_timestamp_utc"] = timestamps[result["source_hour"].to_numpy()].strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    result["execution_timestamp_utc"] = timestamps[result["execution_hour"].to_numpy()].strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    country_array = np.array(countries, dtype=object)
    result["execution_country"] = country_array[result["execution_country_id"].to_numpy()]
    return result.drop(columns=["execution_country_id"])


def run_training_migration_optimization(
    renewable_energy_policy: str,
    scenarios: Sequence[str],
    years: int = 5,
    countries: Optional[Sequence[str]] = None,
    workload_profile_path: Union[str, Path] = ROOT_DIR / "dataset" / "result_df_full_year_2020.pkl",
    workload_year: Optional[int] = 2020,
    year_start: int = 2026,
    output_dir: Union[str, Path] = ROOT_DIR / "results" / "workload_training_migration_optimization",
    save_outputs: bool = True,
    verbose: bool = True,
    hardware_config: Optional[HardwarePowerConfig] = None,
    classification_config: Optional[TaskClassificationConfig] = None,
    task_origin_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
    task_execution_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
    execution_policy: str = "capacity",
    inference_origin_fraction: float = 0.75,
    cpu_data_origin_fraction: float = 0.50,
    capacity_quantile: float = 0.95,
    max_resource_utilization: float = 1.0,
    pue_scale: float = 1.0,
    hourly_carbon_factors_dir: Optional[Union[str, Path]] = ROOT_DIR / "dataset" / "EM-estimate",
    hourly_carbon_scope: str = "direct",
    hourly_carbon_fallback_to_annual: bool = True,
    delay_hours: int = 24,
    migration_constraint: str = "global",
    europe_countries: Sequence[str] = EUROPEAN_COUNTRIES,
    save_schedule: bool = False,
    max_intervals: Optional[int] = None,
    solve_mode: str = "auto",
    commit_hours: int = 24,
    monolithic_variable_limit: int = 1_000_000,
    solver: str = "scipy",
    linprog_options: Optional[Mapping[str, Union[bool, float, int]]] = None,
    gurobi_options: Optional[Mapping[str, Union[bool, float, int, str]]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Optimize training workload migration to minimize global CO2 emissions.

    Training workload is aggregated to hourly resource demand. For each source
    hour, the optimizer can execute the movable share in the same hour or delay
    it by up to ``delay_hours``. ``migration_constraint="global"`` allows all
    selected countries to receive movable training workload. ``"europe_only"``
    allows only European-origin training workload to move within Europe, while
    non-European training remains fixed in its origin country and source hour.
    ``solve_mode="auto"`` uses a single LP for small problems and a rolling
    window LP for full-year runs that would otherwise exceed solver limits.
    """
    if years <= 0:
        raise ValueError("years must be positive.")
    if delay_hours < 0:
        raise ValueError("delay_hours must be non-negative.")
    if migration_constraint not in MIGRATION_CONSTRAINTS:
        raise ValueError(f"migration_constraint must be one of: {MIGRATION_CONSTRAINTS}.")
    if solve_mode not in SOLVE_MODES:
        raise ValueError(f"solve_mode must be one of: {SOLVE_MODES}.")
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of: {SOLVERS}.")
    if commit_hours <= 0:
        raise ValueError("commit_hours must be positive.")
    if monolithic_variable_limit <= 0:
        raise ValueError("monolithic_variable_limit must be positive.")
    if not (0 < max_resource_utilization <= 1):
        raise ValueError("max_resource_utilization must be in (0, 1].")

    data_year_end = DATA_YEAR_START + it_capacity.shape[0] - 1
    if year_start < DATA_YEAR_START or year_start + years - 1 > data_year_end:
        raise ValueError(f"Requested years must be within {DATA_YEAR_START}-{data_year_end}.")

    hardware_config = hardware_config or HardwarePowerConfig()
    hardware_config.validate()
    countries = list(countries or DEFAULT_COUNTRIES)
    for scenario in scenarios:
        if scenario not in SCENARIO_COL_MAP:
            raise ValueError(f"Unknown scenario '{scenario}'. Allowed: {list(SCENARIO_COL_MAP.keys())}")
    unknown_countries = [country for country in countries if country not in it_ratio]
    if unknown_countries:
        raise ValueError(f"Unknown countries: {unknown_countries}")

    origin_weights = _as_task_weight_table(countries, task_origin_weights)
    training_origin_weights = origin_weights[TASK_TYPES.index("training")]
    country_share = _normalize_weights(countries, it_ratio)
    europe_set = set(europe_countries)
    europe_ids = np.array(
        [country_id for country_id, country in enumerate(countries) if country in europe_set],
        dtype=np.int32,
    )
    if migration_constraint == "europe_only" and len(europe_ids) == 0:
        raise ValueError("No selected countries are included in europe_countries.")

    annual_records = []
    country_records = []
    execution_records = []
    schedule_frames = []

    _print_progress(verbose, "Building workload profile from trace.")
    profile = build_workload_profile(
        workload_profile_path=workload_profile_path,
        workload_year=workload_year,
        interval_minutes=15,
        capacity_quantile=capacity_quantile,
        classification_config=classification_config,
        max_intervals=max_intervals,
    )
    _print_progress(
        verbose,
        f"Workload profile ready: {profile.n_intervals} intervals, "
        f"{profile.interval_hours:.2f} hours per interval.",
    )

    for scenario in scenarios:
        _print_progress(verbose, f"Starting scenario '{scenario}'.")
        scenario_col = SCENARIO_COL_MAP[scenario]
        for output_year_idx in range(years):
            year = year_start + output_year_idx
            data_year_idx = year - DATA_YEAR_START
            progress_label = f"{scenario} {year}"
            _print_progress(verbose, f"{progress_label}: loading carbon factors.")
            hourly_timestamps, hourly_emission, used_hourly_factors = _hourly_emission_factors(
                countries=countries,
                renewable_energy_policy=renewable_energy_policy,
                year=year,
                year_idx=data_year_idx,
                hourly_carbon_factors_dir=hourly_carbon_factors_dir,
                hourly_carbon_scope=hourly_carbon_scope,
                hourly_carbon_fallback_to_annual=hourly_carbon_fallback_to_annual,
            )
            if max_intervals is not None:
                horizon_hours = max(1, int(np.ceil(max_intervals * profile.interval_hours)))
                hourly_timestamps = hourly_timestamps[:horizon_hours]
                hourly_emission = hourly_emission[:, :horizon_hours]
            _print_progress(
                verbose,
                f"{progress_label}: carbon factors ready for {len(hourly_timestamps)} hours "
                f"({int(used_hourly_factors.sum())} hourly, {int((~used_hourly_factors).sum())} annual fallback).",
            )

            _print_progress(verbose, f"{progress_label}: building country capacities and PUE.")
            global_it_mw = float(it_capacity[data_year_idx, scenario_col]) * 1e3
            country_it_mw = global_it_mw * country_share
            resource_capacities = _resource_capacity(country_it_mw, hardware_config)
            global_resource_capacity = resource_capacities.sum(axis=0)
            pue = np.array([PUE[country][data_year_idx, scenario_col] for country in countries], dtype=float)
            pue *= pue_scale

            _print_progress(verbose, f"{progress_label}: scaling workload to global capacity.")
            interval_global_load = _scale_workload_to_capacity(
                profile=profile,
                global_resource_capacity=global_resource_capacity,
                max_resource_utilization=max_resource_utilization,
            )
            _print_progress(verbose, f"{progress_label}: aggregating 15-minute workload to hourly demand.")
            hourly_global_type_load = _hourly_load_from_intervals(
                interval_load=interval_global_load,
                interval_index=profile.interval_index,
                interval_hours=profile.interval_hours,
                target_timestamps=hourly_timestamps,
            )
            source_training_load = hourly_global_type_load[TASK_TYPES.index("training")]

            _print_progress(verbose, f"{progress_label}: building fixed workload execution weights.")
            execution_weights = _build_execution_weights(
                countries=countries,
                country_it_mw=country_it_mw,
                origin_weights=origin_weights,
                execution_policy=execution_policy,
                inference_origin_fraction=inference_origin_fraction,
                cpu_data_origin_fraction=cpu_data_origin_fraction,
                task_execution_weights=task_execution_weights,
            )
            country_type_load = (
                hourly_global_type_load[:, None, :, :] * execution_weights[:, :, None, None]
            )
            non_training_ids = [
                task_id for task_id, task_type in enumerate(TASK_TYPES) if task_type != "training"
            ]
            fixed_non_training_load = country_type_load[non_training_ids].sum(axis=0)
            baseline_training_load = (
                source_training_load[None, :, :] * training_origin_weights[:, None, None]
            )
            baseline_total_load = fixed_non_training_load + baseline_training_load
            _print_progress(verbose, f"{progress_label}: evaluating baseline emissions.")
            baseline_eval = _evaluate_country_hourly_carbon(
                country_it_mw=country_it_mw,
                country_resource_load=baseline_total_load,
                resource_capacities=resource_capacities,
                pue=pue,
                hourly_emission_kg_per_mwh=hourly_emission,
                hardware_config=hardware_config,
                max_resource_utilization=max_resource_utilization,
            )

            if migration_constraint == "global":
                movable_share = float(training_origin_weights.sum())
                destination_ids = np.arange(len(countries), dtype=np.int32)
                fixed_training_load = np.zeros_like(baseline_training_load)
            else:
                movable_share = float(training_origin_weights[europe_ids].sum())
                destination_ids = europe_ids
                fixed_training_load = baseline_training_load.copy()
                fixed_training_load[europe_ids] = 0.0
            baseline_movable_load = np.maximum(baseline_training_load - fixed_training_load, 0.0)

            _print_progress(verbose, f"{progress_label}: building migration capacity constraints.")
            fixed_for_optimization = fixed_non_training_load + fixed_training_load
            capacity_limit = resource_capacities[:, :, None] * max_resource_utilization
            fixed_residual_capacity = capacity_limit - fixed_for_optimization
            min_residual = fixed_residual_capacity.min()
            if min_residual < -CAPACITY_TOLERANCE:
                idx = np.unravel_index(np.argmin(fixed_residual_capacity), fixed_residual_capacity.shape)
                raise ValueError(
                    "Fixed workload exceeds capacity before training migration: "
                    f"country={countries[idx[0]]}, resource={RESOURCES[idx[1]]}, "
                    f"hour={idx[2]}, residual={min_residual:.6g}."
                )
            fixed_residual_capacity = np.maximum(fixed_residual_capacity, 0.0)

            active_sources = np.flatnonzero(np.any(source_training_load > 1e-12, axis=0))
            estimated_variables = _estimate_variable_count(
                active_sources=active_sources,
                n_hours=source_training_load.shape[1],
                n_destinations=len(destination_ids),
                delay_hours=delay_hours,
            )
            selected_solve_mode = solve_mode
            if solve_mode == "auto":
                selected_solve_mode = (
                    "windowed" if estimated_variables > monolithic_variable_limit else "monolithic"
                )
            _print_progress(
                verbose,
                f"{progress_label}: selected {selected_solve_mode} solve mode "
                f"for {estimated_variables} estimated variables.",
            )

            if selected_solve_mode == "monolithic":
                _print_progress(verbose, f"{progress_label}: estimating marginal carbon costs.")
                marginal_it = _marginal_it_mwh_per_resource_hour(
                    country_it_mw=country_it_mw,
                    fixed_resource_load=fixed_for_optimization,
                    resource_capacities=resource_capacities,
                    config=hardware_config,
                )
                marginal_carbon_cost = (
                    marginal_it * pue[:, None, None] * hourly_emission[:, None, :] / 1000.0
                )
                movable_training_load, schedule, solver_info = _solve_training_migration_lp(
                    source_training_load=source_training_load,
                    movable_share=movable_share,
                    destination_ids=destination_ids,
                    residual_capacity=fixed_residual_capacity,
                    marginal_carbon_cost=marginal_carbon_cost,
                    delay_hours=delay_hours,
                    return_schedule=save_schedule,
                    verbose=verbose,
                    progress_prefix=progress_label,
                    solver=solver,
                    linprog_options=linprog_options,
                    gurobi_options=gurobi_options,
                )
                solver_info["windows"] = 1
            else:
                movable_training_load, schedule, solver_info = _solve_training_migration_windowed(
                    source_training_load=source_training_load,
                    movable_share=movable_share,
                    destination_ids=destination_ids,
                    capacity_limit=capacity_limit,
                    fixed_for_optimization=fixed_for_optimization,
                    baseline_movable_load=baseline_movable_load,
                    country_it_mw=country_it_mw,
                    resource_capacities=resource_capacities,
                    pue=pue,
                    hourly_emission_kg_per_mwh=hourly_emission,
                    hardware_config=hardware_config,
                    delay_hours=delay_hours,
                    commit_hours=commit_hours,
                    return_schedule=save_schedule,
                    verbose=verbose,
                    progress_prefix=progress_label,
                    solver=solver,
                    linprog_options=linprog_options,
                    gurobi_options=gurobi_options,
                )
            optimized_training_load = fixed_training_load + movable_training_load
            optimized_total_load = fixed_non_training_load + optimized_training_load
            _print_progress(verbose, f"{progress_label}: evaluating optimized emissions.")
            optimized_eval = _evaluate_country_hourly_carbon(
                country_it_mw=country_it_mw,
                country_resource_load=optimized_total_load,
                resource_capacities=resource_capacities,
                pue=pue,
                hourly_emission_kg_per_mwh=hourly_emission,
                hardware_config=hardware_config,
                max_resource_utilization=max_resource_utilization,
            )

            baseline_carbon = float(baseline_eval["carbon_tco2"].sum())
            optimized_carbon = float(optimized_eval["carbon_tco2"].sum())
            carbon_reduction = baseline_carbon - optimized_carbon
            _print_progress(
                verbose,
                f"{progress_label}: completed with {carbon_reduction:.6f} tCO2 reduction "
                f"({carbon_reduction / baseline_carbon:.6%}).",
            )
            annual_records.append(
                {
                    "scenario": scenario,
                    "year": year,
                    "migration_constraint": migration_constraint,
                    "solver": solver,
                    "delay_hours": delay_hours,
                    "solve_mode": selected_solve_mode,
                    "commit_hours": commit_hours if selected_solve_mode == "windowed" else 0,
                    "movable_training_share": movable_share,
                    "baseline_facility_energy_mwh": float(baseline_eval["facility_energy_mwh"].sum()),
                    "optimized_facility_energy_mwh": float(optimized_eval["facility_energy_mwh"].sum()),
                    "baseline_carbon_tco2": baseline_carbon,
                    "optimized_carbon_tco2": optimized_carbon,
                    "carbon_reduction_tco2": carbon_reduction,
                    "carbon_reduction_pct": carbon_reduction / baseline_carbon if baseline_carbon > 0 else 0.0,
                    "linearized_training_objective_tco2": solver_info["objective"],
                    "optimization_variables": solver_info["variables"],
                    "optimization_windows": solver_info["windows"],
                    "hourly_factor_countries": int(used_hourly_factors.sum()),
                    "annual_fallback_countries": int((~used_hourly_factors).sum()),
                    "solver_status": solver_info["status"],
                    "solver_message": solver_info["message"],
                }
            )
            country_records.extend(
                _country_records(
                    scenario=scenario,
                    year=year,
                    countries=countries,
                    baseline=baseline_eval,
                    optimized=optimized_eval,
                )
            )
            execution_records.extend(
                _training_execution_records(
                    scenario=scenario,
                    year=year,
                    countries=countries,
                    training_origin_weights=training_origin_weights,
                    source_training_load=source_training_load,
                    baseline_training_load=baseline_training_load,
                    optimized_training_load=optimized_training_load,
                )
            )
            if save_schedule:
                schedule_frames.append(
                    _schedule_records(
                        schedule=schedule,
                        scenario=scenario,
                        year=year,
                        countries=countries,
                        timestamps=hourly_timestamps,
                    )
                )

    tag = "-".join([scenario.replace(" ", "") for scenario in scenarios]) or "None"
    results = {
        "annual_summary": pd.DataFrame(annual_records),
        "country_summary": pd.DataFrame(country_records),
        "training_execution": pd.DataFrame(execution_records),
        "migration_schedule": (
            pd.concat(schedule_frames, ignore_index=True)
            if schedule_frames
            else pd.DataFrame(
                columns=[
                    "scenario",
                    "year",
                    "source_hour",
                    "execution_hour",
                    "delay_hours",
                    "share",
                    "source_timestamp_utc",
                    "execution_timestamp_utc",
                    "execution_country",
                ]
            )
        ),
        "workload_profile_summary": profile.task_type_summary,
    }

    if save_outputs:
        _print_progress(verbose, "Saving optimization outputs.")
        output_path = Path(output_dir)
        os.makedirs(output_path, exist_ok=True)
        suffix = f"{renewable_energy_policy}_{migration_constraint}_{tag}"
        results["annual_summary"].to_csv(
            output_path / f"Training_Migration_Annual_Summary_{suffix}.csv",
            index=False,
        )
        results["country_summary"].to_csv(
            output_path / f"Training_Migration_Country_Summary_{suffix}.csv",
            index=False,
        )
        results["training_execution"].to_csv(
            output_path / f"Training_Migration_Execution_{suffix}.csv",
            index=False,
        )
        results["workload_profile_summary"].to_csv(
            output_path / "Training_Migration_Workload_Profile_Summary.csv",
            index=False,
        )
        if save_schedule:
            results["migration_schedule"].to_csv(
                output_path / f"Training_Migration_Schedule_{suffix}.csv",
                index=False,
            )
        _print_progress(verbose, f"Output files saved to {os.path.abspath(output_dir)}.")

    if verbose:
        summary = results["annual_summary"][
            [
                "scenario",
                "year",
                "migration_constraint",
                "baseline_carbon_tco2",
                "optimized_carbon_tco2",
                "carbon_reduction_pct",
            ]
        ]
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        if save_outputs:
            print("Saved training migration optimization results to:", os.path.abspath(output_dir))

    return results


if __name__ == "__main__":
    run_training_migration_optimization(
        renewable_energy_policy="CP",
        scenarios=["Base"],
        years=5,
        year_start=2026,
        migration_constraint="europe_only",  #  global  europe_only
        commit_hours=36,
        solver='gurobi',  # scipy  gurobi
    )
