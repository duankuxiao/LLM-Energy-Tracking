"""W05 task attribution of modelled component energy, including idle pools.

Attribution weights are bookkeeping assumptions. They do not turn memory
proxies, unobserved storage activity or modelled power into measurements.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.task_model import TASK_TYPES

IDLE_ALLOCATION_RULES = ("hourly_driver", "annual_driver", "separate_pool")
ENERGY_ATOL_MWH = 1e-6
ENERGY_RTOL = 1e-10


def split_component_power(power, full_power, idle_fractions):
    """Split an unchanged component x country x hour power array."""
    idle = np.broadcast_to(
        np.asarray(full_power).T[:, :, None] * np.asarray(idle_fractions)[:, None, None],
        power.shape)
    dynamic = np.asarray(power) - idle
    if not np.isfinite(power).all() or np.any(dynamic < -1e-10):
        raise ValueError("Component power must be finite and at least its configured idle floor.")
    return idle, np.maximum(dynamic, 0.)


def allocate_component_energy(idle, dynamic, task_load, capacities, interval_hours,
                              components, resources, fan_weights):
    """Compute all idle rules from one power result; dynamic weights are shared."""
    if not np.isfinite(interval_hours) or interval_hours <= 0:
        raise ValueError("Annualized interval hours must be finite and positive.")
    normalized = task_load / capacities[None, :, :, None]
    if not np.isfinite(normalized).all() or np.any(normalized < 0):
        raise ValueError("Attribution loads must be finite and non-negative.")
    n_tasks, n_countries, _, n_hours = normalized.shape
    shape = (n_tasks, n_countries, len(components))
    dynamic_energy = np.zeros(shape)
    hourly_idle = np.zeros(shape)
    annual_idle = np.zeros(shape)
    metadata = []
    for component_id, component in enumerate(components):
        if component == "it_fan":
            driver = sum(weight * normalized[:, :, resources.index(resource), :]
                         for resource, weight in fan_weights.items())
        else:
            driver = normalized[:, :, resources.index(component), :]
        for country_id in range(n_countries):
            local = driver[:, country_id, :]
            total = local.sum(axis=0)
            annual = local.sum(axis=1)
            if annual.sum() > 0:
                fallback = annual / annual.sum()
                basis = "annual_component_driver"
            else:
                proxy = normalized[:, country_id, :, :].sum(axis=(1, 2))
                if proxy.sum() > 0:
                    fallback = proxy / proxy.sum()
                    basis = "capacity_normalized_resource_proxy_no_component_observation"
                else:
                    fallback = np.full(n_tasks, 1. / n_tasks)
                    basis = "equal_task_reference_no_activity_observed"
            active = total > 0
            shares = np.broadcast_to(fallback[:, None], (n_tasks, n_hours)).copy()
            shares[:, active] = local[:, active] / total[active][None, :]
            if np.any(dynamic[component_id, country_id, ~active] > 1e-10):
                raise ValueError(f"Positive dynamic power without a component driver: {component}.")
            dynamic_energy[:, country_id, component_id] = (
                shares @ dynamic[component_id, country_id]) * interval_hours
            hourly_idle[:, country_id, component_id] = (
                shares @ idle[component_id, country_id]) * interval_hours
            annual_idle[:, country_id, component_id] = (
                fallback * idle[component_id, country_id].sum() * interval_hours)
            metadata.append({"country_id": country_id, "component": component,
                             "component_driver_available": bool(annual.sum() > 0),
                             "zero_driver_intervals": int((~active).sum()),
                             "zero_driver_annualized_hours": float((~active).sum() * interval_hours),
                             "fallback_basis": basis,
                             "interpretation": "model_energy_attribution_not_measured_task_power"})
    physical_idle = idle.sum(axis=2).T * interval_hours
    physical_dynamic = dynamic.sum(axis=2).T * interval_hours
    rules = {}
    for rule, assigned_idle in (("hourly_driver", hourly_idle), ("annual_driver", annual_idle),
                                ("separate_pool", np.zeros(shape))):
        pool = physical_idle.copy() if rule == "separate_pool" else np.zeros_like(physical_idle)
        total = assigned_idle + dynamic_energy
        if not np.allclose(assigned_idle.sum(axis=0) + pool, physical_idle,
                           rtol=ENERGY_RTOL, atol=ENERGY_ATOL_MWH):
            raise ValueError(f"Idle attribution does not close for rule {rule}.")
        if not np.allclose(dynamic_energy.sum(axis=0), physical_dynamic,
                           rtol=ENERGY_RTOL, atol=ENERGY_ATOL_MWH):
            raise ValueError(f"Dynamic attribution does not close for rule {rule}.")
        if not np.allclose(total.sum(axis=0) + pool, physical_idle + physical_dynamic,
                           rtol=ENERGY_RTOL, atol=ENERGY_ATOL_MWH):
            raise ValueError(f"Task/component energy does not close for idle rule {rule}.")
        rules[rule] = {"idle": assigned_idle, "dynamic": dynamic_energy,
                       "total": total, "pool": pool}
    return {"rules": rules, "metadata": metadata,
            "physical_idle": physical_idle, "physical_dynamic": physical_dynamic}


def allocation_records(details, countries, components, pue, context, selected_rule):
    """Produce full 4 x 5 matrices and pool/closure rows for every rule."""
    matrix, pools, checks = [], [], []
    physical = details.get("physical_total", details["physical_idle"] + details["physical_dynamic"])
    for rule, values in details["rules"].items():
        for country_id, country in enumerate(countries):
            prefix = {**context, "country": country, "idle_allocation_rule": rule,
                      "is_selected_rule": rule == selected_rule, "pue": float(pue[country_id])}
            for component_id, component in enumerate(components):
                pool = float(values["pool"][country_id, component_id])
                pools.append({**prefix, "component": component,
                              "unallocated_idle_it_energy_mwh": pool,
                              "unallocated_idle_facility_energy_mwh": pool * pue[country_id],
                              "share_of_system_it_energy_pct": pool / physical[country_id].sum() * 100. if physical[country_id].sum() > 0 else np.nan,
                              "share_of_component_physical_it_energy_pct": pool / physical[country_id, component_id] * 100. if physical[country_id, component_id] > 0 else np.nan})
                actual = values["total"][:, country_id, component_id].sum() + pool
                expected = physical[country_id, component_id]
                idle_actual = values["idle"][:, country_id, component_id].sum() + pool
                dynamic_actual = values["dynamic"][:, country_id, component_id].sum()
                checks.append({**prefix, "component": component,
                               "check": "sum_tasks_plus_idle_pool_equals_component",
                               "expected_it_energy_mwh": expected, "accounted_it_energy_mwh": actual,
                               "expected_facility_energy_mwh": expected * pue[country_id],
                               "accounted_facility_energy_mwh": actual * pue[country_id],
                               "expected_idle_it_energy_mwh": details["physical_idle"][country_id, component_id],
                               "accounted_idle_it_energy_mwh": idle_actual,
                               "idle_residual_mwh": idle_actual - details["physical_idle"][country_id, component_id],
                               "expected_dynamic_it_energy_mwh": details["physical_dynamic"][country_id, component_id],
                               "accounted_dynamic_it_energy_mwh": dynamic_actual,
                               "dynamic_residual_mwh": dynamic_actual - details["physical_dynamic"][country_id, component_id],
                               "it_residual_mwh": actual - expected,
                               "facility_residual_mwh": (actual - expected) * pue[country_id],
                               "status": "PASS" if np.isclose(actual, expected, rtol=ENERGY_RTOL, atol=ENERGY_ATOL_MWH) else "ERROR"})
                for task_id, task in enumerate(TASK_TYPES):
                    idle_value = float(values["idle"][task_id, country_id, component_id])
                    dynamic_value = float(values["dynamic"][task_id, country_id, component_id])
                    matrix.append({**prefix, "task_type": task, "component": component,
                                   "idle_it_energy_mwh": idle_value, "dynamic_it_energy_mwh": dynamic_value,
                                   "it_energy_mwh": idle_value + dynamic_value,
                                   "idle_facility_energy_mwh": idle_value * pue[country_id],
                                   "dynamic_facility_energy_mwh": dynamic_value * pue[country_id],
                                   "facility_energy_mwh": (idle_value + dynamic_value) * pue[country_id],
                                   "energy_origin": "modelled_component_power_and_explicit_attribution_rule"})
            expected = float(physical[country_id].sum())
            accounted = float(values["total"][:, country_id, :].sum() + values["pool"][country_id].sum())
            checks.append({**prefix, "component": "ALL_COMPONENTS", "check": "sum_tasks_plus_idle_pool_equals_total_IT",
                           "expected_it_energy_mwh": expected, "accounted_it_energy_mwh": accounted,
                           "expected_facility_energy_mwh": expected * pue[country_id],
                           "accounted_facility_energy_mwh": accounted * pue[country_id],
                           "expected_idle_it_energy_mwh": float(details["physical_idle"][country_id].sum()),
                           "accounted_idle_it_energy_mwh": float(values["idle"][:, country_id, :].sum() + values["pool"][country_id].sum()),
                           "idle_residual_mwh": float(values["idle"][:, country_id, :].sum() + values["pool"][country_id].sum() - details["physical_idle"][country_id].sum()),
                           "expected_dynamic_it_energy_mwh": float(details["physical_dynamic"][country_id].sum()),
                           "accounted_dynamic_it_energy_mwh": float(values["dynamic"][:, country_id, :].sum()),
                           "dynamic_residual_mwh": float(values["dynamic"][:, country_id, :].sum() - details["physical_dynamic"][country_id].sum()),
                           "it_residual_mwh": accounted - expected,
                           "facility_residual_mwh": (accounted - expected) * pue[country_id],
                           "status": "PASS" if np.isclose(accounted, expected, rtol=ENERGY_RTOL, atol=ENERGY_ATOL_MWH) else "ERROR"})
    return matrix, pools, checks


def allocation_share_tables(matrix, pools, keys):
    """Sum absolute energy first, then compute task/component/system shares."""
    energy_columns = ["idle_it_energy_mwh", "dynamic_it_energy_mwh", "it_energy_mwh",
                      "idle_facility_energy_mwh", "dynamic_facility_energy_mwh", "facility_energy_mwh"]
    cells = matrix.groupby(keys + ["task_type", "component"], as_index=False, observed=True)[energy_columns].sum()
    task_totals = cells.groupby(keys + ["task_type"], as_index=False, observed=True)[energy_columns].sum()
    components = cells.groupby(keys + ["component"], as_index=False, observed=True)[energy_columns].sum()
    pool_columns = ["unallocated_idle_it_energy_mwh", "unallocated_idle_facility_energy_mwh"]
    pool = pools.groupby(keys + ["component"], as_index=False, observed=True)[pool_columns].sum()
    components = components.merge(pool, on=keys + ["component"], how="left", validate="one_to_one")
    components[pool_columns] = components[pool_columns].fillna(0.)
    components["physical_it_energy_mwh"] = components["it_energy_mwh"] + components["unallocated_idle_it_energy_mwh"]
    components["physical_idle_it_energy_mwh"] = components["idle_it_energy_mwh"] + components["unallocated_idle_it_energy_mwh"]
    components["physical_facility_energy_mwh"] = components["facility_energy_mwh"] + components["unallocated_idle_facility_energy_mwh"]
    system = components.groupby(keys, as_index=False, observed=True)[
        ["physical_it_energy_mwh", "physical_facility_energy_mwh"]].sum().rename(
            columns={"physical_it_energy_mwh": "system_it_energy_mwh", "physical_facility_energy_mwh": "system_facility_energy_mwh"})
    cells = cells.merge(task_totals[keys + ["task_type", "it_energy_mwh"]].rename(
        columns={"it_energy_mwh": "task_it_energy_mwh"}), on=keys + ["task_type"], validate="many_to_one")
    cells = cells.merge(system, on=keys, validate="many_to_one")
    cells = cells.merge(components[keys + ["component", "physical_it_energy_mwh"]].rename(
        columns={"physical_it_energy_mwh": "component_physical_it_energy_mwh"}), on=keys + ["component"], validate="many_to_one")
    for name, denominator in (("share_within_task_pct", "task_it_energy_mwh"),
                              ("share_of_system_it_energy_pct", "system_it_energy_mwh"),
                              ("share_of_component_physical_it_energy_pct", "component_physical_it_energy_mwh")):
        cells[name] = _percent(cells["it_energy_mwh"], cells[denominator])
    task_totals = task_totals.merge(system, on=keys, validate="many_to_one")
    task_totals["task_share_of_system_it_energy_pct"] = _percent(task_totals["it_energy_mwh"], task_totals["system_it_energy_mwh"])
    components = components.merge(system, on=keys, validate="many_to_one")
    components["component_share_of_system_it_energy_pct"] = _percent(components["physical_it_energy_mwh"], components["system_it_energy_mwh"])
    return cells, task_totals, components


def _percent(numerator, denominator):
    return np.divide(np.asarray(numerator) * 100., np.asarray(denominator),
                     out=np.full(len(numerator), np.nan), where=np.asarray(denominator) > 0)


def resource_proxy_metadata():
    return pd.DataFrame([
        {"resource": "cpu", "source_basis": "requested_cores_and_request_utilization", "source_unit": "core_hours",
         "model_unit": "core_hours", "interpretation": "trace_derived_activity"},
        {"resource": "gpu", "source_basis": "requested_accelerators_and_SM_utilization", "source_unit": "SM_equivalent_GPU_hours",
         "model_unit": "SM_equivalent_GPU_hours", "interpretation": "trace_derived_activity"},
        {"resource": "memory", "source_basis": "CPU_weighted_memory_utilization_proxy", "source_unit": "CPU_weighted_memory_proxy_hours",
         "model_unit": "model_capacity_GB_equivalent_hours",
         "interpretation": "not_measured_host_memory_capacity_or_GPU_memory_activity"},
        {"resource": "storage", "source_basis": "no_observed_dynamic_storage_activity", "source_unit": "not_observed",
         "model_unit": "model_TB_hours", "interpretation": "configured_idle_power_and_proxy_attribution_only"},
    ])
