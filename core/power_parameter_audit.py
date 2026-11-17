"""W07 parameter dependencies and accounting checks, not external validation."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_hardware_config(path, baseline):
    """Apply explicit JSON overrides; reject unknown fields through replace."""
    with Path(path).open(encoding="utf-8-sig") as handle:
        overrides = json.load(handle)
    if not isinstance(overrides, dict):
        raise ValueError("Hardware configuration must be a JSON object.")
    config = replace(baseline, **overrides)
    config.validate()
    return config


def hardware_parameter_table(config):
    return pd.DataFrame([{"parameter": name, "value": value,
                          "source": "configured_assumption_not_fitted_to_MLPerf"}
                         for name, value in asdict(config).items()])


def power_curve_metadata(config):
    return {
        "cpu": {"power_curve": "u_power", "dynamic_response_exponent": config.cpu_dynamic_exponent,
                "dynamic_response_formula": "u**cpu_dynamic_exponent"},
        "gpu": {"power_curve": "log2_one_plus_u_power", "dynamic_response_exponent": config.gpu_dynamic_exponent,
                "dynamic_response_formula": "log2(1+u**gpu_dynamic_exponent)"},
        "memory": {"power_curve": "linear", "dynamic_response_exponent": 1., "dynamic_response_formula": "u"},
        "storage": {"power_curve": "linear", "dynamic_response_exponent": 1., "dynamic_response_formula": "u"},
        "it_fan": {"power_curve": "cubic_heat_load", "dynamic_response_exponent": 3., "dynamic_response_formula": "weighted_heat_load**3"},
    }


def parameter_dependency_table(config):
    rows = []
    for component, full_name, idle_name, share_name in (
        ("cpu", "cpu_full_power_w_per_core", "cpu_idle_power_w_per_core", "cpu_power_share"),
        ("gpu", "gpu_full_power_w", "gpu_idle_power_w", "gpu_power_share"),
        ("memory", "memory_power_w_per_gb", "memory_idle_fraction", "memory_power_share"),
        ("storage", "storage_power_w_per_tb", "storage_idle_fraction", "storage_power_share"),
    ):
        full, idle, share = getattr(config, full_name), getattr(config, idle_name), getattr(config, share_name)
        fraction = idle / full if component in ("cpu", "gpu") else idle
        rows.append({"component": component, "full_unit_power_parameter": full_name,
                     "full_unit_power_w": full, "idle_parameter": idle_name, "idle_parameter_value": idle,
                     "idle_fraction": fraction, "power_share_parameter": share_name, "power_share": share,
                     "capacity_equation": "installed_IT_MW*component_share*1e6/full_unit_W",
                     "load_equation": "clipped_trace_load/trace_capacity*model_resource_capacity",
                     "utilization_equation": "realized_model_resource_load/model_resource_capacity",
                     "power_equation": "component_budget_MW*(idle_fraction+(1-idle_fraction)*response(u))",
                     "unit_wattage_cancels_from_electricity_when": "budget_trace_denominator_idle_fraction_response_execution_weights_fixed",
                     "native_resource_hours_change_when_unit_wattage_changes": True,
                     "fixed_idle_watts_changes_idle_fraction": component in ("cpu", "gpu"),
                     "identity_is_external_validation": False, **power_curve_metadata(config)[component]})
    rows.append({"component": "it_fan", "power_share_parameter": "it_fan_power_share",
                 "power_share": config.it_fan_power_share, "idle_fraction": config.it_fan_idle_fraction,
                 "capacity_equation": "no_separate_fan_resource_capacity",
                 "power_equation": "fan_budget_MW*(idle_fraction+(1-idle_fraction)*weighted_heat_load**3)",
                 "identity_is_external_validation": False, **power_curve_metadata(config)["it_fan"]})
    return pd.DataFrame(rows)


def capacity_parameter_records(context, countries, country_it_mw, capacities, utilization, config, trace_capacity):
    rows = []
    units = ("model_CPU_cores", "model_GPU_equivalents", "model_GB", "model_TB")
    dependency = parameter_dependency_table(config).set_index("component")
    for country_id, country in enumerate(countries):
        for resource_id, resource in enumerate(("cpu", "gpu", "memory", "storage")):
            parameters = dependency.loc[resource]
            budget = float(country_it_mw[country_id] * parameters["power_share"])
            capacity = float(capacities[country_id, resource_id])
            reconstructed = capacity * parameters["full_unit_power_w"] / 1e6
            rows.append({**context, "country": country, "resource": resource,
                         "installed_it_mw": float(country_it_mw[country_id]), "component_budget_mw": budget,
                         "full_unit_power_w": float(parameters["full_unit_power_w"]),
                         "idle_fraction": float(parameters["idle_fraction"]),
                         "model_resource_capacity": capacity, "model_resource_unit": units[resource_id],
                         "trace_resource_capacity_at_quantile": float(trace_capacity[resource_id]),
                         "mean_realized_utilization": float(utilization[country_id, resource_id].mean()),
                         "peak_realized_utilization": float(utilization[country_id, resource_id].max()),
                         "budget_reconstruction_mw": reconstructed,
                         "budget_identity_residual_mw": reconstructed - budget,
                         "budget_identity_is_external_validation": False,
                         **power_curve_metadata(config)[resource]})
    return rows


def mechanism_conservation_table(results, policy):
    """Check valid accounting identities, never equate source and model units."""
    rows = []
    keys = ["scenario", "year", "country"]
    annual = results["annual_summary"].set_index(keys)
    components = results["component_energy"].groupby(keys)[["it_energy_mwh", "facility_energy_mwh"]].sum()
    tasks = results["task_component_energy"].groupby(keys)[["it_energy_mwh", "facility_energy_mwh"]].sum()
    pue = results["task_component_energy"].groupby(keys)["pue"].first()
    capacity = results["capacity_parameter_audit"].set_index(keys + ["resource"])
    pools = results["idle_pool"]
    pools = pools.loc[pools["is_selected_rule"]].groupby(keys)[["unallocated_idle_it_energy_mwh", "unallocated_idle_facility_energy_mwh"]].sum()

    def check(context, identity, actual, expected, unit):
        actual, expected = float(actual), float(expected)
        finite = np.isfinite(actual) and np.isfinite(expected)
        passed = finite and np.isclose(actual, expected, rtol=1e-10, atol=1e-6)
        rows.append({"policy": policy, **context, "identity": identity, "actual": actual, "expected": expected,
                     "residual": actual - expected, "unit": unit, "absolute_tolerance": 1e-6,
                     "relative_tolerance": 1e-10, "status": "PASS" if passed else "FAIL",
                     "aggregation_scope": "selected_countries" if context.get("country") == "ALL" else "country",
                     "interpretation": "model_accounting_identity_not_external_accuracy"})

    for index, row in annual.iterrows():
        context = dict(zip(keys, index))
        check(context, "sum_component_IT_equals_annual_IT", components.loc[index, "it_energy_mwh"], row["it_energy_mwh"], "MWh")
        check(context, "sum_component_facility_equals_annual_facility", components.loc[index, "facility_energy_mwh"], row["facility_energy_mwh"], "MWh")
        check(context, "idle_plus_dynamic_equals_IT", row["idle_it_energy_mwh"] + row["dynamic_it_energy_mwh"], row["it_energy_mwh"], "MWh")
        check(context, "facility_equals_IT_times_PUE", row["facility_energy_mwh"], row["it_energy_mwh"] * pue.loc[index], "MWh")
        check(context, "task_plus_idle_pool_equals_IT", tasks.loc[index, "it_energy_mwh"] + pools.loc[index, "unallocated_idle_it_energy_mwh"], row["it_energy_mwh"], "MWh")
        check(context, "task_plus_idle_pool_equals_facility", tasks.loc[index, "facility_energy_mwh"] + pools.loc[index, "unallocated_idle_facility_energy_mwh"], row["facility_energy_mwh"], "MWh")
    for row in results["resource_audit"].to_dict("records"):
        context = {name: row[name] for name in keys + ["resource"]}
        check(context, "requested_equals_realized_plus_country_clipping", row["realized_resource_hours"] + row["country_clipped_resource_hours"], row["requested_resource_hours"], "model_resource_hours")
        check(context, "task_resource_sum_equals_realized", row["realized_resource_hours"] + row["task_sum_residual_resource_hours"], row["realized_resource_hours"], "model_resource_hours")
        parameters = capacity.loc[tuple(row[name] for name in keys + ["resource"])]
        check(context, "realized_resource_equals_capacity_times_mean_utilization_8760",
              row["realized_resource_hours"], parameters["model_resource_capacity"] * parameters["mean_realized_utilization"] * 8760., "model_resource_hours")
    for row in results["global_resource_audit"].to_dict("records"):
        context = {name: row[name] for name in ("scenario", "year", "resource")}
        context["country"] = "ALL"
        check(context, "before_global_cap_equals_after_plus_clipping", row["after_global_cap_resource_hours"] + row["global_clipped_resource_hours"], row["before_global_cap_resource_hours"], "model_resource_hours")
        check(context, "origin_equals_execution", row["origin_resource_hours"], row["execution_resource_hours"], "model_resource_hours")
        check(context, "after_global_cap_equals_execution", row["after_global_cap_resource_hours"], row["execution_resource_hours"], "model_resource_hours")
        check(context, "execution_equals_realized_plus_country_clipping", row["realized_resource_hours"] + row["country_clipped_resource_hours"], row["execution_resource_hours"], "model_resource_hours")
    for row in results["grid_carbon_audit"].to_dict("records"):
        context = {name: row[name] for name in keys}
        check(context, "hourly_energy_equals_annual_facility", row["hourly_energy_sum_mwh"], row["annual_energy_mwh"], "MWh")
        check(context, "mean_CI_plus_covariance_equals_matched_carbon",
              row["matched_carbon_tco2"], row["mean_ci_carbon_tco2"] + row["hours"] * row["load_ci_covariance"] / 1000., "tCO2")
    table = pd.DataFrame(rows)
    if table["status"].ne("PASS").any():
        failed = table.loc[table["status"].ne("PASS")].head().to_dict("records")
        raise ValueError(f"W07 mechanism conservation failed: {failed}")
    return table
