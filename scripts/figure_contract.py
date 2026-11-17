"""W08 source/display contract shared by data exporters and the plotting CLI.

Numeric source cells retain computation precision. Formatting is applied only
to display strings; exact zero, undefined and small nonzero values stay distinct.
This module does not run models, write files or import a plotting backend.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dataset.Installed_capacity_data import (
    BASE_ELECTRICITY_REFERENCE_ROLE,
    BASE_GLOBAL_ELECTRICITY_REFERENCE_TWH,
    IT_CAPACITY, TOTAL_CAPACITY, SCENARIO_COLUMNS,
)

SCHEMA_VERSION = "W08-1"
SCENARIO_COLORS = {"Base": "#315B7D", "Lift-Off": "#B66B35",
                   "High Efficiency": "#3F8070", "Headwinds": "#866A99"}
POLICY_COLORS = {"CP": "#52677B", "NDC": "#C28C48", "NZ": "#428C7B"}
COMPONENT_LABELS = {"cpu": "CPU", "gpu": "GPU", "memory": "Memory",
                    "storage": "Storage", "it_fan": "IT fan"}
TASK_LABELS = {"training": "Training", "inference": "Inference", "other": "Other",
               "unclassified": "Unclassified", "idle_pool": "Unallocated idle"}
EFFECT_LABELS = {
    "power_configuration": "Power configuration",
    "mean_workload_representation": "Mean workload",
    "load_temporal_variability": "Load timing",
    "hourly_carbon_matching": "Hourly carbon matching",
}
PARAMETER_LABELS = {
    "cpu_unit_power_scale_fixed_idle_ratio": "CPU unit watts: fixed idle ratio",
    "cpu_unit_power_scale_fixed_idle_watts": "CPU unit watts: fixed idle watts",
    "gpu_unit_power_scale_fixed_idle_ratio": "GPU unit watts: fixed idle ratio",
    "gpu_unit_power_scale_fixed_idle_watts": "GPU unit watts: fixed idle watts",
    "cpu_dynamic_exponent": "CPU response exponent",
    "gpu_dynamic_exponent": "GPU response exponent",
    "cpu_idle_power_ratio": "CPU idle/full power ratio",
    "cpu_power_share": "CPU share of IT power budget",
    "pre_power_resource_variability_scale": "Resource variability before power conversion",
    "gpu_idle_power_ratio": "GPU idle/full power ratio",
    "training_gpu_load_scale": "Training GPU load",
    "inference_gpu_load_scale": "Inference GPU load",
    "classified_inference_gpu_load_share": "Inference share of classified GPU load",
    "gpu_power_share": "GPU share of IT power budget",
    "hourly_carbon_variability_scale": "Hourly carbon variability",
}


def display_number(value: float, significant_digits: int = 3) -> str:
    """Never label a finite nonzero value as 0 or -0; never pad source anchors."""
    value = float(value)
    if not math.isfinite(value):
        return "NA"
    if value == 0:
        return "0"
    return format(value, f".{significant_digits}g")


def ordered_names(values, canonical) -> list[str]:
    names = set(values)
    unknown = names - set(canonical)
    if unknown:
        raise ValueError(f"Unrecognised display identifiers: {sorted(unknown)}")
    return [name for name in canonical if name in names]


def require_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame.empty or frame.duplicated(keys).any() or frame[keys].isna().any().any():
        raise ValueError(f"{name} requires nonempty, unique, nonmissing keys {keys}.")


def _figure1_tables(sheets: dict[str, pd.DataFrame]) -> None:
    capacity = sheets["Fig1a_Capacity"]
    require_unique(capacity, ["scenario", "year"], "Figure 1 capacity")
    ordered_names(capacity["scenario"], SCENARIO_COLUMNS)
    audit = capacity.copy()
    audit["expected_source_column"] = audit["scenario"].map(SCENARIO_COLUMNS)
    audit["scenario_column_matches"] = audit["source_scenario_column"] == audit["expected_source_column"]
    rows = audit["year"].astype(int).to_numpy() - 2025
    columns = audit["expected_source_column"].astype(int).to_numpy()
    if np.any(rows < 0) or np.any(rows >= len(IT_CAPACITY)):
        raise ValueError("Figure 1 capacity year is outside the source matrix.")
    audit["source_IT_matrix_residual_gw"] = audit["source_global_it_capacity_gw"] - IT_CAPACITY[rows, columns]
    audit["source_total_matrix_residual_gw"] = audit["source_global_total_capacity_gw"] - TOTAL_CAPACITY[rows, columns]
    audit["total_capacity_residual_gw"] = audit["total_capacity_gw"] - audit["source_global_total_capacity_gw"] * audit["total_capacity_share_sum"]
    audit["it_capacity_residual_gw"] = audit["it_capacity_gw"] - audit["source_global_it_capacity_gw"] * audit["it_capacity_share_sum"]
    audit["ai_it_residual_gw"] = audit["ai_gpu_it_capacity_gw"] - (
        audit["source_global_it_capacity_gw"] * audit["ai_capacity_factor"] * audit["it_capacity_share_sum"])
    audit["AI_is_subset_of_IT"] = audit["ai_gpu_it_capacity_gw"] <= audit["it_capacity_gw"] + 1e-10
    if (not audit["scenario_column_matches"].all() or not audit["AI_is_subset_of_IT"].all()
            or not np.allclose(audit[["ai_it_residual_gw", "source_IT_matrix_residual_gw",
                                      "source_total_matrix_residual_gw", "total_capacity_residual_gw",
                                      "it_capacity_residual_gw"]], 0, atol=1e-10, rtol=0)):
        raise ValueError("Figure 1 capacity labels, matrix columns or AI/IT subset audit failed.")
    audit["status"] = "PASS"
    sheets["W08_Scenario_Capacity_Audit"] = audit

    task_frames = [sheets[name] for name in
                   ("Fig1c_Training", "Fig1d_Inference", "Fig1e_Other", "Fig1_Unclassified")]
    cells = pd.concat(task_frames, ignore_index=True)
    require_unique(cells, ["task_type", "component"], "Figure 1 Base task matrix")
    expected = {(task, component) for task in TASK_LABELS if task != "idle_pool"
                for component in COMPONENT_LABELS}
    if set(zip(cells["task_type"], cells["component"])) != expected:
        raise ValueError("Figure 1 requires all four tasks and all five IT components.")
    rule = str(cells["idle_allocation_rule"].iloc[0])
    denominator = float(cells["annual_avg_system_it_energy_mwh"].iloc[0])
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Figure 1 system IT energy must be positive and finite.")
    if not np.allclose(cells["annual_avg_system_it_energy_mwh"], denominator):
        raise ValueError("Figure 1 task cells have inconsistent system denominators.")
    pools = sheets["W05_Idle_Pool"]
    pools = pools.loc[(pools["scenario"] == "Base") & (pools["idle_allocation_rule"] == rule)]
    # Sum countries within each year before averaging years, never average country rows.
    pool = pools.groupby(["year", "component"])["unallocated_idle_it_energy_mwh"].sum()
    pool = pool.groupby("component").mean().reindex(COMPONENT_LABELS)
    if pool.isna().any():
        raise ValueError("Figure 1 idle pool is missing components.")
    pool_cells = pd.DataFrame({
        "scenario": "Base", "policy": str(cells["policy"].iloc[0]),
        "year_start": int(cells["year_start"].iloc[0]),
        "year_end": int(cells["year_end"].iloc[0]),
        "years_averaged": int(cells["years_averaged"].iloc[0]),
        "task_type": "idle_pool", "component": pool.index,
        "annual_avg_it_energy_mwh": pool.to_numpy(),
        "annual_avg_idle_it_energy_mwh": pool.to_numpy(),
        "annual_avg_dynamic_it_energy_mwh": 0.0,
        "annual_avg_system_it_energy_mwh": denominator,
        "share_of_system_it_energy_pct": pool.to_numpy() / denominator * 100,
        "idle_allocation_rule": rule,
    })
    cells = pd.concat([cells, pool_cells], ignore_index=True)
    cells["system_share_denominator"] = "all_four_tasks_five_components_plus_unallocated_idle"
    if not np.isclose(cells["annual_avg_it_energy_mwh"].sum(), denominator, rtol=1e-10, atol=1e-6):
        raise ValueError("Figure 1 plotted task matrix plus idle pool does not close to system IT energy.")
    sheets["W08_Task_Component_Display"] = cells
    columns = ["annual_avg_it_energy_mwh", "annual_avg_idle_it_energy_mwh", "annual_avg_dynamic_it_energy_mwh"]
    for key, name in (("component", "W08_Component_Display"), ("task_type", "W08_Task_Display")):
        table = cells.groupby(key, as_index=False)[columns].sum()
        for part in ("", "idle_", "dynamic_"):
            table[f"{part}share_of_system_pct"] = table[f"annual_avg_{part}it_energy_mwh"] / denominator * 100
        table["denominator_it_energy_mwh"] = denominator
        table["idle_allocation_rule"] = rule
        for column in ("scenario", "policy", "year_start", "year_end", "years_averaged"):
            table[column] = cells[column].iloc[0]
        sheets[name] = table

    energy, carbon = sheets["Fig1f_Energy"], sheets["Fig1g_Carbon"]
    keys = ["scenario", "policy", "year"]
    require_unique(energy, keys, "Figure 1 energy")
    require_unique(carbon, keys, "Figure 1 emissions")
    ci = energy.merge(carbon, on=keys, validate="one_to_one", how="outer")
    if ci[["facility_energy_twh", "carbon_mtco2"]].isna().any().any():
        raise ValueError("Figure 1 energy and carbon paths have different coverage.")
    ci["load_weighted_ci_kg_per_mwh"] = ci["carbon_mtco2"].div(ci["facility_energy_twh"].where(ci["facility_energy_twh"] > 0)) * 1000
    values = ci["load_weighted_ci_kg_per_mwh"].to_numpy()
    fixed = bool(np.isfinite(values).all() and np.allclose(values, values[0], rtol=1e-10, atol=1e-10))
    ci["fixed_across_all_years_and_scenarios"] = fixed
    ci["panel_decision"] = ("aligned_panels_energy_and_carbon_are_proportional" if fixed
                            else "aligned_panels_carbon_also_reflects_variable_load_weighted_CI")
    sheets["W08_Load_Weighted_CI"] = ci


def _figure4_tables(sheets: dict[str, pd.DataFrame]) -> None:
    frame = sheets["Fig4a_Sensitivity"].copy()
    require_unique(frame, ["experiment_order"], "Figure 4 experiments")
    ordered_names(frame["level"], ("Low", "Base", "High"))
    frame["parameter_label"] = frame["parameter_name"].map(PARAMETER_LABELS)
    if frame["parameter_label"].isna().any():
        raise ValueError("Every Figure 4 parameter requires an explicit display label.")
    frame["parameter_delta"] = frame["parameter_value"] - frame["baseline_parameter_value"]
    for metric in ("facility_energy_mwh", "carbon_tco2"):
        baseline = frame[f"baseline_annual_mean_{metric}"]
        expected = (frame[f"annual_mean_{metric}"] - baseline).div(baseline.where(baseline != 0)) * 100
        if not np.allclose(frame[f"{metric}_change_vs_baseline_pct"], expected, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError(f"Figure 4 percentage denominator does not reproduce {metric} responses.")
    frame["aggregation_definition"] = "mean_over_years(sum_over_selected_countries(annual_metric))"
    frame["percentage_definition"] = "100*(perturbed_annual_mean-baseline_annual_mean)/baseline_annual_mean"
    frame["uncertainty_interpretation"] = "deterministic_one_factor_contrasts_not_confidence_intervals"
    frame["gpu_budget_interpretation"] = "increasing_GPU_budget_reduces_other_component_budgets_proportionally_response_sign_is_computed"
    sheets["Fig4a_Sensitivity"] = frame
    # A long table binds a/b to explicit metrics, retaining the legacy wide sheet.
    parts = []
    for panel, metric, unit in (("a", "facility_energy_mwh", "MWh/year"), ("b", "carbon_tco2", "tCO2/year")):
        part = frame.copy()
        part["panel"] = panel
        part["metric"] = metric
        part["annual_mean"] = frame[f"annual_mean_{metric}"]
        part["baseline_annual_mean"] = frame[f"baseline_annual_mean_{metric}"]
        part["annual_mean_unit"] = unit
        part["change_vs_baseline_pct"] = frame[f"{metric}_change_vs_baseline_pct"]
        part["response_direction"] = np.select(
            [part["change_vs_baseline_pct"].isna(), part["change_vs_baseline_pct"] > 0,
             part["change_vs_baseline_pct"] < 0], ["undefined", "increase", "decrease"], default="exact_zero")
        parts.append(part)
    sheets["W08_Sensitivity_Panels"] = pd.concat(parts, ignore_index=True)


def _panel_rows(number: int) -> list[tuple]:
    # Panel, source sheet, metric columns/units, scientific interpretation.
    return {
        1: [
            ("a", "Fig1a_Capacity", {"total_capacity_gw": "GW", "it_capacity_gw": "GW", "ai_gpu_it_capacity_gw": "GW"},
             "Latest modelled year; parallel total-facility, all-IT and AI-focused IT capacity bars; AI is a subset of IT. Country allocation shares differ by capacity boundary."),
            ("b", "Fig1b_Country_Share", {"it_capacity_share_pct": "% of original global IT allocation"},
             "Fixed input shares; source base year is not documented. Capacity coverage does not establish electricity coverage."),
            ("c", "W08_Task_Component_Display", {"share_of_system_it_energy_pct": "% of system IT energy"},
             "Base scenario; four tasks by five components plus explicit unallocated idle. Ratios of summed energy, including storage and unclassified tasks; memory is a proxy."),
            ("d", "W08_Component_Display", {"idle_share_of_system_pct": "% of system IT energy", "dynamic_share_of_system_pct": "% of system IT energy", "share_of_system_pct": "% of system IT energy"},
             "All five physical components; idle plus dynamic energy. Unallocated idle remains in its physical component."),
            ("e", "W08_Task_Display", {"idle_share_of_system_pct": "% of system IT energy", "dynamic_share_of_system_pct": "% of system IT energy", "share_of_system_pct": "% of system IT energy"},
             "All four tasks plus unallocated idle, using the same system denominator as c and d."),
            ("f", "Fig1f_Energy", {"facility_energy_twh": "TWh/year"},
             "Demand paths conditioned on capacity factors; the default Base electricity path was calibrated to an external global reference. It is not an independent electricity forecast."),
            ("g", "Fig1g_Carbon", {"carbon_mtco2": "MtCO2/year"},
             "AI-focused facility emissions from energy and grid factors; relation to f is audited in W08_Load_Weighted_CI. Aligned axes avoid a dual-axis correlation claim."),
        ],
        2: [
            ("a", "Fig2a_CF_Map", {"annual_avg_carbon_factor_kg_per_mwh": "kgCO2/MWh"},
             "CP annual direct grid-factor inputs averaged across the stated years, shown by country; these are not load-weighted model emissions."),
            ("b", "Fig2b_Country_Abatement", {"cp_carbon_mtco2": "MtCO2/year", "ndc_carbon_mtco2": "MtCO2/year", "nz_carbon_mtco2": "MtCO2/year"},
             "AI-focused facilities only, not national economy emissions. Top countries plus the exact remaining-country sum. Policy columns are selected by name; no redundant avoided-emission line."),
            ("c", "Fig2c_Selected_Countries_Carbon", {"carbon_mtco2": "MtCO2/year"},
             "Sum over selected countries under fixed demand and stylized CP/NDC/NZ input paths, not independently identified policy effects."),
            ("d", "Fig2d_Weighted_CI", {"load_weighted_carbon_intensity_kg_per_mwh": "kgCO2/MWh"},
             "Aggregate AI facility emissions divided by aggregate facility electricity, with unit conversion; distinct from the annual grid input in a."),
        ],
        3: [
            ("a", "Fig3a_Model_Comparison", {"annual_cpu_facility_energy_twh": "TWh/year", "hourly_workload_resolved_facility_energy_twh": "TWh/year", "annual_cpu_carbon_mtco2": "MtCO2/year", "hourly_workload_resolved_carbon_mtco2": "MtCO2/year"},
             "Conditional representation comparison using common capacity factors. Hourly electricity is a calibrated reference under default factors, not independent validation."),
            ("b", "Fig3b_Effect_Decomposition", {"energy_effect_pct_of_hourly_reference": "% of hourly reference", "carbon_effect_pct_of_hourly_reference": "% of hourly reference"},
             "Ordered configuration differences (next minus previous), divided by the corresponding hourly-reference annual mean; not unique causal attribution. Signed nonzero contributions retain significant digits."),
            ("c", "Fig3c_Operating_Shape", {"peak_power_gw": "GW", "max_abs_ramp_gw_per_h": "GW h^-1", "max_abs_ramp_excluding_seams_gw_per_h": "GW h^-1", "max_abs_ramp_excluding_alignment_gw_per_h": "GW h^-1"},
             "Aggregate peak and maximum absolute hourly ramp are separate metrics. Raw, repeat-seam-excluded and all-alignment-discontinuities-excluded ramps remain distinct; the last excludes repeat seams too."),
            ("d", "Fig3d_Country_Peak_Ramp", {"peak_power_gw": "GW", "max_abs_ramp_gw_per_h": "GW h^-1", "max_abs_ramp_excluding_seams_gw_per_h": "GW h^-1", "max_abs_ramp_excluding_alignment_gw_per_h": "GW h^-1"},
             "Country metrics for the stated detail year under the supplied shared workload-shape construction; not independent national load observations."),
        ],
        4: [
            ("a", "W08_Sensitivity_Panels", {"change_vs_baseline_pct": "%", "parameter_value": "row.parameter_unit"},
             "Electricity: 100*(perturbed annual mean - baseline annual mean)/baseline annual mean. Sum countries before averaging years; not the mean of yearly percentages. Deterministic one-factor contrasts, not confidence intervals."),
            ("b", "W08_Sensitivity_Panels", {"change_vs_baseline_pct": "%", "parameter_value": "row.parameter_unit"},
             "Emissions: same aggregation and percentage definition as a, using the emissions baseline denominator. Low/High describe parameter direction, not response sign; GPU budget changes renormalize other component budgets."),
        ],
    }[number]


def prepare_figure_sources(number: int, source: dict[str, pd.DataFrame], *, carbon_scope: str) -> dict[str, pd.DataFrame]:
    """Append versioned W08 metadata and exact-value lookup tables before export."""
    sheets = dict(source)
    boundary = sheets["Accounting_Boundary"].iloc[0]
    if number == 1:
        _figure1_tables(sheets)
    elif number == 4:
        _figure4_tables(sheets)
    if number in (1, 3):
        sheets["W08_Global_Reference"] = pd.DataFrame([
            {"year": year, "global_reference_twh": value,
             "display_label": display_number(value), "role": BASE_ELECTRICITY_REFERENCE_ROLE[year],
             "accounting_scope": "global_external_reference_not_observed_selected_country_energy"}
            for year, value in BASE_GLOBAL_ELECTRICITY_REFERENCE_TWH.items()
        ])
    if number == 3:
        table = sheets["Fig3a_Model_Comparison"]
        table = table.copy()
        table["energy_difference_definition"] = "100*(annual_CPU-hourly_reference)/hourly_reference"
        table["hourly_reference_role"] = str(boundary["calibration_basis"])
        sheets["Fig3a_Model_Comparison"] = table
    metadata, displayed = [], []
    for panel, sheet, metrics, note in _panel_rows(number):
        table = sheets[sheet]
        for metric, unit in metrics.items():
            if metric not in table:
                raise ValueError(f"Figure {number}{panel} requires {sheet}.{metric}.")
            metadata.append({"figure": number, "panel": panel, "source_sheet": sheet,
                             "source_column": metric, "unit": unit, "interpretation": note,
                             "accounting_scope": boundary["accounting_scope"],
                             "country_count": boundary["country_count"],
                             "carbon_scope": "direct" if number == 2 and panel == "a" else carbon_scope,
                             "calibration_basis": boundary["calibration_basis"]})
            for position, (_, row) in enumerate(table.iterrows(), start=2):
                if number == 4 and row["panel"] != panel:
                    continue
                value = float(row[metric])
                record = {"figure": number, "panel": panel, "source_sheet": sheet,
                          "source_excel_row": position, "source_column": metric,
                          "raw_value": value, "unit": row["parameter_unit"] if unit == "row.parameter_unit" else unit,
                          "display_text": display_number(value),
                          "value_status": ("undefined" if not math.isfinite(value) else
                                           "exact_zero" if value == 0 else "nonzero")}
                for key in ("scenario", "policy", "year", "year_start", "year_end", "country", "display_label",
                            "task_type", "component", "profile", "effect", "experiment_order", "parameter_name", "level",
                            "parameter_value", "parameter_unit", "baseline_parameter_value", "annual_mean",
                            "baseline_annual_mean", "annual_mean_unit"):
                    if key in row:
                        record[key] = row[key]
                displayed.append(record)
    sheets["W08_Panel_Metadata"] = pd.DataFrame(metadata)
    sheets["W08_Display_Values"] = pd.DataFrame(displayed)
    sheets["W08_Figure_Contract"] = pd.DataFrame([{
        "schema_version": SCHEMA_VERSION, "figure": number,
        "source_precision": "unrounded_numeric_cells_display_strings_only",
        "display_precision": "3_significant_digits_scientific_notation_for_small_nonzero_values",
        "zero_rule": "only_exact_zero_is_labelled_zero_missing_is_NA",
        "plot_backend": "python_matplotlib", "rendered": False,
        "caption_abbreviations": "CPU: central processing unit; GPU: graphics processing unit; IT: information technology; CI: carbon intensity; CP/NDC/NZ: stylized grid input scenarios",
    }])
    return sheets
