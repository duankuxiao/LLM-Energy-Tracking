"""Original manuscript composition with only evidence-required corrections.

The source JPEGs have no editable plotting project. Rebuild their axes and
marks with Arial text, Times New Roman panel letters, and the sampled original
palette. Model values are always read from the frozen revision workbooks.
"""
from __future__ import annotations

import math
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

BLUE = "#46788F"
CYAN = "#78B7C9"
SALMON = "#E48B7B"
PALE = "#F6E094"
GOLD = "#FFC000"
STORAGE = "#B8B8B8"  # New measured category; existing four component colours stay unchanged.
SCENARIOS = ("Base", "Lift-Off", "High Efficiency", "Headwinds")
SCENARIO_LABELS = ("Base", "Lift-off", "High efficiency", "Headwinds")
SCENARIO_PAIRS = ((BLUE, "#2F5363"), (CYAN, "#418EA2"), (PALE, "#EDC12A"), (SALMON, "#D5472F"))
POLICIES = ("CP", "NDC", "NZ")
COLORS = (CYAN, PALE, SALMON)
COMPONENTS = ("gpu", "memory", "it_fan", "storage", "cpu")
COMPONENT_COLORS = {"cpu": SALMON, "gpu": BLUE, "memory": CYAN, "it_fan": PALE, "storage": STORAGE}
MAIN_PARAMETERS = ("gpu_idle_power_ratio", "training_gpu_load_scale", "inference_gpu_load_scale",
                   "classified_inference_gpu_load_share", "gpu_power_share", "hourly_carbon_variability_scale")
MAIN_LABELS = ("GPU idle power", "Training GPU utilization", "Inference GPU utilization",
               "Training–inference mix", "GPU power share", "Hourly carbon variability")
ORIGINAL_PIXELS = {1: (6816, 7386), 2: (5794, 5350), 3: (6708, 5857), 4: (6054, 5277)}
ORIGINAL_TITLES = {
    1: "Demand pathways shape capacity, electricity use and emissions across 24 major AI countries.",
    2: "Grid decarbonization lowers load-weighted carbon intensity and produces increasing avoided emissions.",
    3: "Annual and hourly model differences, their bridge decomposition and the operational shape of the hourly trace.",
    4: "One-at-a-time sensitivity of annual electricity use and emissions from the hourly workload-resolved model.",
}


def adapt_source_contract(source):
    """Bind the restored main panels to their actual cells, keeping full audits."""
    source.audit_values = source.values.copy()
    source.audit_metadata = source.metadata.copy()
    if source.number == 1:
        for sheet in ("Fig1c_Training", "Fig1d_Inference", "Fig1e_Other"):
            source.tables[sheet] = pd.read_excel(source.path, sheet_name=sheet)
    if source.number == 3:
        frame = source.tables["Fig3a_Model_Comparison"]
        source.tables["Original_Bridge_Endpoints"] = pd.DataFrame([
            {"metric": metric, "annual_mean": frame[f"annual_cpu_{column}"].mean(),
             "hourly_mean": frame[f"hourly_workload_resolved_{column}"].mean(),
             "definition": "mean over the five yearly selected-country totals; no model rerun"}
            for metric, column in (("electricity_TWh", "facility_energy_twh"), ("emissions_MtCO2", "carbon_mtco2"))])
    if source.number == 4:
        table = source.tables["W08_Sensitivity_Panels"]
        source.tables["Original_Main_Sensitivity"] = table.loc[table.parameter_name.isin(MAIN_PARAMETERS)].reset_index(drop=True)
    definitions = {
        1: [("a", "Fig1a_Capacity", {"total_capacity_gw": "GW", "ai_gpu_it_capacity_gw": "GW"}, "All five years; AI is shown inside the true total capacity, never added to total. Scenario columns are corrected."),
            ("b", "Fig1b_Country_Share", {"it_capacity_share_pct": "%"}, "Fixed shares of the original global IT-capacity allocation; source base year undocumented; not electricity coverage."),
            ("c", "Fig1c_Training", {"display_share_pct": "% of task IT energy"}, "Training: all five components; storage added to preserve closure, existing component colours retained."),
            ("d", "Fig1d_Inference", {"display_share_pct": "% of task IT energy"}, "Inference: all five components; task-internal shares are not system shares."),
            ("e", "Fig1e_Other", {"display_share_pct": "% of task IT energy"}, "Other tasks: all five components. Unclassified tasks and system closure appear in Supplementary Fig. S2."),
            ("f", "Fig1f_Energy", {"facility_energy_twh": "TWh"}, "Original grouped annual bars; new experimental electricity values and legacy calibration."),
            ("g", "Fig1g_Carbon", {"carbon_mtco2": "MtCO2"}, "Original grouped annual bars; direct operational AI facility emissions within 24 countries.")],
        2: [("a", "Fig2a_CF_Map", {"annual_avg_carbon_factor_kg_per_mwh": "kgCO2/MWh"}, "Original ranked horizontal bars; all 24 countries; 2026-2030 mean CP input."),
            ("b", "Fig2b_Country_Abatement", {name: "MtCO2" for name in ("cp_carbon_mtco2", "ndc_carbon_mtco2", "nz_carbon_mtco2", "ndc_avoided_vs_cp_mtco2", "nz_avoided_vs_cp_mtco2")}, "Original country grouped bars and derived avoided-emission lines; eight countries plus the other 16; AI facility scope."),
            ("c", "Fig2c_Selected_Countries_Carbon", {"carbon_mtco2": "MtCO2", "avoided_vs_cp_mtco2": "MtCO2"}, "Original annual grouped bars and avoided-emission lines; proportional reductions follow prescribed inputs."),
            ("d", "Fig2d_Weighted_CI", {"load_weighted_carbon_intensity_kg_per_mwh": "kgCO2/MWh"}, "Original CP/NDC/NZ line and marker styles; prescribed annual grid means.")],
        3: [("a", "Fig3a_Model_Comparison", {"annual_cpu_facility_energy_twh": "TWh", "hourly_workload_resolved_facility_energy_twh": "TWh", "annual_cpu_carbon_mtco2": "MtCO2", "hourly_workload_resolved_carbon_mtco2": "MtCO2", "annual_vs_hourly_facility_energy_difference_pct": "%", "annual_vs_hourly_carbon_difference_pct": "%"}, "Original endpoint grouped bars and yellow percentage lines; annual-minus-hourly divided by hourly."),
            ("b", "Fig3b_Effect_Decomposition", {"annual_avg_energy_effect_twh": "TWh", "annual_avg_carbon_effect_mtco2": "MtCO2"}, "Original waterfall; ordered next-minus-previous differences; all small nonzero steps retain significant digits."),
            ("b", "Original_Bridge_Endpoints", {"annual_mean": "row.metric", "hourly_mean": "row.metric"}, "Waterfall endpoints are means of the five frozen annual totals."),
            ("c", "Fig3c_Operating_Shape", {"mean_power_gw": "GW", "peak_power_gw": "GW", "p99_power_gw": "GW", "max_abs_ramp_gw_per_h": "GW/h"}, "Original four operating-profile panels; only incorrect Maximum Power title is corrected to maximum absolute hourly ramp. Filters and high percentiles appear in S4."),
            ("d", "Fig3d_Country_Peak_Ramp", {"peak_power_gw": "GW", "max_abs_ramp_gw_per_h": "GW/h"}, "Original horizontal country bars; 2030 shared-shape reference; not independent national observations.")],
        4: [("a", "Original_Main_Sensitivity", {"change_vs_baseline_pct": "%", "parameter_value": "row.parameter_unit"}, "Original six-parameter tornado layout; electricity; low/high colours describe parameter level, not response sign."),
            ("b", "Original_Main_Sensitivity", {"change_vs_baseline_pct": "%", "parameter_value": "row.parameter_unit"}, "Original six-parameter tornado layout; emissions; remaining nine new parameter experiments appear in S3.")],
    }[source.number]
    metadata, values = [], []
    for panel_name, sheet, metrics, interpretation in definitions:
        for column, unit in metrics.items():
            metadata.append({"figure": source.number, "panel": panel_name, "source_sheet": sheet,
                             "source_column": column, "unit": unit, "interpretation": interpretation,
                             "accounting_scope": "selected_countries", "country_count": 24, "carbon_scope": "direct",
                             "calibration_basis": source.boundary["calibration_basis"]})
            for index, row in source.tables[sheet].iterrows():
                if source.number == 4 and row["panel"] != panel_name:
                    continue
                value = float(row[column])
                display = number(value)
                if source.number == 1 and sheet == "Fig1b_Country_Share":
                    display = f"{value:.1f}"
                elif source.number == 1 and panel_name in "cde":
                    display = f"{value:.1f}" if value >= 1 else number(value)
                elif source.number == 2 and ((sheet == "Fig2c_Selected_Countries_Carbon" and column == "carbon_mtco2")
                                             or (sheet == "Fig2b_Country_Abatement" and column.endswith("_carbon_mtco2"))):
                    display = compact_number(value)
                record = {"figure": source.number, "panel": panel_name, "source_sheet": sheet,
                          "source_excel_row": None if sheet.startswith("Original_") else index + 2,
                          "source_csv_row": index + 2,
                          "source_location_kind": "derived_export_table" if sheet.startswith("Original_") else "frozen_workbook_cell",
                          "source_column": column,
                          "raw_value": value, "unit": unit, "display_text": display,
                          "value_status": "exact_zero" if value == 0 else "nonzero" if math.isfinite(value) else "undefined"}
                for key in ("year", "scenario", "policy", "country", "task_type", "component", "level", "parameter_name", "metric"):
                    if key in row:
                        record[key] = row[key]
                values.append(record)
    source.metadata, source.values = pd.DataFrame(metadata), pd.DataFrame(values)


def caption(source):
    lines = [f"Figure {source.number}. {ORIGINAL_TITLES[source.number]}"]
    for letter, table in source.metadata.groupby("panel", sort=False):
        lines.append(f"{letter}. " + " ".join(table.interpretation.drop_duplicates()))
    lines.append("Study boundary: 24 major AI countries with available capacity and grid-carbon inputs. "
                 "The new experiment values are retained. Legacy factors fit the global reference electricity pathway "
                 "to these countries' aggregate; this does not establish observed national electricity targets or global coverage. "
                 "The original 93.18% IT-capacity allocation is not renormalized and does not establish electricity coverage.")
    lines.append("NDC and NZ are prescribed uniform scalings of CP, reaching 0.89 and 0.53 in 2030. "
                 "Model figures are deterministic conditional calculations. CPU: central processing unit; "
                 "GPU: graphics processing unit; IT: information technology. Source numeric cells retain computation precision.")
    lines.extend(source.run_notes)
    return "\n\n".join(lines) + "\n"


def number(value):
    value = float(value)
    if not math.isfinite(value):
        return "NA"
    if value == 0:
        return "0"
    return f"{value:.3g}"


def configure(plt, font_size=12):
    plt.rcParams.update({"font.family": "Arial", "font.size": font_size,
                         "axes.labelsize": font_size, "axes.titlesize": font_size,
                         "xtick.labelsize": font_size, "ytick.labelsize": font_size,
                         "legend.fontsize": font_size, "legend.frameon": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 1.2, "lines.linewidth": 1.3,
                         "xtick.major.size": 3, "ytick.major.size": 3})


def canvas(plt, n):
    configure(plt, 12 if n in (1, 4) else 10.5)
    w, h = ORIGINAL_PIXELS[n]
    return plt.figure(figsize=(w / 600, h / 600), facecolor="white")


def panel(fig, letter, x, y):
    fig.text(x, y, f"({letter})", family="Times New Roman", fontsize=20,
             fontweight="bold", ha="left", va="top")


def compact_number(value):
    value = float(value)
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}"
    return number(value)


def bar_values(ax, bars, *, inside=False, fontsize=None, compact=False):
    top = ax.get_ylim()[1]
    for bar in bars:
        value = bar.get_height()
        text = ax.annotate(compact_number(value) if compact else number(value), (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, -3 if inside else 2), textcoords="offset points",
                    ha="center", va="top" if inside else "bottom", fontsize=fontsize)
        text.set_gid("bar_value")


def pie(ax, values, *, text_size=11):
    values = np.asarray(values, dtype=float)
    wedges, _ = ax.pie(values, colors=[COMPONENT_COLORS[name] for name in COMPONENTS],
                       startangle=154, counterclock=False,
                       wedgeprops={"edgecolor": "white", "linewidth": .6})
    small = []
    for index, (wedge, value) in enumerate(zip(wedges, values)):
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        label = (f"{value:.1f}" if value >= 1 else number(value)) + "%"
        if value >= 4:
            ax.text(.60 * np.cos(angle), .60 * np.sin(angle), label,
                    color="white" if COMPONENTS[index] == "gpu" else "black",
                    ha="center", va="center", fontsize=text_size)
        elif value > 0:
            small.append((angle, label))
    # Storage was absent from the old pies; separate the two small adjacent
    # labels without changing any existing component colour or omitting energy.
    for side in (-1, 1):
        previous = -2.
        group = [(angle, label) for angle, label in small if (1 if np.cos(angle) >= 0 else -1) == side]
        for angle, label in sorted(group, key=lambda item: np.sin(item[0])):
            y = max(float(np.sin(angle)), previous + .18)
            previous = y
            ax.annotate(label, (np.cos(angle), np.sin(angle)), xytext=(side * 1.06, y),
                        ha="right" if side < 0 else "left", va="center", fontsize=text_size - 1,
                        arrowprops={"arrowstyle": "-", "color": "#555555", "lw": .5})
    ax.set_aspect("equal")
    ax.set_xlim(-1.30, 1.30 if any(np.cos(angle) > 0 for angle, _ in small) else 1.05)


def figure1(source, plt):
    fig = canvas(plt, 1)
    a = fig.add_axes([.075, .682, .913, .286])
    b = fig.add_axes([.075, .311, .913, .334])
    f = fig.add_axes([.075, .035, .425, .210])
    g = fig.add_axes([.565, .035, .423, .210])
    panel(fig, "a", .002, .992); panel(fig, "b", .002, .665)
    panel(fig, "f", .002, .270); panel(fig, "g", .515, .270)
    capacity = source.table("Fig1a_Capacity", ["scenario", "year"])
    years = sorted(capacity.year.unique())
    x, width = np.arange(len(years)), .19
    for j, scenario in enumerate(SCENARIOS):
        rows = capacity.loc[capacity.scenario.eq(scenario)].set_index("year").loc[years]
        positions = x + (j - 1.5) * .20
        light, dark = SCENARIO_PAIRS[j]
        # Total is drawn at its true height and AI occupies its lower subset.
        # The two capacities are never added, correcting R2.16/R3.04.
        bars = a.bar(positions, rows.total_capacity_gw, width, color=light, edgecolor="#333333", linewidth=.7)
        a.bar(positions, rows.ai_gpu_it_capacity_gw, width, color=dark, edgecolor="#333333", linewidth=.7)
        for bar, total, ai in zip(bars, rows.total_capacity_gw, rows.ai_gpu_it_capacity_gw):
            xpos = bar.get_x() + width / 2
            a.text(xpos, total + 3, number(total), ha="center", va="bottom", fontsize=11)
            a.text(xpos, ai - 2, number(ai), ha="center", va="top", fontsize=10.5,
                   color="white" if j == 0 else "black")
    a.set_xticks(x, years); a.set_ylabel("Capacity (GW)")
    a.set_ylim(0, math.ceil(capacity.total_capacity_gw.max() / 50) * 50 + 50)
    a.set_xlim(-.5, len(years) - .5)
    handles = [Patch(facecolor=color, edgecolor="#333333", label=f"{label}-{part}")
               for label, pair in zip(("Base", "Lift-off", "High-efficiency", "Headwinds"), SCENARIO_PAIRS)
               for part, color in zip(("Total", "AI"), pair)]
    a.legend(handles=handles, ncol=4, loc="upper left", bbox_to_anchor=(.005, 1),
             columnspacing=1.5, handlelength=2.1, handletextpad=.35, borderaxespad=.3, fontsize=11.5)
    shares = source.table("Fig1b_Country_Share", ["country"]).sort_values("it_capacity_share_pct", ascending=False)
    y = np.arange(len(shares))
    b.barh(y, shares.it_capacity_share_pct, height=.80, color=BLUE, edgecolor="#333333", linewidth=.65)
    b.set_yticks(y, shares.country)
    b.set_ylim(len(y) - .5, -.5); b.set_xlim(0, 50)
    b.set_xlabel("Share of capacity(%)", labelpad=2)
    for i, value in enumerate(shares.it_capacity_share_pct):
        b.text(value + .55, i, f"{value:.1f}", ha="left", va="center", fontsize=11)
    for j, (task, letter) in enumerate(zip(("training", "inference", "other"), "cde")):
        ax = fig.add_axes([.180 + .265 * j, .365, .245, .230], facecolor="none")
        data = source.table("W08_Task_Component_Display")
        data = data.loc[data.task_type.eq(task)].set_index("component").loc[list(COMPONENTS)]
        values = data.annual_avg_it_energy_mwh / data.annual_avg_it_energy_mwh.sum() * 100
        pie(ax, values)
        panel(fig, letter, .180 + .265 * j, .588)
    legend_order = ("cpu", "gpu", "memory", "it_fan", "storage")
    legend_labels = ("CPU", "GPU", "Memory", "ITFan", "Storage")
    fig.legend([Patch(facecolor=COMPONENT_COLORS[key]) for key in legend_order], legend_labels,
               ncol=5, loc="center", bbox_to_anchor=(.675, .355), fontsize=11.5,
               columnspacing=.8, handletextpad=.3, handlelength=1.9)
    for ax, sheet, metric, ylabel, upper in ((f, "Fig1f_Energy", "facility_energy_twh", "Energy consumption (TWh)", 700),
                                           (g, "Fig1g_Carbon", "carbon_mtco2", "CO$_2$ emissions (MtCO$_2$)", 250)):
        data = source.table(sheet)
        for j, scenario in enumerate(SCENARIOS):
            rows = data.loc[data.scenario.eq(scenario)].set_index("year").loc[years]
            ax.bar(x + (j - 1.5) * .20, rows[metric], .18,
                   color=SCENARIO_PAIRS[j][0], edgecolor="#333333", linewidth=.65, label=SCENARIO_LABELS[j])
        ax.set_xticks(x, years); ax.set_xlim(-.5, len(years) - .5)
        ax.set_ylabel(ylabel); ax.set_ylim(0, upper)
        ax.legend(ncol=4, loc="upper left", handlelength=1.7, columnspacing=.3,
                  handletextpad=.25, fontsize=10, borderaxespad=.1)
    return fig


def policy_bars(ax, data, x, *, column_prefix=False):
    containers = []
    for j, (policy, color) in enumerate(zip(POLICIES, COLORS)):
        values = data[f"{policy.lower()}_carbon_mtco2"] if column_prefix else data[policy]
        bars = ax.bar(x + (j - 1) * .22, values, .205, color=color, edgecolor="black", linewidth=1.0,
                      label=f"{policy} (emissions)" if column_prefix else policy)
        bar_values(ax, bars, fontsize=9.5, compact=True)
        containers.append(bars)
    return containers


def avoided_lines(ax, ndc, nz, x):
    handles = []
    for values, name, color, marker in ((ndc, "NDC (avoided)", PALE, "o"), (nz, "NZ (avoided)", SALMON, "^")):
        line, = ax.plot(x, values, ls=(0, (6, 5)), color=color, marker=marker, markersize=9,
                        markeredgecolor="black", markeredgewidth=.7, label=name, lw=1.6)
        for xpos, value in zip(x, values):
            annotation = ax.annotate(number(value), (xpos, value), xytext=(0, -6 if marker == "o" else 4),
                        textcoords="offset points", ha="center", va="top" if marker == "o" else "bottom", fontsize=10)
            annotation.set_gid("avoided_label")
        handles.append(line)
    ax.spines["right"].set_visible(True)
    ax.set_ylabel("Avoided emissions relative to CP (MtCO$_2$)")
    return handles


def figure2(source, plt):
    fig = canvas(plt, 2)
    a = fig.add_axes([.075, .680, .880, .300])
    b = fig.add_axes([.075, .375, .855, .255])
    c = fig.add_axes([.075, .035, .380, .272])
    d = fig.add_axes([.585, .035, .380, .272])
    for letter, x, y in (("a", .002, .993), ("b", .002, .645), ("c", .002, .345), ("d", .550, .345)):
        panel(fig, letter, x, y)
    ci = source.table("Fig2a_CF_Map", ["country"]).sort_values("annual_avg_carbon_factor_kg_per_mwh", ascending=False)
    y = np.arange(len(ci))
    a.barh(y, ci.annual_avg_carbon_factor_kg_per_mwh, .78, color=BLUE, edgecolor="#222222", linewidth=.7)
    a.set_yticks(y, ci.country); a.set_ylim(len(y) - .5, -.5); a.set_xlim(0, 700)
    a.set_axisbelow(True); a.grid(axis="x", color="#AAAAAA", linestyle=(0, (4, 4)), lw=.5)
    for i, value in enumerate(ci.annual_avg_carbon_factor_kg_per_mwh):
        a.text(value + 3, i, number(value), va="center", fontsize=10)
    a.text(.32, .025, "Average direct carbon intensity (kg CO$_2$ MWh$^{-1}$)", transform=a.transAxes, fontsize=10.5)
    countries = source.table("Fig2b_Country_Abatement", ["display_label"]).sort_values("display_rank")
    x = np.arange(len(countries))
    bars = policy_bars(b, countries, x, column_prefix=True)
    b.set_ylabel("Carbon emissions (MtCO$_2$)"); b.set_ylim(0, 80)
    labels = ["Other\ncountries" if str(label).startswith("Other") else label for label in countries.display_label]
    b.set_xticks(x, labels); b.set_xlim(-.5, len(x) - .5)
    right = b.twinx(); right.set_ylim(-10, 38)
    lines = avoided_lines(right, countries.ndc_avoided_vs_cp_mtco2, countries.nz_avoided_vs_cp_mtco2, x)
    b.legend([*bars, *lines], ["CP (emissions)", "NDC (emissions)", "NZ (emissions)", "NDC (avoided)", "NZ (avoided)"],
             ncol=2, loc="upper right", bbox_to_anchor=(1, 1.09), columnspacing=.5, handletextpad=.3, fontsize=9.5)
    total = source.table("Fig2c_Selected_Countries_Carbon")
    annual = total.pivot(index="year", columns="policy", values="carbon_mtco2").sort_index()
    x = np.arange(len(annual)); bars = policy_bars(c, annual, x)
    c.set_ylabel("Carbon emissions (MtCO$_2$)"); c.set_ylim(0, 165)
    c.set_xticks(x, annual.index); c.set_xlim(-.5, len(x) - .5)
    avoided = total.pivot(index="year", columns="policy", values="avoided_vs_cp_mtco2").loc[annual.index]
    right = c.twinx(); right.set_ylim(-5, 75)
    lines = avoided_lines(right, avoided.NDC, avoided.NZ, x)
    c.legend([*bars, *lines], ["CP", "NDC", "NZ", "NDC", "NZ"], ncol=2, loc="upper left", fontsize=9.5,
             columnspacing=.4, handletextpad=.25)
    weighted = source.table("Fig2d_Weighted_CI")
    for policy, color, marker in zip(POLICIES, COLORS, ("s", "o", "^")):
        rows = weighted.loc[weighted.policy.eq(policy)].sort_values("year")
        d.plot(rows.year, rows.load_weighted_carbon_intensity_kg_per_mwh,
               color=color, marker=marker, markersize=8, markeredgecolor="black", markeredgewidth=.65, lw=1.8, label=policy)
        for xpos, value in zip(rows.year, rows.load_weighted_carbon_intensity_kg_per_mwh):
            d.annotate(number(value), (xpos, value), xytext=(0, 5 if policy == "CP" else -6),
                        textcoords="offset points", ha="center", va="bottom" if policy == "CP" else "top", fontsize=10)
    d.set_ylabel("Load-weighted carbon intensity (kg CO$_2$ MWh$^{-1}$)")
    d.set_ylim(140, 360); d.set_yticks(np.arange(160, 361, 40)); d.set_xticks(sorted(weighted.year.unique()))
    d.set_axisbelow(True); d.grid(axis="y", ls=(0, (4, 4)), color="#999999", lw=.5)
    d.legend(loc="lower left", bbox_to_anchor=(0, .20), fontsize=10, handletextpad=.3)
    return fig


def endpoint(ax, frame, metric, ylabel, title, upper):
    x = np.arange(len(frame))
    for offset, prefix, label, color in ((-.20, "annual_cpu", "Annual CPU model", CYAN),
                                         (.20, "hourly_workload_resolved", "Hourly workload-resolved model", SALMON)):
        bars = ax.bar(x + offset, frame[f"{prefix}_{metric}"], .39, color=color, edgecolor="black", linewidth=1.1, label=label)
        bar_values(ax, bars, inside=True, fontsize=10)
    ax.set_ylim(0, upper); ax.set_xlim(-.5, len(frame) - .5); ax.set_xticks(x, frame.year)
    ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontweight="bold", fontsize=11, pad=10)
    right = ax.twinx()
    suffix = "facility_energy" if metric == "facility_energy_twh" else "carbon"
    values = frame[f"annual_vs_hourly_{suffix}_difference_pct"]
    line, = right.plot(x, values, color=GOLD, marker="o", markersize=9, lw=1.7, ls=(0, (6, 5)), label="Difference (%)")
    right.set_ylim(2.5, max(12, math.ceil(values.max() / 2) * 2)); right.set_ylabel("Difference (%)", labelpad=3)
    right.spines["right"].set_visible(True)
    for xpos, value in zip(x, values):
        text = right.annotate(number(value), (xpos, value), xytext=(0, 7), textcoords="offset points", ha="center", color=GOLD, fontsize=10)
        text.set_gid("difference_label")
    handles, labels = ax.get_legend_handles_labels()
    if metric == "facility_energy_twh":
        ax.legend([*handles, line], [*labels, "Difference (%)"], loc="upper left", fontsize=9.5,
                  borderaxespad=.35, handletextpad=.35)
    else:
        ax.legend(loc="upper left", fontsize=9.5, borderaxespad=.35, handletextpad=.35)


def waterfall(ax, start, end, differences, title, ylabel, step):
    values = np.asarray(differences, dtype=float)
    level = float(start)
    levels = [level]
    ax.bar(0, level, .48, color=BLUE, edgecolor="black", linewidth=1.0)
    ax.text(0, level + step * .025, number(level), ha="center", va="bottom", fontsize=10)
    for index, value in enumerate(values, start=1):
        new = level + value
        ax.plot([index - .76, index - .24], [level, level], color="#888888", ls=(0, (5, 5)), lw=1.4)
        ax.bar(index, abs(value), .48, bottom=min(level, new), color=CYAN if value >= 0 else PALE,
               edgecolor="black", linewidth=1.0)
        ax.text(index, max(level, new) + step * .025 if value >= 0 else min(level, new) - step * .025,
                number(value), ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
        level = new; levels.append(level)
    if not np.isclose(level, end, rtol=1e-12, atol=1e-10):
        raise ValueError("Waterfall does not close to the frozen hourly endpoint.")
    ax.plot([4.24, 4.76], [level, level], color="#888888", ls=(0, (5, 5)), lw=1.4)
    ax.bar(5, end, .48, color=SALMON, edgecolor="black", linewidth=1.0)
    ax.text(5, end + step * .025, number(end), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(6), ("Annual\nCPU model", "Component\npower", "Mean\nworkload", "Load\nvariability", "Carbon\nmatching", "Hourly-resolved\nmodel"), fontsize=9.5)
    ax.set_ylim(0, math.ceil(max(levels) / step) * step + step)
    ax.set_yticks(np.arange(0, ax.get_ylim()[1] + 1, step))
    ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontweight="bold", fontsize=11, pad=5)
    ax.set_axisbelow(True); ax.grid(axis="y", ls=(0, (2, 5)), color="#BBBBBB", lw=.6)


def figure3(source, plt):
    fig = canvas(plt, 3)
    a1 = fig.add_axes([.070, .775, .405, .195]); a2 = fig.add_axes([.565, .775, .380, .195])
    b1 = fig.add_axes([.070, .542, .435, .185]); b2 = fig.add_axes([.565, .542, .405, .185])
    panel(fig, "a", .002, .995); panel(fig, "b", .002, .744)
    panel(fig, "c", .002, .500); panel(fig, "d", .002, .287)
    frame = source.table("Fig3a_Model_Comparison", ["year"]).sort_values("year")
    endpoint(a1, frame, "facility_energy_twh", "Facility electricity use (TWh)", "Facility electricity use", 600)
    endpoint(a2, frame, "carbon_mtco2", "CO$_2$ emissions (MtCO$_2$)", "CO$_2$ emissions", 180)
    bridge = source.table("Fig3b_Effect_Decomposition", ["effect"]).sort_values("effect_order")
    waterfall(b1, frame.annual_cpu_facility_energy_twh.mean(), frame.hourly_workload_resolved_facility_energy_twh.mean(),
               bridge.annual_avg_energy_effect_twh, "Energy bridge", "Electricity use (TWh)", 50)
    waterfall(b2, frame.annual_cpu_carbon_mtco2.mean(), frame.hourly_workload_resolved_carbon_mtco2.mean(),
               bridge.annual_avg_carbon_effect_mtco2, "Carbon bridge", "CO$_2$ emissions (MtCO$_2$)", 20)
    shape = source.table("Fig3c_Operating_Shape", ["profile", "year"])
    specifications = (("mean_power_gw", "Mean Power", (18, 57)), ("peak_power_gw", "Peak Power", (18, 65)),
                      ("p99_power_gw", "P99 Power", (18, 65)),
                      ("max_abs_ramp_gw_per_h", "Maximum absolute hourly ramp", (-2, 15)))
    for j, (column, title, limits) in enumerate(specifications):
        ax = fig.add_axes([.070 + .237 * j, .323, .190, .164])
        for profile, color, label in (("flat_trace_mean_workload", CYAN, "Flat mean workload"),
                                      ("hourly_trace_workload", SALMON, "Full hourly trace")):
            rows = shape.loc[shape.profile.eq(profile)].sort_values("year")
            ax.plot(rows.year, rows[column], marker="o", markersize=9, color=color,
                    markeredgecolor="black", markeredgewidth=.7, lw=.7, label=label)
            for xpos, value in zip(rows.year, rows[column]):
                ax.annotate(number(value), (xpos, value), xytext=(0, 5 if profile.startswith("hourly") else -5),
                            textcoords="offset points", ha="center", va="bottom" if profile.startswith("hourly") else "top",
                            color=color, fontsize=8)
        ax.set_xticks(sorted(shape.year.unique())); ax.set_ylim(*limits)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10 if j == 3 else 10.5, pad=5)
        ax.legend(loc="upper left", fontsize=8.5, handletextpad=.25, borderaxespad=.1)
        if j == 0:
            ax.set_ylabel("Power (GW)")
        elif j == 3:
            ax.set_ylabel("Ramp (GW h$^{-1}$)")
    country = source.table("Fig3d_Country_Peak_Ramp", ["country"]).sort_values("peak_power_gw")
    for j, (metric, title, xlabel, upper) in enumerate((("peak_power_gw", "Peak Power in 2030", "Peak power (GW)", 33),
                                                     ("max_abs_ramp_gw_per_h", "Maximum absolute hourly ramp in 2030", "Maximum (GW h$^{-1}$)", 7))):
        ax = fig.add_axes([.070 + .495 * j, .040, .435 if j == 0 else .405, .235])
        y = np.arange(len(country))
        ax.barh(y, country[metric], .80, color=BLUE, edgecolor="black", linewidth=.65)
        ax.set_yticks(y, country.country, fontsize=8); ax.set_ylim(len(y) - .5, -.5)
        ax.set_xlim(0, upper); ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5, pad=5)
        ax.set_axisbelow(True); ax.grid(axis="y", color="#DDDDDD", lw=.4, ls=(0, (3, 4)))
    return fig


def tornado(fig, ax, data, parameters, labels, panel_name, *, font_size=12, supplemental=False):
    span = max(8, float(np.max(np.abs(data.change_vs_baseline_pct))))
    ax.set_xlim(-span * 1.1, span * 1.1)
    ax.set_ylim(len(parameters) - .55, -.55)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_yticks(range(len(parameters)), labels, fontsize=font_size)
    ax.tick_params(axis="y", length=2)
    ax.set_xlabel("Change (%)", fontsize=font_size + 1)
    ax.set_axisbelow(True); ax.grid(axis="x", color="#DDDDDD", lw=.5, ls=(0, (3, 4)))
    for index, name in enumerate(parameters):
        rows = data.loc[data.parameter_name.eq(name)].set_index("level")
        zero_drawn = False
        values = {level: float(rows.loc[level, "change_vs_baseline_pct"]) for level in ("Low", "High")}
        same_side = values["Low"] * values["High"] >= 0 and max(abs(v) for v in values.values()) < .15 and any(v != 0 for v in values.values())
        for level, color in (("High", SALMON), ("Low", CYAN)):
            value = values[level]
            if value == 0 and zero_drawn:
                continue
            ax.barh(index, value, .64, color=color, edgecolor="black", linewidth=.9, zorder=3)
            text = number(value) + "%"
            y = index + ((-.17 if level == "Low" else .17) if same_side else 0)
            if value == 0:
                ax.text(0, y, text, ha="center", va="center", fontsize=font_size, zorder=4)
                zero_drawn = True
            else:
                inside = abs(value) > (2 if supplemental else 1)
                offset = -3 if (value > 0 and inside) or (value < 0 and not inside) else 3
                align = "right" if offset < 0 else "left"
                ax.annotate(text, (value, y), xytext=(offset, 0), textcoords="offset points",
                            va="center", ha=align, fontsize=font_size, zorder=4)
        baseline = rows.loc["Base", "parameter_value"]
        description = f"Low={number(rows.loc['Low', 'parameter_value'])}; Base={number(baseline)}; High={number(rows.loc['High', 'parameter_value'])}"
        if len(description) > 43 or name == "classified_inference_gpu_load_share":
            description = description.replace("; High=", ";\nHigh=")
        ax.text(-.015, index + (.36 if supplemental else .22), description, transform=ax.get_yaxis_transform(), ha="right", va="top",
                color="#888888", fontsize=font_size - 2)
    ax.legend([Patch(facecolor=SALMON, edgecolor="black"), Patch(facecolor=CYAN, edgecolor="black")],
              ["High value (%)", "Low value (%)"], ncol=2, loc="lower center", bbox_to_anchor=(.50, 1.0),
              handletextpad=.3, columnspacing=1.3, fontsize=font_size)
    ax.annotate("Decrease", (.15, -.13), xycoords="axes fraction", ha="right", color=CYAN, fontsize=font_size + 1)
    ax.annotate("", (.21, -.12), (.34, -.12), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": CYAN, "lw": 1.2})
    ax.annotate("Increase", (.90, -.13), xycoords="axes fraction", ha="left", color=SALMON, fontsize=font_size + 1)
    ax.annotate("", (.87, -.12), (.74, -.12), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": SALMON, "lw": 1.2})


def figure4(source, plt):
    fig = canvas(plt, 4)
    data = source.table("W08_Sensitivity_Panels", ["panel", "experiment_order"])
    for letter, bottom in (("a", .565), ("b", .075)):
        ax = fig.add_axes([.235, bottom, .752, .385])
        tornado(fig, ax, data.loc[data.panel.eq(letter) & data.parameter_name.isin(MAIN_PARAMETERS)],
                MAIN_PARAMETERS, MAIN_LABELS, letter, font_size=12)
        panel(fig, letter, .004, bottom + .435)
    return fig


def clear_endpoint_annotation_collisions(fig):
    """Move only updated numeric annotations that overlap other numeric labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bars = [text for ax in fig.axes for text in ax.texts if text.get_gid() == "bar_value"]
    for text in (text for ax in fig.axes for text in ax.texts if text.get_gid() in ("difference_label", "avoided_label")):
        for _ in range(8):
            bbox = text.get_window_extent(renderer)
            if not any(bbox.overlaps(other.get_window_extent(renderer)) for other in bars):
                break
            x, y = text.get_position()
            text.set_position((12, y)) if text.get_gid() == "avoided_label" and x == 0 else text.set_position((x, y + 4))
            fig.canvas.draw()


def supplement_attribution(source, plt):
    configure(plt, 10.5)
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.2), gridspec_kw={"hspace": .46, "wspace": .45})
    for ax, sheet, key, title in ((axes[0, 0], "W08_Component_Display", "component", "(a) All physical components"),
                                 (axes[0, 1], "W08_Task_Display", "task_type", "(b) All tasks and unallocated idle")):
        data = source.table(sheet).copy()
        labels = data[key].replace({"cpu": "CPU", "gpu": "GPU", "memory": "Memory", "storage": "Storage", "it_fan": "ITFan",
                                    "training": "Training", "inference": "Inference", "other": "Other", "unclassified": "Unclassified", "idle_pool": "Unallocated idle"})
        y = np.arange(len(data))
        ax.barh(y, data.idle_share_of_system_pct, color=CYAN, edgecolor="black", linewidth=.6, label="Idle")
        ax.barh(y, data.dynamic_share_of_system_pct, left=data.idle_share_of_system_pct,
                color=BLUE, edgecolor="black", linewidth=.6, label="Dynamic")
        for i, value in enumerate(data.share_of_system_pct):
            ax.text(value + .6, i, number(value), va="center", fontsize=10)
        ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0, max(data.share_of_system_pct) * 1.22)
        ax.set_xlabel("Share of system IT energy (%)"); ax.set_title(title, loc="left", fontweight="bold")
        ax.legend(loc="upper right" if key == "task_type" else "lower right", fontsize=9)
    cells = source.table("W08_Task_Component_Display")
    table = cells.loc[cells.task_type.eq("unclassified")].set_index("component").loc[list(COMPONENTS)]
    pie(axes[1, 0], table.annual_avg_it_energy_mwh / table.annual_avg_it_energy_mwh.sum() * 100, text_size=10)
    axes[1, 0].set_title("(c) Unclassified task components", loc="left", fontweight="bold")
    tasks = ["training", "inference", "other", "unclassified", "idle_pool"]
    matrix = cells.pivot(index="task_type", columns="component", values="share_of_system_it_energy_pct").loc[tasks, list(COMPONENTS)]
    ax = axes[1, 1]
    cmap = LinearSegmentedColormap.from_list("Original blue", ["white", CYAN, BLUE])
    ax.imshow(matrix, cmap=cmap, vmin=0, aspect="auto")
    ax.set_xticks(range(5), ("GPU", "Memory", "ITFan", "Storage", "CPU"), fontsize=9)
    ax.set_yticks(range(5), ("Training", "Inference", "Other", "Unclassified", "Idle pool"), fontsize=9)
    for (i, j), value in np.ndenumerate(matrix.to_numpy()):
        ax.text(j, i, number(value), ha="center", va="center", fontsize=8.5, color="white" if value > 20 else "black")
    ax.set_title("(d) Task × component: system IT energy (%)", loc="left", fontweight="bold", fontsize=10)
    return fig


def supplement_parameters(source, plt):
    configure(plt, 11)
    data = source.table("W08_Sensitivity_Panels")
    parameters = data.loc[~data.parameter_name.isin(MAIN_PARAMETERS)].sort_values("experiment_order").parameter_name.drop_duplicates().tolist()
    labels = data.drop_duplicates("parameter_name").set_index("parameter_name").parameter_label
    labels = [textwrap.fill(labels[name], 26) for name in parameters]
    fig = plt.figure(figsize=(10.09, 13.2))
    for letter, bottom in (("a", .56), ("b", .065)):
        ax = fig.add_axes([.31, bottom, .675, .39])
        tornado(fig, ax, data.loc[data.panel.eq(letter) & data.parameter_name.isin(parameters)],
                parameters, labels, letter, font_size=10, supplemental=True)
        panel(fig, letter, .005, bottom + .435)
    return fig


def supplement_ramps(source, plt):
    configure(plt, 10.5)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), gridspec_kw={"wspace": .33})
    rows = source.table("Fig3c_Operating_Shape")
    rows = rows.loc[rows.profile.eq("hourly_trace_workload")].sort_values("year")
    for column, label, color, style in (("max_abs_ramp_gw_per_h", "Raw maximum", BLUE, "-"),
                                       ("max_abs_ramp_excluding_seams_gw_per_h", "Exclude repeat seams", CYAN, "--"),
                                       ("max_abs_ramp_excluding_alignment_gw_per_h", "Exclude all joins", SALMON, ":")):
        axes[0].plot(rows.year, rows[column], color=color, ls=style, marker="o", label=label, lw=1.4)
    for column, label, color, style in (("p95_abs_ramp_gw_per_h", "P95", CYAN, "-"),
                                       ("p99_abs_ramp_gw_per_h", "P99", SALMON, "-"),
                                       ("p95_abs_ramp_excluding_seams_gw_per_h", "P95 excluding seams", CYAN, "--"),
                                       ("p99_abs_ramp_excluding_seams_gw_per_h", "P99 excluding seams", SALMON, "--")):
        axes[1].plot(rows.year, rows[column], color=color, ls=style, marker="o", label=label)
    for ax, title in zip(axes, ("(a) Maximum absolute hourly ramp", "(b) High-percentile hourly ramps")):
        ax.set_ylabel("Absolute hourly ramp (GW h$^{-1}$)"); ax.set_xlabel("Year"); ax.set_xticks(rows.year)
        ax.set_ylim(bottom=0); ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.legend(fontsize=8.5)
    return fig


BUILDERS = {1: figure1, 2: figure2, 3: figure3, 4: figure4}
SUPPLEMENT_BUILDERS = {"supplementary_task_energy_closure": (1, supplement_attribution),
                       "supplementary_additional_sensitivity": (4, supplement_parameters),
                       "supplementary_ramp_diagnostics": (3, supplement_ramps)}
