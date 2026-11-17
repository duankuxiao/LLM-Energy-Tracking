"""Render W08 figures from regenerated source workbooks without running models.

Examples (run only after regenerating the four workbooks):
    python -B scripts/plot_figures.py --input-dir results --figures 1 2 3 4
    python -B scripts/plot_figures.py --figures 1 --figure1-workbook results/my_fig1.xlsx

Outputs default to results/revision_w08/figures, leaving manuscript JPGs intact.
SVG/PDF retain editable text; PNG is a preview. Use --formats svg pdf tiff
for TIFF export. Existing outputs require --overwrite. Every figure receives
a caption, exact-value CSV, panel metadata CSV and a source-hash manifest.
Rendering and final-size visual review are separate from code-only preparation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import textwrap

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.research_io import sha256_file as _hash
from scripts.figure_contract import (
    COMPONENT_LABELS, EFFECT_LABELS, POLICY_COLORS, SCENARIO_COLORS,
    SCENARIO_COLUMNS, SCHEMA_VERSION, TASK_LABELS,
    display_number, ordered_names, require_unique,
)


class FigureSource:
    def __init__(self, path: Path, number: int):
        self.path, self.number = path, number
        self.sha256 = _hash(path)
        self.tables: dict[str, pd.DataFrame] = {}
        with pd.ExcelFile(path, engine="openpyxl") as book:
            required = {"W08_Figure_Contract", "W08_Panel_Metadata", "W08_Display_Values", "Accounting_Boundary"}
            if not required <= set(book.sheet_names):
                raise ValueError(f"{path.name} predates W08; regenerate it with figure{number}_data.py first.")
            contract = pd.read_excel(book, sheet_name="W08_Figure_Contract", keep_default_na=False)
            if len(contract) != 1 or contract.iloc[0]["schema_version"] != SCHEMA_VERSION or int(contract.iloc[0]["figure"]) != number:
                raise ValueError(f"{path.name} has the wrong figure number or W08 schema.")
            self.metadata = pd.read_excel(book, sheet_name="W08_Panel_Metadata")
            self.values = pd.read_excel(book, sheet_name="W08_Display_Values", keep_default_na=False,
                                       dtype={"display_text": str})
            self.boundary = pd.read_excel(book, sheet_name="Accounting_Boundary").iloc[0]
            names = set(self.metadata["source_sheet"])
            names.update({"W08_Load_Weighted_CI", "W08_Scenario_Capacity_Audit"} if number == 1 else ())
            for name in names:
                self.tables[name] = pd.read_excel(book, sheet_name=name)
            for _, row in self.metadata.iterrows():
                if row["source_column"] not in self.tables[row["source_sheet"]]:
                    raise ValueError(f"Missing contracted source column: {row['source_sheet']}.{row['source_column']}")
            for _, row in self.values.iterrows():
                actual = self.tables[row["source_sheet"]].iloc[int(row["source_excel_row"]) - 2][row["source_column"]]
                raw = row["raw_value"]
                if raw == "" and pd.isna(actual):
                    continue
                if not np.isclose(float(actual), float(raw), rtol=1e-12, atol=0, equal_nan=True):
                    raise ValueError(f"Display value differs from its source cell: {row['source_sheet']}!{row['source_excel_row']}")
            # Carry truncation and grid fallback evidence into captions/manifests.
            self.run_notes = []
            for name in book.sheet_names:
                if name.endswith("Run_Metadata") or name == "Grid_Scenario_Definition":
                    table = pd.read_excel(book, sheet_name=name)
                    if "trace_truncated_by_max_intervals" in table and table["trace_truncated_by_max_intervals"].astype(str).str.lower().eq("true").any():
                        self.run_notes.append("Trace was truncated for this source workbook; this is a diagnostic figure.")
                if name.endswith("Grid_Input_Audit"):
                    table = pd.read_excel(book, sheet_name=name)
                    if "carbon_factor_source" in table and table["carbon_factor_source"].eq("annual_fallback").any():
                        self.run_notes.append("Some hourly grid inputs used annual-factor fallback; consult the workbook grid audit.")
                    if "calendar_status" in table and table["calendar_status"].ne("full_calendar").any():
                        self.run_notes.append("Some grid series do not cover a full calendar year; consult the workbook grid audit.")

    def table(self, name: str, keys: list[str] | None = None) -> pd.DataFrame:
        result = self.tables[name].copy()
        if keys:
            require_unique(result, keys, name)
        return result

    def context(self) -> str:
        scenarios, policies, years = set(), set(), set()
        for table in self.tables.values():
            for column, result in (("scenario", scenarios), ("policy", policies)):
                if column in table:
                    result.update(table[column].dropna().astype(str))
            for column in ("year", "year_start", "year_end"):
                if column in table:
                    years.update(table[column].dropna().astype(int))
        period = f"{min(years)}–{max(years)}" if years else "fixed input shares"
        scope = str(self.boundary["scope_label"])
        return (f"{scope} | {period} | "
                f"{', '.join(ordered_names(scenarios, SCENARIO_COLUMNS))} | "
                f"{', '.join(ordered_names(policies, POLICY_COLORS))}")

    def caption(self) -> str:
        lines = [f"Figure {self.number}. {self.context()}.",
                 f"Calibration basis: {self.boundary['calibration_basis']}."]
        lines.append("The study boundary comprises the 24 major AI countries with available country-capacity and grid-carbon inputs. "
                     "Results retain the new experiment values. The original country allocation is not renormalized; its 93.18% IT-capacity share "
                     "does not establish electricity or emissions coverage. No residual region is modelled.")
        if str(self.boundary["calibration_basis"]).startswith("legacy_"):
            lines.append("Legacy capacity factors fit the global reference electricity pathway to the selected-country aggregate. "
                         "Absolute results are conditional on this calibration convention, not observed 24-country electricity or a reconstructed global total.")
        for panel, rows in self.metadata.groupby("panel", sort=False):
            row = rows.iloc[0]
            lines.append(f"{panel}. {row['interpretation']} Carbon scope: {row['carbon_scope']}.")
        if self.number == 1:
            ci = self.table("W08_Load_Weighted_CI")
            fixed = ci["fixed_across_all_years_and_scenarios"].astype(str).str.lower().eq("true").all()
            lines.append("The load-weighted carbon intensity is " + ("constant" if fixed else "not constant")
                         + " across all displayed years and scenarios, within rtol=1e-10 and atol=1e-10 kgCO2/MWh.")
            rule = self.tables["W08_Task_Display"]["idle_allocation_rule"].iloc[0]
            lines.append(f"Task attribution uses {rule}; unallocated idle is shown explicitly. Country-share source year is undocumented.")
        if self.number == 4:
            lines.append("Markers identify actual parameter values. Zero is exact zero; scientific notation retains small nonzero responses. NA denotes an undefined percentage.")
            lines.append("Responses of order 1e-14% are floating-point residuals. These deterministic one-factor contrasts are not confidence intervals or joint uncertainty estimates.")
        if self.number == 3:
            lines.append("In source-table endpoint percentages, the sign is annual CPU minus hourly reference, divided by the hourly reference. Bridge contributions instead use next configuration minus previous configuration.")
            lines.append("Power profiles use the complete UTC calendar (8,784 hours in 2028), with annual electricity retaining its 8,760-hour annualization. "
                         "Identical raw, repeat-seam-excluded and all-join-excluded maximum-ramp curves are drawn once and labelled All filters; all three values are retained in source data. "
                         "A shared normalized country workload shape is a reference assumption, not independently observed national behaviour.")
        if self.number == 2:
            lines.append("NDC and NZ are stylized uniform input scalings of CP, with 2030 factors 0.89 and 0.53. Their proportional reductions follow the scenario definition; they are not independently estimated policy effects.")
        lines.extend(dict.fromkeys(self.run_notes))
        lines.append("CPU, central processing unit; GPU, graphics processing unit; IT, information technology; CI, carbon intensity. CP, NDC and NZ denote stylized grid input scenarios.")
        lines.append("Raw numeric values, units and workbook row/column references accompany the figure; display labels use three significant digits. Model panels are deterministic conditional calculations; no sampling-based error bars are inferred.")
        return "\n\n".join(lines) + "\n"


def _title(ax, letter: str, text: str) -> None:
    ax.set_title(f"{letter}  {text}", loc="left", fontweight="bold", fontsize=8)


def _format_axes(ax, axis: str = "y") -> None:
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(FuncFormatter(lambda value, _pos: display_number(value)))
    target.set_major_locator(MaxNLocator(nbins=5))


def _lines(ax, table, names, key, metric, colors) -> None:
    require_unique(table, [key, "year"], metric)
    years = sorted(table["year"].unique())
    for name in names:
        data = table.loc[table[key] == name].set_index("year").reindex(years)
        if not np.isfinite(data[metric].to_numpy(dtype=float)).all():
            raise ValueError(f"Incomplete annual {metric} path for {name}.")
        ax.plot(years, data[metric], marker="o", markersize=2.5, label=name, color=colors[name])
    ax.set_xticks(years)
    ax.set_xlabel("Year")
    _format_axes(ax)


def _stacked_share(ax, table, key, labels) -> None:
    data = table.set_index(key).reindex(labels)
    if data[["idle_share_of_system_pct", "dynamic_share_of_system_pct"]].isna().any().any():
        raise ValueError("Incomplete physical/task energy display table.")
    x = np.arange(len(data))
    ax.barh(x, data["idle_share_of_system_pct"], color="#B6C5D0", label="Idle")
    ax.barh(x, data["dynamic_share_of_system_pct"], left=data["idle_share_of_system_pct"], color="#386D82", label="Dynamic")
    ax.set_yticks(x, [labels[name] for name in data.index])
    ax.invert_yaxis()
    ax.set_xlabel("Share of system IT energy (%)")
    ax.set_xlim(0, max(1, float(data["share_of_system_pct"].max()) * 1.2))
    for y, value in enumerate(data["share_of_system_pct"]):
        ax.annotate(display_number(value), (value, y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6)
    ax.legend(fontsize=6)
    _format_axes(ax, "x")


def figure1(source, plt):
    fig = plt.figure(figsize=(180 / 25.4, 275 / 25.4), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=(1.35, .9, 1, 1))
    a, b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    c = fig.add_subplot(grid[1, :])
    d, e = fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])
    f, g = fig.add_subplot(grid[3, 0]), fig.add_subplot(grid[3, 1])
    capacity = source.table("Fig1a_Capacity", ["scenario", "year"])
    year = int(capacity["year"].max())
    names = ordered_names(capacity["scenario"], SCENARIO_COLUMNS)
    capacity = capacity.loc[capacity["year"] == year].set_index("scenario").reindex(names)
    x, width = np.arange(len(names)), .24
    for offset, (column, label, color) in enumerate((
        ("total_capacity_gw", "Total facility", "#CCD2D5"),
        ("it_capacity_gw", "All IT", "#7E9AAD"),
        ("ai_gpu_it_capacity_gw", "AI-focused IT", "#315B7D"))):
        bars = a.bar(x + (offset - 1) * width, capacity[column], width, label=label, color=color)
        a.bar_label(bars, labels=[display_number(v) for v in capacity[column]], padding=3, fontsize=5.5)
    a.set_xticks(x, [name.replace(" ", "\n") for name in names], fontsize=6)
    a.set_ylabel("Capacity (GW)")
    a.margins(y=.3)
    a.legend(fontsize=6, loc="upper left")
    _title(a, "a", f"Capacity inputs, {year}")
    _format_axes(a)
    shares = source.table("Fig1b_Country_Share", ["country"]).sort_values("it_capacity_share_pct")
    b.barh(shares["country"], shares["it_capacity_share_pct"], color="#7E9AAD")
    b.tick_params(axis="y", labelsize=6)
    b.set_xlabel("Original global IT allocation (%)")
    _title(b, "b", "Fixed country input shares")
    _format_axes(b, "x")
    cells = source.table("W08_Task_Component_Display", ["task_type", "component"])
    matrix = cells.pivot(index="task_type", columns="component", values="share_of_system_it_energy_pct").reindex(index=TASK_LABELS, columns=COMPONENT_LABELS)
    if matrix.isna().any().any():
        raise ValueError("Task/component matrix is incomplete; regenerate source workbook.")
    im = c.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto", vmin=0)
    c.set_xticks(range(len(COMPONENT_LABELS)), list(COMPONENT_LABELS.values()))
    c.set_yticks(range(len(TASK_LABELS)), list(TASK_LABELS.values()))
    for (i, j), value in np.ndenumerate(matrix.to_numpy()):
        c.text(j, i, display_number(value), ha="center", va="center", fontsize=6.5,
               color="white" if value > im.norm.vmax * .6 else "#202020")
    fig.colorbar(im, ax=c, label="System IT energy (%)", fraction=.03, pad=.02)
    _title(c, "c", "Base: task × component attribution")
    _stacked_share(d, source.table("W08_Component_Display", ["component"]), "component", COMPONENT_LABELS)
    _title(d, "d", "Base: physical components")
    _stacked_share(e, source.table("W08_Task_Display", ["task_type"]), "task_type", TASK_LABELS)
    _title(e, "e", "Base: tasks and idle pool")
    for ax, sheet, metric, label, letter in (
        (f, "Fig1f_Energy", "facility_energy_twh", "Electricity (TWh/year)", "f"),
        (g, "Fig1g_Carbon", "carbon_mtco2", "Emissions (MtCO2/year)", "g")):
        table = source.table(sheet)
        _lines(ax, table, names, "scenario", metric, SCENARIO_COLORS)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        _title(ax, letter, "Electricity: common capacity factors" if letter == "f" else "AI facility emissions")
    f.legend(fontsize=6)
    return fig


def figure2(source, plt):
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 190 / 25.4), layout="constrained")
    a, b, c, d = axes.flat
    ci = source.table("Fig2a_CF_Map", ["country"]).sort_values("annual_avg_carbon_factor_kg_per_mwh")
    a.barh(ci["country"], ci["annual_avg_carbon_factor_kg_per_mwh"], color=POLICY_COLORS["CP"])
    a.set_xlabel("Direct grid CI (kgCO2/MWh)")
    a.tick_params(axis="y", labelsize=6)
    _format_axes(a, "x")
    _title(a, "a", "CP: mean annual grid input")
    countries = source.table("Fig2b_Country_Abatement", ["display_label"]).sort_values("display_rank")
    x = np.arange(len(countries))
    for index, policy in enumerate(POLICY_COLORS):
        b.bar(x + (index - 1) * .25, countries[f"{policy.lower()}_carbon_mtco2"], .25,
              color=POLICY_COLORS[policy], label=policy)
    b.set_xticks(x, ["Other\nmodelled" if name == "Other modelled countries" else name for name in countries["display_label"]], rotation=45, rotation_mode="anchor", ha="right", fontsize=6)
    b.set_ylabel("AI facility emissions (MtCO2/year)")
    b.legend(fontsize=6)
    _format_axes(b)
    _title(b, "b", f"Country emissions, {int(countries['year'].iloc[0])}")
    for ax, sheet, metric, label, letter, title in (
        (c, "Fig2c_Selected_Countries_Carbon", "carbon_mtco2", "AI facility emissions (MtCO2/year)", "c", "Stylized policy input paths"),
        (d, "Fig2d_Weighted_CI", "load_weighted_carbon_intensity_kg_per_mwh", "Load-weighted CI (kgCO2/MWh)", "d", "Energy–carbon relationship")):
        table = source.table(sheet)
        _lines(ax, table, ordered_names(table["policy"], POLICY_COLORS), "policy", metric, POLICY_COLORS)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=6)
        _title(ax, letter, title)
    return fig


def _signed_effects(ax, table, metric, color):
    values = table[metric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Undefined bridge percentages in {metric}.")
    labels = [EFFECT_LABELS[name] for name in table["effect"]]
    y = np.arange(len(table))
    ax.barh(y, values, color=color)
    ax.set_yticks(y, labels, fontsize=6)
    ax.invert_yaxis()
    ax.axvline(0, color="#555555", linewidth=.6)
    ax.margins(x=.35)
    for index, value in enumerate(values):
        ax.annotate(display_number(value), (value, index), xytext=(3 if value >= 0 else -3, 0),
                    textcoords="offset points", ha="left" if value >= 0 else "right", va="center", fontsize=6)
    ax.set_xlabel("Change / hourly reference (%)")
    _format_axes(ax, "x")


def _ramp_lines(ax, table, x, *, label_prefix="", color="#315B7D"):
    specifications = (
        ("max_abs_ramp_gw_per_h", "Raw", "-"),
        ("max_abs_ramp_excluding_seams_gw_per_h", "Exclude repeat seams", "--"),
        ("max_abs_ramp_excluding_alignment_gw_per_h", "Exclude all joins", ":"))
    if all(np.array_equal(table[specifications[0][0]].to_numpy(), table[column].to_numpy())
           for column, _, _ in specifications[1:]):
        specifications = ((specifications[0][0], "All filters", "-"),)
    for column, label, style in specifications:
        ax.plot(x, table[column], style, marker="o", markersize=2,
                color=color, label=f"{label_prefix}{label}")
    ax.set_ylabel("Max. |hourly ramp| (GW per hour)")
    _format_axes(ax)


def figure3(source, plt):
    fig = plt.figure(figsize=(180 / 25.4, 255 / 25.4), layout="constrained")
    grid = fig.add_gridspec(4, 2)
    a1, a2, b1, b2, c1, c2, d1, d2 = [fig.add_subplot(grid[i, j]) for i in range(4) for j in range(2)]
    comparison = source.table("Fig3a_Model_Comparison", ["year"]).sort_values("year")
    hourly_label = ("Hourly: legacy calibration" if str(source.boundary["calibration_basis"]).startswith("legacy_")
                    else "Hourly: supplied capacity factors")
    for ax, suffix, ylabel in ((a1, "facility_energy_twh", "Electricity (TWh/year)"), (a2, "carbon_mtco2", "Emissions (MtCO2/year)")):
        for prefix, label, color in (("annual_cpu", "Annual CPU", "#A07449"),
                                     ("hourly_workload_resolved", hourly_label, "#315B7D")):
            ax.plot(comparison["year"], comparison[f"{prefix}_{suffix}"], marker="o", markersize=2, label=label, color=color)
        ax.set_xticks(comparison["year"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=6)
        _format_axes(ax)
    _title(a1, "a", "Common-input endpoint comparison")
    _title(a2, "", "Emissions comparison")
    bridge = source.table("Fig3b_Effect_Decomposition", ["effect"]).sort_values("effect_order")
    ordered_names(bridge["effect"], EFFECT_LABELS)
    _signed_effects(b1, bridge, "energy_effect_pct_of_hourly_reference", "#315B7D")
    _signed_effects(b2, bridge, "carbon_effect_pct_of_hourly_reference", "#428C7B")
    _title(b1, "b", "Ordered differences: electricity")
    _title(b2, "", "Ordered differences: emissions")
    shape = source.table("Fig3c_Operating_Shape", ["profile", "year"])
    profiles = {"flat_trace_mean_workload": ("Flat", "#A07449"), "hourly_trace_workload": ("Hourly", "#315B7D")}
    for profile in ordered_names(shape["profile"], profiles):
        label, color = profiles[profile]
        table = shape.loc[shape["profile"] == profile].sort_values("year")
        c1.plot(table["year"], table["peak_power_gw"], marker="o", markersize=2, color=color, label=f"{label} peak")
        if profile == "hourly_trace_workload":
            c1.plot(table["year"], table["p99_power_gw"], "--", color=color, label="Hourly P99")
            c1.plot(table["year"], table["mean_power_gw"], ":", color=color, label="Hourly mean")
        _ramp_lines(c2, table, table["year"], label_prefix=f"{label}: ", color=color)
    c1.set_ylabel("Facility power (GW)")
    c1.legend(fontsize=6)
    c2.legend(fontsize=5.5)
    for ax in (c1, c2):
        ax.set_xticks(sorted(shape["year"].unique()))
        ax.set_ylim(bottom=0)
        _format_axes(ax)
    _title(c1, "c", "Aggregate operating shape")
    _title(c2, "", "Absolute hourly ramp")
    country = source.table("Fig3d_Country_Peak_Ramp", ["country"]).sort_values("rank")
    x = np.arange(len(country))
    d1.bar(x, country["peak_power_gw"], color="#7E9AAD")
    d1.set_ylabel("Peak power (GW)")
    _ramp_lines(d2, country, x)
    d2.legend(fontsize=5.5)
    for ax in (d1, d2):
        ax.set_xticks(x, country["country"], rotation=90, rotation_mode="anchor", ha="right", va="center", fontsize=5.5)
        ax.set_ylim(bottom=0)
        _format_axes(ax)
    _title(d1, "d", f"Country peaks, {int(country['year'].iloc[0])}")
    _title(d2, "", "Country ramps: shared-shape case")
    return fig


def figure4(source, plt):
    data = source.table("W08_Sensitivity_Panels", ["panel", "experiment_order"])
    parameters = data.sort_values("experiment_order")["parameter_name"].drop_duplicates().tolist()
    height = max(180, len(parameters) * 13)
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, height / 25.4), layout="constrained", sharey=True)
    labels = data.drop_duplicates("parameter_name").set_index("parameter_name")["parameter_label"]
    styles = {"Low": ("#315B7D", "v", -.24), "Base": ("#777777", "o", 0), "High": ("#B66B35", "^", .24)}
    for ax, panel, title in zip(axes, ("a", "b"), ("Electricity sensitivity", "Emissions sensitivity")):
        frame = data.loc[data["panel"] == panel]
        finite_values = frame["change_vs_baseline_pct"].dropna().to_numpy(dtype=float)
        span = max(1., float(np.max(np.abs(finite_values))) if finite_values.size else 1.)
        ax.set_xlim(-span * 1.5, span * 1.5)
        ax.axvline(0, color="#888888", linewidth=.6)
        for index, parameter in enumerate(parameters):
            rows = frame.loc[frame["parameter_name"] == parameter]
            for _, row in rows.iterrows():
                color, marker, offset = styles[row["level"]]
                value = float(row["change_vs_baseline_pct"])
                y = index + offset
                if np.isfinite(value):
                    ax.plot([0, value], [y, y], color=color, linewidth=.7)
                    ax.plot(value, y, marker=marker, color=color, markersize=3)
                    # Numeric labels keep even sub-pixel nonzero responses legible.
                    text = f"{display_number(value)} ({display_number(row['parameter_value'])})"
                    ax.annotate(text, (value, y), xytext=(4 if value >= 0 else -4, 0), textcoords="offset points",
                                ha="left" if value >= 0 else "right", va="center", fontsize=5.5)
                else:
                    ax.text(0, y, f"NA ({display_number(row['parameter_value'])})", fontsize=5.5, va="center")
        for level, (color, marker, _) in styles.items():
            ax.plot([], [], linestyle="none", marker=marker, color=color, label=level, markersize=3)
        ax.legend(fontsize=6, loc="lower right")
        ax.set_xlabel("Change in annual mean (%)\nLabel: response (parameter value)")
        ax.set_yticks(range(len(parameters)), [textwrap.fill(labels[p], 26) for p in parameters], fontsize=6)
        ax.set_ylim(len(parameters) - .4, -.7)
        _format_axes(ax, "x")
        _title(ax, panel, title)
    return fig


def main(single_number=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=ROOT_DIR / "results" / "all_experiments" / "data" if single_number else ROOT_DIR / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results" / "revision_style" / "figures")
    parser.add_argument("--figures", type=int, nargs="+", choices=(single_number,) if single_number else (1, 2, 3, 4),
                        default=[single_number] if single_number else [1, 2, 3, 4])
    for number in range(1, 5):
        parser.add_argument(f"--figure{number}-workbook", type=Path)
    parser.add_argument("--formats", nargs="+", choices=("svg", "pdf", "png", "tiff", "jpg"), default=["svg", "pdf", "png"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--style", choices=("original", "expanded"), default="original",
                        help="Preserve the original manuscript style; expanded layout is retained only as an optional diagnostic view.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("Raster exports require at least 300 dpi.")
    numbers, formats = list(dict.fromkeys(args.figures)), list(dict.fromkeys(args.formats))
    suffixes = formats + ["caption.txt", "source_values.csv", "panel_metadata.csv", "manifest.json"]
    outputs = [args.output_dir / f"figure{number}.{suffix}" for number in numbers for suffix in suffixes]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        parser.error("Outputs already exist; choose a new directory or --overwrite: " + ", ".join(existing))
    sources = [FigureSource(getattr(args, f"figure{number}_workbook") or args.input_dir / f"figure{number}_data.xlsx", number)
               for number in numbers]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "legend.fontsize": 6, "axes.titlesize": 8,
                         "legend.frameon": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "lines.linewidth": 1.1})
    builders = {1: figure1, 2: figure2, 3: figure3, 4: figure4}
    if args.style == "original":
        from scripts import original_figure_style
        builders = original_figure_style.BUILDERS
        for source in sources:
            original_figure_style.adapt_source_contract(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        figure = builders[source.number](source, plt)
        if args.style == "original" and source.number in (2, 3):
            original_figure_style.clear_endpoint_annotation_collisions(figure)
        try:
            if args.style != "original":
                figure.suptitle("\n".join(textwrap.wrap(source.context(), 95)), fontsize=7)
            if source.run_notes:
                figure.supxlabel("\n".join(textwrap.wrap(" ".join(dict.fromkeys(source.run_notes)), 100)), fontsize=6)
            stem = args.output_dir / f"figure{source.number}"
            export_paths = {extension: stem.with_suffix(suffix) for extension, suffix in
                            (("svg", ".svg"), ("pdf", ".pdf"), ("png", ".png"), ("tiff", ".tiff"), ("jpg", ".jpg"))}
            for extension in formats:
                options = {"compression": "tiff_lzw"} if extension == "tiff" else ({"quality": 95} if extension == "jpg" else None)
                figure.savefig(export_paths[extension], dpi=args.dpi, facecolor="white",
                               **({"pil_kwargs": options} if options else {}))
            caption_text = original_figure_style.caption(source) if args.style == "original" else source.caption()
            stem.with_suffix(".caption.txt").write_text(caption_text, encoding="utf-8")
            source.values.to_csv(stem.with_suffix(".source_values.csv"), index=False)
            source.metadata.to_csv(stem.with_suffix(".panel_metadata.csv"), index=False)
            if args.style == "original":
                source.audit_values.to_csv(stem.with_suffix(".audit_source_values.csv"), index=False)
            source_dir = args.output_dir / f"figure{source.number}_sources"
            source_dir.mkdir(parents=True, exist_ok=True)
            for name, table in source.tables.items():
                table.to_csv(source_dir / f"{name}.csv", index=False)
            manifest = {"figure": source.number, "schema_version": SCHEMA_VERSION,
                        "source_workbook": source.path.name, "source_sha256": source.sha256,
                        "plot_script_sha256": _hash(Path(__file__)),
                        "display_contract_sha256": _hash(Path(__file__).with_name("figure_contract.py")),
                        "formats": formats, "dpi": args.dpi, "backend": "matplotlib",
                        "matplotlib_version": matplotlib.__version__, "pandas_version": pd.__version__,
                        "calibration_basis": str(source.boundary["calibration_basis"]),
                        "source_notes": list(dict.fromkeys(source.run_notes)),
                        "source_context": source.context(), "visual_QA": "pending_final_size_inspection",
                        "style": args.style,
                        "style_script_sha256": _hash(Path(__file__).with_name("original_figure_style.py")) if args.style == "original" else None,
                        "source_tables": {str(path.relative_to(args.output_dir)): _hash(path)
                                          for path in sorted(source_dir.glob("*.csv"))},
                        "panel_1a_year_selection": ("all_five_years_total_with_AI_subset" if args.style == "original" else "latest_year") if source.number == 1 else None,
                        "panel_2a_display": "country_bars_of_direct_annual_grid_inputs" if source.number == 2 else None,
                        "outputs": {f"figure{source.number}.{suffix}": _hash(args.output_dir / f"figure{source.number}.{suffix}")
                                    for suffix in suffixes if suffix != "manifest.json"}}
            stem.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Figure {source.number} exported to {args.output_dir}; final-size visual review is pending.")
        finally:
            plt.close(figure)


if __name__ == "__main__":
    main()
