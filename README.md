<p align="center">
  <img src="logo.png" alt="Project logo" width="180"/>
</p>

# LLM-Energy-Tracking

[中文说明](README.zh-CN.md)

A Python toolkit for estimating AI data centre electricity use and operational
carbon emissions. It combines capacity scenarios, GPU-cluster workload traces,
component power models, power usage effectiveness (PUE), and grid carbon intensity.

The toolkit supports annual and hourly calculations, model comparisons,
sensitivity analysis, and electricity-load diagnostics. Results are exported as
CSV tables, Excel workbooks, and figures.

## Features

| Function | Description | Implementation |
| --- | --- | --- |
| Annual model | Estimate country-level annual electricity and emissions from capacity and utilization assumptions | [Annual CPU model](core/annual_cpu_model.py) |
| Hourly model | Calculate CPU, GPU, memory, storage, and IT-fan electricity from workload traces | [Hourly workload-resolved model](core/hourly_workload_resolved_model.py) |
| Demand scenarios | Compare capacity, electricity, emissions, and task/component energy composition | [Demand analysis](scripts/experiment_figure1.py) |
| Grid scenarios | Compare emissions, avoided emissions, and load-weighted carbon intensity | [Grid analysis](scripts/experiment_figure2.py) |
| Model decomposition | Separate power configuration, mean workload, load variability, and hourly carbon-matching contributions; calculate peaks and ramps | [Model and load-profile analysis](scripts/experiment_figure3.py) |
| Sensitivity analysis | Vary power budgets, idle power, response curves, workload parameters, and carbon variability | [Sensitivity analysis](scripts/experiment_figure4.py) |
| External diagnostics | Compare reference power configurations with MLPerf Power data and inspect national electricity-data scopes | [Power diagnostics](scripts/validate_energy_model.py) |

The supplied configuration covers 2026–2030 and 24 countries. Demand scenarios
are `Base`, `Lift-Off`, `High Efficiency`, and `Headwinds`; grid scenarios are
`CP`, `NDC`, and `NZ`. Country identifiers and capacity inputs are defined in
[Installed_capacity_data.py](dataset/Installed_capacity_data.py).

## Installation

The recorded execution environment uses Python 3.9.25 with the package versions
below. Run all commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4 pandas==2.3.3 pyarrow==21.0.0 openpyxl==3.1.5 matplotlib==3.9.2
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Input data

| Input | Location | Preparation |
| --- | --- | --- |
| Capacity paths, country shares, and AI-capacity factors | [dataset/Installed_capacity_data.py](dataset/Installed_capacity_data.py) | Included |
| Annual grid carbon intensity and PUE | [dataset/Factors.py](dataset/Factors.py) | Included |
| GPU-pod workload records | `dataset/asi_opensource_pod_hourly/` | Obtain from the Alibaba Cluster Trace Program |
| Server-resource inventories | `dataset/asi_opensource_server_hourly/` | Obtain from the Alibaba Cluster Trace Program |
| Hourly grid source files | Paths listed in [configs/hourly_grid_sources.csv](configs/hourly_grid_sources.csv) | Supply locally, subject to the provider's access conditions |
| MLPerf Power and national electricity-data inputs | [dataset/validation/](dataset/validation/) | Included; sources are documented in that directory |
| Precomputed figure source data | [results/](results/) | Compact workbooks for inspecting existing figure values |

Workload data are available from the [Alibaba Cluster Trace Program](https://github.com/alibaba/clusterdata).
Grid data are available through [Electricity Maps](https://www.electricitymaps.com/data).
External workload and hourly grid files are excluded from version control.

Expected workload-directory layout:

```text
dataset/
├── asi_opensource_pod_hourly/
│   └── day=0/hour=00/part-000.parquet
├── asi_opensource_server_hourly/
│   └── day=0/hour=00/part-000.parquet
└── EM-CPNDCNZ/
    └── ... hourly grid source CSV files
```

The hourly-source manifest contains `country`, `path`, `source_year`,
`source_kind`, and `zone_id`. Paths in this CSV are relative to the manifest
directory. The supplied manifest selects processed 2026 CP templates; update
the paths and source metadata to match your local inputs.

The grid constructor maps source profiles to complete UTC calendars and
normalizes them to the annual factors in `dataset/Factors.py`. It generates
360 direct-factor files for the default configuration, with 8,784 hours in
2028 and 8,760 hours in other years.

## Quick start

### Run the annual model

This calculation uses the included annual inputs and does not require workload
traces or hourly grid files.

```powershell
python -B -m core.annual_cpu_model --policy CP --scenarios Base --year-start 2026 --years 5 --activity-mode calendar --output-dir results/annual_run
```

Outputs include `Annual_CPU_Model_Country_Annual.csv`,
`Annual_CPU_Model_Selected_Countries_Annual.csv`, task-energy tables, and the
activity and parameter settings.

### Run the complete analysis

Prepare the external inputs first, then construct the hourly grid files and
run the configured analyses:

```powershell
python -B scripts/build_hourly_grid_scenarios.py --source-manifest configs/hourly_grid_sources.csv --output-dir results/grid_hourly_calendar --scopes direct
python -B scripts/run_experiments.py --config configs/experiments.json --output-dir results/example_run
```

Reuse an existing generated grid directory when its source inputs and settings
are unchanged. Each experiment run requires a new output directory.

The calculation suite writes results without drawing figures. It runs the
selected analyses sequentially in one process, sharing the workload-trace
aggregation and hourly grid inputs in memory.

### Draw figures

Use the complete workbooks from the calculation run:

```powershell
python -B scripts/plot_figures.py --input-dir results/example_run/data --output-dir results/example_run/figures --figures 1 2 3 4 --style original --formats svg pdf png --dpi 600
python -B scripts/plot_revision_supplements.py --input-dir results/example_run/data --output-dir results/example_run/diagnostics --dpi 600
```

The first command draws the four main figure sets. The second draws task-energy
closure, additional sensitivity, and ramp-diagnostic figures. Drawing commands
read exported workbooks without rerunning the models.

The four compact workbooks directly under `results/` contain figure source
values. They do not contain the complete tables required by the drawing
commands; use `<run-directory>/data/` as the drawing input.

### Select individual analyses

```powershell
python -B scripts/run_experiments.py --config configs/experiments.json --only figure1 figure4 --output-dir results/sensitivity_run
```

Valid selections are `figure1`, `figure2`, `figure3`, `figure4`, `comparison`,
`boundary`, and `validation`. Without `--only`, the suite runs entries with
`enabled: true`; an explicit selection can also enable a configured entry.

Each main figure also has a separate calculation and drawing entry. For example:

```powershell
python -B scripts/experiment_figure4.py --config configs/experiments.json --output-dir results/figure4_run
python -B scripts/plot_figure4.py --input-dir results/figure4_run/data --output-dir results/figure4_run/figures
```

Replace `4` with `1`, `2`, or `3` to use the corresponding entries.

### Compare the annual and hourly models

```powershell
python -B run.py --policy CP --scenarios Base --year-start 2026 --years 5 --annual-activity-mode calendar --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output-dir results/comparison_run
```

This writes each model's results and comparison tables under `summary/`.
Positive percentage differences indicate that the annual result exceeds the
hourly reference. The sequential decomposition is generated by the `figure3`
analysis.

### Run power diagnostics

The included benchmark and national inputs can be inspected without loading
the workload trace or drawing a figure:

```powershell
python -B scripts/validate_energy_model.py --skip-trace --skip-figure --output-dir results/power_diagnostics
```

Remove `--skip-trace` to include the workload-trace boundary diagnostic.
Remove `--skip-figure` to render its figure, and set `--figure-stem` to choose
the figure's output path.

## Configuration

[configs/experiments.json](configs/experiments.json) controls the analysis suite.

| Block | Settings |
| --- | --- |
| `inputs` | Workload, server, grid, benchmark, and national-data paths; optional capacity-factor file |
| `common` | Countries, years, PUE scale, capacity quantile, utilization cap, idle allocation, strict hourly mode, and trace-reading settings |
| `hardware` | Component power budgets, unit power, idle fractions, and response curves |
| `trace` | Trace filtering and resource-activity settings |
| `annual_activity` | Activity-factor configuration for the annual model |
| `experiments` | Analysis switches, demand/grid scenarios, sensitivity levels, and diagnostic settings |
| `output` | Calculation output directory; `--output-dir` overrides it |

JSON configuration paths are relative to the repository root. By default,
hourly grid inputs are read from `results/grid_hourly_calendar`, and calculation
outputs are written to `results/all_experiments_calendar`.

Useful common settings include:

| Setting | Default | Meaning |
| --- | --- | --- |
| `common.year_start` / `common.years` | `2026` / `5` | Calculation period |
| `common.countries` | `null` | Use all countries in the capacity-input table |
| `common.pue_scale` | `1.0` | Multiply the configured PUE inputs |
| `common.hourly_carbon_scope` | `direct` | Use direct grid-emission factors in the configured analysis |
| `common.strict_hourly_carbon` | `true` | Reject missing hourly grid files and incomplete calendars |
| `common.max_intervals` | `null` | Use the full trace; a finite value limits the trace for debugging |
| `common.hash_inputs` | `false` | Enable an additional content-hashing pass over workload files |
| `inputs.ai_capacity_factors_file` | `null` | Use the built-in factors unless a replacement JSON is supplied |

The `figure3` analysis requires the default 24-country, 2026–2030 coverage.
For a smaller calculation window, select an applicable subset of analyses.
When truncating the trace, disable `experiments.validation.trace_diagnostic`
or retain enough hours for both diagnostic windows.

The built-in capacity factors calibrate the selected-country electricity sum
to an external global reference. Country shares, PUE, power curves, and grid
pathways are model inputs. Aggregated outputs describe the selected countries.
The grid scenarios are deterministic assumptions, and the default country
loads share one normalized trace shape.

## Outputs

The complete analysis and drawing examples produce this layout:

```text
results/example_run/
├── data/
│   ├── figure1_data.xlsx
│   ├── figure2_data.xlsx
│   ├── figure3_data.xlsx
│   └── figure4_data.xlsx
├── comparison/
│   ├── annual_cpu_model/
│   ├── hourly_workload_resolved_model/
│   └── summary/
├── boundary/
├── validation/
├── provenance/
│   └── <run-id>/
│       ├── resolved_experiment_config.json
│       ├── Experiment_Status.csv
│       ├── input_manifest.csv
│       ├── output_manifest.csv
│       └── run_metadata.json
├── figures/
└── diagnostics/
```

`figures/` and `diagnostics/` are created by the separate drawing commands.

| Output | Contents |
| --- | --- |
| `data/figure1_data.xlsx` | Capacity, annual electricity/emissions, and task/component energy attribution |
| `data/figure2_data.xlsx` | Grid factors, scenario emissions, avoided emissions, and load-weighted carbon intensity |
| `data/figure3_data.xlsx` | Model comparisons, decomposition steps, peak/P99 power, ramps, and timing controls |
| `data/figure4_data.xlsx` | Parameter sensitivities, energy composition, resource diagnostics, and conservation checks |
| `comparison/summary/` | Country and selected-country aggregate results and differences between the two models |
| `boundary/` | Calibration-factor comparisons, country totals, and capacity-factor JSON files |
| `validation/` | Benchmark power summaries, error metrics, national-data scope checks, and enabled trace diagnostics |
| `provenance/<run-id>/` | Resolved configuration, experiment status/timing, input/output manifests, and execution metadata |
| `figures/` | Main figure exports, plotted source tables, and figure metadata |
| `diagnostics/` | Task/component closure, additional sensitivity, and ramp-diagnostic figures and source tables |

Power is reported in MW or GW, electricity in MWh or TWh, emissions in
tCO₂ or MtCO₂, and carbon intensity in kgCO₂/MWh. Field suffixes such as
`_mw`, `_mwh`, `_twh`, `_tco2`, and `_mtco2` identify the unit. Hourly ramp
figures use GW/h. Annual electricity uses an 8,760-hour convention and is
distributed over the complete UTC calendar for hourly carbon matching.

## Python interfaces

The model functions return dictionaries containing result tables.

```python
from core.annual_cpu_model import run_annual_cpu_model

annual = run_annual_cpu_model(
    renewable_energy_policy="CP",
    scenarios=["Base"],
    year_start=2026,
    years=5,
    save_outputs=False,
)
country_totals = annual["annual_summary"]
```

For an hourly calculation, prepare the external inputs first:

```python
from core.hourly_workload_resolved_model import run_workload_component_footprint

hourly = run_workload_component_footprint(
    renewable_energy_policy="CP",
    scenarios=["Base"],
    year_start=2026,
    years=5,
    hourly_carbon_factors_dir="results/grid_hourly_calendar",
    hourly_carbon_fallback_to_annual=False,
    save_outputs=False,
)
component_energy = hourly["component_energy"]
```

Hourly CSV export is controlled by `save_hourly_outputs` and is disabled by
default.

## Project structure

```text
configs/       Analysis configuration and hourly-source manifest
core/          Annual/hourly models, power curves, accounting, and data handling
dataset/       Capacity/PUE/grid constants and external-data locations
scripts/       Analysis entries, grid construction, diagnostics, and plotting
results/       Precomputed source data and generated results
manuscript/    LaTeX manuscript and figure assets
run.py         Annual–hourly model comparison entry
```

## License

The project code is released under the [MIT License](LICENSE). External data
remain subject to their providers' access, citation, and redistribution terms.
