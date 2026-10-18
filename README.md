<p align="center">
  <img src="logo.png" alt="Project logo" width="180"/>
</p>

# AI Data Centre Energy and Emissions across 24 Major AI Countries

[中文说明](README.zh-CN.md)

This repository accompanies a study of electricity use and operational carbon
emissions from artificial-intelligence (AI) data centres across 24 major AI
countries worldwide. Selection follows the availability of country-capacity
inputs and historical grid carbon-intensity data; comparable detailed inputs
were unavailable for additional countries. The implementation
combines projected data-centre capacity, country allocation, annual AI-capacity
factors, public GPU-cluster workload traces, component-level power modelling,
power usage effectiveness (PUE), and annual or hourly grid carbon intensity.

The repository uses the same descriptive terminology as the manuscript. The
main comparison connects a conventional **annual CPU model** to an **hourly
workload-resolved model**. An ordered bridge then introduces component power,
trace-derived mean workloads, hourly load variability, and hourly carbon
matching in sequence. These are modelling configurations, not claims that the
hourly endpoint is measured ground truth.

> **Scope.** The study quantifies electricity use, operating load, and
> location-based operational carbon emissions during 2026–2030. It excludes
> embodied emissions, water use, contractual procurement, market-based Scope 2
> claims, and strict 24/7 carbon-free electricity matching.

## Accounting framework

### Endpoint models

| Endpoint | Power and workload representation | Grid carbon intensity | Purpose |
| --- | --- | --- | --- |
| Annual CPU model | Aggregate CPU-style utilisation–power equation with fixed task and utilisation assumptions | Annual mean | Conventional annual accounting baseline |
| Hourly workload-resolved model | Trace-driven CPU, GPU, memory, storage, and IT-fan component power | Hourly | Reference configuration for annual outcomes and operating-shape metrics |

Both endpoints use the same demand pathways, country-capacity boundary, annual
AI-capacity factors, PUE inputs, and electricity-system pathways. Their
difference is decomposed explicitly because several representation choices
change together.

### Ordered bridge

Figure 3 constructs five consecutive configurations. Each effect is the latter
configuration minus the preceding one.

| Order | Configuration identifier in source data | Change introduced |
| --- | --- | --- |
| 1 | `annual_cpu_baseline` | Starting annual CPU representation |
| 2 | `component_power_with_legacy_load` | Replace aggregate power with component power while retaining annual-model task and utilisation assumptions |
| 3 | `flat_trace_annual_carbon` | Replace fixed assumptions with trace-derived mean task–resource loads while keeping the workload flat and carbon intensity annual |
| 4 | `hourly_trace_annual_carbon` | Restore the complete hourly workload shape while retaining annual carbon intensity |
| 5 | `hourly_trace_hourly_carbon` | Match hourly electricity with hourly grid carbon intensity |

The four signed bridge effects are named `power_configuration`,
`mean_workload_representation`, `load_temporal_variability`, and
`hourly_carbon_matching`. Their sum is checked against the direct
hourly-endpoint-minus-annual-baseline difference. Because the bridge is
sequential, it is not an order-invariant causal decomposition.

## Reviewer roadmap

| Manuscript result or method | Primary implementation | Reviewer-facing output |
| --- | --- | --- |
| Demand pathways, country allocation, task and component energy (Fig. 1) | [`dataset/Installed_capacity_data.py`](dataset/Installed_capacity_data.py), [`scripts/figure1_data.py`](scripts/figure1_data.py) | [`results/figure1_data.xlsx`](results/figure1_data.xlsx) |
| Grid pathways and avoided emissions (Fig. 2) | [`dataset/Factors.py`](dataset/Factors.py), [`scripts/figure2_data.py`](scripts/figure2_data.py) | [`results/figure2_data.xlsx`](results/figure2_data.xlsx) |
| Annual–hourly endpoint comparison, ordered bridge, peak and ramp metrics (Fig. 3) | [`core/annual_cpu_model.py`](core/annual_cpu_model.py), [`core/hourly_workload_resolved_model.py`](core/hourly_workload_resolved_model.py), [`scripts/figure3_data.py`](scripts/figure3_data.py) | [`results/figure3_data.xlsx`](results/figure3_data.xlsx) |
| One-at-a-time sensitivity tests (Fig. 4) | [`scripts/figure4_data.py`](scripts/figure4_data.py) | [`results/figure4_data.xlsx`](results/figure4_data.xlsx) |
| MLPerf power diagnostic and trace-boundary audit (Supplementary Fig. S1) | [`scripts/validate_energy_model.py`](scripts/validate_energy_model.py), [`dataset/validation/`](dataset/validation/) | `<output-dir>/validation/` for a suite; `results/validation_run/` in the standalone example |
| Full task/component closure, additional sensitivities and ramp diagnostics (Supplementary Figs. S2–S4) | [`scripts/plot_revision_supplements.py`](scripts/plot_revision_supplements.py) | Rendered from full Figure 1, 4 and 3 experimental workbooks, respectively |

The four workbooks directly under `results/` are reviewer-facing source data for
the displayed main-figure panels only. They contain plotted values and their
coordinates, labels and units, with 7, 4, 4 and 2 worksheets for Figures 1–4,
respectively. They omit full experimental matrices, unplotted results and audit
metadata. Figure 4 contains the six parameters displayed in the main figure;
the other nine mechanism controls are shown in Supplementary Fig. S3.

Run the code to obtain complete experimental outputs. Each configured run writes
full workbooks to `<output-dir>/data/`, validation tables to `validation/`,
and execution records to `provenance/`. The plotting scripts require those full
workbooks, including their W08 contracts; the compact reviewer workbooks are not
plotting inputs. The complete local run used for the revised figures is
`results/all_experiments_calendar_20261017_092038_285/`; this generated directory
is not a substitute for preparing the external inputs and running the code.

The current text and figure descriptions are in the
[main manuscript](manuscript/sn-article-template/sn-article.tex) and
[Supplementary Information](manuscript/sn-article-template/supplementary-information.tex).

## Study scope and scenarios

### Countries

The default boundary contains the following 24 major AI study countries. This
description defines the study scope rather than a ranking of national AI usage:

`USA`, `China`, `Japan`, `France`, `India`, `Singapore`, `Canada`, `Germany`,
`United_Kingdom`, `Australia`, `Italy`, `South_Korea`, `South_Africa`, `Ireland`,
`UAE`, `Brazil`, `Israel`, `Netherlands`, `Spain`, `Sweden`, `Belgium`, `Norway`,
`Poland`, and `Switzerland`.

These identifiers are the canonical keys used by
[`dataset/Installed_capacity_data.py`](dataset/Installed_capacity_data.py) and
[`dataset/Factors.py`](dataset/Factors.py). The modelled country allocation does
not include countries outside this study boundary. Outputs are selected-country
aggregates. Original IT-capacity shares sum to 0.9318338 and retain their source
denominator; this is not an independently established electricity coverage
fraction. Global IEA values are identified separately as external references.

### AI data-centre demand pathways

| Identifier | Interpretation |
| --- | --- |
| `Base` | Baseline capacity-growth pathway |
| `Lift-Off` | Faster demand-growth pathway |
| `High Efficiency` | Efficiency-improvement pathway |
| `Headwinds` | Constrained-growth pathway |

Scenario names passed to Python must use the spelling shown above.

### Electricity-system pathways

| Identifier | Interpretation |
| --- | --- |
| `CP` | Current Policies |
| `NDC` | Nationally Determined Contributions |
| `NZ` | Net Zero |

## Data and code availability

| Resource | Repository status | Access or provenance |
| --- | --- | --- |
| Endpoint-model and ordered-bridge source code | Included | This repository |
| Annual carbon-intensity and PUE inputs | Included | [`dataset/Factors.py`](dataset/Factors.py) |
| Capacity and demand-pathway inputs | Included | [`dataset/Installed_capacity_data.py`](dataset/Installed_capacity_data.py) |
| External-diagnostic inputs | Included | [`dataset/validation/`](dataset/validation/) |
| Compact source data for displayed Figure 1–4 panels | Included | [`results/`](results/); complete experiments are generated by running the code |
| Electricity Maps hourly data and derived hourly scenario files | **Not distributed** | Obtain authorised data directly from Electricity Maps |
| Alibaba GPU-pod and server traces | Not bundled | Obtain from the Alibaba Cluster Trace Program |
| IEA data annex | Not redistributed | Obtain from the International Energy Agency |

The annual CPU model can run from the included annual inputs. Complete
execution of the hourly workload-resolved model and the ordered bridge requires
the external workload traces and hourly carbon-intensity files described below.
The included figure-source workbooks allow the reported values to be inspected
without redistributing restricted inputs.

### Capacity and growth inputs

[`dataset/Installed_capacity_data.py`](dataset/Installed_capacity_data.py)
contains total, IT, and non-IT capacity for four demand pathways from 2025 to
2030, fixed country shares, and annual AI-capacity factors.

The default AI-capacity calibration follows the AI-dedicated data-centre
electricity pathway discussed in the International Energy Agency report
[Key Questions on Energy and AI](https://www.iea.org/reports/key-questions-on-energy-and-ai).
The retained factors fit the global Base electricity reference to the sum of
the 24 study countries: 195, 249.600, 314.496, 386.830 and 465 TWh for 2026–2030.
The approximately 465-TWh endpoint is an external global projection; intermediate
targets are author-defined interpolation inputs. This calibration does not
establish observed electricity coverage. The same factors apply to both model
endpoints and all demand pathways, without recalibration during sensitivity
tests. Country shares and intermediate scenario values are model inputs or
derived parameters, not country forecasts published by the original providers.

### GPU workload and server traces

The hourly workload-resolved model reads two Alibaba Serverless Infrastructure
tables:

- `dataset/asi_opensource_pod_hourly/`: hourly GPU-pod records;
- `dataset/asi_opensource_server_hourly/`: hourly server-resource inventories.

The traces are available from the
[Alibaba Cluster Trace Program](https://github.com/alibaba/clusterdata). Their
system context is described in
[Heterogeneity at Hyperscale: Characterization and Scheduling of Large Production AI Clusters at Alibaba](https://www.usenix.org/conference/osdi26/presentation/li-suyi).

The energy boundary retains all active pods, including CPU-only pods with zero
GPU request. The manuscript task-mix audit uses a separate, stricter filter:
classified tasks within the paper window must have requested GPU memory and
valid used-GPU hours. Training remains training; online and offline inference
are combined as inference; `dev` and explicit `other` labels are combined as
other; missing, `unknown`, or invalid labels remain unclassified. Unclassified
activity contributes to total energy but not to the classified task-share
denominator.

The public trace lacks complete storage activity and host-memory capacity. The
implementation therefore uses idle storage power and a CPU-weighted memory
activity proxy. These assumptions are recorded in the returned
`trace_resource_capacity` table.

### Annual carbon intensity and PUE

[`dataset/Factors.py`](dataset/Factors.py) provides annual grid carbon intensity
for the 24 countries under `CP`, `NDC`, and `NZ`, together with country-, year-,
and demand-pathway-specific PUE inputs. Carbon intensity is expressed as
`kg CO₂/MWh`.

The CP values for 2026–2030 are stored fixed inputs (`CF_CP`), not extrapolations
re-estimated from historical observations during execution. The annual CPU model
uses these annual factors directly. They also provide the
annual means used by bridge configurations. This table has a direct-emissions
boundary: fallback is allowed only in `direct` mode. Missing life-cycle hourly
inputs raise an error instead of substituting direct factors.

### Hourly grid carbon intensity

The supplied configuration reads 360 generated local CSV files from
`results/grid_hourly_calendar/`: 24 countries × three electricity-system pathways
× five years (2026–2030). The manuscript uses direct grid factors. The separate
hourly interface also supports life-cycle factors when suitable inputs are supplied.

The current source manifest reconstructs relative hourly shapes from complete,
processed 2026 CP files under `dataset/EM-CPNDCNZ/`, attributed to Electricity
Maps. The original 2025 acquisition artifacts were unavailable; the reconstruction
does not recover or authenticate those raw observations. Shapes are mapped to
complete UTC calendars and normalized to the fixed annual pathways in
[`dataset/Factors.py`](dataset/Factors.py). The processed hourly files are not
redistributed. Researchers must obtain authorised inputs through the Electricity
Maps [data portal](https://www.electricitymaps.com/data) or
[academic-access guidance](https://help.electricitymaps.com/en/articles/13168512-academic-data-access-and-availability)
and comply with the provider's current terms.

### W03 grid-scenario construction and audit

NDC and NZ are stylized scenarios, with the CP trajectory multiplied by
`endpoint**((year-2025)/5)` after 2025. The 2030 endpoints are 0.89 and 0.53.
The existing annual arrays remain unchanged and rounded to two decimals, so
actual country ratios include small rounding differences. The approximate
11%/47% reductions are input constraints, not independently identified policy effects.

Audit the existing files with:

```powershell
python -B scripts/audit_grid_scenarios.py --output-dir results/revision_audit/grid_baseline
```

The audit exports annual and hourly policy identities, UTC calendar coverage,
direct annual-mean closure, and shape/phase controls with a disclosed synthetic
fixed load. These controls test numerical identities; the model exports actual
workload covariance separately. Failed checks return a nonzero exit status and
retain the report.

Legacy 2028 files contain 8760 consecutive hours from January 1 and omit
December 31. Compatibility mode warns and records
`legacy_8760_truncated_leap_year`; `--strict-hourly-carbon` rejects missing files
and incomplete calendars. The generator produces full UTC calendars (8760 or
8784 hours), matches month/day/hour, copies February 28 when the source lacks
February 29, and then renormalizes to the annual target. The workload model
retains its existing 8760-hour annualized electricity convention and distributes
that unchanged annual electricity over the requested calendar.

Prepare a source CSV manifest with one row per country and fields
`country,path,source_year,source_kind,zone_id`. Paths are absolute or relative
to the manifest. `zone_id` must match the source; the UK folder maps to
`Great Britain`. Use `raw_hourly` for raw templates and `reconstructed_shape`
for shapes recovered from existing derived scenario files. The latter does not
recover or authenticate the original raw observations. Sources must cover the
declared year and have timestamps declaring UTC or explicit offsets, and a direct
carbon-intensity column with units. Life-cycle columns are optional.

```powershell
python -B scripts/build_hourly_grid_scenarios.py --source-manifest source_manifest.csv --output-dir results/revision_audit/grid_hourly --scopes direct life_cycle
python -B scripts/audit_grid_scenarios.py --hourly-carbon-factors-dir results/revision_audit/grid_hourly --output-dir results/revision_audit/grid_checked
python -B scripts/figure2_data.py --hourly-carbon-factors-dir results/revision_audit/grid_hourly --strict-hourly-carbon --output results/revision_audit/figure2_grid.xlsx
```

Missing hours, duplicate/unordered timestamps, negative and non-finite values
are rejected. Existing source estimates are retained and their fraction recorded
when supplied. `construction_audit.csv` and `construction_metadata.json` record
source/output hashes, zone mapping, source year, parameters and means. Life-cycle
targets use the direct annual target times the source life-cycle/direct mean ratio;
they are explicitly derived targets, not closure against the direct annual table.
Use `--variability-scale 0` for flat CI, `--variability-scale 2` for larger
variability, or `--phase-hours 12` for a circular 12-hour phase shift. Negative
values are clipped before restoring the target mean. These are stress tests,
not grid forecasts.

Construction and audit require new output directories and do not overwrite
inputs. The model returns `grid_carbon_audit` and saves `Grid_Carbon_Audit_*.csv`
even when hourly detail is not saved. It records actual source/fallback, coverage,
annual electricity closure, mean CI and the load-CI covariance identity.
The complete Figure 2 experimental workbook includes `Grid_Scenario_Definition`, `Grid_Input_Audit`,
`Annual_Policy_Identity` and `Policy_Effect_Audit` worksheets, including trace
truncation settings. Figure 2a remains a labelled direct annual CP reference
when the model uses life-cycle hourly factors.

Regression tests: `python -B -m unittest discover -s tests -v`, using the Python
dependencies listed below. Runs with `--max-intervals` are debugging outputs,
not formal paper results.

## Repository structure

```text
.
├── core/
│   ├── annual_cpu_model.py
│   ├── hourly_workload_resolved_model.py
│   ├── past_research_data_center_energy_carbon_model.py
│   └── task_model.py
├── dataset/
│   ├── Factors.py
│   ├── Installed_capacity_data.py
│   ├── EM-CPNDCNZ/                    # Local authorised hourly inputs
│   ├── validation/
│   ├── asi_opensource_pod_hourly/     # External local input
│   └── asi_opensource_server_hourly/  # External local input
├── configs/                           # Experiment settings and grid source manifest
├── results/
│   ├── figure1_data.xlsx ... figure4_data.xlsx  # Compact reviewer source data
│   ├── grid_hourly_calendar/          # Generated complete-calendar grid inputs
│   └── <run>/                        # Full experiments, audits and provenance
├── scripts/
│   ├── figure1_data.py
│   ├── figure2_data.py
│   ├── figure3_data.py
│   ├── figure4_data.py
│   ├── run_experiments.py
│   ├── experiment_figure1.py ... experiment_figure4.py
│   ├── plot_figures.py
│   ├── plot_revision_supplements.py
│   └── validate_energy_model.py
├── run.py                             # Two-endpoint comparison entry point
├── LICENSE
├── README.md
└── README.zh-CN.md
```

## Reproducibility environment

The completed manuscript run records Python 3.9.25 and the following versions
in its `provenance/<run-id>/run_metadata.json`:

- `numpy==1.26.4`
- `pandas==2.3.3`
- `pyarrow==21.0.0`
- `openpyxl==3.1.5`
- `matplotlib==3.9.2`

Using Python 3.9.25, create an isolated environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4 pandas==2.3.3 pyarrow==21.0.0 openpyxl==3.1.5 matplotlib==3.9.2
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

## Preparing external inputs

The restricted or external local inputs are excluded by `.gitignore`:

```text
dataset/EM-CPNDCNZ.zip
dataset/EM-CPNDCNZ/
dataset/asi_opensource_pod_hourly/
dataset/asi_opensource_server_hourly/
```

Expected layout:

```text
dataset/
├── EM-CPNDCNZ/
│   ├── USA/
│   │   ├── ...-US-CP-2026-hourly.csv
│   │   ├── ...-US-NDC-2026-hourly.csv
│   │   └── ...-US-NZ-2026-hourly.csv
│   └── Great Britain/
│       └── ...-GB-CP-2026-hourly.csv
├── asi_opensource_pod_hourly/
│   └── day=0/hour=00/part-000.parquet
└── asi_opensource_server_hourly/
    └── day=0/hour=00/part-000.parquet
```

The model key `United_Kingdom` maps to the hourly-data directory
`Great Britain`.

## Running the analysis

Run commands from the repository root.

### Recommended execution workflow

For a complete configured run, prefer the shared-data workflow below. The
individual commands in this section remain available for separate analyses.

### One data pass for the complete experiment suite

From the repository root, construct the hourly grid inputs once, then run the suite:

```powershell
python -B scripts/build_hourly_grid_scenarios.py --source-manifest configs/hourly_grid_sources.csv --output-dir results/grid_hourly_calendar --scopes direct
python -B scripts/run_experiments.py --config configs/experiments.json --output-dir results/reproduction_run
python -B scripts/plot_figures.py --input-dir results/reproduction_run/data --output-dir results/reproduction_run/figures --figures 1 2 3 4 --style original --formats svg pdf png tiff jpg --dpi 600
python -B scripts/plot_revision_supplements.py --input-dir results/reproduction_run/data --output-dir results/reproduction_run/figures --dpi 600
```

The first command constructs grid inputs; the second computes experiments; the
last two render the main figures and Supplementary Figs. S2–S4. Supplementary
Fig. S1 is produced by the external-diagnostic command below. Use a new experiment
output directory on each run. If the grid directory already contains a successful
construction from unchanged inputs, reuse it and start with the experiment command.
Always pass the full run's `data/` directory to the plotting scripts.

[`configs/hourly_grid_sources.csv`](configs/hourly_grid_sources.csv) lists the
complete 2026 CP file for each of the 24 countries as `reconstructed_shape`.
The legacy 2028 files end on December 30 and cannot serve strict-calendar runs.
The constructor maps each country's 2026 shape by UTC month/day/hour onto
2026–2030, copies February 28 for February 29, and normalizes to each CP/NDC/NZ
annual direct-CI target. The generated 2028 calendar has 8784 hours. These are
reconstructed scenario inputs; construction records and source hashes are
written alongside them, and the original files are preserved. After a successful
construction with unchanged inputs, it does not need to be repeated. For another
experiment run, use `--output-dir results/new_run` to select a new result directory.

The default suite enables Figures 1–4, the endpoint comparison, boundary and
calibration comparisons, and external validation. It writes calculation results
without drawing figures. Experiments execute sequentially in one process and
share one streamed trace reduction and one in-memory store of hourly grid
inputs. Raw Parquet frames are released after each batch; sensitivity cases
work from the aggregate profile and copies of transformed arrays. Model
calculations still execute for their distinct configurations.

JSON configuration paths are relative to the repository root; source CSV paths
are relative to the manifest directory. The supplied
[`configs/experiments.json`](configs/experiments.json) selects
`results/grid_hourly_calendar` for grid inputs and `results/all_experiments_calendar`
for outputs unless overridden by `--output-dir`, as in the example above.
Set `inputs.ai_capacity_factors_file` if replacement factors have been adopted.
The supplied configuration requires strict hourly/direct
inputs, all 24 countries and 2026–2030; these are also the Figure 3 contract.
Calendar and CI-mean checks occur before the large trace read. Select a smaller
suite with `--only` when using a smaller country/year/debug window:

```powershell
python -B scripts/run_experiments.py --config configs/experiments.json --only figure1 figure4 --output-dir results/my_sensitivity_run
```

| Configuration block | Controls |
| --- | --- |
| `inputs` | Workload/server/grid paths, optional capacity-factor JSON, MLPerf and national comparison inputs |
| `common` | Countries, years, direct-CI strictness, capacity quantile, utilization cap, PUE scale, idle rule, trace limit and Parquet batch size |
| `hardware`, `trace`, `annual_activity` | Partial dataclass overrides, applied consistently to baseline cases; explicit experimental contrasts override the appropriate setting |
| `experiments` | Enabled analyses, demand/policy selections, country display count, validation seed/bootstrap/window settings and Figure 4 sensitivity ranges |
| `output` | A new directory for the complete run; existing directories are rejected |

`--only` explicitly selects configured experiments, including entries whose
`enabled` value is false. Without it, the enabled entries run in their documented
order. Configuration loading keeps the direct-CI and Figure 3 scope constraints;
model functions check physical parameters and accounting identities.
With debug truncation, disable the validation trace diagnostic or supply enough
observed hours for both endpoint windows. The validation activity filter is
preserved separately from configurable model activity filtering: its hourly
statistics are collected from the same first-pass batches.

Each figure has an independent experiment entry and an independent drawing entry:

| Figure | Experiment source | Drawing source |
| --- | --- | --- |
| 1 | [`scripts/experiment_figure1.py`](scripts/experiment_figure1.py) | [`scripts/plot_figure1.py`](scripts/plot_figure1.py) |
| 2 | [`scripts/experiment_figure2.py`](scripts/experiment_figure2.py) | [`scripts/plot_figure2.py`](scripts/plot_figure2.py) |
| 3 | [`scripts/experiment_figure3.py`](scripts/experiment_figure3.py) | [`scripts/plot_figure3.py`](scripts/plot_figure3.py) |
| 4 | [`scripts/experiment_figure4.py`](scripts/experiment_figure4.py) | [`scripts/plot_figure4.py`](scripts/plot_figure4.py) |

For example, calculate only Figure 4, then draw it from
the exported workbook in a separate command:

```powershell
python -B scripts/experiment_figure4.py --config configs/experiments.json --output-dir results/figure4_recalculated
python -B scripts/plot_figure4.py --input-dir results/figure4_recalculated/data --output-dir results/figure4_drawn
```

The Figure 1/2/3 experiment and drawing commands use the corresponding suffix.
Standalone experiment entries default to `output.directory` suffixed with the
figure name. Drawing entries read the exported workbooks and never rerun models
or read the original trace. For this configuration, pass
`--input-dir <output-dir>/data` (for example, `results/reproduction_run/data`);
the standalone drawing scripts' own default remains `results/all_experiments/data`.
The original `figure*_data.py`
interfaces remain available; use `experiment_figure*.py` for the shared configuration
workflow. Each separate execution reads its selected inputs once.

Default calculation outputs are:

```text
results/all_experiments_calendar/
├── data/figure1_data.xlsx ... figure4_data.xlsx
├── comparison/
├── boundary/
├── validation/
└── provenance/<run-id>/
    ├── resolved_experiment_config.json
    ├── Experiment_Status.csv
    ├── Shared_Data_IO.csv
    ├── Shared_Grid_Source_Selection.csv
    ├── Shared_Validation_Hour_Aggregates.csv
    ├── input_manifest.csv
    ├── output_manifest.csv
    └── run_metadata.json
```

The full experimental workbooks include `Shared_Experiment_Configuration` alongside their W08
source contracts. The suite record includes actual configuration, per-experiment
status/timing, data-read counters, source versions and input/output manifests.
A failed experiment stops the suite and records the failure and completed outputs.
The endpoint comparison shares the suite record; standalone comparison runs keep their own W09 record.
Boundary experiments deliberately compare their own legacy/reconstructed factor
sets; they do not silently replace the factors used by the main figures.

Inputs are shared only in memory during a single process. There is no disk cache,
cache versioning or refresh option. To generate all results with one raw-data
pass, use the complete suite. Running independent experiment commands starts
separate processes and therefore reads the inputs again. Setting
`common.hash_inputs=true` adds a full Parquet hashing pass; leave it false to
avoid that extra I/O. Grid CSVs are hashed from the byte buffer used for parsing.
The shared context retains aggregate inputs rather than the large result tables
from all sensitivity cases.

The W03–W09 worksheet names below refer to these full experimental outputs,
not to the compact reviewer workbooks at the root of `results/`.

### Standalone boundary and calibration comparison

The stored default factors retain the legacy convention of fitting a global
electricity reference to the selected-country sum. Compare that convention,
freshly reconstructed global-reference factors, and a homogeneous-intensity
boundary case with:

```powershell
python -B scripts/analyze_boundary_calibration.py --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output-dir results/revision_audit/boundary
```

Review `calibration_audit.csv`, `endpoint_comparison.csv` and
`invariant_checks.csv`. The homogeneous case assumes equal facility electricity
per unit IT capacity inside and outside the study boundary. It is a conditional
comparison, not observed electricity coverage or an automatically adopted main
result. The same factors apply to both endpoints. Output directories must be
new; smoke-test factors cannot be loaded as research inputs.

After selecting that conditional case, run the endpoint comparison with:

```powershell
python -B run.py --policy CP --scenarios Base --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --ai-capacity-factors-file results/revision_audit/boundary/ai_capacity_factors_homogeneous_intensity_boundary.json --output-dir results/revision_audit/boundary_models
```

Figure-data scripts also accept `--ai-capacity-factors-file`. Full workbooks export
`Accounting_Boundary` and `AI_Capacity_Factors` sheets. Figure 1a now reports
selected-country capacities, retaining explicit `source_global_*` references.
Figure 1b's full table distinguishes original allocation shares from shares
within the study countries; the main panel and compact reviewer table retain
the original global denominator. Full Figure 2c data use
`Fig2c_Selected_Countries_Carbon`; the compact reviewer table uses
`Fig2c_Annual_Emissions`. Reviewer tables retain only displayed data.

### Compare the two endpoint models

```powershell
python -B run.py --policy CP --scenarios Base --year-start 2026 --years 5 --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output-dir results/comparison_run
```

The entry point reads the workload trace once, runs the annual CPU and hourly
workload-resolved endpoints, and writes results under
the chosen `--output-dir` (`results/model_comparison/` if omitted). Multiple demand pathways may be supplied:

```powershell
python -B run.py --policy NDC --scenarios Base "Lift-Off" "High Efficiency" --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output-dir results/comparison_ndc_run
```

For a quick execution check, restrict countries and trace intervals:

```powershell
python -B run.py --countries USA --year-start 2026 --years 1 --max-intervals 24 --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output-dir results/comparison_debug_run
```

`max_intervals` changes the trace sampling basis and is only for smoke tests and
debugging. Omit it for manuscript results.

### Standalone figure-data interfaces

The configured suite above generates all four full workbooks. To use a legacy
standalone data interface, explicitly choose a separate output file. Its default
path is the compact reviewer workbook under `results/`, so a bare invocation
would replace that reviewer export with a full experimental workbook.

```powershell
python -B scripts/figure1_data.py --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output results/standalone_data/figure1_data.xlsx
python -B scripts/figure2_data.py --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output results/standalone_data/figure2_data.xlsx
python -B scripts/figure3_data.py --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output results/standalone_data/figure3_data.xlsx
python -B scripts/figure4_data.py --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output results/standalone_data/figure4_data.xlsx
```

Figure 3 is the authoritative implementation of the ordered bridge. The script
checks variant coverage, country aggregation, hourly-to-annual energy closure,
and additive bridge closure before writing its workbook.

### W04 annual activity and controlled endpoint comparisons

The annual model and Figure 3 share [`core/annual_activity.py`](core/annual_activity.py).
Default `calendar` activity uses fixed 2025/2030 anchors and denominator 5,
reaching training activity 0.925 and inference/other/unclassified activity 0.7
in 2030. Requested output windows only select years. `fixed` holds the 2025
coefficients at 0.9/0.5. `legacy_six_year` fixes denominator 6, reproducing
the old complete 2025–2030 run and its 0.9208333/0.6666667 endpoints. These
coefficients are author-defined reference assumptions.

Run the following manually from the repository root in the environment above.
The annual-only commands do not read workload traces:

```powershell
foreach ($activityMode in @("calendar", "fixed", "legacy_six_year")) {
    python -B -m core.annual_cpu_model --policy CP --scenarios Base --year-start 2026 --years 5 --activity-mode $activityMode --output-dir "results/revision_w04/annual_$activityMode"
}
```

Each directory contains `Annual_Activity_Factors.csv`,
`Annual_Reference_Parameters.csv` and country/selected-country annual outputs.
Task outputs disclose activity coefficients and utilization contributions;
country outputs also report electricity and emissions per installed IT MW.

Generate Figure 3 and all W04 controls with the complete-calendar W03 inputs:

```powershell
python -B scripts/figure3_data.py --annual-activity-mode calendar --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output results/revision_w04/figure3_w04.xlsx
```

The script reads the full workload trace once. It compares all three annual
activity configurations against the unchanged hourly endpoint, and adds common
mean-load controls with fixed activity and symmetric task activity trends.
Common means are calculated after the hourly model's resource-utilization cap.
The aggregate annual equation uses the common CPU mean; the component model
also uses the disclosed GPU/memory/storage means. Symmetric trends multiply
each task's contributions in both endpoints by `a(year)/a(2025)`, with requested
and realized loads exported around clipping. Both common-mean configurations
use flat workloads and the same annual direct CI. Runtime guards check that
CPU activity and installed capacity agree between those endpoints.

Additional worksheets include:

| Worksheet | Content |
|---|---|
| `W04_Activity_Factors` | All three activity sequences, 2025–2030 anchors and denominators |
| `W04_Parameter_Assumptions`, `W04_Configuration_Matrix` | Author-defined coefficients and dimensions changed by each comparison |
| `W04_Control_Comparison` | Country and selected-country gaps, percentages with both explicit endpoint denominators, capacity-normalized metrics and changes over time |
| `W04_Baseline_Impact` | Annual outcomes relative to fixed activity and the historical six-year denominator |
| `W04_Common_Mean_Load` | Common means, task multipliers, clipping and realized resource activity |
| `W04_Run_Metadata`, `W04_Grid_Input_Audit` | Trace truncation, shared calibration and hourly source/fallback |
| `Source_Bridge_Variants`, `Source_Country_Bridge` | Full five-stage bridge inputs and ordered contributions |

All cases share one capacity-factor set without separate recalibration. If
W02/W01 selected a new factor file, pass the same `--ai-capacity-factors-file`
to the annual, Figure 3 and `run.py` commands; omission retains the repository
defaults. Gaps quantify conditional representation differences. Ordered bridge
contributions depend on replacement order. Comparisons require `direct` scope
because no corresponding annual life-cycle table is supplied; the standalone
hourly model retains its life-cycle input option.

Figure 3 accepts full UTC calendars, including 8784 hours in 2028; compatibility
mode also labels legacy truncated 8760-hour inputs. `run.py` exposes
`--annual-activity-mode calendar|fixed|legacy_six_year`. To inspect a shortened
trace run, add `--max-intervals 24` to Figure 3 and choose a separate output file;
these outputs cannot be used as formal manuscript results.

### W05 full task/component attribution and conservation

[`core/energy_attribution.py`](core/energy_attribution.py) exports the complete
four-task by five-component matrix, including unclassified and storage.
`task_component_energy` contains the selected idle rule;
`task_component_energy_all_rules` contains all three rules. Absolute idle,
dynamic, IT and PUE-scaled facility energy and explicit task/component/system
share denominators accompany the matrix. Attribution comparisons reuse the
same component-power calculation.

| `--idle-allocation-rule` | Idle treatment | Closure |
|---|---|---|
| `hourly_driver` (default) | Hour-specific component driver shares | Four task totals equal component/IT energy |
| `annual_driver` | Fixed annual component driver shares | Four task totals equal component/IT energy |
| `separate_pool` | Idle retained in a separate pool; tasks receive dynamic energy | Task dynamic energy plus idle pool equals component/IT energy |

Dynamic energy always follows hourly driver shares. Missing component drivers
use capacity-normalized resource proxy shares, replacing the previous mixed-unit
raw-resource sum. Completely inactive profiles use an explicitly labelled equal
four-task reference. `allocation_audit` records driver availability, inactive
intervals and fallback basis. Memory remains a CPU-weighted proxy; storage lacks
observed dynamic activity. GPU-memory fields do not substitute for host memory
capacity or this proxy. These are model attribution assumptions.

The full Figure 1 workbook reads the model matrix directly, includes all five components in
`display_share_pct`, and adds `Fig1_Unclassified`. Shares across countries and
years use summed absolute energy before division. In separate-pool mode task
shares sum to less than 100%; the remainder is explicit in `W05_Idle_Pool`.
Facility attribution uses proportional country/year PUE scaling and does not
measure component-specific cooling energy.

Run manually from the repository root in the configured Python environment:

```powershell
python -B scripts/figure1_data.py --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --idle-allocation-rule hourly_driver --output results/revision_w05/figure1_w05.xlsx
```

The command includes all four default demand scenarios, reads the trace once
and automatically exports all three idle rules. To select separate-pool task
panels, use `--idle-allocation-rule separate_pool` and a different output file.
New sheets provide country/all-rule/annual matrices, task and system component
shares, physical model component energy, idle pools, allocation fallbacks,
component/system energy closure, resource clipping and proxy metadata.

Run Figure 4 sensitivities and mechanism diagnostics with:

```powershell
python -B scripts/figure4_data.py --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --idle-allocation-rule hourly_driver --output results/revision_w05/figure4_w05.xlsx
```

`W05_Resource_Changes` compares source activity, capacity-normalized activity,
clipping and realized resources. `W05_Control_Checks` separately records source
resource, realized normalized activity, electricity and mean-CI preservation.
Task-mix changes must preserve source resource totals and unclassified activity;
CI-only controls must preserve each country's annual electricity and mean CI.
Resource-total preservation does not impose electricity preservation for other
parameter changes.

`W05_Power_Nonlinearity` compares integrated hourly energy with power evaluated
at the same post-clipping mean utilizations, labelling GPU log2, fan cubic and
CPU/memory/storage linear curves. This distinguishes nonlinear power effects
from resource clipping. The sensitivity sheet also exports actual model GPU,
unclassified, idle and idle-pool fractions. A GPU-budget OAT simultaneously
redistributes non-GPU budgets, so its slope does not directly identify a real
GPU energy fraction.

Figure 4 exports per-experiment matrices, component energy, pools, attribution,
energy/resource closure, grid diagnostics and proxy metadata. `run.py` accepts
the same idle-rule option and saves diagnostic CSVs under the hourly model's
output directory. Pass one common `--ai-capacity-factors-file` to all entry points
if W02/W01 selected a factor file. Formal results require the full trace;
`W05_Run_Metadata` records trace-truncation settings.

### Run external diagnostics

```powershell
python -B scripts/validate_energy_model.py --bootstrap-replicates 10000 --seed 20260819 --output-dir results/validation_run --figure-stem results/validation_run/figures/supplementary_energy_validation
```

This standalone command produces Supplementary Fig. S1 and diagnostic tables.
It reads its own trace windows; the configured suite instead exports validation
tables from its shared data pass without rendering. The diagnostic uses random
seed `20260819` and 10,000 Public-ID bootstrap replicates. Input provenance and
filtering rules are documented under [`dataset/validation/`](dataset/validation/).

### W06 trace seams and temporal robustness

The hourly model adds `temporal_alignment`, `temporal_metrics`, and
`trace_seams`, saved as CSVs with those names. They record source-hour counts,
repeat boundaries and power jumps, calendar copying/positional resampling,
annual electricity normalization and closure. `hourly_carbon` includes source
positions, repeat cycles and incoming-edge seam/alignment flags. Annual-CI
mode also diagnoses the trace shape rather than substituting a flat profile.

Ramps are absolute differences between adjacent UTC hourly powers; no annual
end-to-start ramp is included. Outputs distinguish the raw maximum, maximum
excluding repeat seams, maximum excluding all alignment artifacts, and raw
and nonseam P95/P99 absolute ramps. Empty filtered edge sets return missing
values. Figure 3c/d retain the raw metric and add these diagnostics, using
`Maximum absolute hourly ramp` and `GW h^-1`; power uses `GW`. Shared normalized
country shape is a diagnostic rather than a required identity. Aggregate
peaks sum country powers at the same UTC timestamp before taking the maximum;
the sum of individual country maxima is reported only as a coincidence control.

Figures 3 and 4 reuse captured hourly electricity and CI for 17 configurations
including the baseline: load shifts of -6/+6/+12 h, country phase dispersion,
country shape dispersion, load variability 0.5/1.5, 3 h smoothing, CI shifts
of -6/+6 h, equal/opposite load-CI shifts, flat CI, CI variability, and CI mean
changes of +/-10%. Positive shifts move later in UTC using a circular shift;
the added finite-calendar wrap join is flagged separately. Country offsets
span -12 to +12 h in sorted country order; country variability cycles through
0.5, 1, 1.5. These are synthetic stress assumptions, not empirical time zones
or national behavior. Smoothing is a sensitivity treatment; affected adjacent
artifact edges are excluded by the broader alignment filter.

| Worksheet | Content |
|---|---|
| `W06_Trace_Alignment` | Alignment, seams, annual normalization and closure |
| `W06_Baseline_Peak_Ramp`, `W06_Baseline_Seams` | Figure 3 baseline idle floors, peak/mean, full-calendar ramps and seam events |
| `W06_Timing_Country`, `W06_Timing_Aggregate` | Country and simultaneous aggregate stresses, peaks, ramps, emissions and covariance identity |
| `W06_Timing_Parameters`, `W06_Timing_Metadata` | Actual country phases, variability, smoothing, CI means, constraints and calendar conventions |
| `W06_Seam_Events` | Repeat seams, phase wrap joins and smoothing-affected edges in timing controls |
| `W06_Model_Temporal_Metrics`, `W06_Model_Seams`, `W06_Metric_Changes` | Figure 4 model experiments, idle floors, peak/mean, ramps and baseline differences |

The two stages impose different constraints. Post-power timing stresses hold
each country-year's facility electricity fixed. Load-shape changes do not
infer resource-hour conservation or enforce physical idle floors. CI means
are fixed except in the explicit +/-10% mean scenarios; matched emissions
may change through load-CI covariance. Figure 4 additionally varies
`pre_power_resource_variability_scale=0.5/1/1.5`, preserving each source
task/resource total before power conversion while holding trace-capacity
denominators and hardware fixed. Clipping and nonlinear power are recalculated;
W05/W06 report realized resources, electricity and idle shares separately.
Existing GPU idle-ratio experiments now also export temporal metrics.

Matching uses the full UTC calendar, including 8,784 h in 2028, while preserving
the total obtained by annualizing the trace to 8,760 h. No annual recalibration
is applied. Legacy truncated leap-year inputs and annual-CI fallbacks remain
explicitly flagged. Use the Figure 3/4 entry points with
`--hourly-carbon-factors-dir results/grid_hourly_calendar` and
`--strict-hourly-carbon` for formal results. Supplementary Fig. S4 displays raw,
artifact-filtered and P95/P99 ramp diagnostics; full W06 tables retain the timing
controls and seam records.

The validation script still reads only the two endpoint windows. Its new
proxy P95/P99 and seam flags are explicitly local resource diagnostics, not
full-trace power or national-ramp validation; use the W06 tables for full-calendar
model diagnostics.

### W07 parameter mechanisms, stratified diagnostics and conservation

`core/power_parameter_audit.py` supplies four new hourly-model outputs:
`hardware_parameters`, `parameter_dependencies`, `capacity_parameter_audit`
and `mechanism_conservation`. They record actual assumptions, equations,
country-year budgets, unit watts, trace-capacity denominators, model resource
capacities, idle fractions and realized utilization. Reconstructing a budget
from capacity times unit watts is explicitly a configuration identity, not
external validation.

CPU/GPU dynamic responses expose `cpu_dynamic_exponent` and
`gpu_dynamic_exponent`, both defaulting to one. These reproduce the original
CPU linear and GPU `log2(1+u)` responses. Stress responses use `u**alpha` and
`log2(1+u**alpha)`, preserving zero/full-utilization endpoints. They are
assumption contrasts, not curves fitted to MLPerf system power. Nonlinearity
outputs record the actual exponent and formula rather than a fixed concavity label.

Figure 4 reuses one loaded trace for CPU/GPU unit-wattage controls at
0.8/1/1.2 times baseline. `*_unit_power_scale_fixed_idle_ratio` scales idle and
full watts together: native model capacity/resource-hours must scale inversely,
while normalized activity, component/facility electricity and matched carbon
remain invariant under fixed budgets, trace denominators, responses and
execution weights. `*_unit_power_scale_fixed_idle_watts` instead holds idle
watts fixed, changing idle/full ratio; electricity invariance is not imposed.
Positive contrasts cover CPU/GPU budget shares, idle fractions and response
exponents (default 0.75/1/1.25). Other budget shares rescale proportionally.
`configured_system_W_per_GPU` makes explicit that the external configured
reference can change while capacity-normalized electricity remains invariant.

| Figure 4 worksheet | Content |
|---|---|
| `W07_Hardware_Parameters`, `W07_Parameter_Dependencies` | Actual assumptions and dependencies for every experiment |
| `W07_Capacity_Parameters`, `W07_Parameter_Changes` | Country capacities, trace denominators, utilization, idle fractions, exponents and baseline differences |
| `W07_Control_Evidence` | Control types, preserved quantities, inverse-scaling checks and detected component/facility energy responses |
| `W07_Mechanism_Conservation` | Resource allocation/clipping, tasks plus idle pool, component/IT/facility/PUE, hourly electricity and carbon covariance identities |

Accounting checks distinguish source-trace and model resource units. Realized
model activity must equal capacity times mean utilization times 8,760 h;
facility electricity closes through PUE. Absolute tolerance is `1e-6` plus
relative tolerance `1e-10`; failed identities stop the run with diagnostic
records. Conservation establishes internal accounting consistency, not
external accuracy. Positive-control outputs separately report component and
facility responses; no response under one load condition does not establish
general parameter ineffectiveness.

MLPerf filtering removes exact duplicate raw rows before eligibility filtering and audits consistent
hardware/count/node/version metadata within Public IDs. Conflicting IDs are
exported and excluded rather than reconciled by median accelerator counts.
Each stratum first aggregates within Public ID, then resamples those IDs.
Hardware family, explicit SXM/PCI labels, single/multiple-node boundary,
Server/Offline and inference tasks are stratified. Reported system-power
units do not identify cooling or GPU-only measurement boundaries. Different
IDs are not proven to represent independent physical hardware, and task
strata can share IDs.

New CSVs include `mlperf_public_id_audit`, `mlperf_stratified_metrics`,
`mlperf_hardware_coverage`, `mlperf_a100_h100_comparison`,
`validation_claim_scope` and `validation_run_metadata`. Family summaries
include errors and intervals; A100/H100 coverage and median differences are
reported separately. A single ID has a point estimate but no confidence
interval; either comparison group with fewer than two IDs prevents an interval
for the group difference. Seeded batched bootstrap bounds memory usage.

Legacy filenames and fields remain for compatibility. `observed_power_max_w`
means the largest retained reported workload power within an ID, not a
measured time-series peak or evidence of full GPU utilization. Supplementary
figure labels now state this configuration-diagnostic scope. These data do
not validate idle/dynamic GPU curves, training, PUE/facility energy or future
national accuracy/conservativeness. No curve is fitted; future fitting requires
utilization/power observations and independent-system train/validation splits.

Figure 4 and validation accept `--hardware-config-file` containing JSON
`HardwarePowerConfig` overrides and export every actual configuration. Custom
baselines retain only legal share/idle-fraction contrast levels. Validation also
accepts `--skip-figure`, alongside `--skip-trace` for tables without a trace read
or rendering. The configured suite applies its `hardware` settings to both
the model experiments and external diagnostics.

### W08 reproducible rendering from full experimental workbooks

After completing the configured run, render from its full workbooks:

```powershell
python -B scripts/plot_figures.py --input-dir results/reproduction_run/data --figures 1 2 3 4 --style original --formats svg pdf png tiff jpg --dpi 600 --output-dir results/reproduction_run/figures
python -B scripts/plot_revision_supplements.py --input-dir results/reproduction_run/data --output-dir results/reproduction_run/figures --dpi 600
```

The renderer requires `W08_Figure_Contract`, `W08_Panel_Metadata`,
`W08_Display_Values` and `Accounting_Boundary`. Compact reviewer workbooks lack
these sheets and cannot be used as inputs. Use
`--figure1-workbook`, `--figure2-workbook`, `--figure3-workbook` or
`--figure4-workbook` to select files from separate revision directories.
Default outputs are editable SVG/PDF and a PNG preview; `--formats svg pdf tiff`
selects TIFF instead. Captions, source values, panel metadata and hash manifests
accompany each figure. Existing outputs require `--overwrite` or a new directory.

`--style original` is the default and matches the current main-figure layout.
Figure 1a places the darker AI-focused IT region within total facility capacity;
the two capacities are not added. Panels c–e show five-component energy shares
within training, inference and other tasks. Supplementary Fig. S2 retains all
four task classes, system-level component shares and the idle-pool closure.
Figure 2 shows the annual CP reference, avoided emissions and load-weighted CI.
Figure 3 separates peak power (GW) from absolute hourly ramp (GW h⁻¹).

The full sensitivity experiment varies 15 parameters. Figure 4 displays six
parameters in separate electricity and emissions panels; Supplementary Fig. S3
shows the other nine mechanism controls. Each percentage is relative to that
metric's own baseline: first sum the study countries in each year, then take the
2026–2030 mean. These are deterministic one-at-a-time responses, not joint
uncertainty intervals. `--style expanded` provides an optional diagnostic layout
and does not reproduce the current main panels. Inspect exports at final size.

`scripts/refresh_manuscript_figures.py --sync-manuscript` also copies full
experimental workbooks to the four root-level reviewer paths and rewrites LaTeX
captions. It does not preserve the compact reviewer exports. For the present
data layout, use the plotting commands above and prepare reviewer tables from
displayed values separately.

### W09 unified endpoint configuration and execution records

[`run.py`](run.py) retains the two-endpoint comparison and existing Python API.
It now connects the W04 activity, W05 attribution and W07 hardware settings to
one resolved run record through [`core/run_provenance.py`](core/run_provenance.py).
Model outputs retain their previous directory layout. Every saved invocation
also creates `output-dir/provenance/<UTC timestamp + unique ID>/`.

Example for a full trace, after preparing the W03 full-calendar grid inputs:

```powershell
python -B run.py --policy CP --scenarios Base --year-start 2026 --years 5 --annual-activity-mode calendar --idle-allocation-rule hourly_driver --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --hash-inputs --output-dir results/revision_w09/base_cp
```

Pass the selected W01/W02 `--ai-capacity-factors-file` explicitly when using
replacement factors. Omission retains the repository's **legacy global-reference
calibration to the selected-country sum**. Supplying a new hardware configuration
does not automatically recalibrate those factors. Endpoint comparison requires
`direct` carbon factors; `life_cycle` remains available only through the hourly
model's separate interface.

| Option | Meaning |
| --- | --- |
| `--hardware-config-file` | JSON overrides for `HardwarePowerConfig`, using the schema accepted by Figure 4 and validation |
| `--trace-config-file` | JSON overrides for `Alibaba2026TraceConfig`, including CPU-only pods, active states and GPU clipping |
| `--annual-activity-mode` | `calendar` (default), `fixed` or `legacy_six_year` |
| `--annual-activity-config-file` | JSON activity endpoints and optional mode; mutually exclusive with `--annual-activity-mode` |
| `--idle-allocation-rule` | `hourly_driver`, `annual_driver` or `separate_pool`; all models retain the existing accounting rules |
| `--parquet-batch-size` | Projected Parquet rows per batch, default 65,536; distinct from the trace-hour limit |
| `--trace-read-log` | Optional **new** JSONL filename; defaults to the unique provenance directory; existing logs are refused |
| `--hash-inputs` | Additionally compute SHA-256 for each Parquet file actually opened; this adds full-file I/O |
| `--max-intervals` | Debug trace-hour limit, recorded as truncation; omit for manuscript runs |

Configuration files contain partial overrides. Unspecified fields retain their
dataclass defaults; unknown keys are errors. For example, the trace JSON can be:

```json
{"trace_anchor_year": 2001, "include_zero_gpu_pods": true, "active_states": ["Running"], "cap_gpu_sm_by_request": true}
```

An activity JSON can be:

```json
{"mode": "calendar", "training_start": 0.90, "training_target": 0.925, "inference_start": 0.50, "inference_target": 0.70}
```

CLI syntax errors, including mutually exclusive flags, are rejected before a
run starts. Configuration files are loaded inside the recorded run, so malformed
JSON and object/file conflicts fail with a recorded stage. Python
callers may still pass configuration objects; a file and object for the same
setting cannot both be supplied. `save_outputs=False` returns the run metadata
and provenance tables in memory and rejects an explicit `trace_read_log`.

| Provenance artifact | Contents |
| --- | --- |
| `run_metadata.json` | Requested and resolved settings, start/end UTC, elapsed time, status, failure stage/traceback, Python/package versions, CLI argv or API identity, Git HEAD and local changes, untracked source list, coverage and fallback counts |
| `input_manifest.csv` | Local source/configuration files, selected grid CSVs and Parquet files observed by the actual reader; size, mtime, hash status and post-run change checks |
| `Resolved_Model_Inputs.csv` | Country/scenario/year PUE before and after scaling, IT capacity and allocation shares, AI factors and annual direct CI values |
| `AI_Capacity_Factors.csv`, `Annual_Activity_Factors.csv`, `Annual_Reference_Parameters.csv`, `Hardware_Parameters.csv` | Numeric parameters used by this execution |
| `Grid_Input_Selection.csv`, `Grid_Input_Audit.csv` | Pre-run file selection, then actual hourly/annual source, calendar coverage, CI closure and model-reported hashes |
| `Trace_Resource_Capacity.csv`, `Workload_Trace_Metadata.csv`, `Temporal_Metrics.csv` | Actual trace size, annualization, inventory/proxy origin and temporal diagnostics |
| `Allocation_Audit.csv`, `Fallback_Summary.csv` | Grid fallback, calendar limitations, CPU/GPU inventory fallback, memory/storage proxy assumptions and idle-attribution fallback |
| `events.jsonl`, `trace_read.jsonl` | Flushed stage events and per-file/per-batch trace-reader progress; these are structured logs, not a copy of stdout |
| `output_manifest.csv` | SHA-256 of observed new/modified model outputs and provenance artifacts; pre-existing unchanged model files are marked and not attributed to this run |

Source/configuration files and selected grid CSVs are always hashed. Without
`--hash-inputs`, Parquet entries explicitly say that content hashing was not
requested; size/mtime do not substitute for a content hash. The trace inventory
includes only files opened, so debug truncation does not imply a full-trace
audit. Grid coverage counts unique policy/country/year/scope inputs, avoiding
double counting across demand scenarios. Trace proxies are distinguished from
missing-input fallback.

Use a fresh `--output-dir` for each formal configuration. Reusing a directory
preserves historical provenance records but can overwrite the model CSVs in
the existing model/summary subdirectories; older output hashes then refer to
older content. A caught Python exception records `failed`; keyboard interruption
records `interrupted`. A native crash or forced termination can leave `running`
without an end time: inspect the last flushed trace event rather than treating
it as completion. Hash/mtime change flags identify inputs altered during a run.
`run_metadata.json` and `output_manifest.csv` exclude self-referential hashes.

These records establish what local files and settings were used, not the source
licence, completeness of a submission archive, remote availability or scientific
validity of the inputs. The source register from `audit_revision_inputs.py`
continues to record unresolved provenance gaps. Independent figure-data and
validation commands remain separate; transfer the relevant settings explicitly
and inspect their own metadata. `run.py` does not silently dispatch them or
apply unsupported options to them.

## Python interfaces

### Annual CPU model

```python
from core.annual_cpu_model import run_annual_cpu_model

result = run_annual_cpu_model(
    renewable_energy_policy="CP",
    scenarios=["Base", "Lift-Off"],
    year_start=2026,
    years=5,
)
```

### Hourly workload-resolved model

```python
from core.hourly_workload_resolved_model import run_workload_component_footprint

result = run_workload_component_footprint(
    renewable_energy_policy="CP",
    scenarios=["Base"],
    year_start=2026,
    years=5,
    hourly_carbon_factors_dir="results/grid_hourly_calendar",
    hourly_carbon_fallback_to_annual=False,
    output_dir="results/hourly_api_run",
    save_hourly_outputs=False,
)
```

The hourly interface accepts `hourly_carbon_scope="direct"` or
`"life_cycle"`. Setting `hourly_carbon_fallback_to_annual=False` rejects missing
files and incomplete calendars, corresponding to CLI `--strict-hourly-carbon`.
Hourly CSV output is disabled by default because it can be large.

## Principal parameters

| Parameter | Meaning | Common value or default |
| --- | --- | --- |
| `renewable_energy_policy` | Electricity-system pathway | `CP`, `NDC`, or `NZ` |
| `scenarios` | Demand pathway | `Base`, `Lift-Off`, `High Efficiency`, or `Headwinds` |
| `year_start` | First output year | 2026 for manuscript analyses |
| `years` | Consecutive years | 5 |
| `countries` | Modelled countries | All 24 by default |
| `pue_scale` | PUE sensitivity multiplier | `1.0` |
| `ai_capacity_factors` | Annual AI-capacity calibration | `DEFAULT_AI_CAPACITY_FACTORS` |
| `hourly_carbon_scope` | Hourly carbon-intensity boundary | `direct` or `life_cycle` |
| `hourly_carbon_factors_dir` | Processed hourly grid input directory | Supplied config: `results/grid_hourly_calendar` |
| `hourly_carbon_fallback_to_annual` | Use the annual direct factor if an hourly file is missing | API default `True`; disabled for formal strict runs |
| `capacity_quantile` | Trace-capacity inventory quantile | `0.96` |
| `max_resource_utilization` | Cap on normalized resource utilization | `1.0` |
| `max_intervals` | Trace-hour limit for smoke tests | Full trace by default |
| `include_zero_gpu_pods` | Retain CPU-only pods in the energy boundary | `True` |

The annual CPU model additionally exposes fixed task shares, task-specific
utilisation, idle-power ratio, and maximum-power ratio. The hourly model exposes
component and trace-processing assumptions through `HardwarePowerConfig` and
`Alibaba2026TraceConfig`.

## Outputs

The endpoint runner writes:

```text
results/model_comparison/
├── annual_cpu_model/
├── hourly_workload_resolved_model/
├── provenance/
│   └── <run-id>/
│       ├── run_metadata.json
│       ├── input_manifest.csv
│       └── output_manifest.csv
└── summary/
    ├── Endpoint_Models_Country_Annual.csv
    ├── Endpoint_Models_Selected_Countries_Annual.csv
    ├── Endpoint_Comparison_Country_Annual.csv
    ├── Endpoint_Comparison_Selected_Countries_Annual.csv
    └── Accounting_Boundary.csv
```

Comparison tables report annual CPU-model differences relative to the hourly
workload-resolved endpoint. A positive percentage means the annual result is
higher than the hourly reference.

The hourly model can return or save:

- `annual_summary`: country-year electricity, emissions, and resource use;
- `component_energy`: CPU, GPU, memory, storage, and IT-fan energy;
- `task_type_energy`: training, inference, other, and unclassified task energy;
- `task_demand` and `task_execution`: resource-hours by task origin and location;
- `capacity_overflow`: resource-capacity diagnostics;
- `hourly_carbon`: hourly electricity, carbon intensity, and emissions;
- `workload_profile_summary`: workload composition by reporting basis;
- `trace_resource_capacity`: estimated capacity and provenance.

### Units

| Field suffix | Unit |
| --- | --- |
| `_mw` | MW |
| `_mwh` | MWh |
| `_twh` | TWh |
| `_kg_per_mwh` | kg CO₂/MWh |
| `_tco2` | t CO₂ |
| `_mtco2` | Mt CO₂ |

## Interpretation boundaries

- Capacity inputs cover 2025–2030 and should not be extrapolated without new evidence.
- Fixed country shares exclude spatial migration, changing national shares, and the residual global capacity share.
- Under default capacity allocation, countries inherit scaled shapes from one relative trace. W06 country heterogeneity is a synthetic stress, not an independent national behavior prediction.
- The 4,440-h Alibaba trace is repeated over the CI UTC calendar (8,784 h in 2028), preserving 8,760-h annualized electricity. Repeat seams can determine raw maximum ramps; inspect seam flags and filtered metrics together.
- Future hourly grid series inherit reconstructed shapes from processed 2026 CP templates; original 2025 acquisition artifacts were unavailable. They do not forecast changes in dispatch, storage, imports, or variable-generation profiles.
- Component shares, unit power, idle fractions, fan response, memory activity, and storage activity are configurable reference assumptions.
- The MLPerf diagnostic assesses order of magnitude and bias; it is not a calibration fit and does not validate future national facility electricity.
- Compatibility mode permits annual direct-factor fallback; the formal configuration uses strict hourly inputs and rejects missing files or incomplete calendars. Inspect `carbon_factor_source` before interpreting temporal results.
- The hourly workload-resolved endpoint is a transparent reference configuration, not measured ground truth.

## Licence

The project code is released under the [MIT License](LICENSE). Third-party data
remain subject to the original providers' access, citation, and redistribution
terms.
