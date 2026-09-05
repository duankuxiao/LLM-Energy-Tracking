<p align="center">
  <img src="logo.png" alt="Project logo" width="180"/>
</p>

# Energy and Carbon Accounting for Global AI Data Centres

[中文说明](README.zh-CN.md)

This repository accompanies a study of electricity use and operational carbon
emissions from artificial-intelligence (AI) data centres. The implementation
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
> operational carbon emissions. It does not assess water use or water usage
> effectiveness.

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
| MLPerf power diagnostic and trace-boundary audit (Supplementary Fig. S1) | [`scripts/validate_energy_model.py`](scripts/validate_energy_model.py), [`dataset/validation/`](dataset/validation/) | Generated under `results/model_validation/` |

## Study scope and scenarios

### Countries

The default model boundary contains 24 countries:

`USA`, `China`, `Japan`, `France`, `India`, `Singapore`, `Canada`, `Germany`,
`United_Kingdom`, `Australia`, `Italy`, `South_Korea`, `South_Africa`, `Ireland`,
`UAE`, `Brazil`, `Israel`, `Netherlands`, `Spain`, `Sweden`, `Belgium`, `Norway`,
`Poland`, and `Switzerland`.

These identifiers are the canonical keys used by
[`dataset/Installed_capacity_data.py`](dataset/Installed_capacity_data.py) and
[`dataset/Factors.py`](dataset/Factors.py). The modelled country allocation does
not include the residual global capacity share, so repository outputs should be
described as totals across the modelled countries rather than complete global
totals.

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
| Figure 1–4 source-data workbooks | Included | [`results/`](results/) |
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
Calibration endpoints and interpolation assumptions are documented in the
source. Country shares and intermediate scenario values are model inputs or
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

The annual CPU model uses these annual factors directly. They also provide the
annual means used by bridge configurations and the documented fallback when an
hourly file is unavailable.

### Hourly grid carbon intensity

The hourly model expects 360 local CSV files under `dataset/EM-CPNDCNZ/`: 24
countries × three electricity-system pathways × five years (2026–2030). Files
contain UTC timestamps and direct and life-cycle carbon-intensity columns.

The study constructs future hourly series from the relative hourly pattern of
authorised 2025 Electricity Maps data, scaled to the annual pathways in
[`dataset/Factors.py`](dataset/Factors.py). The processed hourly files are not
redistributed. Eligible researchers must obtain access through the Electricity
Maps [data portal](https://www.electricitymaps.com/data) or
[academic-access guidance](https://help.electricitymaps.com/en/articles/13168512-academic-data-access-and-availability)
and comply with the provider's current terms.

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
├── results/                           # Generated outputs and figure workbooks
├── scripts/
│   ├── figure1_data.py
│   ├── figure2_data.py
│   ├── figure3_data.py
│   ├── figure4_data.py
│   └── validate_energy_model.py
├── run.py                             # Two-endpoint comparison entry point
├── LICENSE
├── README.md
└── README.zh-CN.md
```

## Reproducibility environment

The submission package was validated with Python 3.12.3 and:

- `numpy==2.1.3`
- `pandas==2.2.3`
- `pyarrow==17.0.0`
- `openpyxl==3.1.5`
- `matplotlib==3.9.2`

Create an isolated environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy==2.1.3 pandas==2.2.3 pyarrow==17.0.0 openpyxl==3.1.5 matplotlib==3.9.2
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

### Compare the two endpoint models

```powershell
python run.py --policy CP --scenarios Base --year-start 2026 --years 5
```

The entry point reads the workload trace once, runs the annual CPU and hourly
workload-resolved endpoints, and writes results under
`results/model_comparison/`. Multiple demand pathways may be supplied:

```powershell
python run.py --policy NDC --scenarios Base "Lift-Off" "High Efficiency"
```

For a quick execution check, restrict countries and trace intervals:

```powershell
python run.py --countries USA --year-start 2026 --years 1 --max-intervals 24
```

`max_intervals` changes the trace sampling basis and is only for smoke tests and
debugging. Omit it for manuscript results.

### Reproduce figure source data

```powershell
python scripts/figure1_data.py
python scripts/figure2_data.py
python scripts/figure3_data.py
python scripts/figure4_data.py
```

Figure 3 is the authoritative implementation of the ordered bridge. The script
checks variant coverage, country aggregation, hourly-to-annual energy closure,
and additive bridge closure before writing its workbook.

### Run external diagnostics

```powershell
python scripts/validate_energy_model.py --bootstrap-replicates 10000
```

The diagnostic workflow uses random seed `20260819`. Input provenance and
filtering rules are documented under [`dataset/validation/`](dataset/validation/).

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
    save_hourly_outputs=False,
)
```

The hourly interface accepts `hourly_carbon_scope="direct"` or
`"life_cycle"`. Hourly CSV output is disabled by default because it can be
large.

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
| `hourly_carbon_fallback_to_annual` | Use the annual factor if an hourly file is missing | `True` |
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
└── summary/
    ├── Endpoint_Models_Country_Annual.csv
    ├── Endpoint_Models_Global_Annual.csv
    ├── Endpoint_Comparison_Country_Annual.csv
    └── Endpoint_Comparison_Global_Annual.csv
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
- All countries inherit the same normalized relative workload shape; the model does not represent time zones or country-specific task mixes.
- The 4,440-h Alibaba trace is repeated to 8,760 h. The repeat seam can determine an unfiltered maximum ramp even though it does not change annual energy.
- Future hourly grid series preserve a 2025 relative shape and do not forecast changes in dispatch, storage, imports, or variable-generation profiles.
- Component shares, unit power, idle fractions, fan response, memory activity, and storage activity are configurable reference assumptions.
- The MLPerf diagnostic assesses order of magnitude and bias; it is not a calibration fit and does not validate future national facility electricity.
- Missing hourly carbon data can fall back to annual factors. Inspect `carbon_factor_source` before interpreting temporal results.
- The hourly workload-resolved endpoint is a transparent reference configuration, not measured ground truth.

## Licence

The project code is released under the [MIT License](LICENSE). Third-party data
remain subject to the original providers' access, citation, and redistribution
terms.
