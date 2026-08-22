<p align="center">
  <img src="logo.png" alt="Project logo" width="180"/>
</p>

# Energy and Carbon Accounting for Global AI Data Centres

[中文说明](README.zh-CN.md)

This repository accompanies a research study of electricity use and carbon
emissions from artificial intelligence (AI) data centres. It combines projected
data-centre capacity, country allocation, the AI share of installed capacity,
server power characteristics, power usage effectiveness (PUE), and annual or
hourly grid carbon intensity to evaluate electricity use and carbon emissions
from 2025 to 2030.

The central research design is a two-by-two accounting matrix that varies the
temporal resolution of grid carbon intensity and the representation of server
power. Comparing models M1–M4 separates the effects of annual carbon averaging
from those of using a conventional central processing unit (CPU)-based power
approximation for graphics processing unit (GPU)-intensive AI workloads.

> **Scope.** The study quantifies electricity use, load, and carbon emissions.
> It does not assess water consumption or water usage effectiveness.

## Accounting framework

| Model | Energy model | Grid carbon intensity | Role in the comparison |
| --- | --- | --- | --- |
| M1 | Conventional CPU utilisation–power model | Annual mean | Simplified accounting baseline |
| M2 | Conventional CPU utilisation–power model | Hourly | Isolates temporal aggregation in carbon accounting |
| M3 | GPU workload and component-level power model | Annual mean | Isolates AI-hardware representation |
| M4 | GPU workload and component-level power model | Hourly | Reference configuration for model comparison |

All four models use the same data-centre capacity, country allocation, AI
capacity boundary, PUE assumptions, demand scenarios, and electricity-policy
pathways. They differ only in server-energy representation and the temporal
resolution of grid carbon intensity.

### M1: annual CPU model

[`core/m1_annual_cpu_model.py`](core/m1_annual_cpu_model.py) implements the
conventional utilisation–power method in
[`core/past_research_data_center_energy_carbon_model.py`](core/past_research_data_center_energy_carbon_model.py).
It calculates annual information technology (IT) electricity use from
idle-power, maximum-power, and utilisation parameters; applies PUE; and assigns
annual grid carbon intensities from [`dataset/Factors.py`](dataset/Factors.py).

The model uses a common four-class task taxonomy: training, inference, other,
and unclassified. Its default classified workload shares are 20% training, 75%
inference, and 5% other.

### M2: hourly-carbon CPU model

[`core/m2_hourly_carbon_cpu_model.py`](core/m2_hourly_carbon_cpu_model.py) uses
the same annual energy model and task classification as M1. Annual electricity
use is therefore identical by design. M2 represents annual electricity demand
as constant hourly power and matches it to hourly grid carbon intensity, thereby
isolating the effect of temporal resolution in carbon accounting.

### M3: annual GPU model

[`core/m3_annual_gpu_model.py`](core/m3_annual_gpu_model.py) uses the GPU
workload and component-level power calculations implemented for M4, but applies
annual grid carbon intensities. Public cluster traces are used to estimate CPU,
GPU, memory, storage, and IT fan power before the workload is scaled to national
AI IT capacity.

### M4: hourly GPU model

[`core/m4_hourly_gpu_model.py`](core/m4_hourly_gpu_model.py) is the complete
hourly configuration. It constructs hourly resource demand for training,
inference, other, and unclassified tasks from GPU-pod and server traces;
calculates component-level power; applies PUE; and matches facility electricity
use to country-specific hourly carbon intensity.

## Study scope and scenarios

### Countries

The default analysis covers 24 major data-centre countries:

`USA`, `China`, `Japan`, `France`, `India`, `Singapore`, `Canada`, `Germany`,
`United_Kingdom`, `Australia`, `Italy`, `South_Korea`, `South_Africa`, `Ireland`,
`UAE`, `Brazil`, `Israel`, `Netherlands`, `Spain`, `Sweden`, `Belgium`, `Norway`,
`Poland`, and `Switzerland`.

These identifiers are the canonical country keys used in
[`dataset/Installed_capacity_data.py`](dataset/Installed_capacity_data.py) and
[`dataset/Factors.py`](dataset/Factors.py).

### AI data-centre demand scenarios

| Scenario | Interpretation |
| --- | --- |
| `Base` | Baseline capacity-growth pathway |
| `Lift-Off` | Faster demand-growth pathway |
| `High Efficiency` | Efficiency-improvement pathway |
| `Headwinds` | Constrained-growth pathway |

Scenario names passed to the code must use the exact spelling shown above.

### Electricity-system pathways

| Pathway | Interpretation |
| --- | --- |
| `CP` | Current Policies |
| `NDC` | Nationally Determined Contributions |
| `NZ` | Net Zero |

Annual carbon intensities, future hourly scaling, and model outputs are
organised using these three pathway identifiers.

## Data and code availability

| Resource | Repository status | Access or provenance |
| --- | --- | --- |
| M1–M4 source code | Included | This repository |
| Annual carbon-intensity and PUE inputs | Included | [`dataset/Factors.py`](dataset/Factors.py) |
| Capacity and scenario inputs | Included | [`dataset/Installed_capacity_data.py`](dataset/Installed_capacity_data.py) |
| External-validation inputs | Included | [`dataset/validation/`](dataset/validation/) |
| Figure 1–4 source-data workbooks | Included | [`results/`](results/) |
| Electricity Maps hourly data and derived hourly scenario files | **Not distributed** | Obtain authorised source data directly from Electricity Maps |
| Alibaba GPU-pod and server traces | Not bundled | Obtain from the Alibaba Cluster Trace Program |
| IEA data annex | Not redistributed | Obtain from the International Energy Agency |

The code is public, but complete execution of M2 and M4 requires hourly data
that this repository cannot redistribute. The source-data workbooks allow the
reported main-figure values to be inspected without those restricted files.

### Capacity and growth inputs

[`dataset/Installed_capacity_data.py`](dataset/Installed_capacity_data.py)
contains total, IT, and non-IT capacity for four growth scenarios from 2025 to
2030, together with country shares and the default AI-capacity calibration
factors.

The default calibration follows the AI-dedicated data-centre electricity
pathway discussed in the International Energy Agency report
[Key Questions on Energy and AI](https://www.iea.org/reports/key-questions-on-energy-and-ai).
Calibration endpoints and assumptions for intermediate years are documented in
the code. Country shares and scenario interpolation are model inputs or derived
parameters; they should not be interpreted as country-level forecasts published
by the original data providers.

### GPU workload and server traces

M3 and M4 use two components of the Alibaba Serverless Infrastructure traces:

- `dataset/asi_opensource_pod_hourly/`: hourly GPU-pod traces;
- `dataset/asi_opensource_server_hourly/`: hourly server-resource and device
  inventories.

The traces are available from the
[Alibaba Cluster Trace Program](https://github.com/alibaba/clusterdata). Their
background and system context are described in
[Heterogeneity at Hyperscale: Characterization and Scheduling of Large Production AI Clusters at Alibaba](https://www.usenix.org/conference/osdi26/presentation/li-suyi).

The default energy boundary includes every active pod. CPU and memory activity
from CPU-only pods remains included when `gpu_request == 0`. Task-share
statistics use a separate, stricter paper-specific filter: within the study
window, only classified tasks with requested GPU memory and valid GPU-use hours
enter the reported classified workload shares. Training remains training;
online and offline inference are combined as inference; `dev` and explicitly
labelled `other` tasks are combined as other; and missing, `unknown`, or invalid
labels remain unclassified. Unclassified load contributes to total energy use
but not to the classified task-share denominator.

Because the public traces do not include complete storage activity or host
memory capacity, the implementation uses idle storage power and a CPU-weighted
memory-activity proxy. M4 records these assumptions in its trace-capacity source
output.

### Annual carbon intensity and PUE

[`dataset/Factors.py`](dataset/Factors.py) provides annual grid carbon intensity
for the 24 countries under the `CP`, `NDC`, and `NZ` pathways, together with PUE
inputs by country, year, and demand scenario. Carbon intensity is expressed as
`kg CO₂/MWh`. These values are used directly by M1 and M3 and as the documented
fallback when hourly data are unavailable.

### Hourly grid carbon intensity

M2 and M4 expect 360 local hourly files under `dataset/EM-CPNDCNZ/`, covering 24
countries, three electricity-system pathways, and five years from 2026 to 2030.
The study constructs future hourly scenarios from the relative hourly pattern
of authorised 2025 Electricity Maps downloads, scaled to the annual pathways in
[`dataset/Factors.py`](dataset/Factors.py).

This project does not hold permission to redistribute the downloaded or derived
hourly files. Neither `dataset/EM-CPNDCNZ.zip` nor the hourly CSV files are
included in the GitHub repository. Eligible researchers must request access
through the Electricity Maps
[data portal](https://www.electricitymaps.com/data) or
[academic-access guidance](https://help.electricitymaps.com/en/articles/13168512-academic-data-access-and-availability)
and comply with the provider's current terms.

## Repository structure

```text
.
├── core/
│   ├── past_research_data_center_energy_carbon_model.py
│   ├── m1_annual_cpu_model.py
│   ├── m2_hourly_carbon_cpu_model.py
│   ├── m3_annual_gpu_model.py
│   ├── m4_hourly_gpu_model.py
│   └── task_model.py
├── dataset/
│   ├── Factors.py
│   ├── Installed_capacity_data.py
│   ├── EM-CPNDCNZ/                    # Local authorised hourly inputs; CSVs ignored by Git
│   ├── validation/
│   ├── asi_opensource_pod_hourly/     # External local input; ignored by Git
│   └── asi_opensource_server_hourly/  # External local input; ignored by Git
├── results/                           # Generated outputs and Figure 1–4 source workbooks
├── scripts/                           # Figure-data and validation scripts
├── run.py                             # Unified M1–M4 entry point
├── LICENSE
├── README.md                          # Default English documentation
└── README.zh-CN.md                    # Chinese documentation
```

## Reproducibility environment

The submission package was validated with Python 3.12.3 and the following exact
dependency versions:

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

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Preparing external inputs

The following local inputs are excluded by `.gitignore`:

```text
dataset/EM-CPNDCNZ.zip
dataset/EM-CPNDCNZ/
dataset/asi_opensource_pod_hourly/
dataset/asi_opensource_server_hourly/
```

After obtaining the relevant permissions and source data, prepare the following
directory structure:

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

Run all commands from the repository root.

### Unified M1–M4 run

```powershell
python run.py --policy CP --scenarios Base --year-start 2026 --years 5
```

Multiple demand scenarios can be supplied in one run. Quote names that contain
spaces:

```powershell
python run.py --policy NDC --scenarios Base "Lift-Off" "High Efficiency"
```

The unified entry point constructs one in-memory `WorkloadProfile` from the
Alibaba traces and passes the same object to M3 and M4. The large Parquet inputs
are therefore read once, while annual-carbon and hourly-carbon calculations
remain separate. Outputs are written to `results/m1_m4_comparison/`.

For a smoke test, restrict the country set and trace intervals:

```powershell
python run.py --countries USA --year-start 2026 --years 1 --max-intervals 24
```

`max_intervals` changes the trace sampling basis and is intended only for smoke
tests and debugging. Omit it when generating manuscript results. Run
`python run.py --help` for the complete command-line interface.

### External validation

Validation inputs are documented under
[`dataset/validation/`](dataset/validation/), and the executable validation
workflow is [`scripts/validate_energy_model.py`](scripts/validate_energy_model.py).

```powershell
python scripts/validate_energy_model.py --bootstrap-replicates 10000
```

The script uses the fixed random seed `20260819`. It writes diagnostic tables to
`results/model_validation/` and exports the supplementary validation figure to
`figures/`. The validation documentation defines data provenance, filtering,
the independent system-submission unit, and the scope boundary of national
statistics.

### Figure source data

The analysed source data for the main figures are provided as existing Excel
workbooks:

| Workbook | Contents |
| --- | --- |
| [`results/figure1_data.xlsx`](results/figure1_data.xlsx) | Figure 1a–g |
| [`results/figure2_data.xlsx`](results/figure2_data.xlsx) | Figure 2a–d and the complete country-level source table for Figure 2b |
| [`results/figure3_data.xlsx`](results/figure3_data.xlsx) | Figure 3a–d |
| [`results/figure4_data.xlsx`](results/figure4_data.xlsx) | Figure 4a |

Other generated files under `results/` remain excluded from Git.

## Model-specific Python interfaces

### M1

```python
from core.m1_annual_cpu_model import run_m1_annual_cpu_model

result = run_m1_annual_cpu_model(
    renewable_energy_policy="CP",
    scenarios=["Base", "Lift-Off"],
    year_start=2026,
    years=5,
)
```

### M2

```python
from core.m2_hourly_carbon_cpu_model import run_m2_hourly_carbon_cpu_model

result = run_m2_hourly_carbon_cpu_model(
    renewable_energy_policy="NDC",
    scenarios=["Base"],
    year_start=2026,
    years=5,
    hourly_carbon_scope="direct",
    save_hourly_outputs=False,
)
```

`hourly_carbon_scope` accepts `direct` or `life_cycle`. Hourly outputs are
disabled by default because they can be large.

### M3

```python
from core.m3_annual_gpu_model import run_m3_annual_gpu_model

result = run_m3_annual_gpu_model(
    renewable_energy_policy="NZ",
    scenarios=["High Efficiency"],
    year_start=2026,
    years=5,
)
```

M3 requires the GPU traces but uses annual carbon intensity and does not produce
an hourly carbon-emissions table.

### M4

```python
from core.m4_hourly_gpu_model import run_workload_component_footprint

result = run_workload_component_footprint(
    renewable_energy_policy="CP",
    scenarios=["Base"],
    year_start=2026,
    years=5,
    save_hourly_outputs=False,
)
```

A reduced smoke test can be run with:

```python
result = run_workload_component_footprint(
    renewable_energy_policy="CP",
    scenarios=["Base"],
    countries=["USA"],
    year_start=2026,
    years=1,
    max_intervals=24,
    save_outputs=False,
)
```

Do not use `max_intervals` when generating manuscript results.

## Principal parameters

| Parameter | Meaning | Common value or default |
| --- | --- | --- |
| `renewable_energy_policy` | Electricity-system pathway | `CP`, `NDC`, or `NZ` |
| `scenarios` | AI data-centre demand scenario | `Base`, `Lift-Off`, `High Efficiency`, or `Headwinds` |
| `year_start` | First output year | M1/M2 support 2025–2030; M3/M4 default to 2026 |
| `years` | Number of consecutive years | 5 |
| `countries` | Countries included | All 24 by default |
| `pue_scale` | PUE sensitivity multiplier | `1.0` |
| `ai_capacity_factors` | Annual AI-capacity calibration | `DEFAULT_AI_CAPACITY_FACTORS` |
| `hourly_carbon_scope` | Hourly carbon-intensity boundary | `direct` or `life_cycle` |
| `hourly_carbon_fallback_to_annual` | Use annual value when hourly data are missing | `True` |
| `max_intervals` | Limit the GPU-trace intervals read | Full trace by default |
| `default_p_infer` | Default M1/M2 inference share | `0.75` |
| `default_p_other` | Default M1/M2 other-task share | `0.05` |
| `task_ratio_by_country` | Country-specific M1/M2 task-share override | None by default |
| `include_zero_gpu_pods` | Include CPU-only pods in M3/M4 energy boundary | `True` |

M1 and M2 also expose task shares, task-specific utilisation, idle-power ratios,
and maximum-power ratios. M3 and M4 expose hardware and trace-processing
assumptions through `HardwarePowerConfig` and `Alibaba2026TraceConfig`.

## Outputs

The unified runner writes model-specific results to
`results/m1_m4_comparison/m1_annual_cpu/` through
`results/m1_m4_comparison/m4_hourly_gpu/`. The `summary/` directory contains:

- `All_Models_Country_Annual.csv`
- `All_Models_Global_Annual.csv`
- `Model_Comparison_Country_Annual.csv`
- `Model_Comparison_Global_Annual.csv`

The comparison tables report electricity use, load-weighted carbon intensity,
and carbon emissions for all four models. For M1–M3, absolute and percentage
differences are calculated relative to M4 as
`(model result - M4 result) / M4 result × 100%`; a positive value indicates a
result above M4.

M4 can additionally return or save:

- `annual_summary`: country-year energy, emissions, and resource utilisation;
- `component_energy`: CPU, GPU, memory, storage, and IT fan energy;
- `task_type_energy`: energy for training, inference, other, and unclassified
  tasks;
- `task_demand` and `task_execution`: resource-hours by task origin and execution
  location;
- `capacity_overflow`: capacity-overflow diagnostics;
- `hourly_carbon`: optional hourly electricity, carbon intensity, and emissions;
- `workload_profile_summary`: workload composition under each reporting basis;
- `trace_resource_capacity`: estimated trace capacity and provenance.

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

- Capacity inputs cover 2025–2030 and should not be extrapolated beyond that
  interval without additional evidence.
- The country allocation of IT capacity is represented by
  `IT_RATIO`; use of fixed shares should be reported as a limitation.
- M2 tests the effect of hourly carbon intensity under constant hourly power. It
  does not represent diurnal variation in training or inference demand.
- The Alibaba traces use relative time. M3 and M4 annualise the traces to 8,760
  hours and repeat their relative hourly pattern when matching future calendar
  years.
- Component-power shares, unit full-load power, idle power, and fan curves are
  configurable model assumptions and should be evaluated through sensitivity
  or uncertainty analysis.
- Annual M4 totals can conceal cancellation between positive and negative
  country-hour deviations; formal analysis should examine country and global
  scales together.
- When hourly carbon intensity is missing, the code can fall back to annual
  values. Inspect `carbon_factor_source` before interpreting an output as based
  on measured hourly variation.
- Full GPU traces require substantial storage, memory, and processing time.

## Version

The manuscript-submission reproducibility package is fixed at Git tag
[`submission-2026-08-22`](https://github.com/duankuxiao/LLM-Energy-Tracking/tree/submission-2026-08-22).
The tag is immutable; later documentation corrections remain on the `main`
branch unless released under a new tag.

## Licence

The project code is released under the [MIT License](LICENSE). Third-party data
are not automatically covered by this software licence. Use, citation, and
redistribution of Alibaba, International Energy Agency, Electricity Maps, and
other external resources remain subject to the original providers' terms.
