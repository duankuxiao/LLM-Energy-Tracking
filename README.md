# AI Data Center Footprint Model

This repository computes the projected energy, carbon, and water footprints of AI data centers across 24 countries for the manuscript figures.

It supports:
- demand scenarios from 2025 to 2030
- power-sector policy pathways (`CP`, `NDC`, `NZ`)
- country-level training and inference allocation
- deployment optimization for cross-border workload allocation
- sensitivity analysis for key technical assumptions

## Repository Layout

```text
Code/
|- core/        # Core calculation and optimization modules
|- scripts/     # Entry scripts for Fig. 1 to Fig. 5
|- dataset/     # Input datasets and scenario factors
|- paper/       # Manuscript and supplementary information
|- results/     # Generated outputs
`- run_example.py
```

## Quick Start

Run the manuscript figure scripts directly:

```bash
python scripts/Fig1_Results.py
python scripts/Fig2_Results.py
python scripts/Fig3_Results.py
python scripts/Fig4_Results.py
python scripts/Fig5_Results.py
```

Run a minimal custom example:

```bash
python run_example.py
```

## Core API

The main entry point is:

```python
from core import AIFootprint
```

Typical usage:

```python
result = AIFootprint(
    renewable_energy_policy="CP",
    scenarios=["Base"],
    years=6,
    countries=[...],
    default_p_infer=0.7,
    save_outputs=False,
    verbose=False,
    return_results=True,
)
```

## Main Inputs

### Scenario inputs

- `renewable_energy_policy`: `CP`, `NDC`, or `NZ`
- `scenarios`: one or more of `Base`, `Lift-Off`, `High Efficiency`, `Headwinds`
- `years`: usually `6` for `2025-2030`
- `countries`: list of the 24 modeled countries

### Workload allocation inputs

- `infer_ratio_by_country`: optional country-specific inference share
- `default_p_infer`: default inference share for countries not explicitly specified

### Sensitivity-analysis inputs

- `u_train`
- `u_infer`
- `idle_power_rate`
- `max_power_rate`
- `pue_scale`

### Output controls

- `output_dir`
- `save_outputs`
- `verbose`
- `return_results`

## Main Outputs

When `save_outputs=True`, the model writes CSV files such as:

- `Country_PowerUsage_<policy>_<scenario>.csv`
- `Country_WaterUsage_<policy>_<scenario>.csv`
- `Country_DirectWaterUsage_<policy>_<scenario>.csv`
- `Country_CarbonEmission_<policy>_<scenario>.csv`
- `Country_PowerUsage_Train_<policy>_<scenario>.csv`
- `Country_PowerUsage_Infer_<policy>_<scenario>.csv`
- `Total_Summary_Results_<policy>_<scenario>.csv`

Units:

- Power: `TWh`
- Carbon: `MtCO2`
- Water: `million m3`

## Output Examples

Aggregate summary:

```csv
,Power_Base,Water_Base,DirectWater_Base,Carbon_Base
2025,416.0249,1828.3867,644.5379,171.7015
2026,491.4298,2155.0306,760.6908,199.5889
...
2030,909.2719,3953.4636,1399.4678,346.6982
```

Country-level output:

```csv
,USA,China,Japan,France,India,...,Switzerland
2025,203.3815,116.3443,9.1011,6.5388,9.1364,...,2.8044
2026,240.1922,137.5570,10.7403,7.7206,10.8107,...,3.3095
...
```

Fig. 5 sensitivity summary:

```csv
parameter_key,parameter_name,argument_name,low_value,applied_value,high_value,...
training_utilization,Training utilization,u_train,0.6,0.8,0.9,...
inference_utilization,Inference utilization,u_infer,0.4,0.5,0.6,...
idle_power_rate,Server idle power,idle_power_rate,0.1,0.23,0.35,...
```

## Which File Should I Use?

- Use `scripts/` if you want the manuscript figure result tables.
- Use `core.AIFootprint(...)` if you want to build a custom scenario.
- Use `core/modeling.py`, `core/modeling_parition.py`, and `core/optimize.py` if you want the deployment optimization logic.

## Additional Notes

- The workflow is CSV-first, so plotting and downstream analysis can be done separately.
