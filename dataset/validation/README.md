# External validation data

This directory contains public data used by `scripts/validate_energy_model.py`.

## MLPerf Power

`raw/mlperf_power_raw_data.csv` is the public aggregate table released with the
MLPerf Power analysis repository:

- Repository: https://github.com/aryatschand/MLPerf-Power-HPCA-2025
- Raw table: https://raw.githubusercontent.com/aryatschand/MLPerf-Power-HPCA-2025/main/raw_data.csv
- Accessed: 2026-08-19

The validation script retains Closed-division, data-centre inference submissions
with Server or Offline measurements, NVIDIA accelerators, and reported whole-
system power. `Public ID` is the independent unit; repeated workload rows are
aggregated within a system submission.

## National data-centre electricity statistics

`national_data_center_energy.csv` transcribes annual totals from official
statistical releases for Ireland and the Netherlands. These observations cover
all data centres within the agencies' stated scopes, whereas the model estimates
AI-related electricity in selected countries and future years. The series is
therefore used only for a scope and order-of-magnitude audit, not to calculate
prediction errors.

## Reproduce

Run from the repository root:

```powershell
python -m pip install numpy==2.1.3 pandas==2.2.3 pyarrow==17.0.0 openpyxl==3.1.5 matplotlib==3.9.2
python scripts\validate_energy_model.py --bootstrap-replicates 10000
```

The script writes system-level source data and diagnostic tables to
`results/model_validation/` and exports the supplementary figure as editable
SVG/PDF plus 600-dpi TIFF and preview PNG under `figures/`.
