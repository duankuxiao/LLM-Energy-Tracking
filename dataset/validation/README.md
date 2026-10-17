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

W07 treats this comparison as an external configuration diagnostic. The
configured full-load reference is `gpu_full_power_w / gpu_power_share` per
accelerator; it is not fitted to the benchmark table. Source unit watts may
cancel from capacity-normalized model electricity when idle/full ratio is
held fixed, so a match to this reference does not establish energy accuracy.

Exact duplicate rows are removed. IDs with inconsistent hardware/count/node/
version metadata are audited and excluded. Hardware families, explicit form
factor labels, single/multiple nodes, Server/Offline and task models have
separate metrics. Within each stratum, one Public ID contributes one aggregated
reported power value. IDs may reuse physical hardware; independence is an
analysis assumption. A single-ID stratum has no bootstrap confidence interval.
A100/H100 sample coverage and their descriptive median difference are exported
separately, including missing/insufficient-group flags.

`observed_power_max_w` retains its legacy name but denotes the maximum of
reported workload powers within an ID, not a measured time-series peak or
known full-utilization power. The aggregate table does not identify GPU idle
power, utilization-response curves, training power, PUE or facility energy.
No curve fit or train/validation split is performed; those require additional
independent-system utilization/power measurements. Bootstrap cannot establish
representativeness of future fleets or conservativeness.

Additional outputs under `results/model_validation/`:

- `mlperf_public_id_audit.csv`: consistency and excluded conflicting IDs;
- `mlperf_stratified_metrics.csv`, `mlperf_family_summary.csv`: stratum errors,
  independent-unit counts, uncertainty and within-ID aggregation rules;
- `mlperf_hardware_coverage.csv`, `mlperf_a100_h100_comparison.csv`: hardware
  coverage and descriptive differences;
- `validation_claim_scope.csv`: supported diagnostic roles and data gaps;
- `validation_run_metadata.csv`: input hash, reference, seed and assumptions;
- `validation_hardware_parameters.csv`, `validation_parameter_dependencies.csv`:
  actual hardware configuration and its model dependencies.

Optional `--hardware-config-file` supplies JSON overrides of the shared
hardware configuration. `--skip-figure` produces tables only; `--skip-trace`
also avoids the endpoint parquet read. The code revision has not regenerated
these outputs; run it after the planned source updates are complete.

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
