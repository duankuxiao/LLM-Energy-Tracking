# Manuscript and supplementary source tables

These 24 compact CSV tables were exported from the completed manuscript experiment
`all_experiments_calendar_20261017_092038_285`. They preserve the reported numerical
results while avoiding large intermediate workbooks. `manifest.json` records each
table's row count, SHA-256, source file and, where relevant, workbook sheet.

Exporting tables does not rerun the workload model. The source experiment used all
24 study countries during 2026–2030, the retained AI-capacity factors, direct grid
emissions and complete UTC calendars. The global electricity reference was fitted
to the selected-country aggregate; the 93.18% capacity share is not observed
electricity coverage. Absolute results remain conditional on that calibration.

| Files | Manuscript/SI evidence |
| --- | --- |
| `S1_*.csv` | MLPerf filtering, Public-ID measurements, bootstrap metrics, hardware strata, the local trace-boundary proxy and national scope audits |
| `S2_*.csv` | Task/component energy, idle/dynamic shares, unclassified activity and system closure |
| `S3_sensitivity.csv` | All 15 parameters, with Low/Base/High levels and electricity/emissions responses (45 rows); the main Figure 4 displays six and Figure S3 displays the additional nine |
| `S4_peak_ramp.csv` | Calendar means, peaks, P99 power, signed/absolute ramps and repeat/alignment-join filtering |
| `S4_timing_aggregate.csv` | Synthetic fixed-electricity timing controls and coincident aggregate power metrics |
| `annual_endpoint_comparison.csv` | Annual CPU versus hourly endpoints and percentage gaps |
| `annual_activity_controls.csv`, `annual_common_mean_loads.csv` | Calendar/fixed/historical activity and common-mean controls |
| `boundary_calibration.csv`, `boundary_invariants.csv` | Calibration boundary comparisons and common-scaling invariants |

Column names identify units. Electricity is generally MWh (or explicitly TWh),
power is MW and ramps are MW per hour; divide MW by 1000 for GW. Sensitivity
percentages use each metric's own mean of annual country sums over 2026–2030.
Low/High denotes parameter level, not response direction. Small nonzero responses
are retained rather than rounded to zero.

The MLPerf analysis unit is a Public ID, not proven independent physical hardware.
Its intervals use 10,000 Public-ID bootstrap resamples with seed 20260819. These
diagnostics do not validate training, idle/dynamic curves, PUE or future national
accuracy. The local trace seam proxy is separate from full-calendar power ramps.
Deterministic parameter and timing contrasts are not joint uncertainty intervals.

The current regression suite checks the annual endpoint and S1 bootstrap results
against these tables, executes all 15 sensitivities on a synthetic trace, and
checks source-table integrity. Reproducing the full original trace experiment
requires the external Alibaba and hourly grid inputs documented in the root README.

To export the same table set from another completed full experiment:

```powershell
python -B scripts/export_supplementary_sources.py --experiment-dir results/example_run --output-dir results/new_supplementary_sources
```

Use a new output directory. The exporter never edits manuscript or SI files.
