"""Export compact manuscript/SI source tables from a completed experiment.

This copies existing numerical results, without rerunning models, drawing figures,
or editing manuscript files. Raw workload and hourly grid inputs are not copied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research_io import sha256_file


CSV_SOURCES = {
    "S1_filter_audit.csv": "validation/validation_filter_audit.csv",
    "S1_metrics.csv": "validation/validation_metric_summary.csv",
    "S1_systems.csv": "validation/mlperf_system_validation.csv",
    "S1_family_summary.csv": "validation/mlperf_family_summary.csv",
    "S1_stratified_metrics.csv": "validation/mlperf_stratified_metrics.csv",
    "S1_hardware_coverage.csv": "validation/mlperf_hardware_coverage.csv",
    "S1_a100_h100_comparison.csv": "validation/mlperf_a100_h100_comparison.csv",
    "S1_trace_boundary.csv": "validation/trace_boundary_diagnostics.csv",
    "S1_trace_seam_window.csv": "validation/trace_seam_window.csv",
    "S1_national_scope.csv": "validation/national_scope_audit.csv",
    "S1_national_series.csv": "validation/national_official_energy_series.csv",
    "annual_endpoint_comparison.csv": "comparison/summary/Endpoint_Comparison_Selected_Countries_Annual.csv",
    "boundary_calibration.csv": "boundary/calibration_audit.csv",
    "boundary_invariants.csv": "boundary/invariant_checks.csv",
}
WORKBOOK_SOURCES = {
    1: {
        "S2_task_component_annual.csv": "W05_Task_Component_Annual",
        "S2_task_system_shares.csv": "W05_Task_System_Shares",
        "S2_component_system_shares.csv": "W05_Component_System_Shares",
        "S2_system_closure.csv": "W05_System_Closure",
        "S2_task_component_display.csv": "W08_Task_Component_Display",
    },
    3: {
        "S4_peak_ramp.csv": "W06_Baseline_Peak_Ramp",
        "S4_timing_aggregate.csv": "W06_Timing_Aggregate",
        "annual_activity_controls.csv": "W04_Control_Comparison",
        "annual_common_mean_loads.csv": "W04_Common_Mean_Load",
    },
    4: {"S3_sensitivity.csv": "Fig4a_Sensitivity"},
}


def export_sources(experiment_dir, output_dir):
    source, output = Path(experiment_dir).resolve(), Path(output_dir).resolve()
    if not source.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError("Source and output directories must be inside the workspace.")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    source_paths = [source / path for path in CSV_SOURCES.values()]
    source_paths.extend(source / f"data/figure{number}_data.xlsx" for number in WORKBOOK_SOURCES)
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Read all selected tables before creating an output directory. Large hourly
    # and task-level diagnostic sheets are deliberately never loaded.
    tables, origins = {}, {}
    for name, relative in CSV_SOURCES.items():
        tables[name] = pd.read_csv(source / relative)
        origins[name] = {"source": relative}
    for number, entries in WORKBOOK_SOURCES.items():
        relative = f"data/figure{number}_data.xlsx"
        with pd.ExcelFile(source / relative, engine="openpyxl") as book:
            for name, sheet in entries.items():
                tables[name] = pd.read_excel(book, sheet_name=sheet)
                origins[name] = {"source": relative, "sheet": sheet}
    sensitivity = tables["S3_sensitivity.csv"]
    if sensitivity["parameter_name"].nunique() != 15 or len(sensitivity) != 45:
        raise ValueError("The manuscript export requires all 15 Low/Base/High sensitivity experiments.")

    output.mkdir(parents=True)
    records = []
    for name, frame in tables.items():
        path = output / name
        frame.to_csv(path, index=False, lineterminator="\n")
        records.append({"file": name, "rows": len(frame), "sha256": sha256_file(path), **origins[name]})
    manifest = {
        "schema_version": 1,
        "source_experiment": source.relative_to(ROOT).as_posix(),
        "role": "compact_source_tables_from_completed_experiment_not_a_new_model_run",
        "accounting_scope": "24_selected_countries_2026_2030",
        "source_sha256": {path.relative_to(source).as_posix(): sha256_file(path) for path in source_paths},
        "tables": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/supplementary")
    args = parser.parse_args()
    result = export_sources(args.experiment_dir, args.output_dir)
    print(f"Exported {len(result['tables'])} compact source tables to {args.output_dir}")


if __name__ == "__main__":
    main()
