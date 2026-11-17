"""Run configured experiments in one process; this entry point never draws.

    python -B scripts/run_experiments.py --config configs/experiments.json
    python -B scripts/run_experiments.py --config configs/experiments.json --only figure1 figure4 --output-dir results/my_run

Inputs are shared in memory within each execution. Figure drawing uses plot_figure1.py
through plot_figure4.py separately, after all desired data have been generated.
"""

import argparse
from dataclasses import asdict
from importlib import import_module
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.experiment_config import EXPERIMENT_NAMES, load_experiment_config
from core.experiment_context import ExperimentContext
from core.run_provenance import RunProvenance, json_value


def _comparison(context, output):
    from run import run_model_comparison
    cfg, entry = context.config, context.config.experiments["comparison"]
    result = run_model_comparison(**context.figure_arguments(),
                                  renewable_energy_policy=entry["policy"], scenarios=entry["scenarios"],
                                  hardware_config=cfg.hardware, trace_config=cfg.trace, activity_config=cfg.activity,
                                  pue_scale=cfg.common["pue_scale"], idle_allocation_rule=cfg.common["idle_allocation_rule"],
                                  parquet_batch_size=cfg.common["parquet_batch_size"], hash_inputs=cfg.common["hash_inputs"],
                                  output_dir=output, save_hourly_outputs=entry["save_hourly_outputs"])
    del result
    return output


def _boundary(context, output):
    from scripts.analyze_boundary_calibration import analyze_boundary_calibration, write_analysis
    cfg, common = context.config, context.config.common
    result = analyze_boundary_calibration(
        countries=common["countries"], year_start=common["year_start"], years=common["years"],
        policy=cfg.experiments["boundary"]["policy"], workload_profile_path=cfg.inputs["workload_profile_path"],
        server_profile_path=cfg.inputs["server_profile_path"], hourly_carbon_factors_dir=cfg.inputs["hourly_carbon_factors_dir"],
        hourly_carbon_scope=common["hourly_carbon_scope"], strict_hourly_carbon=common["strict_hourly_carbon"],
        max_intervals=common.get("max_intervals"), workload_profile=context.profile,
        verbose=common["verbose"], experiment_context=context)
    return write_analysis(result, output)


def _validation(context, output):
    from scripts.validate_energy_model import audit_national_scope, diagnose_trace_boundary, validate_mlperf
    cfg, entry = context.config, context.config.experiments["validation"]
    frame, digest = context.csv(cfg.inputs["mlperf"])
    validate_mlperf(cfg.inputs["mlperf"], output, entry["bootstrap_replicates"], entry["seed"],
                    hardware_config=cfg.hardware, input_frame=frame, input_sha256=digest)
    frame, _ = context.csv(cfg.inputs["national"])
    audit_national_scope(cfg.inputs["national"], output, input_frame=frame)
    if entry["trace_diagnostic"]:
        _, diagnostics = diagnose_trace_boundary(cfg.inputs["workload_profile_path"], output,
                                                  entry["trace_window_hours"], hour_aggregates=context.validation_hours)
        diagnostics["trace_truncated_by_max_intervals"] = cfg.common.get("max_intervals") is not None
        diagnostics["source"] = "first_pass_validation_filter_aggregates"
        diagnostics.to_csv(output / "trace_boundary_diagnostics.csv", index=False)
    return output


def run_experiments(config_path=ROOT_DIR / "configs" / "experiments.json", *, only=None, output_dir=None,
                    single_name=None, invocation=None):
    cfg = load_experiment_config(config_path, only=only, output_dir=output_dir)
    if single_name is not None and output_dir is None:
        cfg.output_dir = cfg.output_dir.with_name(f"{cfg.output_dir.name}_{single_name}")
    if not cfg.output_dir.is_relative_to(ROOT_DIR):
        raise ValueError("Experiment outputs must be inside the repository workspace.")
    if cfg.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {cfg.output_dir}")
    requested = {"config_file": cfg.path, "config_sha256": cfg.sha256, "document": cfg.document,
                 "selected_experiments": cfg.selected}
    record = RunProvenance(ROOT_DIR, cfg.output_dir, save_outputs=True, hash_inputs=cfg.common["hash_inputs"],
                           requested=requested, invocation=invocation)
    record.output_roots = [cfg.output_dir / name for name in ("data", "comparison", "boundary", "validation")]
    statuses, outputs = [], {}
    with record:
        row = record.add_input(cfg.path, "experiment_configuration", hash_content=False, usage="loaded")
        row.update(sha256=cfg.sha256, hash_status="sha256_of_loaded_configuration")
        if cfg.inputs["ai_capacity_factors_file"]:
            record.add_input(cfg.inputs["ai_capacity_factors_file"], "capacity_factor_configuration", hash_content=True, usage="loaded")
        record.configure({"inputs": cfg.inputs, "common": cfg.common, "hardware": asdict(cfg.hardware),
                          "trace": asdict(cfg.trace), "annual_activity": asdict(cfg.activity),
                          "ai_capacity_factors": cfg.factors, "experiments": cfg.experiments,
                          "selected": cfg.selected,
                          "output_directory": cfg.output_dir})
        (record.directory / "resolved_experiment_config.json").write_text(
            json.dumps(json_value(record.record["resolved_configuration"]), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        context = ExperimentContext(cfg, record)
        try:
            context.prepare()
        finally:
            record.record["shared_data_io"] = context.io_stats()
            record.table("Shared_Data_IO", pd.DataFrame([context.io_stats()]))
            record.table("Shared_Grid_Source_Selection", pd.DataFrame(context.grid.source_rows))
        for name in cfg.selected:
            record.stage(f"experiment_{name}")
            started = time.perf_counter()
            status = {"experiment": name, "status": "running", "output": None, "elapsed_seconds": None}
            statuses.append(status)
            try:
                if name.startswith("figure"):
                    path = cfg.output_dir / "data" / f"{name}_data.xlsx"
                    outputs[name] = import_module(f"scripts.experiment_{name}").run(context, path)
                else:
                    outputs[name] = {"comparison": _comparison, "boundary": _boundary, "validation": _validation}[name](context, cfg.output_dir / name)
                status.update(status="completed", output=str(outputs[name]))
            except BaseException as exc:
                status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                status["elapsed_seconds"] = time.perf_counter() - started
                record.table("Experiment_Status", pd.DataFrame(statuses))
                record.record["shared_data_io"] = context.io_stats()
                record.table("Shared_Data_IO", pd.DataFrame([context.io_stats()]))
                record._flush_metadata()
        record.stage("experiment_outputs_complete")
    print(f"Experiment outputs: {cfg.output_dir}\nRun record: {record.directory}")
    return {"outputs": outputs, "run_metadata": record.record, "provenance_directory": record.directory}


def main(single_name=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "configs" / "experiments.json")
    if single_name is None:
        parser.add_argument("--only", nargs="+", choices=EXPERIMENT_NAMES)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_experiments(args.config, only=[single_name] if single_name else args.only,
                    output_dir=args.output_dir, single_name=single_name,
                    invocation={"kind": "cli", "executable": sys.executable, "argv": list(sys.argv)})


if __name__ == "__main__":
    main()
