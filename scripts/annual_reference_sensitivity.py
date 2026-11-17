"""Run annual-reference controls against an existing hourly country/year CSV.

No trace files or hourly grids are reread and no hourly model is rerun.
    python -B scripts/annual_reference_sensitivity.py --hourly-reference <CSV> --output-dir results/reference_controls
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.annual_reference_controls import annual_reference_controls
from core.experiment_config import load_experiment_config
from core.research_io import sha256_file


def workspace_path(path):
    path = Path(path)
    resolved = (path if path.is_absolute() else ROOT_DIR / path).resolve()
    if not resolved.is_relative_to(ROOT_DIR):
        raise ValueError("Reference-control inputs and outputs must be inside the repository workspace.")
    return resolved


def run_reference_sensitivity(reference_path, output_dir, config_path=ROOT_DIR / "configs/experiments.json"):
    """Calculate in memory first, then write to a new directory only."""
    reference_path, output_dir, config_path = map(workspace_path, (reference_path, output_dir, config_path))
    if output_dir.exists():
        raise FileExistsError(f"Choose a new reference-control output directory: {output_dir}")
    cfg = load_experiment_config(config_path, only=["figure3"])
    payload = reference_path.read_bytes()
    reference = pd.read_csv(io.BytesIO(payload))
    policy_source = "input_column" if "policy" in reference else "explicit_Figure_3_CP_scope"
    if "policy" not in reference:
        # Native hourly exports store policy in their filename, not a column.
        # This entry point is explicitly the Base-CP Figure 3 comparison.
        reference["policy"] = "CP"
    if "model" in reference and not reference["model"].eq("hourly_workload_resolved_model").all():
        raise ValueError("The input must contain only the hourly workload-resolved country/year endpoint.")
    tables = annual_reference_controls(
        reference, countries=cfg.common["countries"], year_start=cfg.common["year_start"], years=cfg.common["years"],
        activity_config=cfg.activity, ai_capacity_factors=cfg.factors, pue_scale=cfg.common["pue_scale"],
        config=cfg.experiments["figure3"]["reference_sensitivity"])
    if not tables:
        raise ValueError("Annual reference sensitivity is disabled in the configuration.")
    code_files = ("core/annual_reference_controls.py", "core/annual_activity.py", "core/annual_cpu_model.py",
                  "core/past_research_data_center_energy_carbon_model.py", "core/task_model.py",
                  "dataset/Installed_capacity_data.py", "dataset/Factors.py",
                  "scripts/annual_reference_sensitivity.py", "core/experiment_config.py")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_file": reference_path.relative_to(ROOT_DIR).as_posix(),
        "reference_sha256": hashlib.sha256(payload).hexdigest(),
        "reference_policy_source": policy_source,
        "reference_provenance_status": "existing_model_endpoint_not_independently_validated_measurement",
        "config_file": config_path.relative_to(ROOT_DIR).as_posix(), "config_sha256": cfg.sha256,
        "activity": asdict(cfg.activity), "reference_sensitivity": cfg.experiments["figure3"]["reference_sensitivity"],
        "countries": cfg.common["countries"], "pue_scale": cfg.common["pue_scale"],
        "ai_capacity_factors": cfg.factors, "python_version": sys.version,
        "source_sha256": {name: sha256_file(ROOT_DIR / name) for name in code_files},
        "trace_reads": 0, "hourly_model_runs": 0,
        "annual_model_runs": len(tables["W04_Reference_Cases"]),
        "status": "completed", "tables": {name: len(frame) for name, frame in tables.items()},
    }
    factor_file = cfg.inputs.get("ai_capacity_factors_file")
    if factor_file is not None:
        factor_file = workspace_path(factor_file)
        metadata["capacity_factor_file_sha256"] = sha256_file(factor_file)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig", mode="x")
    with (output_dir / "run_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "configs/experiments.json")
    parser.add_argument("--hourly-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_reference_sensitivity(args.hourly_reference, args.output_dir, args.config)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Reference controls failed: {exc}\n")
    print(f"Reference controls: {result}\nNo trace reads or hourly model runs.")


if __name__ == "__main__":
    main()
