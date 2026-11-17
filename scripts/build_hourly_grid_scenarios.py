"""Construct reproducible, full-calendar stylized grid scenarios from a manifest.

Required CSV fields: country,path,source_year,source_kind,zone_id.
source_kind is raw_hourly or reconstructed_shape. Paths are relative to the
manifest (or absolute). No source is fetched and no existing output is replaced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.grid_carbon import (
    CI_COLUMNS, COUNTRY_DIRS, SCOPES, map_template_to_year, read_hourly_file,
    transform_shape,
)
from core.research_io import sha256_file
from dataset.Factors import GRID_SCENARIO_KIND, get_carbon_factor


def new_output_directory(path):
    output = Path(path).resolve()
    if not output.is_relative_to(ROOT_DIR):
        raise ValueError("Output must be inside the workspace.")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    return output


def build_scenarios(source_manifest, output_dir, *, years=range(2026, 2031),
                    policies=("CP", "NDC", "NZ"), scopes=("direct",),
                    variability_scale=1.0, phase_hours=0):
    manifest_path = Path(source_manifest).resolve()
    output = new_output_directory(output_dir)
    years, policies, scopes = tuple(years), tuple(policies), tuple(scopes)
    if not years or not policies or not scopes or not set(scopes).issubset(SCOPES):
        raise ValueError("Specify nonempty years, policies and valid scopes.")
    if any(len(set(items)) != len(items) for items in (years, policies, scopes)):
        raise ValueError("Years, policies and scopes must be unique.")
    if not np.isfinite(variability_scale) or variability_scale < 0 or not isinstance(phase_hours, int):
        raise ValueError("Use a finite non-negative variability scale and an integer phase shift.")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    required = {"country", "path", "source_year", "source_kind", "zone_id"}
    if not required.issubset(manifest.columns) or manifest.empty:
        raise ValueError(f"Source manifest must have rows and columns: {sorted(required)}")
    if manifest["country"].duplicated().any():
        raise ValueError("Provide exactly one explicitly mapped source per country.")

    templates = []
    for row in manifest.to_dict("records"):
        country = row["country"]
        if row["source_kind"] not in ("raw_hourly", "reconstructed_shape"):
            raise ValueError("source_kind must be raw_hourly or reconstructed_shape.")
        source = Path(row["path"])
        source = (source if source.is_absolute() else manifest_path.parent / source).resolve()
        source_year = int(row["source_year"])
        # Validate all sources and requested annual targets before creating outputs.
        for year in years:
            for policy in policies:
                get_carbon_factor(policy, country, year)
        frame = pd.read_csv(source)
        if not row["zone_id"] or "Zone id" not in frame or set(frame["Zone id"].astype(str)) != {str(row["zone_id"])}:
            raise ValueError(f"Zone id does not match the manifest mapping for {country}: {source}")
        timestamps, direct = read_hourly_file(source, source_year, "direct")
        values = {"direct": direct}
        if "life_cycle" in scopes:
            _, values["life_cycle"] = read_hourly_file(source, source_year, "life_cycle")
        if direct.mean() <= 0:
            raise ValueError(f"A positive direct template mean is required for scaling: {source}")
        estimated = None
        if "Data estimated" in frame:
            flags = frame["Data estimated"].astype(str).str.lower()
            if not flags.isin(("true", "false", "1", "0")).all():
                raise ValueError(f"Unknown Data estimated values in {source}")
            estimated = float(flags.isin(("true", "1")).mean())
        templates.append((row, source, timestamps, values, estimated, sha256_file(source)))

    output.mkdir(parents=True)
    records = []
    for row, source, timestamps, values, estimated, digest in templates:
        country = row["country"]
        folder = output / COUNTRY_DIRS.get(country, country)
        folder.mkdir()
        for year in years:
            mapped = {scope: map_template_to_year(timestamps, values[scope], year) for scope in scopes}
            for policy in policies:
                target_direct = get_carbon_factor(policy, country, year)
                target_index = mapped[scopes[0]][0]
                frame = pd.DataFrame({"Datetime (UTC)": target_index.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                      "Country": country, "Zone id": row["zone_id"]})
                scope_records = []
                for scope in scopes:
                    ratio = float(values[scope].mean() / values["direct"].mean())
                    target = target_direct * ratio
                    generated = transform_shape(mapped[scope][1], target_mean=target,
                                                variability_scale=variability_scale, phase_hours=phase_hours)
                    frame[CI_COLUMNS[scope]] = generated
                    scope_records.append({
                        "country": country, "policy": policy, "year": year, "scope": scope,
                        "source_file": str(source), "source_sha256": digest,
                        "source_year": int(row["source_year"]), "source_kind": row["source_kind"],
                        "zone_id": row["zone_id"], "source_estimated_fraction": estimated,
                        "hours": len(target_index), "target_mean_kg_per_mwh": target,
                        "actual_mean_kg_per_mwh": float(generated.mean()),
                        "mean_residual_kg_per_mwh": float(generated.mean() - target),
                        "target_basis": ("annual_direct_table" if scope == "direct" else
                                         "annual_direct_times_source_lifecycle_to_direct_mean_ratio"),
                        "template_scope_to_direct_mean_ratio": ratio,
                        "variability_scale": variability_scale, "phase_hours": phase_hours,
                    })
                frame["Data estimated"] = True
                path = folder / f"stylized-{country}-{policy}-{year}-hourly.csv"
                frame.to_csv(path, index=False)
                output_hash = sha256_file(path)
                for record in scope_records:
                    record.update(output_file=str(path.relative_to(output)), output_sha256=output_hash)
                records.extend(scope_records)
    pd.DataFrame(records).to_csv(output / "construction_audit.csv", index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "scenario_kind": GRID_SCENARIO_KIND,
        "interpretation": "conditional_scenarios_and_shape_stress_tests_not_grid_forecasts",
        "source_manifest": str(manifest_path), "source_manifest_sha256": sha256_file(manifest_path),
        "annual_inputs_sha256": sha256_file(ROOT_DIR / "dataset/Factors.py"),
        "generator_sha256": sha256_file(Path(__file__)),
        "grid_helpers_sha256": sha256_file(ROOT_DIR / "core/grid_carbon.py"),
        "timezone": "UTC", "units": "kg CO2eq/MWh = g CO2eq/kWh",
        "calendar_mapping": "UTC month/day/hour; absent February 29 copies February 28; normalize after mapping",
        "missing_data_rule": "reject missing, duplicate, invalid, negative or incomplete hours",
        "estimated_data_rule": "retain source estimates; record fraction when supplied; all future hours are synthetic",
        "scope_rule": "direct annual targets; life-cycle targets derived only from the explicitly supplied source ratio",
        "years": years, "policies": policies, "scopes": scopes,
        "variability_scale": variability_scale, "phase_hours": phase_hours,
        "source_kinds": sorted(set(manifest["source_kind"])), "files_written": len(records) // len(scopes),
    }
    (output / "construction_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2026, 2031)))
    parser.add_argument("--policies", choices=("CP", "NDC", "NZ"), nargs="+", default=["CP", "NDC", "NZ"])
    parser.add_argument("--scopes", choices=SCOPES, nargs="+", default=["direct"])
    parser.add_argument("--variability-scale", type=float, default=1.0)
    parser.add_argument("--phase-hours", type=int, default=0)
    args = parser.parse_args()
    print(build_scenarios(args.source_manifest, args.output_dir, years=args.years, policies=args.policies,
                         scopes=args.scopes, variability_scale=args.variability_scale, phase_hours=args.phase_hours))


if __name__ == "__main__":
    main()
