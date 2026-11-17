"""Audit policy identities, UTC coverage, means and fixed-energy CI controls.

Returns a nonzero status for missing/invalid files, truncated leap calendars,
direct annual-mean mismatch or failed scalar identities. Auditing writes only
to a new report directory; it never repairs or overwrites the source inputs.
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

from core.accounting_boundary import resolve_countries
from core.grid_carbon import (
    MEAN_ATOL, calendar_hours, calendar_status, find_hourly_file, matching_diagnostics,
    read_hourly_file, transform_shape,
)
from core.research_io import sha256_file
from dataset.Factors import annual_policy_audit, get_carbon_factor
from scripts.build_hourly_grid_scenarios import new_output_directory


def audit_grid_scenarios(directory, *, countries=None, years=range(2026, 2031), scope="direct"):
    countries, years = resolve_countries(countries), tuple(years)
    if not years or len(set(years)) != len(years) or scope not in ("direct", "life_cycle"):
        raise ValueError("Supply unique years and a valid scope.")
    annual = pd.DataFrame(annual_policy_audit(countries, years))
    coverage, identities, stresses, loaded = [], [], [], {}
    for country in countries:
        for year in years:
            for policy in ("CP", "NDC", "NZ"):
                record = {"country": country, "year": year, "policy": policy, "scope": scope,
                          "expected_calendar_hours": len(calendar_hours(year)),
                          "annual_fallback_available": scope == "direct",
                          "annual_target_ci_kg_per_mwh": get_carbon_factor(policy, country, year) if scope == "direct" else np.nan,
                          "annual_mean_comparison_available": scope == "direct",
                          "status": "ERROR", "error": "", "input_file": "", "input_sha256": ""}
                try:
                    path = find_hourly_file(directory, country, policy, year)
                    if path is None:
                        raise FileNotFoundError("missing_hourly_file")
                    record.update(input_file=str(path.resolve()), input_sha256=sha256_file(path))
                    timestamps, values = read_hourly_file(path, year, scope, allow_legacy=True)
                    status = calendar_status(timestamps, year)
                    mean = float(values.mean())
                    mean_ok = scope != "direct" or bool(np.isclose(mean, record["annual_target_ci_kg_per_mwh"], rtol=1e-9, atol=MEAN_ATOL))
                    record.update(hours=len(timestamps), calendar_status=status,
                                  hourly_mean_ci_kg_per_mwh=mean,
                                  annual_mean_residual_kg_per_mwh=mean - record["annual_target_ci_kg_per_mwh"],
                                  annual_mean_closed=mean_ok if scope == "direct" else None,
                                  status="PASS" if status == "full_calendar" and mean_ok else "ERROR")
                    if status != "full_calendar":
                        record["error"] = "legacy_calendar_truncates_24_hours"
                    if not mean_ok:
                        record["error"] += "; direct_mean_mismatch"
                    loaded[(country, year, policy)] = (timestamps, values)
                except (ValueError, OSError) as exc:
                    record["error"] = str(exc)
                coverage.append(record)
            cp = loaded.get((country, year, "CP"))
            for policy in ("NDC", "NZ"):
                other = loaded.get((country, year, policy))
                ratio = get_carbon_factor(policy, country, year) / get_carbon_factor("CP", country, year)
                row = {"country": country, "year": year, "policy": policy, "scope": scope,
                       "expected_actual_annual_ratio": ratio, "status": "UNAVAILABLE",
                       "interpretation": "same_load_implies_this_emission_ratio_only_if_hourly_scalar_identity_holds"}
                if cp is not None and other is not None and cp[0].equals(other[0]):
                    residual = float(np.max(np.abs(other[1] - cp[1] * ratio)))
                    row.update(max_hourly_scalar_residual_kg_per_mwh=residual,
                               status="PASS" if np.allclose(other[1], cp[1] * ratio, rtol=1e-9, atol=MEAN_ATOL) else "ERROR")
                identities.append(row)
            if cp is not None:
                # A disclosed synthetic positive load tests the identity only;
                # actual workload covariance is exported by the hourly model.
                h = np.arange(len(cp[1]))
                energy = 1 + 0.35 * np.cos(2 * np.pi * h / 24)
                for name, scale, phase in (("reference", 1., 0), ("flat", 0., 0),
                                           ("variability_x2", 2., 0), ("phase_12h", 1., 12)):
                    transformed = transform_shape(cp[1], variability_scale=scale, phase_hours=phase)
                    diagnostic = matching_diagnostics(energy, transformed, float(energy.sum()))
                    mean_residual = float(transformed.mean() - cp[1].mean())
                    passed = abs(mean_residual) < MEAN_ATOL and abs(diagnostic["covariance_identity_residual_tco2"]) < 1e-8
                    if name == "flat":
                        passed = passed and abs(diagnostic["hourly_matching_effect_tco2"]) < 1e-8
                    stresses.append({"country": country, "year": year, "scope": scope, "case": name,
                                     "load_kind": "synthetic_identity_control_not_research_result",
                                     "variability_scale": scale, "phase_hours": phase,
                                     "mean_preservation_residual_kg_per_mwh": mean_residual,
                                     "status": "PASS" if passed else "ERROR", **diagnostic})
            # Only this country/year pair is needed in memory.
            for policy in ("CP", "NDC", "NZ"):
                loaded.pop((country, year, policy), None)
    tables = {"input_coverage": pd.DataFrame(coverage), "annual_policy_identity": annual,
              "hourly_policy_identity": pd.DataFrame(identities), "shape_stress_controls": pd.DataFrame(stresses)}
    failures = sum(int((tables[name]["status"] != "PASS").sum()) for name in
                   ("input_coverage", "hourly_policy_identity") if not tables[name].empty)
    failures += int((~annual["within_rounding_tolerance"]).sum())
    if not tables["shape_stress_controls"].empty:
        failures += int((tables["shape_stress_controls"]["status"] != "PASS").sum())
    return tables, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly-carbon-factors-dir", type=Path, default=ROOT_DIR / "dataset/EM-CPNDCNZ")
    parser.add_argument("--countries", nargs="+")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2026, 2031)))
    parser.add_argument("--scope", choices=("direct", "life_cycle"), default="direct")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only == (args.output_dir is not None):
        parser.error("Choose exactly one of --check-only or --output-dir.")
    output = new_output_directory(args.output_dir) if args.output_dir else None
    tables, failures = audit_grid_scenarios(args.hourly_carbon_factors_dir, countries=args.countries,
                                          years=args.years, scope=args.scope)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "input_directory": str(args.hourly_carbon_factors_dir.resolve()), "scope": args.scope,
               "rows": {name: len(table) for name, table in tables.items()}, "failed_checks": failures,
               "full_calendar_inputs": int((tables["input_coverage"].get("calendar_status", pd.Series(dtype=str)) == "full_calendar").sum()),
               "annual_inputs_sha256": sha256_file(ROOT_DIR / "dataset/Factors.py"),
               "audit_script_sha256": sha256_file(Path(__file__)),
               "grid_helpers_sha256": sha256_file(ROOT_DIR / "core/grid_carbon.py"),
               "note": "Static input audit; fallback availability is not evidence that a model used fallback. Stress loads are synthetic."}
    if output:
        output.mkdir(parents=True)
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=False)
        (output / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
