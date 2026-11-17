"""Inventory revision inputs without importing or running the research models.

Run from the repository root:
    python -B scripts/audit_revision_inputs.py --check-only
    python -B scripts/audit_revision_inputs.py --output-dir results/revision_audit/inputs

Only the Python standard library is required. The default report records input
file sizes and modification times, and hashes source/configuration files. Use
--hash-inputs to also read and hash external CSV/Parquet inputs (potentially slow).
No raw data, model settings, or existing result files are modified. An existing
report is never overwritten. This is an inventory of local defaults, not an
executed model configuration, data-quality validation, or provenance verification.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Mapping


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.research_io import git_snapshot, sha256_file as _sha256
CAPACITY_SOURCE = "dataset/Installed_capacity_data.py"
FACTOR_SOURCE = "dataset/Factors.py"
HOURLY_SOURCE = "core/hourly_workload_resolved_model.py"
ANNUAL_SOURCE = "core/annual_cpu_model.py"
LEGACY_SOURCE = "core/past_research_data_center_energy_carbon_model.py"
REPORT_NAMES = (
    "file_manifest.csv", "model_inputs.csv", "audit_findings.csv",
    "source_register.csv", "audit_metadata.json",
)
MANIFEST_COLUMNS = (
    "path", "category", "exists", "git_tracked", "size_bytes", "modified_utc",
    "sha256", "hash_status",
)
INPUT_COLUMNS = (
    "category", "parameter", "country", "scenario", "policy", "year", "value",
    "unit", "role", "source_file", "source_line",
)
FINDING_COLUMNS = ("severity", "check", "subject", "observed", "expected", "detail")
SOURCE_COLUMNS = ("input_group", "local_evidence", "documented_claim", "status", "needed")


def _workspace_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT_DIR):
        raise ValueError(f"Path resolves outside the workspace: {path}")
    return resolved


def _read_tree(relative: str) -> ast.Module:
    path = _workspace_path(ROOT_DIR / relative)
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)


def _literal(node: ast.AST, values: Mapping[str, Any]) -> Any:
    """Read literal constants and array wrappers; never execute source code."""
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.Dict):
        return {_literal(k, values): _literal(v, values) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_literal(item, values) for item in node.elts]
        return tuple(items) if isinstance(node, ast.Tuple) else items
    if isinstance(node, ast.Call):
        name = ast.unparse(node.func)
        if name in {"array", "np.array"} and len(node.args) == 1 and not node.keywords:
            return _literal(node.args[0], values)
        raise ValueError(f"Unsupported constant expression: {ast.unparse(node)}")
    return ast.literal_eval(node)


def _constants(tree: ast.Module) -> tuple[dict[str, Any], dict[str, int]]:
    values: dict[str, Any] = {}
    lines: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name) or value is None:
            continue
        try:
            values[target.id] = _literal(value, values)
            lines[target.id] = node.lineno
        except (ValueError, TypeError, KeyError):
            continue
    return values, lines


def _named_node(tree: ast.Module, name: str, kind: type) -> Any:
    return next(node for node in tree.body if isinstance(node, kind) and node.name == name)


def _walk_files(directory: Path, suffix: str, findings: list[dict]) -> list[Path]:
    _workspace_path(directory)
    if not directory.exists():
        return []
    result = []

    def on_error(exc: OSError) -> None:
        findings.append(dict(severity="ERROR", check="directory_read", subject=str(exc.filename),
                             detail=str(exc)))

    for current, directories, filenames in os.walk(directory, followlinks=False, onerror=on_error):
        # Junctions/symlinks are not input roots; do not silently traverse another checkout.
        for name in directories[:]:
            candidate = Path(current) / name
            if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
                directories.remove(name)
                findings.append(dict(severity="WARN", check="linked_directory_skipped",
                                     subject=candidate.relative_to(ROOT_DIR).as_posix(),
                                     detail="Linked input directories require separate review."))
        for name in filenames:
            candidate = Path(current) / name
            if candidate.suffix.lower() == suffix:
                result.append(candidate)
    return sorted(result)


def build_audit(hash_inputs: bool = False) -> dict[str, Any]:
    """Build report tables in memory. This function creates no files."""
    findings: list[dict] = []
    inputs: list[dict] = []
    manifest: list[dict] = []
    git = git_snapshot(ROOT_DIR)

    def finding(severity: str, check: str, subject: str, observed: Any = "",
                expected: Any = "", detail: str = "") -> None:
        findings.append(dict(severity=severity, check=check, subject=subject,
                             observed=observed, expected=expected, detail=detail))

    def parameter(category: str, name: str, value: Any, unit: str, role: str,
                  source: str, line: int, **dimensions: Any) -> None:
        inputs.append(dict(category=category, parameter=name, value=value, unit=unit,
                           role=role, source_file=source, source_line=line, **dimensions))

    def record_file(relative: str, category: str, hash_file: bool) -> None:
        path = ROOT_DIR / relative
        row = dict(path=relative, category=category, exists=False,
                   git_tracked=(relative in git["tracked_paths"] if git["available"] else "unknown"))
        try:
            _workspace_path(path)
            if path.is_file():
                info = path.stat()
                row.update(exists=True, size_bytes=info.st_size,
                           modified_utc=datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
                           sha256=_sha256(path) if hash_file else "",
                           hash_status="computed" if hash_file else "not_requested")
            else:
                finding("ERROR", "required_file", relative, False, True)
        except (OSError, ValueError) as exc:
            finding("ERROR", "file_access", relative, detail=str(exc))
            row["hash_status"] = "unavailable"
        manifest.append(row)

    source_files = {"run.py", "README.md", "README.zh-CN.md", ".gitignore",
                    CAPACITY_SOURCE, FACTOR_SOURCE, ANNUAL_SOURCE, HOURLY_SOURCE,
                    LEGACY_SOURCE, "core/task_model.py", "审稿意见汇总与修改计划.md"}
    for directory in ("core", "scripts"):
        source_files.update(path.relative_to(ROOT_DIR).as_posix()
                            for path in _walk_files(ROOT_DIR / directory, ".py", findings))
    for directory, suffix in (("configs", ".json"), ("configs", ".csv"), ("tests", ".py")):
        source_files.update(path.relative_to(ROOT_DIR).as_posix()
                            for path in _walk_files(ROOT_DIR / directory, suffix, findings))
    for relative in sorted(source_files):
        record_file(relative, "source_or_documentation", True)
    for relative in (ANNUAL_SOURCE, HOURLY_SOURCE):
        if git["available"] and relative not in git["tracked_paths"]:
            finding("WARN", "untracked_endpoint", relative, "local_only", "tracked_for_delivery",
                    "Local presence does not establish submission or remote availability.")
    if not git["available"]:
        finding("WARN", "git_unavailable", "repository", detail=git["error"])

    trees = {name: _read_tree(name) for name in
             (CAPACITY_SOURCE, FACTOR_SOURCE, HOURLY_SOURCE, ANNUAL_SOURCE, LEGACY_SOURCE)}
    capacity, capacity_lines = _constants(trees[CAPACITY_SOURCE])
    factors, factor_lines = _constants(trees[FACTOR_SOURCE])
    hourly, hourly_lines = _constants(trees[HOURLY_SOURCE])
    annual, _ = _constants(trees[ANNUAL_SOURCE])
    legacy, _ = _constants(trees[LEGACY_SOURCE])
    countries = capacity["DEFAULT_COUNTRIES"]
    scenarios = capacity["SCENARIO_COLUMNS"]
    start_year = hourly["DATA_YEAR_START"]
    years = list(range(start_year, start_year + len(capacity["IT_CAPACITY"])))
    policies = factors["CARBON_FACTORS"]
    numeric = lambda value: isinstance(value, (int, float)) and math.isfinite(value)
    for source, alias in ((HOURLY_SOURCE, "SCENARIO_COL_MAP"), (LEGACY_SOURCE, "SCENARIO_COLUMN")):
        shared = any(name.name == "SCENARIO_COLUMNS" and name.asname == alias
                     for node in trees[source].body if isinstance(node, ast.ImportFrom)
                     and node.module == "dataset.Installed_capacity_data" for name in node.names)
        finding("PASS" if shared else "ERROR", "scenario_mapping", source,
                "shared capacity scenario mapping" if shared else "shared mapping import missing", scenarios)
    for scenario, column in scenarios.items():
        parameter("mapping", "scenario_column", column, "index", "array_mapping",
                  CAPACITY_SOURCE, capacity_lines["SCENARIO_COLUMNS"], scenario=scenario)
    if annual["DATA_YEAR_START"] != start_year or legacy["DATA_YEAR_START"] != start_year:
        finding("ERROR", "year_origin", "endpoint models", detail="Data year origins disagree.")
    if annual["DATA_YEAR_END"] != years[-1]:
        finding("ERROR", "year_end", ANNUAL_SOURCE, annual["DATA_YEAR_END"], years[-1])

    for name in ("TOTAL_RATIO", "IT_RATIO", "NON_IT_RATIO"):
        shares = capacity[name]
        missing, extra = sorted(set(countries) - set(shares)), sorted(set(shares) - set(countries))
        if missing or extra:
            finding("WARN" if name == "NON_IT_RATIO" else "ERROR", "country_keys", name,
                    f"missing={missing}; extra={extra}", "DEFAULT_COUNTRIES",
                    "No keys or shares have been corrected by this audit.")
        for country, value in shares.items():
            parameter("country_share", name, value, "fraction", "fixed_allocation_input",
                      CAPACITY_SOURCE, capacity_lines[name], country=country)
            if not numeric(value) or not 0 <= value <= 1:
                finding("ERROR", "share_range", f"{name}/{country}", value, "[0, 1]")
        if all(numeric(value) for value in shares.values()):
            total = math.fsum(shares.get(country, 0) for country in countries)
            parameter("coverage", f"{name}_selected_country_sum", total, "fraction",
                      "derived_capacity_coverage", CAPACITY_SOURCE, capacity_lines[name])
            if not math.isclose(total, 1.0, abs_tol=1e-8):
                finding("WARN", "capacity_coverage", name, total, "scope_decision_needed",
                        "Capacity coverage is not an independently measured electricity coverage fraction.")

    valid_matrices = True
    for name in ("TOTAL_CAPACITY", "IT_CAPACITY", "NON_IT_CAPACITY"):
        matrix = capacity[name]
        if len(matrix) != len(years) or any(len(row) != len(scenarios) for row in matrix):
            finding("ERROR", "capacity_shape", name, "invalid", f"{len(years)} x {len(scenarios)}")
            valid_matrices = False
            continue
        for index, year in enumerate(years):
            for scenario, column in scenarios.items():
                value = matrix[index][column]
                parameter("capacity", name, value, "GW", "scenario_input",
                          CAPACITY_SOURCE, capacity_lines[name], year=year, scenario=scenario)
                if not numeric(value) or value < 0:
                    finding("ERROR", "capacity_range", f"{name}/{year}/{scenario}", value, ">= 0")
                    valid_matrices = False
    if valid_matrices:
        residual = max(abs(total - it - non_it)
                       for total_row, it_row, non_it_row in zip(
                           capacity["TOTAL_CAPACITY"], capacity["IT_CAPACITY"], capacity["NON_IT_CAPACITY"])
                       for total, it, non_it in zip(total_row, it_row, non_it_row))
        finding("PASS" if residual <= 0.00010001 else "ERROR", "capacity_closure", "total=IT+non-IT",
                residual, "max absolute residual <= 0.00010001 GW", "Tolerance allows input table rounding.")

    for year, value in capacity["DEFAULT_AI_CAPACITY_FACTORS"].items():
        parameter("calibration", "ai_capacity_factor", value, "fraction", "calibration_input",
                  CAPACITY_SOURCE, capacity_lines["DEFAULT_AI_CAPACITY_FACTORS"], year=year)
        if not numeric(value) or not 0 < value <= 1:
            finding("ERROR", "ai_factor_range", str(year), value, "(0, 1]")
        else:
            parameter("calibration", "inverse_ai_capacity_factor", 1 / value, "ratio",
                      "derived_reciprocal_not_applied_factor", CAPACITY_SOURCE,
                      capacity_lines["DEFAULT_AI_CAPACITY_FACTORS"], year=year)
    missing_factors = sorted(set(years) - set(capacity["DEFAULT_AI_CAPACITY_FACTORS"]))
    if missing_factors:
        finding("ERROR", "ai_factor_years", "capacity factors", missing_factors, years)

    pue = factors["PUE"]
    for country in sorted(set(countries) | set(pue)):
        matrix = pue.get(country, [])
        if country not in countries or len(matrix) != len(years) or any(len(row) != len(scenarios) for row in matrix):
            finding("ERROR", "pue_coverage", country, "invalid country or matrix shape",
                    f"{len(years)} x {len(scenarios)}")
            continue
        for index, year in enumerate(years):
            for scenario, column in scenarios.items():
                value = matrix[index][column]
                parameter("pue", "PUE", value, "ratio", "local_numeric_input_provenance_pending",
                          FACTOR_SOURCE, factor_lines["PUE"], country=country, year=year, scenario=scenario)
                if not numeric(value) or value < 1:
                    finding("ERROR", "pue_range", f"{country}/{year}/{scenario}", value, ">= 1")

    carbon_start, carbon_end = factors["CARBON_FACTOR_YEAR_START"], factors["CARBON_FACTOR_YEAR_END"]
    for policy, country_values in policies.items():
        if set(country_values) != set(countries):
            finding("ERROR", "carbon_country_keys", policy, sorted(country_values), sorted(countries))
        for country, series in country_values.items():
            if len(series) != carbon_end - carbon_start + 1:
                finding("ERROR", "carbon_year_coverage", f"{policy}/{country}", len(series), carbon_end - carbon_start + 1)
            for offset, value in enumerate(series):
                parameter("carbon", "annual_carbon_intensity", value, "kgCO2/MWh", "annual_grid_input",
                          FACTOR_SOURCE, factor_lines[f"CF_{policy}"],
                          country=country, policy=policy, year=carbon_start + offset)
                if not numeric(value) or value < 0:
                    finding("ERROR", "carbon_range", f"{policy}/{country}/{carbon_start + offset}", value, ">= 0")
    for policy, endpoint in (("NDC", 0.89), ("NZ", 0.53)):
        residuals = []
        for country in countries:
            cp, other = policies["CP"].get(country, []), policies[policy].get(country, [])
            for year in years:
                index = year - carbon_start
                if index < len(cp) and index < len(other) and numeric(cp[index]) and numeric(other[index]):
                    residuals.append(abs(other[index] - cp[index] * endpoint ** ((year - 2025) / 5)))
        residual = max(residuals) if residuals else math.nan
        finding("WARN" if residual <= 0.011 else "INFO", "uniform_policy_multiplier", policy,
                residual, "max absolute CI residual <= 0.011 kgCO2/MWh",
                f"Compare local annual arrays with CP * {endpoint}**((year-2025)/5). "
                "This does not verify hourly files or establish policy realism.")

    hardware_node = _named_node(trees[HOURLY_SOURCE], "HardwarePowerConfig", ast.ClassDef)
    for node in hardware_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            name, value = node.target.id, _literal(node.value, {})
            unit = ("W/core" if name.endswith("w_per_core") else "W/GB" if name.endswith("w_per_gb")
                    else "W/TB" if name.endswith("w_per_tb") else "W" if name.endswith("_w") else "fraction")
            parameter("hardware", name, value, unit, "hardware_configuration_default",
                      HOURLY_SOURCE, node.lineno)
    for source, function in ((ANNUAL_SOURCE, "run_annual_cpu_model"),
                             (LEGACY_SOURCE, "calculate_past_research_energy_carbon")):
        node = _named_node(trees[source], function, ast.FunctionDef)
        for argument, default in zip(node.args.args[-len(node.args.defaults):], node.args.defaults):
            try:
                value = _literal(default, {})
            except (ValueError, TypeError):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                unit = "year" if argument.arg == "year_start" else "years" if argument.arg == "years" else "ratio"
                parameter("annual_default", f"{function}.{argument.arg}", value, unit,
                          "function_default_not_executed_configuration", source, default.lineno)
    activity_assignments = [node for node in ast.walk(trees[LEGACY_SOURCE]) if isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id in
                                    {"training_activity", "inference_activity"} for target in node.targets)]
    for node in activity_assignments:
        name = node.targets[0].id
        parameter("annual_activity", name, ast.unparse(node.value), "expression",
                  "implemented_equation", LEGACY_SOURCE, node.lineno)
        if any(isinstance(part, ast.BinOp) and isinstance(part.op, ast.Div)
               and isinstance(part.right, ast.Name) and part.right.id == "years"
               for part in ast.walk(node.value)):
            finding("WARN", "activity_horizon_dependency", name, "denominator=years", "calendar_defined_activity",
                    "Activity depends on the requested native run length; endpoint/ramp choice remains pending.")

    input_groups = (
        ("dataset/EM-CPNDCNZ", ".csv", "hourly_carbon_input"),
        ("dataset/asi_opensource_pod_hourly", ".parquet", "pod_trace_input"),
        ("dataset/asi_opensource_server_hourly", ".parquet", "server_trace_input"),
    )
    for relative, suffix, category in input_groups:
        files = _walk_files(ROOT_DIR / relative, suffix, findings)
        finding("INFO" if files else "ERROR", "input_file_count", relative, len(files), "> 0",
                "File inventory only; file contents, schema, timestamps and values have not been validated.")
        for path in files:
            record_file(path.relative_to(ROOT_DIR).as_posix(), category, hash_inputs)
        if category == "hourly_carbon_input":
            for country in countries:
                directory = hourly.get("HOURLY_CARBON_COUNTRY_DIRS", {}).get(country, country)
                for policy in policies:
                    for year in years:
                        if year < 2026:
                            continue
                        matches = [path for path in files if path.parent == ROOT_DIR / relative / directory
                                   and path.name.endswith(f"-{policy}-{year}-hourly.csv")]
                        finding("PASS" if len(matches) == 1 else "ERROR", "hourly_file_coverage",
                                f"{country}/{policy}/{year}", len(matches), 1)
        elif files:
            partition_re = re.compile(hourly["ALIBABA_2026_PARTITION_RE"] if
                                      "ALIBABA_2026_PARTITION_RE" in hourly else r"day=(\d+)/hour=(\d+)")
            slots = set()
            for path in files:
                match = partition_re.search(path.relative_to(ROOT_DIR).as_posix())
                if match:
                    day, hour = int(match.group(1)), int(match.group(2))
                    if 0 <= hour < 24:
                        slots.add(day * 24 + hour)
                    else:
                        finding("ERROR", "trace_partition_hour", str(path), hour, "0..23")
                else:
                    finding("WARN", "trace_partition_unrecognized", path.relative_to(ROOT_DIR).as_posix())
            gaps = sorted(set(range(min(slots), max(slots) + 1)) - slots) if slots else []
            finding("PASS" if slots and not gaps else "ERROR", "trace_partition_continuity", relative,
                    f"unique_hours={len(slots)}; gaps={len(gaps)}", "no missing hours within observed range",
                    f"First missing hour offsets: {gaps[:24]}. Multiple files in one partition are allowed. "
                    "Continuity does not establish that the full release is present.")

    for relative in ("dataset/validation/README.md", "dataset/validation/national_data_center_energy.csv",
                     "dataset/validation/raw/mlperf_power_raw_data.csv"):
        record_file(relative, "validation_input_or_documentation", True)
    finding("INFO", "audit_scope", "this audit", detail=
            "No models or external services were run. Calibration closure, hourly mean closure, "
            "trace schema, task energy closure, submission version and remote availability remain unverified.")
    source_register = [
        dict(input_group=group, local_evidence=evidence, documented_claim=claim,
             status="LOCAL_DOCUMENTATION_ONLY_NOT_EXTERNALLY_VERIFIED", needed=needed)
        for group, evidence, claim, needed in (
            ("capacity_and_country_shares", CAPACITY_SOURCE, "Local matrices and fixed country shares",
             "Original capacity export, source version/base year, selection and transcription rules."),
            ("calibration", CAPACITY_SOURCE, "Source comments attribute endpoints to IEA; intermediate targets are author estimates",
             "Verify original IEA figure/table, submission-era targets and geographic energy boundary."),
            ("PUE", FACTOR_SOURCE + "; README.zh-CN.md", "Local country/year/scenario arrays exist",
             "Resolve commercial-source versus author-defined attribution; preserve raw export or construction rules."),
            ("annual_carbon", FACTOR_SOURCE, "Local CP/NDC/NZ annual arrays in kg CO2/MWh",
             "Original data/source version and reproducible projection/multiplier rules."),
            ("hourly_carbon", "README.zh-CN.md; dataset/EM-CPNDCNZ", "README describes a scaled 2025 Electricity Maps hourly template",
             "Authorized source version, license/access records, conversion code, timezone, missing-data and scope rules."),
            ("workload_trace", "README.zh-CN.md; " + HOURLY_SOURCE, "Alibaba GPU v2026 relative pod/server traces; CPU-weighted memory proxy",
             "Release version, source hashes, field definitions and preprocessing/partition selection records."),
            ("hardware", HOURLY_SOURCE, "HardwarePowerConfig supplies local power-budget and idle defaults",
             "Per-parameter references or explicit author assumptions; independent mechanism-level evidence."),
            ("MLPerf", "dataset/validation/README.md", "Local documentation records repository URL and access date 2026-08-19",
             "Verify upstream version and system boundaries; local CSV presence is not validation of future hardware."),
        )
    ]
    metadata = {
        "audit_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(ROOT_DIR), "python_version": sys.version,
        "git": {key: value for key, value in git.items() if key != "tracked_paths"},
        "scope": "local source defaults and file inventory; no model execution or external verification",
        "input_hashes_requested": hash_inputs, "countries": countries, "scenario_columns": scenarios,
        "capacity_years": years, "hourly_inventory_years": [year for year in years if year >= 2026],
        "finding_counts": dict(Counter(row["severity"] for row in findings)),
        "table_rows": {"file_manifest": len(manifest), "model_inputs": len(inputs),
                       "audit_findings": len(findings), "source_register": len(source_register)},
    }
    return dict(file_manifest=manifest, model_inputs=inputs, audit_findings=findings,
                source_register=source_register, audit_metadata=metadata)


def write_reports(report: dict[str, Any], directory: Path) -> Path:
    """Write a new report directory inside this checkout; refuse overwrites."""
    output = _workspace_path(directory if directory.is_absolute() else ROOT_DIR / directory)
    if output.exists():
        raise FileExistsError(f"Report directory already exists; choose a new --output-dir: {output}")
    output.mkdir(parents=True, exist_ok=False)
    for name, columns in (("file_manifest", MANIFEST_COLUMNS), ("model_inputs", INPUT_COLUMNS),
                          ("audit_findings", FINDING_COLUMNS), ("source_register", SOURCE_COLUMNS)):
        with (output / f"{name}.csv").open("x", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(report[name])
    with (output / "audit_metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(report["audit_metadata"], stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("results/revision_audit/inputs"))
    parser.add_argument("--check-only", action="store_true", help="Audit in memory; create no output files.")
    parser.add_argument("--hash-inputs", action="store_true", help="Also hash external CSV/Parquet files; may be slow.")
    args = parser.parse_args()
    try:
        report = build_audit(hash_inputs=args.hash_inputs)
        metadata = report["audit_metadata"]
        print("Audit covers local defaults and inventory only; research analyses have not been run.")
        print("Rows:", json.dumps(metadata["table_rows"]))
        print("Findings:", json.dumps(metadata["finding_counts"]))
        for row in report["audit_findings"]:
            if row["severity"] in {"ERROR", "WARN"}:
                print(f"[{row['severity']}] {row['check']} | {row['subject']} | {row.get('observed', '')}")
        if args.check_only:
            print("Check-only: no report files were created.")
        else:
            print("Reports written to:", write_reports(report, args.output_dir))
        return 1 if metadata["finding_counts"].get("ERROR", 0) else 0
    except (OSError, ValueError, KeyError, StopIteration, SyntaxError) as exc:
        print(f"Audit could not complete: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
