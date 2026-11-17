"""W09 execution records; no model execution or filesystem writes on import.

Each saved run owns a new provenance directory. Model result directories retain
their existing layout; output manifests distinguish changed files from stale
files that existed before this run. No external input contents are copied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from uuid import uuid4

import pandas as pd

from core.research_io import git_snapshot, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item"):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _stat(path: Path) -> dict:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}


def _files(directory: Path):
    return (path for path in directory.rglob("*") if path.is_file())


def runtime_environment() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "openpyxl", "matplotlib"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "packages": packages}


class RunProvenance:
    def __init__(self, root: Path, output_dir: Path, *, save_outputs: bool,
                 hash_inputs: bool, requested: dict, trace_read_log=None, invocation=None):
        self.root, self.output_dir = root.resolve(), output_dir.resolve()
        self.save_outputs, self.hash_inputs = save_outputs, hash_inputs
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid4().hex[:8]
        self.directory = self.output_dir / "provenance" / self.run_id if save_outputs else None
        self.trace_path = Path(trace_read_log).resolve() if trace_read_log is not None else (
            self.directory / "trace_read.jsonl" if self.directory else None)
        self.trace_stream = self.event_stream = None
        self.trace_owned = False
        self.directory_owned = False
        self.outputs_snapshot_complete = False
        self.inputs, self.tables, self.outputs_before = {}, {}, {}
        self.output_roots = [self.output_dir / name for name in ("annual_cpu_model", "hourly_workload_resolved_model", "summary")]
        self.started = time.perf_counter()
        self.record = {"schema_version": "W09-1", "run_id": self.run_id, "status": "running",
                       "stage": "initialization", "started_utc": utc_now(),
                       "requested_configuration": json_value(requested),
                       "invocation": json_value(invocation or {"kind": "python_api", "function": "run.run_model_comparison"}),
                       "working_directory": str(Path.cwd()), "output_directory": str(self.output_dir),
                       "save_outputs": save_outputs, "hash_parquet_inputs": hash_inputs,
                       "random_seed": None, "randomness": "deterministic_endpoints_no_sampling",
                       "native_crash_note": "running without finished_utc may mean process termination; inspect the flushed trace log",
                       "provenance_directory": str(self.directory) if self.directory else None}

    def _flush_metadata(self):
        if self.directory and self.directory_owned:
            target = self.directory / "run_metadata.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(json_value(self.record), ensure_ascii=False, indent=2,
                                            allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(target)

    def __enter__(self):
        try:
            if not self.save_outputs and self.trace_path is not None:
                raise ValueError("trace_read_log requires save_outputs=True; in-memory runs create no files.")
            if self.directory:
                self.directory.mkdir(parents=True, exist_ok=False)
                self.directory_owned = True
                self._flush_metadata()
                self.event_stream = (self.directory / "events.jsonl").open("x", encoding="utf-8")
                self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                self.trace_stream = self.trace_path.open("x", encoding="utf-8")
                self.trace_owned = True
                for root in self.output_roots:
                    for path in _files(root):
                        self.outputs_before[str(path.resolve())] = _stat(path)
                self.outputs_snapshot_complete = True
            self.record["trace_read_log"] = str(self.trace_path) if self.trace_path else None
            self.record["environment"] = runtime_environment()
            self.record["git"] = git_snapshot(self.root)
            tracked = set(self.record["git"]["tracked_paths"])
            files = [self.root / "run.py", self.root / "README.md", self.root / "README.zh-CN.md"]
            for folder in ("core", "dataset", "scripts"):
                files.extend(sorted((self.root / folder).glob("*.py")))
            for path in files:
                row = self.add_input(path, "repository_source", hash_content=True, usage="source_snapshot")
                row["git_tracked"] = path.relative_to(self.root).as_posix() in tracked if self.record["git"]["available"] else None
            self.record["untracked_source_files"] = [row["path"] for row in self.inputs.values() if row.get("git_tracked") is False]
            self.stage("configuration")
            return self
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def stage(self, name: str, **details):
        self.record["stage"] = name
        self.event("stage", name=name, **details)
        self._flush_metadata()

    def event(self, event_name: str, **details):
        if self.event_stream:
            self.event_stream.write(json.dumps(json_value({"utc": utc_now(), "event": event_name, **details}), ensure_ascii=False) + "\n")
            self.event_stream.flush()

    def configure(self, configuration: dict):
        self.record["resolved_configuration"] = json_value(configuration)
        self._flush_metadata()

    def add_input(self, path, category: str, *, hash_content: bool, usage: str) -> dict:
        path = Path(path).resolve()
        key = str(path)
        if key in self.inputs:
            self.inputs[key]["usage_status"] = usage
            return self.inputs[key]
        row = {"path": key, "category": category, "usage_status": usage, "exists": path.is_file(),
               "sha256": None, "hash_status": "not_requested"}
        self.inputs[key] = row
        if row["exists"]:
            row.update(_stat(path))
            if hash_content:
                row["sha256"] = sha256_file(path)
                row["hash_status"] = "sha256_before_use"
        else:
            row["hash_status"] = "missing"
        return row

    def trace_event(self, event: dict):
        # Flush file_start before hashing/opening the Parquet input.
        if self.trace_stream:
            self.trace_stream.write(json.dumps(json_value({"utc": utc_now(), **event}), ensure_ascii=False) + "\n")
            self.trace_stream.flush()
        path = event.get("path")
        if path is None:
            return
        name = event["event"]
        if name == "file_start":
            row = self.add_input(path, "trace_parquet", hash_content=self.hash_inputs, usage="read_started")
            self.event("trace_input", **row)
        elif name in ("file_complete", "file_error"):
            row = self.inputs.get(str(Path(path).resolve()))
            if row is not None:
                row["usage_status"] = "read_complete" if name == "file_complete" else "read_error"
                row["rows_read"] = event.get("rows_read")
                row["batches_read"] = event.get("batches_read")

    def table(self, name: str, frame: pd.DataFrame):
        self.tables[name] = frame.copy()
        if self.directory and self.directory_owned:
            frame.to_csv(self.directory / f"{name}.csv", index=False)

    def finish_inputs(self):
        for row in self.inputs.values():
            path = Path(row["path"])
            if row["exists"]:
                row["exists_after_run"] = path.is_file()
                current = _stat(path) if row["exists_after_run"] else {}
                row["size_or_mtime_changed"] = (current.get("size_bytes"), current.get("mtime_ns")) != (row.get("size_bytes"), row.get("mtime_ns"))
        self.table("input_manifest", pd.DataFrame(list(self.inputs.values()), columns=[
            "path", "category", "usage_status", "exists", "size_bytes", "mtime_ns", "modified_utc",
            "sha256", "hash_status", "git_tracked", "rows_read", "batches_read", "exists_after_run",
            "size_or_mtime_changed", "model_reported_sha256", "model_hash_mismatch"], dtype=object))
        self.record["input_count"] = len(self.inputs)
        self.record["input_changes_detected"] = any(row.get("size_or_mtime_changed", False) or row.get("model_hash_mismatch", False) for row in self.inputs.values())
        self.record["input_version_assessment"] = ("changes_detected_review_required" if self.record["input_changes_detected"]
                                                    else "no_changes_detected_by_recorded_checks")
        self.record["trace_hash_coverage"] = {"files_read": sum(row["category"] == "trace_parquet" for row in self.inputs.values()),
                                             "files_hashed": sum(row["category"] == "trace_parquet" and bool(row["sha256"]) for row in self.inputs.values())}

    def finish_outputs(self):
        records = []
        if self.directory and self.directory_owned:
            for root in self.output_roots:
                for path in _files(root):
                    current = _stat(path)
                    before = self.outputs_before.get(str(path.resolve()))
                    changed = self.outputs_snapshot_complete and (before is None or any(current[key] != before[key] for key in ("size_bytes", "mtime_ns")))
                    records.append({"path": str(path.resolve()), **current,
                                    "observation": ("not_attributed_snapshot_incomplete" if not self.outputs_snapshot_complete else
                                                    "created_or_modified_during_run" if changed else "preexisting_unchanged_not_attributed_to_run"),
                                    "sha256": sha256_file(path) if changed else None})
            for path in _files(self.directory):
                if path.name in ("run_metadata.json", "run_metadata.json.tmp", "output_manifest.csv"):
                    continue  # Avoid self hashes and metadata that is still being updated.
                records.append({"path": str(path.resolve()), **_stat(path), "observation": "run_provenance_artifact", "sha256": sha256_file(path)})
            if self.trace_owned and self.trace_path and not self.trace_path.is_relative_to(self.directory):
                records.append({"path": str(self.trace_path), **_stat(self.trace_path),
                                "observation": "run_trace_log", "sha256": sha256_file(self.trace_path)})
        records = list({row["path"]: row for row in records}.values())
        self.table("output_manifest", pd.DataFrame(records, columns=["path", "size_bytes", "mtime_ns", "modified_utc", "observation", "sha256"]))

    def __exit__(self, exc_type, exc, tb):
        self.record.update(status="completed" if exc is None else "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                           finished_utc=utc_now(), elapsed_seconds=time.perf_counter() - self.started)
        if exc is not None:
            self.record["error"] = {"type": type(exc).__name__, "message": str(exc), "stage": self.record["stage"],
                                    "traceback": "".join(traceback.format_exception(exc_type, exc, tb))}
        self.event("execution_finished", status=self.record["status"])
        for stream in (self.trace_stream, self.event_stream):
            if stream:
                stream.close()
        self.finish_inputs()
        self.finish_outputs()
        self.record.update(finished_utc=utc_now(), elapsed_seconds=time.perf_counter() - self.started,
                           output_snapshot_complete=self.outputs_snapshot_complete)
        self._flush_metadata()
        return False
