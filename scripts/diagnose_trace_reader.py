"""Read a few actual trace partitions without running research analyses.

File numbers are ONE-BASED in the model's existing lexicographic path order,
not chronological day/hour order. The default checks files 3552-3554, near
the last progress message from the interrupted run. Nothing is written.
Add --compare-whole-file to compare batched statistics against a whole-file,
single-threaded reference read (which uses more memory).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import faulthandler
from importlib import metadata as package_metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.hourly_workload_resolved_model import (
    ALIBABA_2026_POD_TABLE, ALIBABA_2026_SERVER_TABLE,
    DEFAULT_PARQUET_BATCH_SIZE, POD_PROFILE_COLUMNS, SERVER_PROFILE_COLUMNS,
    Alibaba2026TraceConfig, _aggregate_pod_batch, _alibaba_2026_partition_files,
    _iter_parquet_frames, _resolve_alibaba_2026_table_root,
)


def _report(event: dict) -> None:
    print(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event},
                     ensure_ascii=False, allow_nan=False), flush=True)


def _statistics(frame: pd.DataFrame, table: str, day: int) -> dict:
    if table == "pod":
        return vars(_aggregate_pod_batch(frame, day, Alibaba2026TraceConfig()))
    return {"capacity": np.array([
        pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0).sum()
        for column in ("cpu_capacity_cores", "gpu_count")
    ], dtype=np.float64)}


def diagnose_files(
    root: Path,
    table: str = "pod",
    start_file: int = 3552,
    file_count: int = 3,
    batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
    compare_whole_file: bool = False,
    repeat: int = 1,
) -> list[dict]:
    """Validate row coverage and, optionally, statistics across batch boundaries."""
    if min(start_file, file_count, batch_size, repeat) <= 0:
        raise ValueError("File numbers, counts, batch size and repeat must be positive.")
    if table not in ("pod", "server"):
        raise ValueError("table must be pod or server.")
    table_name = ALIBABA_2026_POD_TABLE if table == "pod" else ALIBABA_2026_SERVER_TABLE
    table_root = _resolve_alibaba_2026_table_root(root, table_name)
    files = _alibaba_2026_partition_files(table_root)
    end_file = start_file + file_count - 1
    if end_file > len(files):
        raise ValueError(f"Requested files {start_file}-{end_file}, but found {len(files)} under {table_root}.")
    columns = POD_PROFILE_COLUMNS if table == "pod" else SERVER_PROFILE_COLUMNS
    summaries = []
    for file_number, (day, hour, path) in enumerate(files[start_file - 1:end_file], start=start_file):
        for repetition in range(1, repeat + 1):
            started = time.perf_counter()
            _report({"event": "diagnostic_file_start", "file_number": file_number,
                     "path": str(path), "day": day, "hour": hour, "repeat": repetition})
            totals = _statistics(pd.DataFrame(columns=columns), table, day)
            rows_read, batches_read = 0, 0
            for frame in _iter_parquet_frames(path, columns, batch_size, _report):
                batch_totals = _statistics(frame, table, day)
                for name in totals:
                    totals[name] += batch_totals[name]
                rows_read += len(frame)
                batches_read += 1
                del frame, batch_totals

            max_absolute_difference = None
            if compare_whole_file:
                import pyarrow.parquet as pq

                _report({"event": "whole_file_reference_start", "path": str(path), "use_threads": False})
                parquet = pq.ParquetFile(path, memory_map=False, pre_buffer=False)
                try:
                    arrow_table = parquet.read(columns=list(columns), use_threads=False)
                    frame = arrow_table.to_pandas(use_threads=False, split_blocks=True)
                    if len(frame) != rows_read:
                        raise AssertionError(f"Row coverage differs for {path}: whole={len(frame)}, batches={rows_read}")
                    reference = _statistics(frame, table, day)
                    max_absolute_difference = 0.0
                    for name in totals:
                        if np.issubdtype(totals[name].dtype, np.integer):
                            np.testing.assert_array_equal(totals[name], reference[name], err_msg=f"{path}: {name}")
                        else:
                            np.testing.assert_allclose(totals[name], reference[name], rtol=1e-10, atol=1e-7,
                                                       err_msg=f"{path}: {name}")
                        difference = np.max(np.abs(totals[name] - reference[name]))
                        max_absolute_difference = max(max_absolute_difference, float(difference))
                    del frame, reference, arrow_table
                finally:
                    parquet.close()

            summary = {"event": "diagnostic_file_pass", "file_number": file_number,
                       "path": str(path), "repeat": repetition, "rows_read": rows_read,
                       "batches_read": batches_read, "row_coverage": "PASS",
                       "whole_file_statistics": "PASS" if compare_whole_file else "not_requested",
                       "max_absolute_difference": max_absolute_difference,
                       "elapsed_seconds": round(time.perf_counter() - started, 3)}
            summaries.append(summary)
            _report(summary)
    return summaries


def main() -> None:
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-profile-path", type=Path, default=ROOT_DIR / "dataset")
    parser.add_argument("--table", choices=("pod", "server"), default="pod")
    parser.add_argument("--start-file", type=int, default=3552)
    parser.add_argument("--file-count", type=int, default=3)
    parser.add_argument("--parquet-batch-size", type=int, default=DEFAULT_PARQUET_BATCH_SIZE)
    parser.add_argument("--compare-whole-file", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if min(args.start_file, args.file_count, args.parquet_batch_size, args.repeat) <= 0:
        parser.error("File numbers, counts, batch size and repeat must be positive.")
    versions = {"python": platform.python_version(), "executable": sys.executable}
    for name in ("numpy", "pandas", "pyarrow"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    _report({"event": "diagnostic_start", "runtime_versions": versions})
    summaries = diagnose_files(args.workload_profile_path, args.table, args.start_file,
                              args.file_count, args.parquet_batch_size, args.compare_whole_file, args.repeat)
    print(f"PASS: {len(summaries)} partition checks completed. This small sample is not a full-trace validation.", flush=True)


if __name__ == "__main__":
    main()
