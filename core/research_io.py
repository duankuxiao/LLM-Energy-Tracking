"""Small file helpers shared by the paper's analysis scripts."""

import hashlib
from pathlib import Path
import subprocess


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_workbook(path, sheets):
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            worksheet = writer.sheets[name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions


def git_snapshot(root):
    def git(*args):
        return subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args],
                              capture_output=True, check=True, timeout=30,
                              encoding="utf-8", errors="replace").stdout
    try:
        return {"available": True, "head_commit": git("rev-parse", "HEAD").strip(),
                "tracked_changes": git("status", "--porcelain=v1", "--untracked-files=no").splitlines(),
                "tracked_paths": sorted(set(git("ls-files", "--cached", "-z").split("\0")) - {""}),
                "remote_availability": "not_checked"}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc), "tracked_paths": [], "remote_availability": "not_checked"}
