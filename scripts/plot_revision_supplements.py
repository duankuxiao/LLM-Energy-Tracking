"""Draw additional reviewer diagnostics as separate figures in the original palette."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/revision_style/figures")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("Raster resolution must be at least 300 dpi.")
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir.parent / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.research_io import sha256_file
    from scripts.plot_figures import FigureSource
    from scripts.original_figure_style import SUPPLEMENT_BUILDERS
    captions = {
        "supplementary_task_energy_closure": "Supplementary Figure S2. All-task, all-component energy closure. "
            "a,b, Idle and dynamic shares of system IT energy by physical component and task, including unclassified "
            "activity and the unallocated idle pool. c, Five-component composition of unclassified tasks. "
            "d, Task-by-component shares of system IT energy. Country totals are summed within year, then averaged "
            "over 2026-2030 under Base-CP. Task-level pie percentages have a task denominator; matrix and bars have "
            "the full-system denominator. All calculations use the new experiment and hourly-driver attribution.",
        "supplementary_additional_sensitivity": "Supplementary Figure S3. Additional one-factor mechanism controls. "
            "a, Electricity; b, direct operational emissions. The nine additional parameters are shown separately "
            "so that Figure 4 retains its original six parameters and tornado composition. Colours indicate Low/High "
            "parameter levels, not the sign of their response. Percentages use each metric's own mean annual "
            "Base-CP total over 2026-2030. Values near 1e-14% are floating-point residuals. These deterministic "
            "contrasts are not confidence intervals or joint uncertainty estimates.",
        "supplementary_ramp_diagnostics": "Supplementary Figure S4. Raw and filtered hourly-ramp diagnostics. "
            "a, Maximum absolute ramp under raw, repeat-seam-excluded and all-join-excluded definitions. "
            "Their curves coincide in this reference case. b, P95 and P99 ramps with and without repeat seams. "
            "Metrics sum country power at the same UTC hour before computing the aggregate ramp. Calendar "
            "coverage includes 8,784 hours in 2028; facility electricity retains 8,760-hour annualization. "
            "The shared national load shape is a reference assumption, not observed national behaviour.",
    }
    existing = [args.output_dir / f"{name}.{extension}" for name in SUPPLEMENT_BUILDERS for extension in ("svg", "pdf", "png", "tiff")]
    if any(path.exists() for path in existing) and not args.overwrite:
        parser.error("Supplement outputs already exist; use --overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    for index, (name, (n, builder)) in enumerate(SUPPLEMENT_BUILDERS.items(), start=2):
        if n not in sources:
            sources[n] = FigureSource(args.input_dir / f"figure{n}_data.xlsx", n)
        source = sources[n]
        figure = builder(source, plt)
        stem = args.output_dir / name
        try:
            figure.savefig(stem.with_suffix(".svg"), facecolor="white")
            figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
            figure.savefig(stem.with_suffix(".png"), dpi=args.dpi, facecolor="white")
            figure.savefig(stem.with_suffix(".tiff"), dpi=args.dpi, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
        finally:
            plt.close(figure)
        sources_dir = args.output_dir / f"{name}_sources"
        sources_dir.mkdir(exist_ok=True)
        for sheet, table in source.tables.items():
            table.to_csv(sources_dir / f"{sheet}.csv", index=False)
        boundary = (" Study boundary: 24 major AI countries. New experiment values retain the legacy calibration "
                    "of global reference electricity to the selected-country sum; absolute results are conditional "
                    "on that assumption. The 93.18% IT-capacity share is not observed electricity coverage.")
        stem.with_suffix(".caption.txt").write_text(captions[name] + boundary + "\n", encoding="utf-8")
        manifest = {"figure": f"S{index}", "source_workbook": source.path.name, "source_sha256": source.sha256,
                    "style": "original_palette_new_supplement", "dpi": args.dpi,
                    "style_script_sha256": sha256_file(ROOT / "scripts/original_figure_style.py"),
                    "plot_script_sha256": sha256_file(Path(__file__)), "visual_QA": "pending_panel_review",
                    "outputs": {str(path.relative_to(args.output_dir)): sha256_file(path)
                                for path in sorted(args.output_dir.glob(name + ".*")) if path.suffix != ".json"},
                    "source_tables": {str(path.relative_to(args.output_dir)): sha256_file(path) for path in sorted(sources_dir.glob("*.csv"))}}
        stem.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Supplementary Figure S{index}: {name}")


if __name__ == "__main__":
    main()
