"""Check clean-checkout completeness and published manuscript reference results."""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from core.annual_cpu_model import run_annual_cpu_model
from core.hourly_workload_resolved_model import WorkloadProfile
from scripts.figure4_data import generate_figure4a_data
from scripts.validate_energy_model import _bootstrap_metrics

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "results/supplementary"


class ReleaseConsistencyTest(unittest.TestCase):
    def test_committed_sources_have_all_internal_dependencies(self):
        tracked = set(subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0"))
        for path in sorted(tracked):
            if not path.endswith(".py"):
                continue
            tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                           else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    if module.split(".")[0] not in ("core", "scripts", "dataset"):
                        continue
                    target = module.replace(".", "/")
                    with self.subTest(path=path, line=node.lineno, module=module):
                        self.assertTrue(target + ".py" in tracked or target + "/__init__.py" in tracked,
                                        f"Unpublished dependency: {module}")

    def test_entrypoints_import_without_external_trace_or_grid_inputs(self):
        for name in ("run", "scripts.run_experiments", "scripts.figure1_data", "scripts.figure2_data",
                     "scripts.figure3_data", "scripts.figure4_data", "scripts.validate_energy_model",
                     "scripts.build_hourly_grid_scenarios", "scripts.plot_figures",
                     "scripts.plot_revision_supplements", "scripts.export_supplementary_sources"):
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_annual_model_matches_published_manuscript_endpoints(self):
        expected = pd.read_csv(SOURCES / "annual_endpoint_comparison.csv").set_index("year")
        actual = run_annual_cpu_model("CP", ["Base"], year_start=2026, years=5,
                                      save_outputs=False, verbose=False)["selected_countries_summary"].set_index("year")
        for metric in ("power_twh", "carbon_mtco2", "installed_it_mw"):
            np.testing.assert_allclose(actual[metric], expected[f"annual_cpu_{metric}"], rtol=1e-11)
        self.assertAlmostEqual(expected.loc[2030, "annual_cpu_carbon_difference_vs_hourly_pct"],
                               13.173924876308279, places=10)

    def test_s1_bootstrap_reproduces_published_intervals(self):
        systems = pd.read_csv(SOURCES / "S1_systems.csv")
        expected = pd.read_csv(SOURCES / "S1_metrics.csv").set_index("metric")
        actual = _bootstrap_metrics(systems["observed_power_max_w"].to_numpy(),
                                    systems["predicted_full_load_reference_w"].to_numpy(),
                                    systems["accelerator_count"].to_numpy(), 10000, 20260819).set_index("metric")
        self.assertEqual(len(systems), 32)
        columns = ["estimate", "ci_95_lower", "ci_95_upper"]
        np.testing.assert_allclose(actual[columns], expected.loc[actual.index, columns], rtol=1e-11)

    def test_published_supplementary_table_hashes_and_coverage(self):
        manifest = json.loads((SOURCES / "manifest.json").read_text(encoding="utf-8"))
        for record in manifest["tables"]:
            path = SOURCES / record["file"]
            with self.subTest(table=path.name):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
                self.assertEqual(len(pd.read_csv(path)), record["rows"])
        sensitivity = pd.read_csv(SOURCES / "S3_sensitivity.csv")
        self.assertEqual(sensitivity["parameter_name"].nunique(), 15)
        self.assertEqual(len(sensitivity), 45)
        self.assertTrue(sensitivity.groupby("parameter_name")["level"].apply(
            lambda values: set(values) == {"Low", "Base", "High"}).all())

    def test_all_fifteen_sensitivities_execute_with_a_synthetic_trace(self):
        hour = np.arange(24)
        load = np.broadcast_to(.05 + .03 * np.cos(2 * np.pi * hour / 24), (4, 4, 24)).copy()
        load[:, 3, :] = 0.
        profile = WorkloadProfile(
            interval_index=pd.date_range("2001-01-01", periods=24, freq="h", tz="UTC"),
            interval_hours=1., load=load, trace_capacity=np.ones(4), task_counts=np.ones(4),
            task_type_summary=pd.DataFrame({"task_type": ("training", "inference", "other", "unclassified")}),
            relative_time=True, trace_capacity_source=("synthetic_regression",) * 4)
        base = ROOT / "results/release_test_tmp"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory, \
                patch("scripts.figure4_data.build_workload_profile", return_value=profile), \
                patch("scripts.figure4_data.write_workbook") as writer, warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            generate_figure4a_data(output_path=Path(directory) / "figure4.xlsx", countries=["USA"],
                                  year_start=2030, years=1, hourly_carbon_factors_dir=directory, verbose=False)
        tables = writer.call_args.args[1]
        result = tables["Fig4a_Sensitivity"]
        self.assertEqual(result["parameter_name"].nunique(), 15)
        self.assertEqual(len(result), 45)
        self.assertTrue(np.isfinite(result["facility_energy_mwh_change_vs_baseline_pct"]).all())


if __name__ == "__main__":
    unittest.main()
