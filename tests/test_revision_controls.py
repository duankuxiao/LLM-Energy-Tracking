"""Scientific regression checks for R1.01 and R2.03–06,10,12–13."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
import pandas as pd

from core.annual_activity import AnnualActivityConfig
from core.annual_cpu_model import run_annual_cpu_model
from core.annual_reference_controls import (
    REFERENCE_DEFAULTS, annual_reference_controls, reference_cases,
)
from core.energy_attribution import allocate_component_energy, allocation_records
from core.experiment_config import load_experiment_config
from core.grid_carbon import calendar_hours
from core import hourly_workload_resolved_model as model
from core.temporal_robustness import power_metrics, timing_stress_tables
from scripts.annual_reference_sensitivity import run_reference_sensitivity
from scripts.figure3_data import _w04_endpoint_pair, generate_figure3_data

ROOT = Path(__file__).resolve().parents[1]


def annual_reference(countries=("USA", "China"), year_start=2029, years=2):
    reference = run_annual_cpu_model("CP", ["Base"], countries=countries, year_start=year_start,
                                     years=years, save_outputs=False, verbose=False)["annual_summary"].copy()
    reference["model"] = "hourly_workload_resolved_model"  # Synthetic endpoint for invariant checks only.
    return reference


def synthetic_profile():
    hour = np.arange(24)
    load = np.broadcast_to((.05 + .03 * np.cos(2 * np.pi * hour / 24)), (4, 4, 24)).copy()
    load[:, 3, :] = 0.  # No observed storage activity.
    return model.WorkloadProfile(
        interval_index=pd.date_range("2001-01-01", periods=24, freq="h", tz="UTC"),
        interval_hours=1., load=load, trace_capacity=np.ones(4), task_counts=np.ones(4),
        task_type_summary=pd.DataFrame({"task_type": model.TASK_TYPES}), relative_time=True,
        trace_capacity_source=("synthetic_test",) * 4)


class ReferenceControlsTest(unittest.TestCase):
    def setUp(self):
        self.reference = annual_reference()

    def run_controls(self, reference=None, **kwargs):
        return annual_reference_controls(self.reference if reference is None else reference,
                                         countries=["USA", "China"], year_start=2029, years=2, **kwargs)

    def test_one_factor_cases_preserve_task_share_simplex(self):
        cases = reference_cases()
        self.assertEqual(len(cases), 9)
        for case in cases:
            changed = [key for key, value in REFERENCE_DEFAULTS.items() if case[key] != value]
            self.assertEqual(len(changed), 0 if case["case"] == "baseline" else 1)
            shares = [case[key] for key in ("training_task_share", "inference_task_share", "other_task_share", "unclassified_task_share")]
            self.assertAlmostEqual(sum(shares), 1.)
            self.assertGreaterEqual(min(shares), 0.)

    def test_sensitivity_matches_analytic_energy_change_and_keeps_reference(self):
        before = self.reference.copy(deep=True)
        tables = self.run_controls()
        pd.testing.assert_frame_equal(before, self.reference)
        country = tables["W04_Reference_Country"]
        base = country.loc[country["case"] == "baseline"].set_index(["year", "country"])
        np.testing.assert_allclose(base["energy_gap_pct_of_reference"], 0., atol=1e-10)
        case = next(case for case in reference_cases() if case["varied_parameter"] == "u_infer" and case["parameter_level"] == "Low")
        actual = country.loc[country["case"] == case["case"]].set_index(["year", "country"])
        for key, row in actual.iterrows():
            inference_activity = AnnualActivityConfig().vector(key[0])[1]
            expected = row["installed_it_mw_annual"] * 8760 * (.88 - .23) * .75 * inference_activity * (case["u_infer"] - .5)
            self.assertAlmostEqual((row["it_energy_mwh_annual"] - base.loc[key, "it_energy_mwh_annual"]) / expected, 1.)
        reference_totals = country.groupby(["case", "year"])["facility_energy_mwh_reference"].sum().unstack("case")
        for column in reference_totals:
            np.testing.assert_allclose(reference_totals[column], reference_totals["baseline"])
        self.assertTrue(tables["W04_Reference_Checks"]["status"].eq("PASS").all())
        self.assertTrue(tables["W04_Reference_Envelope"]["case_count"].eq(9).all())

    def test_invalid_or_duplicate_reference_rejected_before_model_run(self):
        for reference in (self.reference.iloc[:-1], pd.concat([self.reference, self.reference.iloc[[0]]])):
            runner = Mock()
            with self.subTest(rows=len(reference)), self.assertRaisesRegex(ValueError, "exactly one"):
                self.run_controls(reference, annual_runner=runner)
            runner.assert_not_called()

    def test_capacity_and_pue_mismatch_cannot_be_reported_as_model_effect(self):
        for column in ("installed_it_mw", "facility_energy_mwh"):
            bad = self.reference.copy()
            bad[column] *= 1.1
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "identical capacity factors and PUE"):
                self.run_controls(bad)

    def test_disabled_controls_do_not_run_models(self):
        runner = Mock()
        self.assertEqual(self.run_controls(config={"enabled": False}, annual_runner=runner), {})
        runner.assert_not_called()

    def test_invalid_ranges_fail_without_clipping_or_silent_fallback(self):
        configs = [
            {"enable": True}, {"enabled": "true"}, {"levels": {"unknown": [0., 1.]}},
            {"levels": {"u_train": [.64, .96]}}, {"levels": {"u_train": [.8, .8]}},
            {"levels": {"u_train": [.8, True]}}, {"levels": {"u_train": [.8, np.nan]}},
            {"levels": {"default_p_infer": [.75, 1.]}},
        ]
        for config in configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                reference_cases(config)

    def test_zero_reference_carbon_has_no_infinite_percentage(self):
        reference = self.reference.copy()
        reference["carbon_tco2"] = 0.
        tables = self.run_controls(reference)
        self.assertTrue(tables["W04_Reference_Summary"]["carbon_gap_pct_of_reference"].isna().all())

    def test_calendar_endpoint_and_requested_window_are_consistent(self):
        calendar = AnnualActivityConfig()
        np.testing.assert_allclose(calendar.vector(2030)[:2], [.925, .70])
        np.testing.assert_allclose(AnnualActivityConfig(mode="legacy_six_year").vector(2030)[:2], [.9208333333333333, .6666666666666666])
        full = annual_reference(("USA",), 2025, 6).set_index("year")
        sliced = annual_reference(("USA",), 2030, 1).set_index("year")
        self.assertAlmostEqual(full.loc[2030, "facility_energy_mwh"], sliced.loc[2030, "facility_energy_mwh"])

    def test_common_mean_control_requires_matching_CPU_activity(self):
        annual = self.reference.copy()
        annual["composite_utilization"] = .4
        component = annual.copy()
        component["avg_cpu_utilization"] = .4
        pair = _w04_endpoint_pair(annual, component, "common_mean_test")
        np.testing.assert_allclose(pair["cpu_utilization_difference"], 0.)
        component["avg_cpu_utilization"] += .01
        with self.assertRaisesRegex(ValueError, "Common CPU activity"):
            _w04_endpoint_pair(annual, component, "common_mean_test")


class EnergyAndTimingControlsTest(unittest.TestCase):
    def test_every_idle_rule_closes_to_same_physical_components(self):
        idle, dynamic = np.full((5, 2, 24), .2), np.full((5, 2, 24), .1)
        dynamic[3] = 0.
        dynamic[:, :, 0] = 0.
        loads = np.full((4, 2, 4, 24), .1)
        loads[:, :, 3, :] = 0.
        loads[:, :, :, 0] = 0.
        details = allocate_component_energy(idle, dynamic, loads, np.ones((2, 4)), 1.,
                                             model.COMPONENTS, model.RESOURCES,
                                             {"cpu": .35, "gpu": .5, "memory": .15})
        expected = (idle + dynamic).sum(axis=2).T
        for rule, allocation in details["rules"].items():
            with self.subTest(rule=rule):
                np.testing.assert_allclose(allocation["total"].sum(axis=0) + allocation["pool"], expected)
                np.testing.assert_allclose(allocation["dynamic"], details["rules"]["hourly_driver"]["dynamic"])
        self.assertGreater(details["rules"]["separate_pool"]["pool"].sum(), 0.)
        _, _, checks = allocation_records(details, ["USA", "China"], model.COMPONENTS,
                                          np.array([1.2, 1.3]), {"year": 2030}, "hourly_driver")
        self.assertTrue(all(row["status"] == "PASS" for row in checks))

    def test_unit_watts_cancel_only_when_idle_ratio_is_fixed(self):
        def run(hardware):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return model.run_workload_component_footprint(
                    "CP", ["Base"], countries=["USA"], year_start=2030, years=1,
                    workload_profile=synthetic_profile(), hardware_config=hardware,
                    hourly_carbon_factors_dir=None, save_outputs=False, verbose=False)["annual_summary"]
        hardware = model.HardwarePowerConfig()
        base = run(hardware)
        fixed_ratio = run(replace(hardware, gpu_full_power_w=500., gpu_idle_power_w=50.))
        fixed_idle = run(replace(hardware, gpu_full_power_w=500.))
        np.testing.assert_allclose(base["facility_energy_mwh"], fixed_ratio["facility_energy_mwh"], rtol=1e-12)
        self.assertLess(fixed_idle["facility_energy_mwh"].iloc[0], base["facility_energy_mwh"].iloc[0])

    def test_ramp_filters_exclude_incoming_join_without_annual_wrap(self):
        result = power_metrics([10., 30., 15.], [False, True, False], [False, True, False])
        self.assertEqual(result["max_abs_ramp_mw_per_h"], 20.)
        self.assertEqual(result["max_abs_ramp_excluding_seams_mw_per_h"], 15.)
        self.assertEqual(result["raw_ramp_edge_count"], 2)
        all_filtered = power_metrics([10., 30.], [False, True])
        self.assertTrue(np.isnan(all_filtered["max_abs_ramp_excluding_seams_mw_per_h"]))

    def test_timing_stresses_preserve_electricity_and_joint_shift_emissions(self):
        times = calendar_hours(2027)
        hour = np.arange(len(times))
        frames = []
        for country, offset in (("USA", 0), ("China", 8)):
            frames.append(pd.DataFrame({
                "scenario": "Base", "year": 2027, "country": country,
                "timestamp_utc": times.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "facility_energy_mwh": 10. + 5. * np.sin(2 * np.pi * (hour + offset) / 24),
                "carbon_factor_kg_per_mwh": 300. + 100. * np.sin(2 * np.pi * hour / 24),
                "carbon_factor_source": "hourly", "trace_repeat_seam_before": False,
                "alignment_discontinuity_before": False}))
        table = timing_stress_tables(pd.concat(frames, ignore_index=True), policy="CP", carbon_scope="direct")["W06_Timing_Aggregate"].set_index("timing_case")
        np.testing.assert_allclose(table["energy_closure_residual_mwh"], 0., atol=1e-6)
        self.assertAlmostEqual(table.loc["joint_phase_plus6h", "matched_carbon_tco2"], table.loc["baseline", "matched_carbon_tco2"])
        self.assertAlmostEqual(table.loc["ci_flat_annual_mean", "hourly_matching_effect_tco2"], 0.)
        self.assertTrue((table["peak_coincidence_ratio"] <= 1. + 1e-12).all())


class ConfigurationAndEntryTest(unittest.TestCase):
    def setUp(self):
        base = ROOT / "results/reviewer_code_checks_tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.directory = Path(self.temp.name).resolve()
        assert self.directory.is_relative_to(base.resolve())
        self.config = json.loads((ROOT / "configs/experiments.json").read_text(encoding="utf-8"))

    def tearDown(self):
        self.temp.cleanup()

    def write_config(self, data):
        path = self.directory / "experiments.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_selection_validation_fails_before_data_loading(self):
        for selection in ([], ["figure3", "figure3"], ["unknown"], "figure3"):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                load_experiment_config(ROOT / "configs/experiments.json", only=selection)

    def test_invalid_ranges_hardware_and_years_fail_at_configuration_load(self):
        mutations = (
            lambda d: d["common"].update(years=0),
            lambda d: d["common"].update(capacity_quantile=1.1),
            lambda d: d["common"].update(strict_hourly_carbon="true"),
            lambda d: d["hardware"].update(gpu_power_share=.9),
            lambda d: d["experiments"]["figure3"]["reference_sensitivity"].update(levels={"u_train": [.2, .3]}),
        )
        for mutation in mutations:
            document = deepcopy(self.config)
            mutation(document)
            with self.subTest(document=document), self.assertRaises(ValueError):
                load_experiment_config(self.write_config(document))

    def test_standalone_writes_complete_new_output_and_refuses_overwrite(self):
        reference = annual_reference(None, 2026, 5).drop(columns=["policy", "model"])
        source = self.directory / "reference.csv"
        reference.to_csv(source, index=False)
        output = self.directory / "controls"
        with patch("core.hourly_workload_resolved_model.build_workload_profile", side_effect=AssertionError("No trace read permitted")):
            run_reference_sensitivity(source, output)
        metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["annual_model_runs"], 9)
        self.assertEqual(metadata["hourly_model_runs"], 0)
        self.assertEqual(metadata["reference_policy_source"], "explicit_Figure_3_CP_scope")
        self.assertEqual(len(list(output.glob("W04_Reference_*.csv"))), 6)
        with self.assertRaises(FileExistsError):
            run_reference_sensitivity(source, output)

    def test_figure3_exports_reference_controls_with_synthetic_trace(self):
        output = self.directory / "figure3_data.xlsx"
        with patch("scripts.figure3_data.build_workload_profile", return_value=synthetic_profile()), \
                patch("scripts.figure3_data.write_workbook") as writer, warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            generate_figure3_data(output_path=output, hourly_carbon_factors_dir=self.directory,
                                  hourly_carbon_fallback_to_annual=True, verbose=False)
        writer.assert_called_once()
        sheets = writer.call_args.args[1]
        summary = sheets["W04_Reference_Summary"]
        checks = sheets["W04_Reference_Checks"]
        self.assertEqual(len(summary), 45)
        self.assertTrue(checks["status"].eq("PASS").all())
        baseline = summary.loc[summary["case"] == "baseline"].set_index("year")
        reference = sheets["W04_Control_Comparison"]
        reference = reference.loc[(reference["case"] == "calendar_baseline") & (reference["scope"] == "selected_countries")].set_index("year")
        np.testing.assert_allclose(baseline["energy_gap_pct_of_reference"], reference["energy_gap_pct_of_component"])


if __name__ == "__main__":
    unittest.main()
