"""W03 regression tests. Run: python -B -m unittest discover -s tests -v"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from core import hourly_workload_resolved_model as model
from core.grid_carbon import (
    CI_COLUMNS, calendar_hours, map_template_to_year, matching_diagnostics,
    transform_shape,
)
from dataset.Factors import annual_policy_audit, get_carbon_factor, get_policy_multiplier
from scripts.audit_grid_scenarios import audit_grid_scenarios
from scripts.build_hourly_grid_scenarios import build_scenarios
from scripts.figure2_data import generate_figure2_data

ROOT = Path(__file__).resolve().parents[1]


def synthetic_profile():
    shape = .12 + .06 * np.cos(2 * np.pi * np.arange(24) / 24)
    return model.WorkloadProfile(
        interval_index=pd.date_range("2001-01-01", periods=24, freq="h", tz="UTC"),
        interval_hours=1., load=np.tile(shape, (4, 4, 1)), trace_capacity=np.ones(4),
        task_counts=np.ones(4), task_type_summary=pd.DataFrame({"task_type": model.TASK_TYPES}),
        relative_time=True, trace_capacity_source=("synthetic_test",) * 4,
    )


class GridScenariosTest(unittest.TestCase):
    def setUp(self):
        base = ROOT / "results/revision_audit/grid_test_tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(base.resolve())

    def tearDown(self):
        self.temp.cleanup()

    def frame(self, year=2028, country="USA", policy="CP", legacy=False):
        index = calendar_hours(year)
        if legacy:
            index = index[:8760]
        shape = 1 + .8 * np.cos(2 * np.pi * np.arange(len(index)) / 24)
        values = shape * get_carbon_factor(policy, country, year)
        return pd.DataFrame({"Datetime (UTC)": index.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "Zone id": "US", CI_COLUMNS["direct"]: values,
                             CI_COLUMNS["life_cycle"]: values * 1.2, "Data estimated": False})

    def write_input(self, *, year=2028, country="USA", policy="CP", legacy=False):
        folder = self.root / country
        folder.mkdir(exist_ok=True)
        path = folder / f"test-{policy}-{year}-hourly.csv"
        self.frame(year, country, policy, legacy).to_csv(path, index=False)
        return path

    def load(self, **overrides):
        args = dict(countries=["USA"], renewable_energy_policy="CP", year=2028,
                    hourly_carbon_factors_dir=self.root, hourly_carbon_scope="direct",
                    hourly_carbon_fallback_to_annual=False)
        args.update(overrides)
        return model._load_hourly_carbon_factors(**args)

    def run_model(self, **overrides):
        args = dict(renewable_energy_policy="CP", scenarios=["Base"], years=1, year_start=2028,
                    countries=["USA"], workload_profile=synthetic_profile(), save_outputs=False,
                    verbose=False, hourly_carbon_factors_dir=self.root,
                    hourly_carbon_fallback_to_annual=False)
        args.update(overrides)
        return model.run_workload_component_footprint(**args)

    def test_annual_policy_constraints_include_rounding(self):
        self.assertEqual(get_policy_multiplier("NZ", 2030), .53)
        self.assertEqual(get_policy_multiplier("NDC", 2030), .89)
        records = annual_policy_audit(["USA", "Sweden", "France"], range(2026, 2031))
        self.assertTrue(all(r["within_rounding_tolerance"] for r in records))
        self.assertTrue(any(abs(r["rounding_residual_kg_per_mwh"]) > 0 for r in records))

    def test_calendar_maps_leap_day_and_preserves_december_31(self):
        src = calendar_hours(2025)
        values = np.arange(len(src)) + 1.
        target, mapped = map_template_to_year(src, values, 2028)
        self.assertEqual(len(target), 8784)
        self.assertEqual(target[-1], pd.Timestamp("2028-12-31T23:00Z"))
        self.assertEqual(mapped[target.get_loc("2028-02-29T12:00Z")], values[src.get_loc("2025-02-28T12:00Z")])
        self.assertEqual(mapped[-1], values[-1])
        self.assertAlmostEqual(transform_shape(mapped, target_mean=12.).mean(), 12.)

    def test_invalid_hourly_data_never_silently_falls_back(self):
        path = self.write_input()
        original = self.frame()
        for kind in ("negative", "nan", "inf", "duplicate", "gap", "unordered", "wrong_year", "units", "milligrams"):
            frame = original.copy()
            if kind in ("negative", "nan", "inf"):
                frame.loc[0, CI_COLUMNS["direct"]] = {"negative": -1., "nan": np.nan, "inf": np.inf}[kind]
            elif kind == "duplicate":
                frame.loc[1, "Datetime (UTC)"] = frame.loc[0, "Datetime (UTC)"]
            elif kind == "gap":
                frame = frame.drop(20)
            elif kind == "unordered":
                frame = frame.iloc[::-1]
            elif kind == "wrong_year":
                frame.loc[0, "Datetime (UTC)"] = "2027-01-01T00:00:00Z"
            else:
                unit = "mg" if kind == "milligrams" else "kg"
                frame = frame.rename(columns={CI_COLUMNS["direct"]: f"Carbon intensity {unit}CO2eq/kWh (direct)"})
            frame.to_csv(path, index=False)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.load(hourly_carbon_fallback_to_annual=True)

    def test_mean_mismatch_rejected(self):
        path = self.write_input()
        frame = self.frame()
        frame[CI_COLUMNS["direct"]] *= 1.01
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "does not close"):
            self.load()

    def test_legacy_leap_year_is_explicit_and_strict_rejects(self):
        self.write_input(legacy=True)
        with self.assertRaisesRegex(ValueError, "full UTC calendar"):
            self.load()
        with self.assertWarnsRegex(RuntimeWarning, "truncate"):
            result = self.run_model(hourly_carbon_fallback_to_annual=True)
        row = result["grid_carbon_audit"].iloc[0]
        self.assertEqual(row["calendar_status"], "legacy_8760_truncated_leap_year")
        self.assertFalse(row["eligible_for_full_calendar_hourly_claim"])

    def test_inconsistent_calendars_cannot_drop_extra_hours(self):
        self.write_input(legacy=True)
        self.write_input(country="France")
        with self.assertRaisesRegex(ValueError, "calendars differ"):
            self.load(countries=["USA", "France"], hourly_carbon_fallback_to_annual=True)

    def test_missing_files_fallback_and_scope_boundary(self):
        with self.assertRaises(FileNotFoundError):
            self.load()
        with self.assertWarns(RuntimeWarning):
            times, factors, used = self.load(hourly_carbon_fallback_to_annual=True)
        self.assertEqual(len(times), 8784)
        self.assertFalse(used.any())
        self.assertTrue(np.all(factors == get_carbon_factor("CP", "USA", 2028)))
        with self.assertRaisesRegex(ValueError, "life-cycle annual fallback"):
            self.load(hourly_carbon_scope="life_cycle", hourly_carbon_fallback_to_annual=True)
        with self.assertRaisesRegex(ValueError, "Annual-only"):
            self.run_model(hourly_carbon_scope="life_cycle", hourly_carbon_factors_dir=None)

    def test_partial_fallback_is_exported(self):
        self.write_input()
        with self.assertWarns(RuntimeWarning):
            result = self.run_model(countries=["USA", "France"], hourly_carbon_fallback_to_annual=True)
        audit = result["grid_carbon_audit"].set_index("country")
        self.assertEqual(audit.loc["France", "carbon_factor_source"], "annual_fallback")
        self.assertEqual(audit.loc["France", "fallback_reason"], "missing_hourly_file")
        self.assertFalse(audit.loc["France", "eligible_for_full_calendar_hourly_claim"])
        self.assertTrue(audit.loc["USA", "eligible_for_full_calendar_hourly_claim"])

    def test_stress_clipping_phase_and_covariance(self):
        values = np.tile([0., 0., 1., 9.], 6)
        energy = np.arange(1., 25.)
        for scale, phase in ((0., 0), (3., 0), (1., 7)):
            altered = transform_shape(values, variability_scale=scale, phase_hours=phase)
            self.assertAlmostEqual(altered.mean(), values.mean())
            self.assertTrue(np.all(altered >= 0))
            diag = matching_diagnostics(energy, altered, energy.sum())
            self.assertAlmostEqual(diag["covariance_identity_residual_tco2"], 0.)
        np.testing.assert_allclose(transform_shape(values, phase_hours=7), np.roll(values, 7))
        with self.assertRaises(ValueError):
            transform_shape(np.zeros(24), target_mean=1.)

    def test_model_electricity_scalar_identity_and_audit_csv(self):
        self.write_input()
        self.write_input(policy="NZ")
        cp = self.run_model(save_outputs=True, output_dir=self.root / "outputs")
        nz = self.run_model(renewable_energy_policy="NZ")
        annual = self.run_model(hourly_carbon_factors_dir=None)
        for result in (nz, annual):
            np.testing.assert_allclose(result["annual_summary"]["facility_energy_mwh"], cp["annual_summary"]["facility_energy_mwh"], rtol=1e-12)
        ratio = get_carbon_factor("NZ", "USA", 2028) / get_carbon_factor("CP", "USA", 2028)
        np.testing.assert_allclose(nz["annual_summary"]["carbon_tco2"], cp["annual_summary"]["carbon_tco2"] * ratio, rtol=1e-12)
        row = cp["grid_carbon_audit"].iloc[0]
        self.assertAlmostEqual(row["energy_closure_residual_mwh"], 0., delta=1e-6)
        self.assertAlmostEqual(row["covariance_identity_residual_tco2"], 0., delta=1e-6)
        self.assertGreater(abs(row["hourly_matching_effect_tco2"]), 1.)
        self.assertTrue((self.root / "outputs/Grid_Carbon_Audit_CP_Base.csv").is_file())

    def test_lifecycle_audit_does_not_claim_direct_annual_closure(self):
        self.write_input()
        result = self.run_model(hourly_carbon_scope="life_cycle")
        row = result["grid_carbon_audit"].iloc[0]
        self.assertFalse(row["annual_mean_comparison_available"])
        self.assertTrue(np.isnan(row["annual_target_ci_kg_per_mwh"]))
        self.assertAlmostEqual(row["hourly_mean_ci_kg_per_mwh"], get_carbon_factor("CP", "USA", 2028) * 1.2)

    def test_generator_provenance_and_both_scopes(self):
        source = self.write_input(year=2026)
        manifest = self.root / "sources.csv"
        pd.DataFrame([dict(country="USA", path=str(source), source_year=2026,
                           source_kind="reconstructed_shape", zone_id="US")]).to_csv(manifest, index=False)
        output = build_scenarios(manifest, self.root / "generated", years=[2028], scopes=["direct", "life_cycle"])
        audit = pd.read_csv(output / "construction_audit.csv")
        self.assertEqual(len(audit), 6)
        self.assertTrue((audit["hours"] == 8784).all())
        self.assertTrue((audit["source_kind"] == "reconstructed_shape").all())
        self.assertLess(audit["mean_residual_kg_per_mwh"].abs().max(), 1e-10)
        tables, failures = audit_grid_scenarios(output, countries=["USA"], years=[2028])
        self.assertEqual(failures, 0)
        self.assertEqual(len(tables["hourly_policy_identity"]), 2)
        with self.assertRaises(FileExistsError):
            build_scenarios(manifest, output, years=[2028])
        bad = pd.read_csv(manifest)
        bad["zone_id"] = "wrong"
        bad.to_csv(manifest, index=False)
        with self.assertRaisesRegex(ValueError, "Zone id"):
            build_scenarios(manifest, self.root / "invalid", years=[2028])
        self.assertFalse((self.root / "invalid").exists())

    def test_audit_distinguishes_missing_and_legacy_inputs(self):
        self.write_input(legacy=True)
        tables, failures = audit_grid_scenarios(self.root, countries=["USA"], years=[2028])
        self.assertGreater(failures, 0)
        self.assertEqual((tables["input_coverage"]["status"] == "ERROR").sum(), 3)
        self.assertTrue(tables["hourly_policy_identity"]["status"].eq("UNAVAILABLE").all())

    def test_figure2_workbook_exports_policy_and_grid_audits(self):
        for policy in ("CP", "NDC", "NZ"):
            self.write_input(policy=policy)
        with patch("scripts.figure2_data.build_workload_profile", return_value=synthetic_profile()):
            output = generate_figure2_data(output_path=self.root / "figure2.xlsx", countries=["USA"],
                                          year_start=2028, years=1, hourly_carbon_factors_dir=self.root,
                                          hourly_carbon_fallback_to_annual=False, verbose=False)
        effect = pd.read_excel(output, sheet_name="Policy_Effect_Audit")
        audit = pd.read_excel(output, sheet_name="Grid_Input_Audit")
        self.assertEqual(len(audit), 3)
        self.assertTrue(effect["scalar_emissions_identity_holds"].all())
        self.assertTrue(effect["both_policies_have_full_hourly_inputs"].all())

    def test_carbon_variability_preserves_energy_and_annual_mean(self):
        self.write_input()
        reference = self.run_model()
        flat = self.run_model(carbon_variability_scale=0.)
        np.testing.assert_allclose(flat["annual_summary"]["facility_energy_mwh"],
                                   reference["annual_summary"]["facility_energy_mwh"], rtol=1e-12)
        row = flat["grid_carbon_audit"].iloc[0]
        self.assertAlmostEqual(row["annual_mean_residual_kg_per_mwh"], 0., delta=1e-6)
        self.assertAlmostEqual(row["hourly_matching_effect_tco2"], 0., delta=1e-6)

    def test_audit_detects_policy_shape_changes_despite_equal_means(self):
        for policy in ("CP", "NDC", "NZ"):
            path = self.write_input(policy=policy)
            if policy == "NZ":
                frame = pd.read_csv(path)
                frame[CI_COLUMNS["direct"]] = np.roll(frame[CI_COLUMNS["direct"]], 12)
                frame.to_csv(path, index=False)
        tables, failures = audit_grid_scenarios(self.root, countries=["USA"], years=[2028])
        self.assertEqual(failures, 1)
        self.assertTrue(tables["input_coverage"]["status"].eq("PASS").all())
        self.assertEqual(tables["hourly_policy_identity"].set_index("policy").loc["NZ", "status"], "ERROR")


if __name__ == "__main__":
    unittest.main()
