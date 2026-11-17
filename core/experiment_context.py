"""Read inputs once and share their aggregates within one experiment process."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from core.annual_cpu_model import run_annual_cpu_model
from core.grid_carbon import calendar_status, find_hourly_file, read_hourly_file, validate_direct_mean
from core.hourly_workload_resolved_model import (
    _load_hourly_carbon_factors, build_workload_profile, run_workload_component_footprint,
)


class SharedGridData:
    def __init__(self, context):
        self.context = context
        self.series_cache, self.array_cache, self.file_hashes = {}, {}, {}
        self.source_rows = []
        self.raw_csv_reads = 0

    def series(self, *, country, renewable_energy_policy, year, hourly_carbon_factors_dir, hourly_carbon_scope):
        lookup = (str(Path(hourly_carbon_factors_dir).resolve()), country, renewable_energy_policy, year, hourly_carbon_scope)
        if lookup not in self.series_cache:
            source = find_hourly_file(hourly_carbon_factors_dir, country, renewable_energy_policy, year)
            row = {"country": country, "policy": renewable_energy_policy, "year": year,
                   "scope": hourly_carbon_scope, "path": str(source.resolve()) if source else None,
                   "source_status": "hourly_file_missing"}
            self.source_rows.append(row)
            self.series_cache[lookup] = None
            if source is not None:
                payload = source.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                timestamps, factors = read_hourly_file(
                    source, year, hourly_carbon_scope, allow_legacy=True, frame=pd.read_csv(io.BytesIO(payload)))
                if hourly_carbon_scope == "direct":
                    validate_direct_mean(factors, country, renewable_energy_policy, year)
                factors.setflags(write=False)
                self.series_cache[lookup] = (timestamps, factors)
                self.file_hashes[str(source.resolve())] = digest
                self.raw_csv_reads += 1
                row.update(source_status="parsed_grid_csv_once", sha256=digest,
                           calendar_status=calendar_status(timestamps, year))
                entry = self.context.record.add_input(source, "hourly_grid_csv", hash_content=False, usage="parsed_once")
                entry.update(sha256=digest, hash_status="sha256_of_parsed_bytes")
        return self.series_cache[lookup]

    def load(self, **kwargs):
        key = (tuple(kwargs["countries"]), kwargs["renewable_energy_policy"], kwargs["year"],
               str(Path(kwargs["hourly_carbon_factors_dir"]).resolve()), kwargs["hourly_carbon_scope"],
               kwargs["hourly_carbon_fallback_to_annual"])
        if key not in self.array_cache:
            result = _load_hourly_carbon_factors(**kwargs, grid_data=self)
            result[1].setflags(write=False)
            result[2].setflags(write=False)
            self.array_cache[key] = result
        return self.array_cache[key]

    def fingerprint(self, path):
        return self.file_hashes[str(Path(path).resolve())]


class ExperimentContext:
    def __init__(self, config, record):
        self.config, self.record = config, record
        self.grid = SharedGridData(self)
        self.profile = None
        self.validation_hours = pd.DataFrame()
        self.raw_profile_builds = 0
        self._observed_hours, self._csv = {}, {}

    def prepare(self):
        cfg, common = self.config, self.config.common
        policies = set()
        for name in cfg.selected:
            if name == "figure2":
                policies.update(("CP", "NDC", "NZ"))
            elif name != "validation":
                policies.add(cfg.experiments[name].get("policy", "CP"))
        self.record.stage("shared_grid_inputs")
        for policy in sorted(policies):
            for year in range(common["year_start"], common["year_start"] + common["years"]):
                self.grid.load(countries=common["countries"], renewable_energy_policy=policy, year=year,
                               hourly_carbon_factors_dir=cfg.inputs["hourly_carbon_factors_dir"],
                               hourly_carbon_scope=common["hourly_carbon_scope"],
                               hourly_carbon_fallback_to_annual=not common["strict_hourly_carbon"])
        capture_validation = "validation" in cfg.selected and cfg.experiments["validation"]["trace_diagnostic"]
        if any(name != "validation" for name in cfg.selected) or capture_validation:
            self.record.stage("shared_workload_profile")
            self.profile = build_workload_profile(
                workload_profile_path=cfg.inputs["workload_profile_path"], server_profile_path=cfg.inputs["server_profile_path"],
                capacity_quantile=common["capacity_quantile"], trace_config=cfg.trace,
                max_intervals=common.get("max_intervals"), verbose=common["verbose"],
                parquet_batch_size=common["parquet_batch_size"], read_callback=self.record.trace_event,
                batch_observer=self.observe_batch if capture_validation else None)
            self.raw_profile_builds += 1
            for array in (self.profile.load, self.profile.trace_capacity, self.profile.task_counts):
                array.setflags(write=False)
        if capture_validation:
            self.validation_hours = pd.DataFrame([
                {"trace_day": day, "trace_hour": hour, **values}
                for (day, hour), values in sorted(self._observed_hours.items())])
            self.record.table("Shared_Validation_Hour_Aggregates", self.validation_hours)

    def observe_batch(self, frame, *, day, hour, path):
        from scripts.validate_energy_model import aggregate_trace_batch
        values = aggregate_trace_batch(frame)
        total = self._observed_hours.setdefault((day, hour), {name: 0 for name in values})
        for name, value in values.items():
            total[name] += value

    def run_hourly(self, **kwargs):
        return run_workload_component_footprint(**{
            "hardware_config": self.config.hardware, "pue_scale": self.config.common["pue_scale"],
            "idle_allocation_rule": self.config.common["idle_allocation_rule"],
            "workload_profile": self.profile, "grid_data": self.grid,
            "capacity_quantile": self.config.common["capacity_quantile"],
            "max_resource_utilization": self.config.common["max_resource_utilization"], **kwargs})

    def run_annual(self, **kwargs):
        return run_annual_cpu_model(**{
            "pue_scale": self.config.common["pue_scale"], "activity_config": self.config.activity, **kwargs})

    def csv(self, path):
        path = Path(path).resolve()
        if path not in self._csv:
            payload = path.read_bytes()
            row = self.record.add_input(path, "validation_csv", hash_content=False, usage="parsed_once")
            row.update(sha256=hashlib.sha256(payload).hexdigest(), hash_status="sha256_of_parsed_bytes")
            self._csv[path] = pd.read_csv(io.BytesIO(payload), low_memory=False)
        return self._csv[path].copy(), self.record.inputs[str(path)]["sha256"]

    def io_stats(self):
        return {"raw_profile_builds": self.raw_profile_builds, "grid_csv_parses": self.grid.raw_csv_reads,
                "validation_csv_parses": len(self._csv), "retained_raw_parquet_frames": 0}

    def figure_arguments(self):
        cfg, common = self.config, self.config.common
        return {"countries": common["countries"], "year_start": common["year_start"], "years": common["years"],
                "workload_profile_path": cfg.inputs["workload_profile_path"], "server_profile_path": cfg.inputs["server_profile_path"],
                "hourly_carbon_factors_dir": cfg.inputs["hourly_carbon_factors_dir"], "hourly_carbon_scope": common["hourly_carbon_scope"],
                "hourly_carbon_fallback_to_annual": not common["strict_hourly_carbon"],
                "capacity_quantile": common["capacity_quantile"], "max_resource_utilization": common["max_resource_utilization"],
                "max_intervals": common.get("max_intervals"), "verbose": common["verbose"],
                "ai_capacity_factors": cfg.factors, "experiment_context": self}

    def metadata_table(self):
        return pd.DataFrame([{"suite_run_id": self.record.run_id, **self.io_stats(),
                              "pue_scale": self.config.common["pue_scale"],
                              "hardware": json.dumps(asdict(self.config.hardware), sort_keys=True),
                              "trace": json.dumps(asdict(self.config.trace), sort_keys=True)}])
