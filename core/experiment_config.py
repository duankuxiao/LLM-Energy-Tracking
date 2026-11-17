"""Read the settings used by the paper's experiment suite."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from core.accounting_boundary import load_ai_capacity_factors, resolve_countries
from core.annual_activity import AnnualActivityConfig
from core.annual_reference_controls import validate_reference_sensitivity
from core.hourly_workload_resolved_model import Alibaba2026TraceConfig, HardwarePowerConfig
from dataset.Installed_capacity_data import DEFAULT_COUNTRIES

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_NAMES = ("figure1", "figure2", "figure3", "figure4", "comparison", "boundary", "validation")
SENSITIVITY_DEFAULTS = {
    "unit_power_scales": [.8, 1., 1.2], "dynamic_exponent_scales": [.75, 1., 1.25],
    "resource_variability_scales": [.5, 1., 1.5], "gpu_utilization_scales": [.8, 1., 1.2],
    "task_mix_inference_shifts": [-.3, 0., .3], "carbon_variability_scales": [.5, 1., 1.5],
    "idle_ratio_scales": [.5, 1., 1.5], "component_budget_scales": [.8, 1., 1.2],
}


def _path(value):
    if value is None:
        return None
    path = Path(value)
    return (path if path.is_absolute() else ROOT_DIR / path).resolve()


@dataclass
class ExperimentConfig:
    path: Path
    sha256: str
    document: dict
    inputs: dict
    common: dict
    experiments: dict
    selected: tuple
    hardware: HardwarePowerConfig
    trace: Alibaba2026TraceConfig
    activity: AnnualActivityConfig
    factors: dict | None
    output_dir: Path


def load_experiment_config(path, *, only=None, output_dir=None) -> ExperimentConfig:
    path = _path(path)
    payload = path.read_bytes()
    data = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Experiment configuration must be a JSON object.")
    for section in ("inputs", "common", "experiments", "output"):
        if not isinstance(data.get(section), dict):
            raise ValueError(f"Experiment configuration requires an object named {section}.")
    inputs = {key: _path(value) for key, value in data["inputs"].items()}
    common = dict(data["common"])
    common["countries"] = resolve_countries(common.get("countries"))
    if set(data["experiments"]) - set(EXPERIMENT_NAMES):
        raise ValueError("Unknown experiment name in configuration.")
    experiments = {}
    for name, entry in data["experiments"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("enabled", False), bool):
            raise ValueError(f"Experiment {name} requires an object and a boolean enabled flag.")
        experiments[name] = dict(entry)
    if only is not None and (isinstance(only, str) or not isinstance(only, (tuple, list))):
        raise ValueError("Experiment selection must be a list of names.")
    selected = tuple(only) if only is not None else tuple(
        name for name in EXPERIMENT_NAMES if experiments.get(name, {}).get("enabled", False))
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(EXPERIMENT_NAMES):
        raise ValueError("Select at least one unique, known experiment name.")
    if set(selected) - set(experiments):
        raise ValueError("Every selected experiment requires a configuration entry.")
    for name in ("year_start", "years", "parquet_batch_size"):
        if isinstance(common.get(name), bool) or not isinstance(common.get(name), int):
            raise ValueError(f"Common setting {name} must be an integer.")
    if common["years"] <= 0 or not 2025 <= common["year_start"] <= common["year_start"] + common["years"] - 1 <= 2030:
        raise ValueError("Experiment years must be within 2025–2030.")
    if common["parquet_batch_size"] <= 0:
        raise ValueError("parquet_batch_size must be positive.")
    for name in ("capacity_quantile", "max_resource_utilization", "pue_scale"):
        value = common.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Common setting {name} must be finite and positive.")
    if common["capacity_quantile"] > 1:
        raise ValueError("capacity_quantile must be at most one.")
    for name in ("strict_hourly_carbon", "hash_inputs", "verbose"):
        if not isinstance(common.get(name), bool):
            raise ValueError(f"Common setting {name} must be a boolean.")
    if "figure3" in experiments:
        experiments["figure3"]["reference_sensitivity"] = validate_reference_sensitivity(
            experiments["figure3"].get("reference_sensitivity"))
    # These conditions define the comparisons in the paper.
    if common["hourly_carbon_scope"] != "direct":
        raise ValueError("The annual/hourly comparison requires direct carbon factors.")
    if "figure3" in selected and (common["year_start"] != 2026 or common["years"] != 5
                                  or set(common["countries"]) != set(DEFAULT_COUNTRIES)):
        raise ValueError("Figure 3 uses all 24 countries and 2026–2030.")
    trace_values = dict(data.get("trace", {}))
    if "active_states" in trace_values:
        trace_values["active_states"] = tuple(trace_values["active_states"])
    factors_path = inputs.get("ai_capacity_factors_file")
    hardware = HardwarePowerConfig(**data.get("hardware", {}))
    hardware.validate()
    activity = AnnualActivityConfig(**data.get("annual_activity", {}))
    activity.validate()
    return ExperimentConfig(
        path=path, sha256=hashlib.sha256(payload).hexdigest(), document=data,
        inputs=inputs, common=common, experiments=experiments, selected=selected,
        hardware=hardware,
        trace=Alibaba2026TraceConfig(**trace_values),
        activity=activity,
        factors=load_ai_capacity_factors(factors_path, common["countries"]) if factors_path else None,
        output_dir=_path(output_dir if output_dir is not None else data["output"]["directory"]),
    )
