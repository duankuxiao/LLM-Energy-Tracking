"""Calendar-based annual activity assumptions shared by both comparison paths.

All numerical coefficients below are author-defined reference assumptions.
The legacy mode reproduces the original 2025--2030 six-year denominator,
explicitly fixed at six rather than inferred from the requested output window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from core.task_model import TASK_TYPES

ACTIVITY_YEAR_START = 2025
ACTIVITY_YEAR_END = 2030
ACTIVITY_MODES = ("calendar", "fixed", "legacy_six_year")
REFERENCE_TASK_RATIOS = {"training": 0.20, "inference": 0.75, "other": 0.05, "unclassified": 0.0}
REFERENCE_TASK_UTILIZATION = {"training": 0.80, "inference": 0.50, "other": 0.50, "unclassified": 0.50}
REFERENCE_IDLE_POWER_RATE = 0.23
REFERENCE_MAX_POWER_RATE = 0.88


@dataclass(frozen=True)
class AnnualActivityConfig:
    mode: str = "calendar"
    training_start: float = 0.90
    training_target: float = 0.925
    inference_start: float = 0.50
    inference_target: float = 0.70

    def validate(self) -> None:
        if self.mode not in ACTIVITY_MODES:
            raise ValueError(f"Activity mode must be one of {ACTIVITY_MODES}.")
        values = (self.training_start, self.training_target, self.inference_start, self.inference_target)
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Activity endpoints must be finite and in [0, 1].")

    def vector(self, year: int) -> np.ndarray:
        self.validate()
        if not ACTIVITY_YEAR_START <= year <= ACTIVITY_YEAR_END:
            raise ValueError(f"Activity year must be within {ACTIVITY_YEAR_START}-{ACTIVITY_YEAR_END}.")
        denominator = ACTIVITY_YEAR_END - ACTIVITY_YEAR_START
        if self.mode == "legacy_six_year":
            denominator += 1
        progress = 0.0 if self.mode == "fixed" else (year - ACTIVITY_YEAR_START) / denominator
        training = self.training_start + (self.training_target - self.training_start) * progress
        inference = self.inference_start + (self.inference_target - self.inference_start) * progress
        return np.array([training if task == "training" else inference for task in TASK_TYPES], dtype=float)

    def table(self, years: Sequence[int]) -> pd.DataFrame:
        records = []
        for year in years:
            for task, value in zip(TASK_TYPES, self.vector(year)):
                records.append({
                    "year": year, "task_type": task, "activity_factor": value,
                    "activity_mode": self.mode,
                    "anchor_year_start": ACTIVITY_YEAR_START, "anchor_year_end": ACTIVITY_YEAR_END,
                    "denominator_years": (0 if self.mode == "fixed" else
                                          ACTIVITY_YEAR_END - ACTIVITY_YEAR_START + (self.mode == "legacy_six_year")),
                    "definition": ("fixed_at_2025" if self.mode == "fixed" else
                                   "calendar_endpoints" if self.mode == "calendar" else "historical_six_year_denominator"),
                    "parameter_status": "author_defined_reference_assumption",
                })
        return pd.DataFrame(records)


def reference_parameter_table(config: AnnualActivityConfig) -> pd.DataFrame:
    """Make the reference configuration and its evidence boundary inspectable."""
    config.validate()
    records = []
    for name, value in (
        ("training_activity_start", config.training_start), ("training_activity_target", config.training_target),
        ("inference_activity_start", config.inference_start), ("inference_activity_target", config.inference_target),
        ("idle_power_rate", REFERENCE_IDLE_POWER_RATE), ("max_power_rate", REFERENCE_MAX_POWER_RATE),
        *[(f"task_ratio_{task}", value) for task, value in REFERENCE_TASK_RATIOS.items()],
        *[(f"task_utilization_{task}", value) for task, value in REFERENCE_TASK_UTILIZATION.items()],
    ):
        records.append({"parameter": name, "value": value, "activity_mode": config.mode,
                        "parameter_status": "author_defined_reference_assumption",
                        "provenance": "local_reference_configuration; no external numerical validation asserted"})
    return pd.DataFrame(records)
