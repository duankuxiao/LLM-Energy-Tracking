"""UTC trace alignment and explicit, synthetic timing controls (W06).

Electricity controls operate AFTER the component power model. They preserve
country-year electricity, but make no claim about resource-hour conservation
or physical idle floors. Pre-power resource controls are reported separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.grid_carbon import SCOPES, calendar_status, matching_diagnostics, transform_shape


def trace_hour_mapping(interval_index, target_timestamps, relative_trace=False):
    """Map each target hour to the SAME sorted source hours used by alignment.

    ``*_before`` identifies the incoming edge at a target hour. The first
    hour has no incoming edge; an end-to-start annual ramp is never measured.
    Calendar copying, positional resampling and missing source hours are
    distinguished from a last-to-first relative-trace repeat.
    """
    source = pd.DatetimeIndex(interval_index).floor("h").unique().sort_values()
    target = pd.DatetimeIndex(target_timestamps)
    if not len(source) or not len(target) or source.isna().any() or target.isna().any():
        raise ValueError("Source and target hours must be nonempty valid timestamps.")
    if target.has_duplicates or not target.is_monotonic_increasing:
        raise ValueError("Target hours must be unique and increasing.")
    if len(target) > 1 and np.any(np.diff(target.asi8) != pd.Timedelta(hours=1).value):
        raise ValueError("Target timestamps must be consecutive UTC hours.")
    cycles = np.zeros(len(target), dtype=int)
    copied = np.zeros(len(target), dtype=bool)
    if relative_trace:
        positions = np.arange(len(target)) % len(source)
        cycles = np.arange(len(target)) // len(source)
        method = "relative_trace_repeat"
    elif source.equals(target):
        positions = np.arange(len(target))
        method = "exact_timestamp"
    else:
        keys = pd.MultiIndex.from_arrays([source.month, source.day, source.hour])
        target_keys = pd.MultiIndex.from_arrays([target.month, target.day, target.hour])
        positions = keys.get_indexer(target_keys) if keys.is_unique else None
        if positions is not None:
            lookup = {key: pos for pos, key in enumerate(keys)}
            for pos in np.flatnonzero(positions < 0):
                month, day, hour = target_keys[pos]
                if month == 2 and day == 29 and (2, 28, hour) in lookup:
                    positions[pos] = lookup[(2, 28, hour)]
                    copied[pos] = True
        if positions is None or np.any(positions < 0):
            positions = np.rint(np.linspace(0, len(source) - 1, len(target))).astype(int)
            copied[:] = False
            method = "position_resample"
        else:
            method = "month_day_hour_with_feb28_copy" if copied.any() else "month_day_hour"
    repeat = np.zeros(len(target), dtype=bool)
    discontinuity = np.zeros(len(target), dtype=bool)
    if len(target) > 1:
        repeat[1:] = relative_trace & (cycles[1:] != cycles[:-1])
        source_delta = np.diff(source.asi8[positions])
        discontinuity[1:] = (np.diff(positions) != 1) | (source_delta != pd.Timedelta(hours=1).value)
    return pd.DataFrame({
        "hour_index": np.arange(len(target)),
        "timestamp_utc": target.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_hour_position": positions,
        "source_timestamp": source[positions].astype(str),
        "source_repeat_cycle": cycles,
        "trace_repeat_seam_before": repeat,
        "alignment_discontinuity_before": discontinuity,
        "leap_day_copied": copied,
        "alignment_method": method,
        "source_hour_count": len(source),
        "source_missing_hour_edges": int(np.count_nonzero(np.diff(source.asi8) != pd.Timedelta(hours=1).value)),
    })


def _edge_mask(mask, length):
    result = np.zeros(length, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
    if result.shape != (length,):
        raise ValueError("Edge flags must have one entry per hour.")
    result[0] = False
    return result


def power_metrics(hourly_power_mw, seam_before=None, discontinuity_before=None):
    """Power is MW; adjacent one-hour absolute ramps are MW h^-1.

    P95/P99 use absolute adjacent-hour differences, not power percentiles.
    Empty filtered edge sets return NaN, not a misleading zero ramp.
    """
    values = np.asarray(hourly_power_mw, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Hourly power must be a finite non-negative nonempty vector.")
    seams = _edge_mask(seam_before, len(values))[1:]
    artifacts = _edge_mask(discontinuity_before, len(values))[1:] | seams
    differences = np.diff(values)
    ramps = np.abs(differences)
    clean = ramps[~seams]
    unbroken = ramps[~artifacts]
    mean = float(values.mean())
    maximum = float(ramps.max()) if len(ramps) else 0.0
    at_max = np.isclose(ramps, maximum, rtol=1e-10, atol=1e-12) if maximum > 0 else np.zeros(len(ramps), dtype=bool)
    return {
        "mean_power_mw": mean, "minimum_power_mw": float(values.min()),
        "peak_power_mw": float(values.max()), "p99_power_mw": float(np.quantile(values, .99)),
        "peak_to_mean_ratio": float(values.max() / mean) if mean else np.nan,
        "max_ramp_up_mw_per_h": max(float(differences.max()), 0.) if len(ramps) else 0.,
        "max_ramp_down_mw_per_h": min(float(differences.min()), 0.) if len(ramps) else 0.,
        "max_abs_ramp_mw_per_h": maximum,
        "max_abs_ramp_pct_of_mean": maximum * 100 / mean if mean else np.nan,
        "max_abs_ramp_excluding_seams_mw_per_h": float(clean.max()) if len(clean) else np.nan,
        "max_abs_ramp_excluding_alignment_mw_per_h": float(unbroken.max()) if len(unbroken) else np.nan,
        "p95_abs_ramp_mw_per_h": float(np.quantile(ramps, .95)) if len(ramps) else np.nan,
        "p99_abs_ramp_mw_per_h": float(np.quantile(ramps, .99)) if len(ramps) else np.nan,
        "p95_abs_ramp_excluding_seams_mw_per_h": float(np.quantile(clean, .95)) if len(clean) else np.nan,
        "p99_abs_ramp_excluding_seams_mw_per_h": float(np.quantile(clean, .99)) if len(clean) else np.nan,
        "repeat_seam_count": int(seams.sum()), "alignment_discontinuity_count": int(artifacts.sum()),
        "raw_ramp_edge_count": len(ramps), "nonseam_ramp_edge_count": len(clean),
        "max_seam_abs_ramp_mw_per_h": float(ramps[seams].max()) if seams.any() else np.nan,
        "raw_max_occurs_at_repeat_seam": bool(np.any(at_max & seams)),
        "raw_max_occurs_at_alignment_discontinuity": bool(np.any(at_max & artifacts)),
    }


def shared_shape_diagnostics(country_hourly_energy):
    values = np.asarray(country_hourly_energy, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Country profiles must be a finite non-negative country-by-hour array.")
    means = values.mean(axis=1)
    positive = means > 0
    normalized = values[positive] / means[positive, None]
    deviation = float(np.max(np.abs(normalized - normalized[[0]]))) if positive.any() else np.nan
    return {"country_count": len(values), "positive_energy_country_count": int(positive.sum()),
            "all_positive_country_shapes_identical": bool(positive.any() and deviation <= 1e-9),
            "maximum_normalized_shape_deviation": deviation,
            "country_shape_reference": "first_positive_energy_country_in_input_order",
            "country_behavior_evidence": "shared_trace_or_synthetic_stress_not_independent_national_predictions"}


def seam_event_records(values, timestamps, seam_before, discontinuity_before, context, alignment=None):
    values = np.asarray(values, dtype=float)
    seams = _edge_mask(seam_before, len(values))
    artifacts = _edge_mask(discontinuity_before, len(values)) | seams
    times = pd.DatetimeIndex(timestamps)
    records = [{**context, "hour_index": int(pos), "timestamp_utc": times[pos].isoformat(),
             "trace_repeat_seam_before": bool(seams[pos]),
             "alignment_discontinuity_before": bool(artifacts[pos]),
             "power_before_mw": float(values[pos - 1]), "power_after_mw": float(values[pos]),
             "absolute_ramp_mw_per_h": float(abs(values[pos] - values[pos - 1]))}
            for pos in np.flatnonzero(artifacts)]
    if alignment is not None:
        for row in records:
            pos = row["hour_index"]
            for column in ("source_hour_position", "source_timestamp", "source_repeat_cycle", "alignment_method"):
                row[column] = alignment[column].iloc[pos]
            row["previous_source_hour_position"] = int(alignment["source_hour_position"].iloc[pos - 1])
    return records


def _shift_edges(mask, shift):
    shifted = np.roll(mask, shift).copy()
    shifted[0] = False
    return shifted


def _electricity_control(values, seams, artifacts, shift, variability, smooth):
    result = transform_shape(values, variability_scale=variability, phase_hours=shift)
    seams = _shift_edges(seams, shift)
    artifacts = _shift_edges(artifacts, shift)
    # Circular phase shifts create a finite-calendar end/start join as well
    # as moving the original repeat seams. Exclude this additional join too.
    join = shift % len(values)
    if join:
        artifacts[join] = True
    if smooth > 1:
        half = smooth // 2
        result = np.convolve(np.pad(result, (half, half), mode="edge"), np.ones(smooth) / smooth, mode="valid")
        if result.sum() > 0:
            result *= values.sum() / result.sum()
        # Keep the actual repeat locations; expand only the broader artifact
        # mask to cover every edge influenced by smoothing a discontinuity.
        artifacts = np.convolve(artifacts.astype(int), np.ones(smooth, dtype=int), mode="same") > 0
        artifacts[0] = False
    return result, seams, artifacts


def timing_stress_tables(hourly, *, policy, carbon_scope):
    """Reuse captured hourly energy/CI; no trace reads or power-model reruns.

    Country offsets and variability assignments are deterministic stress
    assumptions, deliberately NOT country time-zone or behavioral estimates.
    Positive phase shifts move a series later by that many UTC hours.
    """
    if carbon_scope not in SCOPES:
        raise ValueError(f"Carbon scope must be one of {SCOPES}.")
    required = {"scenario", "year", "country", "timestamp_utc", "facility_energy_mwh",
                "carbon_factor_kg_per_mwh", "carbon_factor_source", "trace_repeat_seam_before",
                "alignment_discontinuity_before"}
    if hourly.empty or not required.issubset(hourly.columns):
        raise ValueError(f"Timing controls require a nonempty hourly table with {sorted(required)}.")
    # case, common load shift, CI shift, load variability, CI variability,
    # smoothing window, heterogeneous phase, heterogeneous variability, CI mean multiplier
    cases = (
        ("baseline", 0, 0, 1., 1., 1, False, False, 1.),
        ("load_phase_minus6h", -6, 0, 1., 1., 1, False, False, 1.),
        ("load_phase_plus6h", 6, 0, 1., 1., 1, False, False, 1.),
        ("load_phase_plus12h", 12, 0, 1., 1., 1, False, False, 1.),
        ("country_phase_spread", 0, 0, 1., 1., 1, True, False, 1.),
        ("country_shape_spread", 0, 0, 1., 1., 1, False, True, 1.),
        ("load_variability_0p5", 0, 0, .5, 1., 1, False, False, 1.),
        ("load_variability_1p5", 0, 0, 1.5, 1., 1, False, False, 1.),
        ("load_smoothing_3h", 0, 0, 1., 1., 3, False, False, 1.),
        ("ci_phase_minus6h", 0, -6, 1., 1., 1, False, False, 1.),
        ("ci_phase_plus6h", 0, 6, 1., 1., 1, False, False, 1.),
        ("joint_phase_plus6h", 6, 6, 1., 1., 1, False, False, 1.),
        ("load_plus6h_ci_minus6h", 6, -6, 1., 1., 1, False, False, 1.),
        ("ci_flat_annual_mean", 0, 0, 1., 0., 1, False, False, 1.),
        ("ci_variability_1p5", 0, 0, 1., 1.5, 1, False, False, 1.),
        ("ci_mean_minus10pct", 0, 0, 1., 1., 1, False, False, .9),
        ("ci_mean_plus10pct", 0, 0, 1., 1., 1, False, False, 1.1),
    )
    country_rows, aggregate_rows, parameter_rows, events, metadata = [], [], [], [], []
    for (scenario, year), group in hourly.groupby(["scenario", "year"], sort=True):
        if group.duplicated(["country", "timestamp_utc"]).any():
            raise ValueError("Duplicate country-hour in timing controls.")
        energy = group.pivot(index="timestamp_utc", columns="country", values="facility_energy_mwh").sort_index()
        countries = list(energy.columns)
        country_offsets = np.rint(np.linspace(-12, 12, len(countries))).astype(int) if len(countries) > 1 else np.zeros(1, dtype=int)
        times = pd.DatetimeIndex(pd.to_datetime(energy.index, utc=True))
        status = calendar_status(times, int(year))
        if status == "invalid_calendar":
            raise ValueError("Timing controls require a full or explicitly legacy UTC calendar.")
        def pivot(column):
            frame = group.pivot(index="timestamp_utc", columns="country", values=column).reindex(index=energy.index, columns=countries)
            if frame.isna().any().any():
                raise ValueError(f"Missing aligned country/hour values in {column}.")
            return frame.to_numpy().T
        base_energy = pivot("facility_energy_mwh").astype(float)
        base_ci = pivot("carbon_factor_kg_per_mwh").astype(float)
        if not np.isfinite(base_energy).all() or not np.isfinite(base_ci).all() or np.any(base_energy < 0) or np.any(base_ci < 0):
            raise ValueError("Timing inputs must be finite and non-negative.")
        base_seams = pivot("trace_repeat_seam_before").astype(bool)
        base_artifacts = pivot("alignment_discontinuity_before").astype(bool)
        sources = pivot("carbon_factor_source")
        base_carbon = np.sum(base_energy * base_ci, axis=1) / 1000.
        base_peaks = np.max(base_energy, axis=1)
        base_aggregate_peak = float(base_energy.sum(axis=0).max())
        for case, load_shift, ci_shift, load_var, ci_var, smooth, hetero_phase, hetero_shape, ci_multiplier in cases:
            context = {"scenario": scenario, "policy": policy, "year": int(year), "timing_case": case,
                       "carbon_factor_scope": carbon_scope,
                       "hours": len(times), "calendar_status": status,
                       "control_stage": "post_power_fixed_country_year_electricity",
                       "resource_hours_constraint": "not_modelled_in_post_power_control",
                       "is_synthetic_stress": case != "baseline", "power_unit": "MW", "ramp_unit": "MW h^-1",
                       "ramp_definition": "Maximum absolute hourly ramp",
                       "alignment_filter_includes_smoothing_influence": smooth > 1,
                       "smoothing_is_observation": False}
            case_energy, case_seams, case_artifacts = [], [], []
            carbon_total = mean_carbon_total = matching_total = covariance_total = 0.
            preserved_ci = []
            for country_id, country in enumerate(countries):
                shift = int(country_offsets[country_id]) if hetero_phase else load_shift
                variability = (.5, 1., 1.5)[country_id % 3] if hetero_shape else load_var
                values, seams, artifacts = _electricity_control(
                    base_energy[country_id], base_seams[country_id], base_artifacts[country_id], shift, variability, smooth)
                factors = transform_shape(base_ci[country_id], phase_hours=ci_shift,
                                          variability_scale=ci_var, target_mean=base_ci[country_id].mean() * ci_multiplier)
                total = float(base_energy[country_id].sum())
                diagnostic = matching_diagnostics(values, factors, total)
                electricity_preserved = bool(np.isclose(values.sum(), total, rtol=1e-10, atol=1e-6))
                ci_preserved = bool(np.isclose(factors.mean(), base_ci[country_id].mean(), rtol=1e-10, atol=1e-8))
                if not electricity_preserved or (ci_multiplier == 1 and not ci_preserved):
                    raise ValueError("Fixed-electricity / fixed-mean CI timing control failed to close.")
                if case == "joint_phase_plus6h" and not np.isclose(diagnostic["matched_carbon_tco2"], base_carbon[country_id], rtol=1e-10, atol=1e-6):
                    raise ValueError("Joint equal circular shifts must preserve matched emissions.")
                country_context = {**context, "scope": "country", "country": country,
                                   "carbon_factor_source": str(sources[country_id, 0]),
                                   "eligible_for_full_calendar_hourly_claim": bool(status == "full_calendar" and np.all(sources[country_id] == "hourly"))}
                country_rows.append({**country_context, **power_metrics(values, seams, artifacts), **diagnostic,
                                     "country_year_electricity_preserved": electricity_preserved,
                                     "country_year_mean_ci_preserved": ci_preserved,
                                     "baseline_carbon_tco2": float(base_carbon[country_id]),
                                     "carbon_difference_tco2": diagnostic["matched_carbon_tco2"] - base_carbon[country_id],
                                     "peak_difference_mw": float(values.max() - base_peaks[country_id])})
                parameter_rows.append({**country_context, "load_phase_hours": shift, "ci_phase_hours": ci_shift,
                                       "load_variability_scale": variability, "ci_variability_scale": ci_var,
                                       "ci_mean_multiplier": ci_multiplier, "smoothing_window_hours": smooth,
                                       "heterogeneity_basis": "sorted_country_order_synthetic_no_timezone_or_behavior_evidence",
                                       "baseline_annual_energy_mwh": total,
                                       "baseline_mean_ci_kg_per_mwh": float(base_ci[country_id].mean()),
                                       "expected_electricity_preserved": True, "expected_mean_ci_preserved": ci_multiplier == 1})
                events.extend(seam_event_records(values, times, seams, artifacts, country_context))
                case_energy.append(values)
                case_seams.append(seams)
                case_artifacts.append(artifacts)
                carbon_total += diagnostic["matched_carbon_tco2"]
                mean_carbon_total += diagnostic["mean_ci_carbon_tco2"]
                matching_total += diagnostic["hourly_matching_effect_tco2"]
                covariance_total += diagnostic["covariance_identity_residual_tco2"]
                preserved_ci.append(ci_preserved)
            values = np.asarray(case_energy)
            aggregate = values.sum(axis=0)  # Same UTC timestamp, BEFORE max/diff.
            seams = np.any(case_seams, axis=0)
            artifacts = np.any(case_artifacts, axis=0)
            aggregate_context = {**context, "scope": "aggregate", "country": "ALL",
                                 "carbon_factor_source": "all_hourly" if np.all(sources == "hourly") else "contains_annual_fallback",
                                 "eligible_for_full_calendar_hourly_claim": bool(status == "full_calendar" and np.all(sources == "hourly"))}
            sum_peaks = float(values.max(axis=1).sum())
            aggregate_rows.append({**aggregate_context, **power_metrics(aggregate, seams, artifacts),
                                   **shared_shape_diagnostics(values),
                                   "peak_construction": "sum_country_power_at_same_UTC_hour_then_max",
                                   "sum_of_individual_country_peaks_mw": sum_peaks,
                                   "peak_coincidence_ratio": float(aggregate.max() / sum_peaks) if sum_peaks else np.nan,
                                   "peak_difference_mw": float(aggregate.max() - base_aggregate_peak),
                                   "annual_energy_mwh": float(aggregate.sum()),
                                   "energy_closure_residual_mwh": float(aggregate.sum() - base_energy.sum()),
                                   "country_year_electricity_preserved": True,
                                   "country_year_mean_ci_preserved": all(preserved_ci),
                                   "matched_carbon_tco2": carbon_total, "mean_ci_carbon_tco2": mean_carbon_total,
                                   "hourly_matching_effect_tco2": matching_total,
                                   "covariance_identity_residual_tco2": covariance_total,
                                   "baseline_carbon_tco2": float(base_carbon.sum()),
                                   "carbon_difference_tco2": carbon_total - base_carbon.sum(),
                                   "aggregate_carbon_construction": "sum_country_hourly_energy_times_own_CI_not_mean_country_CI"})
            events.extend(seam_event_records(aggregate, times, seams, artifacts, aggregate_context))
        metadata.append({"scenario": scenario, "policy": policy, "year": int(year), "hours": len(times),
                         "carbon_factor_scope": carbon_scope,
                         "calendar_status": status, "country_count": len(countries), "case_count": len(cases),
                         "electricity_convention": "8760_hour_annualized_trace_energy_distributed_over_CI_calendar",
                         "phase_convention": "circular_positive_shift_moves_later_UTC_wrap_join_flagged",
                         "filtered_ramp_convention": "exclude_incoming_edges_marked_before_no_annual_end_start_ramp",
                         "heterogeneity_evidence": "synthetic_stress_only_not_independent_country_prediction",
                         "post_power_variability_can_change_idle_floor": True,
                         "aggregate_covariance_convention": "sum_country_matching_effects_each_against_own_mean_CI"})
    return {"W06_Timing_Country": pd.DataFrame(country_rows),
            "W06_Timing_Aggregate": pd.DataFrame(aggregate_rows),
            "W06_Timing_Parameters": pd.DataFrame(parameter_rows),
            "W06_Seam_Events": pd.DataFrame(events),
            "W06_Timing_Metadata": pd.DataFrame(metadata)}
