"""Leakage-safe feature builder (strictly timestamp <= t) for the 6-hour onset target.

Target definition
-----------------
* Predict organ dysfunction onset (first hour with SOFA increase >= 2 above
  baseline) within the next 6 hours: ``t < onset_hour <= t + 6``.
* Excludes stays whose onset occurs in the first 6 ICU hours (or hours after
  onset has already occurred).
* Every feature at hour ``t`` uses ONLY records with ``hours_from_icu <= t``.
"""

from __future__ import annotations

import pandas as pd

from src.ews.alerts import evaluate_ews
from src.ingest.loader import Cohort
from src.scores.aps3 import compute_aps3, extract_aps3_inputs
from src.scores.sofa import SOFA_COMPONENTS, compute_sofa, window_inputs

FEATURE_COLUMNS: tuple[str, ...] = (
    "sofa_total",
    "sofa_delta",
    "sofa_respiration",
    "sofa_coagulation",
    "sofa_liver",
    "sofa_cardiovascular",
    "sofa_cns",
    "sofa_renal",
    "sofa_completeness",
    "sofa_missing_count",
    "aps3_score",
    "aps3_available",
    "deterioration_count",
    "early_warning_flag",
    "vulnerability_elevated",
    "vulnerability_high",
    "icu_acquired_window",
    "device_count",
    "suspected_infection",
    "lactate_latest",
    "lactate_delta_6h",
)


def stay_events(cohort: Cohort, pseudonym: str) -> dict[str, pd.DataFrame]:
    """Slice all event frames for a single stay pseudonym."""
    out: dict[str, pd.DataFrame] = {}
    labs = cohort.labs[cohort.labs["pseudonym"] == pseudonym]
    for var in labs["variable"].unique():
        out[f"lab:{var}"] = labs[labs["variable"] == var]
    vit = cohort.vitals[cohort.vitals["pseudonym"] == pseudonym]
    for var in vit["variable"].unique():
        out[f"chart:{var}"] = vit[vit["variable"] == var]
    out["vaso:all"] = cohort.vasopressors[cohort.vasopressors["pseudonym"] == pseudonym]
    out["urine:all"] = cohort.urine[cohort.urine["pseudonym"] == pseudonym]
    out["devices:all"] = cohort.devices[cohort.devices["pseudonym"] == pseudonym]
    out["micro:all"] = cohort.micro[cohort.micro["pseudonym"] == pseudonym]
    out["anti:all"] = cohort.antimicrobials[cohort.antimicrobials["pseudonym"] == pseudonym]
    return out


def _latest_at(events: dict[str, pd.DataFrame], key: str, t_hours: float, lookback: float = 24.0) -> float | None:
    df = events.get(key)
    if df is None or len(df) == 0:
        return None
    sub = df[(df["hours_from_icu"] > t_hours - lookback) & (df["hours_from_icu"] <= t_hours)]
    if len(sub) == 0:
        return None
    return float(sub.loc[sub["hours_from_icu"].idxmax(), "value"])


def _lactate_delta_6h(events: dict[str, pd.DataFrame], t_hours: float) -> float | None:
    df = events.get("lab:lactate")
    if df is None or len(df) == 0:
        return None
    sub = df[(df["hours_from_icu"] >= t_hours - 6.0) & (df["hours_from_icu"] <= t_hours)]
    if len(sub) < 2:
        return None
    sub = sub.sort_values("hours_from_icu")
    return float(sub["value"].iloc[-1] - sub["value"].iloc[0])


def snapshot_at(
    cohort: Cohort,
    pseudonym: str,
    t_hours: float,
    *,
    events: dict[str, pd.DataFrame] | None = None,
    baseline_total: int | None = None,
) -> dict:
    """Compute a complete leakage-safe snapshot at hour ``t_hours`` (<= t_hours only)."""
    ev = events if events is not None else stay_events(cohort, pseudonym)
    stay_row = cohort.cohort[cohort.cohort["pseudonym"] == pseudonym].iloc[0]
    comorbs = cohort.comorbidities[cohort.comorbidities["pseudonym"] == pseudonym]["flag"].tolist()

    # Baseline uses only data up to min(t_hours, 24.0) so it never leaks future data.
    if baseline_total is None:
        base_inputs = window_inputs(ev, min(t_hours, 24.0), window_hours=24.0)
        baseline_total = compute_sofa(base_inputs).total

    sofa = compute_sofa(window_inputs(ev, t_hours, window_hours=24.0))
    sofa_delta = sofa.total - baseline_total

    aps3_inputs = extract_aps3_inputs(ev, t_hours, window_hours=24.0)
    aps3 = compute_aps3(aps3_inputs)

    lactate = _latest_at(ev, "lab:lactate", t_hours)
    lactate_d6 = _lactate_delta_6h(ev, t_hours)

    dev_df = ev.get("devices:all")
    devices_now = (
        dev_df[dev_df["hours_from_icu"] <= t_hours]["device"].unique().tolist()
        if dev_df is not None and len(dev_df)
        else []
    )
    micro_df = ev.get("micro:all")
    has_micro = bool(micro_df is not None and len(micro_df[micro_df["hours_from_icu"] <= t_hours]) > 0)
    anti_df = ev.get("anti:all")
    has_anti = bool(anti_df is not None and len(anti_df[anti_df["hours_from_icu"] <= t_hours]) > 0)

    ews = evaluate_ews(
        hour=t_hours,
        sofa_total=sofa.total,
        sofa_delta=sofa_delta,
        age=float(stay_row["age"]) if pd.notna(stay_row["age"]) else None,
        admission_type=str(stay_row["admission_type"]),
        comorbidities=comorbs,
        lactate=lactate,
        lactate_delta_6h=lactate_d6,
        respiratory_rate=aps3_inputs["respiratory_rate"],
        heart_rate=aps3_inputs["heart_rate"],
        temp_c=aps3_inputs["temp_c"],
        map_val=aps3_inputs["map"],
        wbc=aps3_inputs["wbc"],
        device_exposures=devices_now,
        has_micro_culture=has_micro,
        has_antimicrobial=has_anti,
    )

    features = {
        "sofa_total": float(sofa.total),
        "sofa_delta": float(sofa_delta),
        **{f"sofa_{c}": float(sofa.components[c]) for c in SOFA_COMPONENTS},
        "sofa_completeness": float(sofa.completeness),
        "sofa_missing_count": float(len(sofa.missing)),
        "aps3_score": float(aps3.score) if aps3.score is not None else 0.0,
        "aps3_available": 1.0 if aps3.score is not None else 0.0,
        "deterioration_count": float(ews.deterioration_count),
        "early_warning_flag": 1.0 if ews.early_warning_alert else 0.0,
        "vulnerability_elevated": 1.0 if ews.vulnerability_tier == "elevated" else 0.0,
        "vulnerability_high": 1.0 if ews.vulnerability_tier == "high" else 0.0,
        "icu_acquired_window": 1.0 if t_hours >= 48.0 else 0.0,
        "device_count": float(len(devices_now)),
        "suspected_infection": 1.0 if ews.suspected_infection else 0.0,
        "lactate_latest": float(lactate) if lactate is not None else 0.0,
        "lactate_delta_6h": float(lactate_d6) if lactate_d6 is not None else 0.0,
    }

    return {
        "pseudonym": pseudonym,
        "patient_id": str(stay_row["patient_id"]),
        "hour": float(t_hours),
        "sofa": sofa,
        "sofa_baseline": int(baseline_total),
        "sofa_delta": int(sofa_delta),
        "aps3": aps3,
        "ews": ews,
        "features": features,
    }


def build_feature_table(
    cohort: Cohort,
    *,
    step_hours: float = 4.0,
    max_stay_hours: float = 120.0,
    horizon_hours: float = 6.0,
    exclude_first_hours: float = 6.0,
) -> pd.DataFrame:
    """Build the leakage-safe modeling dataset across all ICU stays."""
    rows: list[dict] = []
    for _, stay in cohort.cohort.iterrows():
        pseudonym = stay["pseudonym"]
        patient_id = stay["patient_id"]
        los_hours = min(float(stay["los"]) * 24.0, max_stay_hours)
        if los_hours < exclude_first_hours:
            continue

        ev = stay_events(cohort, pseudonym)
        # First pass: hourly SOFA to find baseline and first onset (delta >= 2).
        hourly_scores: list[tuple[float, int]] = []
        h = 0.0
        while h <= los_hours + 1e-9:
            s = compute_sofa(window_inputs(ev, h, window_hours=24.0)).total
            hourly_scores.append((h, s))
            h += 1.0

        first_24 = [s for hr, s in hourly_scores if hr <= 24.0]
        baseline_24 = max(first_24) if first_24 else 0

        onset_hour: float | None = None
        for hr, s in hourly_scores:
            base_at_hr = max(sc for t0, sc in hourly_scores if t0 <= min(hr, 24.0))
            if s - base_at_hr >= 2:
                onset_hour = hr
                break

        # Exclude stays with onset in the first 6 ICU hours.
        if onset_hour is not None and onset_hour <= exclude_first_hours:
            continue

        # Sample prediction times t_hours strictly before onset (if any).
        t = exclude_first_hours
        while t <= los_hours - 1.0:
            if onset_hour is not None and t >= onset_hour:
                break
            base_t = max(sc for t0, sc in hourly_scores if t0 <= min(t, 24.0))
            snap = snapshot_at(cohort, pseudonym, t, events=ev, baseline_total=base_t)
            y = 1 if (onset_hour is not None and t < onset_hour <= t + horizon_hours) else 0
            row = {
                "patient_id": patient_id,
                "pseudonym": pseudonym,
                "hour": t,
                "onset_hour": onset_hour,
                "target_onset_6h": y,
                **snap["features"],
            }
            rows.append(row)
            t += step_hours

    return pd.DataFrame(rows)
