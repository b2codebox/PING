"""Tiered Early Warning System (EWS) & Sepsis/Organ Dysfunction Alert Engine.

Three tiers:
1. Early Warning Alert (Pre-SOFA / Deterioration):
   Flags acute physiological/lab deterioration (rising lactate, tachypnea,
   tachycardia, fever/hypothermia, borderline MAP, abnormal WBC) stratified by
   baseline vulnerability and pre-existing (<48h) vs. ICU-acquired (>=48h)
   infection context.
2. Organ Dysfunction Alert:
   SOFA increase >= 2 points above baseline.
3. Possible Sepsis Alert:
   Organ dysfunction alert + suspected-infection flag (otherwise labeled
   "organ dysfunction risk" or "severity-based sepsis risk").

All thresholds are prototype values marked with # TODO: verify.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.ingest.allowlist import DISCLAIMER

# ---------------------------------------------------------------------------
# Prototype EWS thresholds (# TODO: verify with local ICU clinicians)
# ---------------------------------------------------------------------------
LACTATE_HIGH = 2.0          # mmol/L  # TODO: verify
LACTATE_DELTA_6H = 0.5      # mmol/L rise over 6h  # TODO: verify
RR_HIGH = 22.0              # breaths/min (qSOFA-aligned)  # TODO: verify
HR_HIGH = 100.0             # bpm  # TODO: verify
TEMP_HIGH_C = 38.3          # C  # TODO: verify
TEMP_LOW_C = 36.0           # C  # TODO: verify
MAP_BORDERLINE = 70.0       # mmHg  # TODO: verify
WBC_HIGH = 12.0             # K/uL  # TODO: verify
WBC_LOW = 4.0               # K/uL  # TODO: verify
ELDERLY_AGE = 65.0          # years  # TODO: verify
ICU_ACQUIRED_HOURS = 48.0   # >=48h -> ICU-acquired infection window  # TODO: verify
SOFA_DELTA_ALERT = 2        # >=2 points above baseline (Sepsis-3)


@dataclass
class EWSAssessment:
    disclaimer: str = DISCLAIMER
    hour: float = 0.0
    # 1. Vulnerability context
    vulnerability_factors: list[str] = field(default_factory=list)
    vulnerability_tier: str = "standard"  # "standard" | "elevated" | "high"
    # 2. Early physiological & lab deterioration signals
    deterioration_signals: list[str] = field(default_factory=list)
    deterioration_count: int = 0
    # 3. Infection context (pre-existing vs ICU-acquired)
    infection_window: str = "pre_existing_or_early (<48h)"
    suspected_infection: bool = False
    infection_factors: list[str] = field(default_factory=list)
    device_exposures: list[str] = field(default_factory=list)
    # Tiered alerts
    early_warning_alert: bool = False
    organ_dysfunction_alert: bool = False
    possible_sepsis_alert: bool = False
    alert_tier: str = "none"  # "none" | "early_warning" | "organ_dysfunction" | "possible_sepsis"
    risk_label: str = "low_risk"
    summary: str = ""


def evaluate_ews(
    *,
    hour: float,
    sofa_total: int,
    sofa_delta: int,
    age: float | None = None,
    admission_type: str | None = None,
    comorbidities: list[str] | None = None,
    lactate: float | None = None,
    lactate_delta_6h: float | None = None,
    respiratory_rate: float | None = None,
    heart_rate: float | None = None,
    temp_c: float | None = None,
    map_val: float | None = None,
    wbc: float | None = None,
    device_exposures: list[str] | None = None,
    has_micro_culture: bool = False,
    has_antimicrobial: bool = False,
    clinician_suspected_infection: bool = False,
) -> EWSAssessment:
    """Evaluate tiered alerts from leakage-safe inputs at time ``hour``."""
    comorbidities = list(comorbidities or [])
    device_exposures = sorted(set(device_exposures or []))

    # --- 1. Vulnerability context ------------------------------------------
    vuln: list[str] = []
    if age is not None and not pd.isna(age) and age >= ELDERLY_AGE:
        vuln.append(f"Age >= {int(ELDERLY_AGE)} ({age:.0f}y)")
    if admission_type and any(k in admission_type.upper() for k in ("EMER", "URGENT")):
        vuln.append(f"Emergency/Urgent admission ({admission_type})")
    if "immunosuppression" in comorbidities:
        vuln.append("Immunosuppression (ICD/flag)")
    for c in comorbidities:
        if c != "immunosuppression":
            vuln.append(f"Comorbidity: {c.replace('_', ' ')}")

    if "immunosuppression" in comorbidities or len(vuln) >= 3:
        vuln_tier = "high"
    elif len(vuln) >= 1:
        vuln_tier = "elevated"
    else:
        vuln_tier = "standard"

    # --- 2. Early physiological & lab deterioration ------------------------
    signals: list[str] = []
    if lactate is not None and not pd.isna(lactate) and lactate >= LACTATE_HIGH:
        signals.append(f"Elevated lactate ({lactate:.2f} mmol/L)")
    if lactate_delta_6h is not None and not pd.isna(lactate_delta_6h) and lactate_delta_6h >= LACTATE_DELTA_6H:
        signals.append(f"Rising lactate (+{lactate_delta_6h:.2f} mmol/L / 6h)")
    if respiratory_rate is not None and not pd.isna(respiratory_rate) and respiratory_rate >= RR_HIGH:
        signals.append(f"Tachypnea (RR {respiratory_rate:.0f}/min)")
    if heart_rate is not None and not pd.isna(heart_rate) and heart_rate >= HR_HIGH:
        signals.append(f"Tachycardia (HR {heart_rate:.0f} bpm)")
    if temp_c is not None and not pd.isna(temp_c):
        if temp_c >= TEMP_HIGH_C:
            signals.append(f"Fever ({temp_c:.1f} C)")
        elif temp_c < TEMP_LOW_C:
            signals.append(f"Hypothermia ({temp_c:.1f} C)")
    if map_val is not None and not pd.isna(map_val) and map_val < MAP_BORDERLINE:
        signals.append(f"Borderline/Low MAP ({map_val:.0f} mmHg)")
    if wbc is not None and not pd.isna(wbc):
        if wbc > WBC_HIGH:
            signals.append(f"Leukocytosis (WBC {wbc:.1f} K/uL)")
        elif wbc < WBC_LOW:
            signals.append(f"Leukopenia (WBC {wbc:.1f} K/uL)")

    # --- 3. Infection context: pre-existing (<48h) vs ICU-acquired (>=48h) -
    if hour >= ICU_ACQUIRED_HOURS:
        inf_window = "icu_acquired_risk (>=48h)"
    else:
        inf_window = "pre_existing_or_early (<48h)"

    inf_factors: list[str] = []
    if has_micro_culture:
        inf_factors.append("Structured microbiology culture ordered/collected")
    if has_antimicrobial:
        inf_factors.append("Structured antimicrobial started")
    if clinician_suspected_infection:
        inf_factors.append("Clinician-entered suspected infection flag")
    if device_exposures and hour >= ICU_ACQUIRED_HOURS:
        inf_factors.append(f"Invasive device exposure >=48h ({', '.join(device_exposures)})")

    suspected_infection = bool(
        clinician_suspected_infection or (has_micro_culture and has_antimicrobial)
    )

    # --- Tiered Alert Logic ------------------------------------------------
    # High vulnerability lowers the threshold to 1 deterioration signal.
    ews_threshold = 1 if vuln_tier == "high" else 2
    early_warning = len(signals) >= ews_threshold
    organ_dysfunction = sofa_delta >= SOFA_DELTA_ALERT
    possible_sepsis = organ_dysfunction and suspected_infection

    if possible_sepsis:
        tier = "possible_sepsis"
        risk_label = "Possible Sepsis Alert (SOFA delta >= 2 + suspected infection)"
    elif organ_dysfunction:
        tier = "organ_dysfunction"
        risk_label = "Organ Dysfunction Risk (SOFA delta >= 2; no confirmed infection flag)"
    elif early_warning:
        tier = "early_warning"
        risk_label = "Early Warning Alert (Pre-SOFA physiological/lab deterioration)"
    else:
        tier = "none"
        risk_label = "Low / Routine Monitoring"

    summary = (
        f"Tier={tier} | SOFA={sofa_total} (delta {sofa_delta:+d}) | "
        f"Vulnerability={vuln_tier} | Signals={len(signals)} | Window={inf_window}"
    )

    return EWSAssessment(
        disclaimer=DISCLAIMER,
        hour=hour,
        vulnerability_factors=vuln,
        vulnerability_tier=vuln_tier,
        deterioration_signals=signals,
        deterioration_count=len(signals),
        infection_window=inf_window,
        suspected_infection=suspected_infection,
        infection_factors=inf_factors,
        device_exposures=device_exposures,
        early_warning_alert=early_warning,
        organ_dysfunction_alert=organ_dysfunction,
        possible_sepsis_alert=possible_sepsis,
        alert_tier=tier,
        risk_label=risk_label,
        summary=summary,
    )
