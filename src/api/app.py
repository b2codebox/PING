"""FastAPI service for the Sepsis Onset, Organ Dysfunction & Early Warning prototype.

Endpoints:
* GET  /health
* GET  /patients
* GET  /patients/{stay_id}/scores
* GET  /patients/{stay_id}/risk
* POST /assess
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.ews.alerts import evaluate_ews
from src.features.builder import snapshot_at, stay_events
from src.ingest.allowlist import DISCLAIMER
from src.ingest.loader import Cohort, load_cache
from src.model.baseline import load_artifacts, predict_with_drivers
from src.scores.aps3 import compute_aps3
from src.scores.sofa import SOFA_COMPONENTS, compute_sofa, window_inputs

ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(
    title="Sepsis & Early Warning Prototype API",
    description=DISCLAIMER,
    version="0.1.0",
)


@lru_cache(maxsize=1)
def _get_cohort() -> Cohort:
    return load_cache(ROOT / "data" / "processed")


@lru_cache(maxsize=1)
def _get_model():
    return load_artifacts()


@app.get("/health")
def health():
    cohort = _get_cohort()
    _, metrics = _get_model()
    return {
        "status": "ok",
        "disclaimer": DISCLAIMER,
        "stays_loaded": int(len(cohort.cohort)),
        "patients_loaded": int(cohort.cohort["patient_id"].nunique()),
        "model_metrics": metrics,
    }


@app.get("/patients")
def list_patients():
    cohort = _get_cohort()
    rows = []
    for _, r in cohort.cohort.iterrows():
        rows.append(
            {
                "stay_id": r["pseudonym"],
                "patient_id": r["patient_id"],
                "age": None if r["age"] != r["age"] else round(float(r["age"]), 1),
                "gender": r["gender"],
                "admission_type": r["admission_type"],
                "first_careunit": r["first_careunit"],
                "los_hours": round(float(r["los"]) * 24.0, 1),
            }
        )
    return {"disclaimer": DISCLAIMER, "count": len(rows), "patients": rows}


@app.get("/patients/{stay_id}/scores")
def patient_scores(stay_id: str, step_hours: int = 4, max_hours: int = 72):
    cohort = _get_cohort()
    match = cohort.cohort[cohort.cohort["pseudonym"] == stay_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown pseudonymised stay_id: {stay_id}")

    los_hours = min(float(match.iloc[0]["los"]) * 24.0, float(max_hours))
    ev = stay_events(cohort, stay_id)

    trajectory = []
    h = 0.0
    baseline_so_far = 0
    while h <= los_hours + 1e-9:
        sofa = compute_sofa(window_inputs(ev, h, window_hours=24.0))
        if h <= 24.0:
            baseline_so_far = max(baseline_so_far, sofa.total)
        trajectory.append(
            {
                "hour": h,
                "sofa_total": sofa.total,
                "sofa_baseline": baseline_so_far,
                "sofa_delta": sofa.total - baseline_so_far,
                "components": sofa.components,
                "missing": sofa.missing,
                "approximate": sofa.approximate,
                "completeness": round(sofa.completeness, 2),
            }
        )
        h += max(1, step_hours)

    latest_snap = snapshot_at(cohort, stay_id, los_hours, events=ev, baseline_total=baseline_so_far)
    return {
        "disclaimer": DISCLAIMER,
        "stay_id": stay_id,
        "patient_id": latest_snap["patient_id"],
        "latest_hour": round(los_hours, 1),
        "sofa_latest": asdict(latest_snap["sofa"]),
        "sofa_baseline": latest_snap["sofa_baseline"],
        "sofa_delta": latest_snap["sofa_delta"],
        "aps3": asdict(latest_snap["aps3"]),
        "trajectory": trajectory,
    }


@app.get("/patients/{stay_id}/risk")
def patient_risk(stay_id: str, hour: float | None = None):
    cohort = _get_cohort()
    match = cohort.cohort[cohort.cohort["pseudonym"] == stay_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown pseudonymised stay_id: {stay_id}")

    los_hours = float(match.iloc[0]["los"]) * 24.0
    t = los_hours if hour is None else min(max(0.0, float(hour)), los_hours)
    snap = snapshot_at(cohort, stay_id, t)
    pipe, metrics = _get_model()
    prob, drivers = predict_with_drivers(pipe, snap["features"])

    return {
        "disclaimer": DISCLAIMER,
        "stay_id": stay_id,
        "patient_id": snap["patient_id"],
        "hour": round(t, 1),
        "risk_probability_6h": prob,
        "top_drivers": drivers,
        "sofa_total": snap["sofa"].total,
        "sofa_baseline": snap["sofa_baseline"],
        "sofa_delta": snap["sofa_delta"],
        "aps3": asdict(snap["aps3"]),
        "ews": asdict(snap["ews"]),
        "small_sample_caveats": metrics.get("small_sample_caveats", []),
    }


class AssessRequest(BaseModel):
    icu_hours_elapsed: float = Field(12.0, ge=0.0)
    age: float | None = None
    admission_type: str = "EMERGENCY"
    comorbidities: list[str] = Field(default_factory=list)
    # Vitals & Labs
    pao2: float | None = None
    fio2: float | None = None
    spo2: float | None = None
    platelets: float | None = None
    bilirubin: float | None = None
    map_val: float | None = None
    norepinephrine: float | None = None
    gcs_total: float | None = None
    creatinine: float | None = None
    urine_24h: float | None = None
    heart_rate: float | None = None
    respiratory_rate: float | None = None
    temp_c: float | None = None
    wbc: float | None = None
    lactate: float | None = None
    lactate_delta_6h: float | None = None
    # Context
    sofa_baseline: int = 0
    device_exposures: list[str] = Field(default_factory=list)
    clinician_suspected_infection: bool = False


@app.post("/assess")
def assess(req: AssessRequest):
    comp_inputs = {
        "respiration": {
            "pao2": req.pao2,
            "fio2": req.fio2,
            "spo2": req.spo2,
            "approximate": req.pao2 is None and req.spo2 is not None,
        },
        "coagulation": {"platelets": req.platelets},
        "liver": {"bilirubin": req.bilirubin},
        "cardiovascular": {"map": req.map_val, "norepinephrine": req.norepinephrine},
        "cns": {"gcs_total": req.gcs_total},
        "renal": {"creatinine": req.creatinine, "urine_24h": req.urine_24h},
    }
    sofa = compute_sofa(comp_inputs)
    sofa_delta = sofa.total - req.sofa_baseline

    aps3 = compute_aps3(
        {
            "temp_c": req.temp_c,
            "map": req.map_val,
            "heart_rate": req.heart_rate,
            "respiratory_rate": req.respiratory_rate,
            "oxygenation": req.pao2,
            "wbc": req.wbc,
            "bilirubin": req.bilirubin,
            "creatinine": req.creatinine,
            "urine_24h": req.urine_24h,
            "gcs_total": req.gcs_total,
        }
    )

    ews = evaluate_ews(
        hour=req.icu_hours_elapsed,
        sofa_total=sofa.total,
        sofa_delta=sofa_delta,
        age=req.age,
        admission_type=req.admission_type,
        comorbidities=req.comorbidities,
        lactate=req.lactate,
        lactate_delta_6h=req.lactate_delta_6h,
        respiratory_rate=req.respiratory_rate,
        heart_rate=req.heart_rate,
        temp_c=req.temp_c,
        map_val=req.map_val,
        wbc=req.wbc,
        device_exposures=req.device_exposures,
        clinician_suspected_infection=req.clinician_suspected_infection,
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
        "icu_acquired_window": 1.0 if req.icu_hours_elapsed >= 48.0 else 0.0,
        "device_count": float(len(req.device_exposures)),
        "suspected_infection": 1.0 if ews.suspected_infection else 0.0,
        "lactate_latest": float(req.lactate) if req.lactate is not None else 0.0,
        "lactate_delta_6h": float(req.lactate_delta_6h) if req.lactate_delta_6h is not None else 0.0,
    }

    pipe, _ = _get_model()
    prob, drivers = predict_with_drivers(pipe, features)

    return {
        "disclaimer": DISCLAIMER,
        "sofa": asdict(sofa),
        "sofa_baseline": req.sofa_baseline,
        "sofa_delta": sofa_delta,
        "aps3": asdict(aps3),
        "ews": asdict(ews),
        "risk_probability_6h": prob,
        "top_drivers": drivers,
    }
