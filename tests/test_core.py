"""Core pytest suite for the Sepsis & Early Warning prototype.

Covers:
1. Allowlist enforcement (denied columns, unknown itemids, bad units).
2. SOFA boundaries & explicit missingness / approximate flags.
3. APS III unavailable handling (never labeled APACHE II / full APACHE III).
4. EWS alert logic & pre-existing (<48h) vs ICU-acquired (>=48h) infection split.
5. No-future-data leakage check (adding future records cannot change score at t).
6. Narrow FHIR R4 adapter round-trip & loud failure on disallowed resources/units.
7. FastAPI endpoints & mandatory disclaimer presence.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mock_fhir.adapter import FHIRAllowlistError, build_sample_fhir_bundle, parse_fhir_bundle
from src.api.app import app
from src.ews.alerts import evaluate_ews
from src.ingest.allowlist import (
    DISCLAIMER,
    AllowlistViolation,
    canonical_unit,
    LAB_UNITS,
    resolve_lab,
    validate_column_request,
)
from src.scores.aps3 import compute_aps3
from src.scores.sofa import compute_sofa, window_inputs


def test_allowlist_rejects_free_text_and_unknown_items():
    with pytest.raises(AllowlistViolation):
        validate_column_request("labevents", ["subject_id", "comments"])
    with pytest.raises(AllowlistViolation):
        resolve_lab(999999)
    with pytest.raises(AllowlistViolation):
        canonical_unit("lactate", "mg/dL", LAB_UNITS)


def test_sofa_boundaries_and_missingness():
    res = compute_sofa(
        {
            "respiration": {"pao2": None, "fio2": 50, "spo2": 90, "approximate": True},
            "coagulation": {"platelets": 95},  # 50-99 -> 2
            "liver": {"bilirubin": None},  # missing -> 0
            "cardiovascular": {"map": 65, "norepinephrine": 0.15},  # >0.1 -> 3
            "cns": {"gcs_total": 12},  # 10-12 -> 2
            "renal": {"creatinine": 2.5},  # 2.0-3.4 -> 2
        }
    )
    assert res.components["coagulation"] == 2
    assert res.components["liver"] == 0
    assert "liver" in res.missing
    assert "respiration" in res.approximate
    assert res.components["cardiovascular"] == 3
    assert res.components["cns"] == 2
    assert res.components["renal"] == 2
    assert res.completeness == pytest.approx(5 / 6)


def test_aps3_unavailable_when_core_missing():
    res = compute_aps3({"heart_rate": 110, "respiratory_rate": 24})
    assert res.status == "unavailable"
    assert res.score is None
    assert "map" in res.missing_variables


def test_ews_tiered_alerts_and_infection_window():
    # Pre-SOFA early warning (<48h)
    ews_early = evaluate_ews(
        hour=12.0,
        sofa_total=1,
        sofa_delta=0,
        age=75,
        comorbidities=["immunosuppression"],
        lactate=2.5,
        heart_rate=112,
    )
    assert ews_early.early_warning_alert is True
    assert ews_early.alert_tier == "early_warning"
    assert ews_early.infection_window == "pre_existing_or_early (<48h)"

    # Organ dysfunction without suspected infection (>=48h)
    ews_od = evaluate_ews(
        hour=54.0,
        sofa_total=4,
        sofa_delta=2,
        device_exposures=["central_line"],
        clinician_suspected_infection=False,
    )
    assert ews_od.alert_tier == "organ_dysfunction"
    assert ews_od.infection_window == "icu_acquired_risk (>=48h)"

    # Possible sepsis with suspected infection
    ews_sep = evaluate_ews(
        hour=54.0,
        sofa_total=4,
        sofa_delta=2,
        clinician_suspected_infection=True,
    )
    assert ews_sep.alert_tier == "possible_sepsis"


def test_no_future_data_leakage():
    base_events = {
        "lab:creatinine": pd.DataFrame({"hours_from_icu": [4.0], "value": [1.0]}),
        "chart:map_arterial": pd.DataFrame({"hours_from_icu": [4.0], "value": [75.0]}),
    }
    future_events = {
        "lab:creatinine": pd.DataFrame({"hours_from_icu": [4.0, 12.0], "value": [1.0, 6.0]}),
        "chart:map_arterial": pd.DataFrame({"hours_from_icu": [4.0, 12.0], "value": [75.0, 45.0]}),
    }
    s_base = compute_sofa(window_inputs(base_events, t_hours=6.0))
    s_future = compute_sofa(window_inputs(future_events, t_hours=6.0))
    assert s_base.total == s_future.total
    assert s_base.components == s_future.components


def test_fhir_adapter_roundtrip_and_rejections():
    bundle = build_sample_fhir_bundle("P123456")
    parsed = parse_fhir_bundle(bundle)
    assert parsed.patient_id == "P123456"
    assert parsed.observations["lactate"] == 2.8
    assert parsed.vasopressors["norepinephrine"] == 0.08
    assert "central_line" in parsed.devices

    # Disallowed resource type (free-text DocumentReference)
    bad = build_sample_fhir_bundle("P123456")
    bad["entry"].append({"resource": {"resourceType": "DocumentReference"}})
    with pytest.raises(FHIRAllowlistError):
        parse_fhir_bundle(bad)

    # Disallowed unit
    bad_unit = build_sample_fhir_bundle("P123456")
    bad_unit["entry"][1]["resource"]["valueQuantity"]["unit"] = "mg/dL"
    with pytest.raises(FHIRAllowlistError):
        parse_fhir_bundle(bad_unit)


def test_api_endpoints_and_disclaimer():
    client = TestClient(app)
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["disclaimer"] == DISCLAIMER

    r_pts = client.get("/patients")
    assert r_pts.status_code == 200
    body = r_pts.json()
    assert body["disclaimer"] == DISCLAIMER
    assert body["count"] > 0

    stay_id = body["patients"][0]["stay_id"]
    assert stay_id.startswith("S")

    r_scores = client.get(f"/patients/{stay_id}/scores")
    assert r_scores.status_code == 200
    assert r_scores.json()["disclaimer"] == DISCLAIMER

    r_risk = client.get(f"/patients/{stay_id}/risk")
    assert r_risk.status_code == 200
    assert r_risk.json()["disclaimer"] == DISCLAIMER

    r_assess = client.post(
        "/assess",
        json={
            "icu_hours_elapsed": 50.0,
            "age": 70,
            "lactate": 3.1,
            "heart_rate": 115,
            "respiratory_rate": 24,
            "map_val": 62,
            "creatinine": 2.2,
            "clinician_suspected_infection": True,
        },
    )
    assert r_assess.status_code == 200
    assert r_assess.json()["disclaimer"] == DISCLAIMER
