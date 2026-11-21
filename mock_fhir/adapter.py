"""Narrow FHIR R4 adapter for the Sepsis/EWS prototype.

Accepts ONLY:
* Pseudonymised ``Patient`` (rejects raw ``name``, ``telecom``, ``address``,
  ``birthDate`` or non-pseudonymised IDs).
* Allowlisted ``Observation`` resources (LOINC-mapped vitals/labs with strict
  unit checks).
* Allowlisted vasopressor ``MedicationAdministration`` resources (strict rate
  units).
* Allowlisted ``Device`` and ``Condition`` resources.

Fails loudly with ``FHIRAllowlistError`` on any disallowed resource type,
free-text note (e.g. ``DocumentReference``, ``DiagnosticReport.conclusion``),
identifying field, or unverified unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.ingest.allowlist import DISCLAIMER

ALLOWED_RESOURCE_TYPES = frozenset(
    {"Patient", "Observation", "MedicationAdministration", "Device", "Condition"}
)

DENIED_PATIENT_FIELDS = frozenset({"name", "telecom", "address", "birthDate", "photo", "contact"})

# Prototype LOINC -> canonical variable + allowed UCUM units.
# TODO: verify LOINC codes and UCUM units against target hospital FHIR IG.
ALLOWED_OBSERVATION_LOINC: dict[str, tuple[str, frozenset[str]]] = {
    "8867-4": ("heart_rate", frozenset({"/min", "bpm"})),
    "9279-1": ("respiratory_rate", frozenset({"/min", "insp/min"})),
    "8478-0": ("map", frozenset({"mm[Hg]", "mmHg"})),
    "8310-5": ("temp_c", frozenset({"Cel", "C"})),
    "2708-6": ("spo2", frozenset({"%"})),
    "3150-0": ("fio2", frozenset({"%"})),
    "2524-7": ("lactate", frozenset({"mmol/L"})),
    "777-3": ("platelets", frozenset({"10*3/uL", "K/uL"})),
    "1975-2": ("bilirubin", frozenset({"mg/dL"})),
    "2160-0": ("creatinine", frozenset({"mg/dL"})),
    "6690-2": ("wbc", frozenset({"10*3/uL", "K/uL"})),
    "9269-2": ("gcs_total", frozenset({"{score}", "1", ""})),
}

ALLOWED_VASOPRESSORS: dict[str, frozenset[str]] = {
    "norepinephrine": frozenset({"ug/kg/min", "mcg/kg/min"}),
    "epinephrine": frozenset({"ug/kg/min", "mcg/kg/min"}),
    "dopamine": frozenset({"ug/kg/min", "mcg/kg/min"}),
    "dobutamine": frozenset({"ug/kg/min", "mcg/kg/min"}),
    "vasopressin": frozenset({"U/h", "units/hour"}),
}

ALLOWED_DEVICES = frozenset(
    {"invasive_ventilation", "noninvasive_ventilation", "central_line", "urinary_catheter"}
)

ALLOWED_CONDITIONS = frozenset(
    {
        "immunosuppression",
        "malignancy",
        "chronic_kidney_disease",
        "liver_disease",
        "diabetes",
        "copd",
        "heart_failure",
    }
)


class FHIRAllowlistError(ValueError):
    """Raised when a FHIR resource violates the narrow prototype allowlist."""


@dataclass
class ParsedFHIRBundle:
    patient_id: str
    observations: dict[str, float] = field(default_factory=dict)
    vasopressors: dict[str, float] = field(default_factory=dict)
    devices: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


def parse_fhir_bundle(bundle: dict) -> ParsedFHIRBundle:
    """Validate and parse a narrow FHIR R4 Bundle into structured prototype inputs."""
    if bundle.get("resourceType") != "Bundle":
        raise FHIRAllowlistError("Expected FHIR resourceType 'Bundle'")

    patient_id: str | None = None
    observations: dict[str, float] = {}
    vasopressors: dict[str, float] = {}
    devices: list[str] = []
    conditions: list[str] = []

    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        rtype = res.get("resourceType")
        if rtype not in ALLOWED_RESOURCE_TYPES:
            raise FHIRAllowlistError(
                f"Disallowed FHIR resourceType {rtype!r}. Allowed: {sorted(ALLOWED_RESOURCE_TYPES)}"
            )
        if "note" in res or "text" in res:
            raise FHIRAllowlistError(f"Free-text field ('note'/'text') is disallowed in {rtype}")

        if rtype == "Patient":
            bad_fields = DENIED_PATIENT_FIELDS.intersection(res.keys())
            if bad_fields:
                raise FHIRAllowlistError(
                    f"Identifying Patient field(s) disallowed: {sorted(bad_fields)}"
                )
            pid = str(res.get("id", ""))
            if not pid.startswith("P"):
                raise FHIRAllowlistError("Patient.id must be pseudonymised (prefix 'P')")
            patient_id = pid

        elif rtype == "Observation":
            codings = res.get("code", {}).get("coding", [])
            if not codings:
                raise FHIRAllowlistError("Observation missing code.coding")
            code = codings[0].get("code")
            if code not in ALLOWED_OBSERVATION_LOINC:
                raise FHIRAllowlistError(f"Observation LOINC code {code!r} is not allowlisted")
            var_name, allowed_units = ALLOWED_OBSERVATION_LOINC[code]
            vq = res.get("valueQuantity", {})
            unit = str(vq.get("unit", ""))
            if unit not in allowed_units:
                raise FHIRAllowlistError(
                    f"Disallowed unit {unit!r} for {var_name}; allowed={sorted(allowed_units)}"
                )
            if "value" not in vq:
                raise FHIRAllowlistError(f"Observation {var_name} missing valueQuantity.value")
            observations[var_name] = float(vq["value"])

        elif rtype == "MedicationAdministration":
            drug = str(res.get("medicationCodeableConcept", {}).get("text", "")).lower().strip()
            if drug not in ALLOWED_VASOPRESSORS:
                raise FHIRAllowlistError(f"Medication {drug!r} is not an allowlisted vasopressor")
            rate_q = res.get("dosage", {}).get("rateQuantity", {})
            unit = str(rate_q.get("unit", ""))
            if unit not in ALLOWED_VASOPRESSORS[drug]:
                raise FHIRAllowlistError(
                    f"Disallowed vasopressor unit {unit!r} for {drug}; allowed={sorted(ALLOWED_VASOPRESSORS[drug])}"
                )
            vasopressors[drug] = float(rate_q["value"])

        elif rtype == "Device":
            dev_type = str(res.get("type", {}).get("text", "")).lower().strip()
            if dev_type not in ALLOWED_DEVICES:
                raise FHIRAllowlistError(f"Device {dev_type!r} is not allowlisted")
            devices.append(dev_type)

        elif rtype == "Condition":
            cond = str(res.get("code", {}).get("text", "")).lower().strip()
            if cond not in ALLOWED_CONDITIONS:
                raise FHIRAllowlistError(f"Condition {cond!r} is not allowlisted")
            conditions.append(cond)

    if not patient_id:
        raise FHIRAllowlistError("Bundle must include a pseudonymised Patient resource")

    return ParsedFHIRBundle(
        patient_id=patient_id,
        observations=observations,
        vasopressors=vasopressors,
        devices=sorted(set(devices)),
        conditions=sorted(set(conditions)),
    )


def build_sample_fhir_bundle(patient_id: str = "P000DEMO01") -> dict:
    """Create a valid, pseudonymised mock FHIR R4 bundle for local testing."""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": patient_id}},
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "2524-7"}]},
                    "valueQuantity": {"value": 2.8, "unit": "mmol/L"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8478-0"}]},
                    "valueQuantity": {"value": 64.0, "unit": "mmHg"},
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationAdministration",
                    "medicationCodeableConcept": {"text": "norepinephrine"},
                    "dosage": {"rateQuantity": {"value": 0.08, "unit": "mcg/kg/min"}},
                }
            },
            {
                "resource": {
                    "resourceType": "Device",
                    "type": {"text": "central_line"},
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "code": {"text": "immunosuppression"},
                }
            },
        ],
    }
