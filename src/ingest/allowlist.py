"""Explicit allowlist for the MIMIC-IV demo ingestion path.

Every table, column, itemid, unit and free-text field that may enter the
prototype is declared here.  Anything not declared raises ``AllowlistViolation``
instead of being silently dropped, so a schema drift or an accidental note
column fails loudly.

# TODO: verify  All MIMIC itemids below were checked against the MIMIC-IV demo
# TODO: verify  d_items/d_labitems dictionaries (v2.2).  Re-verify before using
# TODO: verify  a different MIMIC release, because itemids are version-bound.
"""

from __future__ import annotations

from dataclasses import dataclass

DISCLAIMER = (
    "Decision support only. Not validated for clinical use. Requires clinician review."
)


class AllowlistViolation(RuntimeError):
    """Raised when a table, column, itemid or unit is outside the allowlist."""


# ---------------------------------------------------------------------------
# 1. Tables: permitted columns.  Loaders pass these straight to pandas usecols,
#    and any extra requested column is rejected.
# ---------------------------------------------------------------------------
ALLOWED_COLUMNS: dict[str, tuple[str, ...]] = {
    "patients": ("subject_id", "gender", "anchor_age", "anchor_year", "dod"),
    "admissions": (
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "admission_type",
        "hospital_expire_flag",
    ),
    "icustays": (
        "subject_id",
        "hadm_id",
        "stay_id",
        "first_careunit",
        "last_careunit",
        "intime",
        "outtime",
        "los",
    ),
    "diagnoses_icd": ("subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"),
    "labevents": (
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "valuenum",
        "valueuom",
    ),
    "microbiologyevents": (
        "subject_id",
        "hadm_id",
        "charttime",
        "chartdate",
        "spec_type_desc",
        "org_name",
    ),
    "prescriptions": (
        "subject_id",
        "hadm_id",
        "starttime",
        "stoptime",
        "drug",
        "route",
        "dose_val_rx",
        "dose_unit_rx",
    ),
    "chartevents": (
        "subject_id",
        "hadm_id",
        "stay_id",
        "charttime",
        "itemid",
        "value",
        "valuenum",
        "valueuom",
    ),
    "inputevents": (
        "subject_id",
        "hadm_id",
        "stay_id",
        "starttime",
        "endtime",
        "itemid",
        "amount",
        "amountuom",
        "rate",
        "rateuom",
        "statusdescription",
    ),
    "outputevents": (
        "subject_id",
        "hadm_id",
        "stay_id",
        "charttime",
        "itemid",
        "value",
        "valueuom",
    ),
    "procedureevents": (
        "subject_id",
        "hadm_id",
        "stay_id",
        "starttime",
        "endtime",
        "itemid",
        "statusdescription",
    ),
}

# Columns that exist in MIMIC but are deliberately excluded: free-text notes,
# provider identifiers and other identifying / unstructured fields.
DENIED_COLUMNS: frozenset[str] = frozenset(
    {
        "comments",
        "text",
        "note",
        "notes",
        "caregiver_id",
        "order_provider_id",
        "admit_provider_id",
        "storetime",
        "pharmacy_id",
        "poe_id",
        "patientweight",
    }
)

# ---------------------------------------------------------------------------
# 2. Lab itemids (hosp/labevents) -> canonical clinical variable.
# ---------------------------------------------------------------------------
LAB_ITEMIDS: dict[int, str] = {
    50821: "pao2",        # pO2, arterial blood gas (mm Hg)
    50820: "ph",          # pH, arterial blood gas
    50813: "lactate",     # Lactate, blood gas (mmol/L)
    53154: "lactate",     # Lactate, chemistry (mmol/L)  # TODO: verify overlap
    51265: "platelets",   # Platelet Count (K/uL)
    50885: "bilirubin",   # Bilirubin, Total (mg/dL)
    50912: "creatinine",  # Creatinine (mg/dL)
    51006: "bun",         # Urea Nitrogen (mg/dL)
    50983: "sodium",      # Sodium (mEq/L)
    51221: "hematocrit",  # Hematocrit (%)
    51301: "wbc",         # White Blood Cells (K/uL)
    50931: "glucose",     # Glucose (mg/dL)
    50862: "albumin",     # Albumin (g/dL)
}

LAB_UNITS: dict[str, frozenset[str]] = {
    "pao2": frozenset({"mm Hg"}),
    "ph": frozenset({"units"}),
    "lactate": frozenset({"mmol/L"}),
    "platelets": frozenset({"K/uL"}),
    "bilirubin": frozenset({"mg/dL"}),
    "creatinine": frozenset({"mg/dL"}),
    "bun": frozenset({"mg/dL"}),
    "sodium": frozenset({"mEq/L"}),
    "hematocrit": frozenset({"%"}),
    "wbc": frozenset({"K/uL"}),
    "glucose": frozenset({"mg/dL"}),
    "albumin": frozenset({"g/dL"}),
}

# ---------------------------------------------------------------------------
# 3. Chart itemids (icu/chartevents) -> canonical vital variable.
# ---------------------------------------------------------------------------
CHART_ITEMIDS: dict[int, str] = {
    220045: "heart_rate",          # Heart Rate (bpm)
    220210: "respiratory_rate",    # Respiratory Rate (insp/min)
    220052: "map_arterial",        # Arterial Blood Pressure mean (mmHg)
    220181: "map_noninvasive",     # Non Invasive Blood Pressure mean (mmHg)
    223762: "temp_celsius",        # Temperature Celsius
    223761: "temp_fahrenheit",     # Temperature Fahrenheit
    220277: "spo2",                # O2 saturation pulseoxymetry (%)
    223835: "fio2",                # Inspired O2 Fraction (%)
    220739: "gcs_eye",             # GCS - Eye Opening (text)
    223900: "gcs_verbal",          # GCS - Verbal Response (text)
    223901: "gcs_motor",           # GCS - Motor Response (text)
}

CHART_UNITS: dict[str, frozenset[str]] = {
    "heart_rate": frozenset({"bpm"}),
    "respiratory_rate": frozenset({"insp/min"}),
    "map_arterial": frozenset({"mmHg"}),
    "map_noninvasive": frozenset({"mmHg"}),
    "temp_celsius": frozenset({"\u00b0C"}),
    "temp_fahrenheit": frozenset({"\u00b0F"}),
    "spo2": frozenset({"%"}),          # also rejects SpO2 < 50 later as artefact
    "fio2": frozenset({"%"}),          # FiO2 0.21-1.0 stored either as % or fraction
    "gcs_eye": frozenset({""}),
    "gcs_verbal": frozenset({""}),
    "gcs_motor": frozenset({""}),
}

# GCS component text -> integer points.  "No Response-ETT" is the MIMIC coding
# for an intubated patient and maps to verbal score 1 (flagged as intubated).
GCS_TEXT_MAP: dict[str, dict[str, int]] = {
    "gcs_eye": {
        "Spontaneously": 4,
        "To Speech": 3,
        "To Pain": 2,
        "No response": 1,
        "None": 1,
    },
    "gcs_verbal": {
        "Oriented": 5,
        "Confused": 4,
        "Inappropriate Words": 3,
        "Incomprehensible sounds": 2,
        "No Response": 1,
        "No Response-ETT": 1,  # intubated -> scored 1, flagged
        "None": 1,
    },
    "gcs_motor": {
        "Obeys Commands": 6,
        "Localizes Pain": 5,
        "Flex-withdraws": 4,
        "Abnormal Flexion": 3,
        "Abnormal extension": 2,
        "No response": 1,
        "None": 1,
    },
}

# ---------------------------------------------------------------------------
# 4. Vasopressors (icu/inputevents).  Original SOFA units are mcg/kg/min.
# ---------------------------------------------------------------------------
VASOPRESSOR_ITEMIDS: dict[int, str] = {
    221906: "norepinephrine",
    221289: "epinephrine",
    221662: "dopamine",
    221653: "dobutamine",
    222315: "vasopressin",  # NOT in original 1996 SOFA; see scores/sofa.py TODO
}

# Only these rate units are accepted.  Anything else -> AllowlistViolation so a
# silent unit mismatch can never be scored as if it were mcg/kg/min.
VASOPRESSOR_RATE_UNITS: dict[str, frozenset[str]] = {
    "norepinephrine": frozenset({"mcg/kg/min"}),
    "epinephrine": frozenset({"mcg/kg/min"}),
    "dopamine": frozenset({"mcg/kg/min"}),
    "dobutamine": frozenset({"mcg/kg/min"}),
    "vasopressin": frozenset({"units/hour"}),
}

# Normalisation factors to mcg/kg/min.  Vasopressin is kept in units/hour and
# never converted (its SOFA role is unverified).
VASOPRESSOR_TO_MCG_KG_MIN: dict[str, float] = {
    "mcg/kg/min": 1.0,
}

# ---------------------------------------------------------------------------
# 5. Urine output itemids (icu/outputevents), millilitres.
# ---------------------------------------------------------------------------
URINE_OUTPUT_ITEMIDS: frozenset[int] = frozenset(
    {
        226559,  # Foley
        226560,  # Void
        226561,  # Condom Cath
        226584,  # Ileoconduit
        226563,  # Suprapubic
        226564,  # R Nephrostomy
        226565,  # L Nephrostomy
        226567,  # Straight Cath
        226557,  # R Ureteral Stent
        226558,  # L Ureteral Stent
        227488,  # GU Irrigant Volume In (negative)
        227489,  # GU Irrigant/Urine Volume Out
        226627,  # OR Urine
    }
)

# ---------------------------------------------------------------------------
# 6. Device exposure (icu/procedureevents) and ventilation.
# ---------------------------------------------------------------------------
DEVICE_PROCEDURE_ITEMIDS: dict[int, str] = {
    225792: "invasive_ventilation",
    225794: "noninvasive_ventilation",
    224270: "dialysis_catheter",
    224560: "pa_catheter",
    224273: "presep_catheter",
    225203: "pheresis_catheter",
    225441: "hemodialysis",
    225802: "crrt",
    225803: "cvvhd",
    225805: "peritoneal_dialysis",
    225809: "cvvhdf",
    225955: "scuf",
    229351: "urinary_catheter",
}

# Central venous access is charted as a line-placement procedure in MIMIC-IV.
CENTRAL_LINE_ITEMIDS: frozenset[int] = frozenset({224270, 224560, 224273, 225203})

# ---------------------------------------------------------------------------
# 7. Structured microbiology / antimicrobial start.
# ---------------------------------------------------------------------------
# Anti-infective prescriptions are identified from the structured `drug` field.
# TODO: verify  This is a name-substring heuristic, not a validated ATC mapping.
ANTIMICROBIAL_PATTERNS: tuple[str, ...] = (
    "vancomycin",
    "piperacillin",
    "tazobactam",
    "cefepime",
    "ceftriaxone",
    "ceftazidime",
    "meropenem",
    "ertapenem",
    "imipenem",
    "ciprofloxacin",
    "levofloxacin",
    "moxifloxacin",
    "metronidazole",
    "azithromycin",
    "linezolid",
    "ampicillin",
    "amoxicillin",
    "clindamycin",
    "doxycycline",
    "gentamicin",
    "tobramycin",
    "colistin",
    "micafungin",
    "fluconazole",
    "caspofungin",
)

# Structured specimen types that count as a culture for suspected infection.
CULTURE_SPECIMENS: frozenset[str] = frozenset(
    {
        "BLOOD CULTURE",
        "URINE",
        "SPUTUM",
        "TISSUE",
        "SWAB",
        "BRONCHOALVEOLAR LAVAGE",
        "PLEURAL FLUID",
        "CSF;SPINAL FLUID",
        "ABSCESS",
    }
)

# ---------------------------------------------------------------------------
# 8. Structured ICD comorbidity flags (hosp/diagnoses_icd).
# ---------------------------------------------------------------------------
# TODO: verify  ICD-9/10 prefix lists with a clinician / coding specialist.
COMORBIDITY_FLAGS: dict[str, dict[str, tuple[str, ...]]] = {
    "icd9": {
        "immunosuppression": ("279", "V08", "042", "V42"),
        "malignancy": ("140", "141", "142", "143", "144", "145", "146", "147",
                       "148", "149", "150", "151", "152", "153", "154", "155",
                       "156", "157", "158", "159", "160", "161", "162", "163",
                       "164", "165", "170", "171", "172", "174", "175", "176",
                       "179", "180", "181", "182", "183", "184", "185", "186",
                       "187", "188", "189", "190", "191", "192", "193", "194",
                       "195", "196", "197", "198", "199", "200", "201", "202",
                       "203", "204", "205", "206", "207", "208"),
        "chronic_kidney_disease": ("585", "V451", "V56"),
        "liver_disease": ("570", "571", "572", "573"),
        "diabetes": ("250"),
        "copd": ("491", "492", "493", "494", "496"),
        "heart_failure": ("428",),
    },
    "icd10": {
        "immunosuppression": ("D80", "D81", "D82", "D83", "D84", "D89", "B20"),
        "malignancy": ("C",),
        "chronic_kidney_disease": ("N18", "Z99", "Z49"),
        "liver_disease": ("K70", "K71", "K72", "K73", "K74", "K76"),
        "diabetes": ("E10", "E11", "E13"),
        "copd": ("J44",),
        "heart_failure": ("I50", "I11", "I13"),
    },
}


def validate_column_request(table: str, columns) -> tuple[str, ...]:
    """Return the requested columns if every one is allowed, else raise."""
    if table not in ALLOWED_COLUMNS:
        raise AllowlistViolation(f"table not allowlisted: {table!r}")
    allowed = ALLOWED_COLUMNS[table]
    requested = tuple(columns)
    for col in requested:
        if col in DENIED_COLUMNS:
            raise AllowlistViolation(f"column {col!r} in table {table!r} is denied (free text / identifier)")
        if col not in allowed:
            raise AllowlistViolation(f"column {col!r} in table {table!r} is not allowlisted")
    return requested


def canonical_unit(variable: str, unit: str, mapping: dict[str, frozenset[str]]) -> str:
    """Validate a unit for a canonical variable and return it."""
    allowed = mapping.get(variable)
    if allowed is None:
        raise AllowlistViolation(f"variable {variable!r} has no unit allowlist")
    if unit not in allowed:
        raise AllowlistViolation(
            f"unit {unit!r} not allowlisted for {variable!r}; allowed={sorted(allowed)}"
        )
    return unit


@dataclass(frozen=True)
class FieldRef:
    """A resolved, allowlisted clinical field reference."""

    table: str
    itemid: int
    variable: str


def resolve_lab(itemid: int) -> str:
    if itemid not in LAB_ITEMIDS:
        raise AllowlistViolation(f"lab itemid {itemid} is not allowlisted")
    return LAB_ITEMIDS[itemid]


def resolve_chart(itemid: int) -> str:
    if itemid not in CHART_ITEMIDS:
        raise AllowlistViolation(f"chart itemid {itemid} is not allowlisted")
    return CHART_ITEMIDS[itemid]


def resolve_vasopressor(itemid: int) -> str:
    if itemid not in VASOPRESSOR_ITEMIDS:
        raise AllowlistViolation(f"vasopressor itemid {itemid} is not allowlisted")
    return VASOPRESSOR_ITEMIDS[itemid]


def resolve_urine(itemid: int) -> str:
    if itemid not in URINE_OUTPUT_ITEMIDS:
        raise AllowlistViolation(f"urine output itemid {itemid} is not allowlisted")
    return "urine_output"
