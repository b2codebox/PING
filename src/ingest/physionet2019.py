"""Allowlist-enforced loader for the PhysioNet/CinC Challenge 2019 Sepsis dataset.

Maps the structured hourly PSV files into the same pseudonymised ``Cohort``
schema used by the MIMIC-IV Demo loader so SOFA, APS III, EWS, and the
Streamlit dashboard work identically on both datasets.

Note on missingness:
* Challenge 2019 PSV files do not include vasopressor infusions, GCS, urine
  output, microbiology, or ICD codes. Those components are left empty (never
  imputed as normal) and are explicitly flagged as missing by the SOFA/APS III
  engines.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingest.allowlist import AllowlistViolation
from src.ingest.loader import Cohort, Pseudonymizer, write_cache

# Explicit allowlist of Challenge 2019 PSV columns.
PHYSIONET2019_ALLOWED_COLS: tuple[str, ...] = (
    "HR",
    "O2Sat",
    "Temp",
    "MAP",
    "Resp",
    "FiO2",
    "pH",
    "BUN",
    "Creatinine",
    "Glucose",
    "Lactate",
    "Bilirubin_total",
    "Hct",
    "WBC",
    "Platelets",
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "ICULOS",
    "SepsisLabel",
)

# Map PSV columns -> canonical vital/lab names used by SOFA/APS III/EWS.
_VITAL_MAP = {
    "HR": ("heart_rate", "bpm"),
    "Resp": ("respiratory_rate", "insp/min"),
    "MAP": ("map_arterial", "mmHg"),
    "Temp": ("temp_celsius", "°C"),
    "O2Sat": ("spo2", "%"),
    "FiO2": ("fio2", "%"),
}

_LAB_MAP = {
    "Lactate": ("lactate", "mmol/L"),
    "Platelets": ("platelets", "K/uL"),
    "Bilirubin_total": ("bilirubin", "mg/dL"),
    "Creatinine": ("creatinine", "mg/dL"),
    "BUN": ("bun", "mg/dL"),
    "Hct": ("hematocrit", "%"),
    "WBC": ("wbc", "K/uL"),
    "Glucose": ("glucose", "mg/dL"),
    "pH": ("ph", "units"),
}


def build_physionet2019_cohort(
    raw_dir: Path | str,
    pseudonymizer: Pseudonymizer | None = None,
) -> Cohort:
    """Ingest Challenge 2019 .psv files from ``raw_dir`` (recursively)."""
    raw_dir = Path(raw_dir)
    pseudo = pseudonymizer or Pseudonymizer()
    psv_files = sorted(raw_dir.rglob("*.psv"))
    if not psv_files:
        raise FileNotFoundError(f"No .psv files found under {raw_dir}")

    cohort_rows: list[dict] = []
    vital_rows: list[dict] = []
    lab_rows: list[dict] = []

    base_time = pd.Timestamp("2020-01-01 00:00:00")

    for path in psv_files:
        df = pd.read_csv(path, sep="|", usecols=list(PHYSIONET2019_ALLOWED_COLS))
        extra = [c for c in df.columns if c not in PHYSIONET2019_ALLOWED_COLS]
        if extra:
            raise AllowlistViolation(f"Disallowed PhysioNet 2019 columns: {extra}")
        if df.empty:
            continue

        raw_id = path.stem
        site = path.parent.name if path.parent.name in ("A", "B") else "PN19"
        patient_id = pseudo.patient(f"pn19:{site}:{raw_id}")
        stay_id = pseudo.stay(f"pn19:{site}:{raw_id}")

        max_iculos = float(df["ICULOS"].max())
        age_val = float(df["Age"].dropna().iloc[0]) if df["Age"].notna().any() else float("nan")
        gender_val = "M" if (df["Gender"].dropna().iloc[0] == 1) else "F"
        unit = "MICU (Unit1)" if df["Unit1"].fillna(0).max() == 1 else (
            "SICU (Unit2)" if df["Unit2"].fillna(0).max() == 1 else f"ICU ({site})"
        )
        sepsis_rows = df[df["SepsisLabel"] == 1]
        has_sepsis = len(sepsis_rows) > 0
        sepsis_onset_hr = float(sepsis_rows["ICULOS"].min()) if has_sepsis else float("nan")

        cohort_rows.append(
            {
                "first_careunit": unit,
                "intime": base_time,
                "outtime": base_time + pd.Timedelta(hours=max_iculos),
                "los": max_iculos / 24.0,
                "admittime": base_time,
                "admission_type": f"PHYSIONET_2019_{site}",
                "hospital_expire_flag": 0,
                "gender": gender_val,
                "anchor_age": age_val,
                "anchor_year": 2020,
                "age": age_val,
                "patient_id": patient_id,
                "pseudonym": stay_id,
                "has_sepsis_label": has_sepsis,
                "sepsis_onset_hour": sepsis_onset_hr,
            }
        )

        # Hourly observations (hours_from_icu = ICULOS - 1)
        hours = df["ICULOS"].astype(float) - 1.0
        timestamps = base_time + pd.to_timedelta(hours, unit="h")

        for col, (var, unit_str) in _VITAL_MAP.items():
            valid = df[col].notna()
            if valid.any():
                vals = df.loc[valid, col].astype(float)
                # Challenge 2019 FiO2 is often 0.21-1.0; normalise to % for consistency.
                if col == "FiO2":
                    vals = vals.where(vals > 1.0, vals * 100.0)
                for ts_v, hr_v, val_v in zip(timestamps[valid], hours[valid], vals):
                    vital_rows.append(
                        {
                            "ts": ts_v,
                            "variable": var,
                            "value": float(val_v),
                            "unit": unit_str,
                            "pseudonym": stay_id,
                            "hours_from_icu": float(hr_v),
                        }
                    )

        for col, (var, unit_str) in _LAB_MAP.items():
            valid = df[col].notna()
            if valid.any():
                for ts_v, hr_v, val_v in zip(timestamps[valid], hours[valid], df.loc[valid, col].astype(float)):
                    lab_rows.append(
                        {
                            "ts": ts_v,
                            "variable": var,
                            "value": float(val_v),
                            "unit": unit_str,
                            "pseudonym": stay_id,
                            "hours_from_icu": float(hr_v),
                        }
                    )

    empty_vaso = pd.DataFrame(columns=["ts", "endtime", "drug", "rate", "rate_mcg_kg_min", "unit", "statusdescription", "pseudonym", "hours_from_icu"])
    empty_urine = pd.DataFrame(columns=["ts", "value", "pseudonym", "hours_from_icu"])
    empty_dev = pd.DataFrame(columns=["ts", "endtime", "device", "pseudonym", "hours_from_icu"])
    empty_micro = pd.DataFrame(columns=["ts", "spec_type_desc", "pseudonym", "hours_from_icu"])
    empty_anti = pd.DataFrame(columns=["ts", "pseudonym", "hours_from_icu"])
    empty_comorb = pd.DataFrame(columns=["pseudonym", "flag"])

    return Cohort(
        cohort=pd.DataFrame(cohort_rows),
        labs=pd.DataFrame(lab_rows) if lab_rows else pd.DataFrame(columns=["ts", "variable", "value", "unit", "pseudonym", "hours_from_icu"]),
        vitals=pd.DataFrame(vital_rows) if vital_rows else pd.DataFrame(columns=["ts", "variable", "value", "unit", "pseudonym", "hours_from_icu"]),
        vasopressors=empty_vaso,
        urine=empty_urine,
        devices=empty_dev,
        micro=empty_micro,
        antimicrobials=empty_anti,
        comorbidities=empty_comorb,
        issues=[
            "PhysioNet/CinC Challenge 2019: vasopressors, GCS, urine output, microbiology, and ICD comorbidities are not present in source PSV files; marked missing (never imputed)."
        ],
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    c = build_physionet2019_cohort("/tmp/pn2019")
    write_cache(c, root / "data" / "processed_pn2019")
    n_sep = int(c.cohort["has_sepsis_label"].sum())
    print(
        f"PhysioNet 2019 ingested: stays={len(c.cohort)} sepsis_positive={n_sep} "
        f"vitals={len(c.vitals)} labs={len(c.labs)}"
    )
