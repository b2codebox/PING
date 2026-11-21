"""Allowlist-enforced, pseudonymised ingester for the user's ~6.6k Sepsis cohort.

Important temporal note:
    In this dataset, columns ``<var>_1 .. <var>_24`` correspond to hours 1..24
    *after* ``onsettime`` (verified via ``charttime_1_1 - onsettime`` median +0.37h,
    ``charttime_1_24 - onsettime`` median +22.9h). Therefore, this cohort is stored
    as a dedicated post-onset sepsis trajectory & severity dataset and is NEVER
    leaked into pre-onset (``t < onset``) preventive training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest.loader import Cohort, Pseudonymizer, write_cache

HOURS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

VITAL_MAP = {
    "heart_rate": ("heart_rate", "bpm"),
    "resp_rate": ("respiratory_rate", "breaths/min"),
    "mbp": ("map_arterial", "mmHg"),
    "temperature": ("temp_celsius", "C"),
    "spo2": ("spo2", "%"),
    "gcs": ("gcs_total", "points"),
}

LAB_MAP = {
    "lactate": ("lactate", "mmol/L"),
    "platelet": ("platelets", "K/uL"),
    "bilirubin_total": ("bilirubin", "mg/dL"),
    "creatinine": ("creatinine", "mg/dL"),
    "bun": ("bun", "mg/dL"),
    "wbc": ("wbc", "K/uL"),
    "glucose": ("glucose", "mg/dL"),
    "ph": ("ph", "pH"),
    "pao2fio2ratio": ("pao2_fio2_ratio", "mmHg"),
}

VASO_MAP = {
    "norepinephrine": "norepinephrine",
    "epinephrine": "epinephrine",
    "dopamine": "dopamine",
    "dobutamine": "dobutamine",
    "vasopressin": "vasopressin",
    "phenelephrine": "phenylephrine",
}

COMORB_COLS = ["hypertension", "diabetes", "heart_failure", "ckd", "cancer", "copd"]


def build_sepsis6k_cohort(tsv_path: str | Path, max_stays: int | None = None) -> Cohort:
    """Ingest the 6.6k sepsis TSV into the pseudonymised Cohort schema."""
    tsv_path = Path(tsv_path)

    # Only read strictly allowlisted columns (drop ethnicity, raw transformations, etc.)
    base_cols = ["subject_id", "stay_id", "onsettime", "intime", "outtime", "sofa", "gender", "anchor_age"] + COMORB_COLS
    ts_cols: list[str] = []
    for prefix in list(VITAL_MAP) + list(LAB_MAP) + list(VASO_MAP) + ["urine"]:
        for h in HOURS:
            ts_cols.append(f"{prefix}_{h}")

    header = list(pd.read_csv(tsv_path, sep="\t", nrows=0).columns)
    usecols = [c for c in base_cols + ts_cols if c in header]

    raw = pd.read_csv(tsv_path, sep="\t", usecols=usecols, nrows=max_stays)
    raw = raw.dropna(subset=["subject_id", "stay_id"]).drop_duplicates(subset=["stay_id"]).reset_index(drop=True)

    pseudo = Pseudonymizer()
    raw["patient_id"] = [pseudo.patient(int(s)) for s in raw["subject_id"]]
    raw["pseudonym"] = [pseudo.stay(int(s)) for s in raw["stay_id"]]

    t_in = pd.to_datetime(raw["intime"], errors="coerce")
    t_out = pd.to_datetime(raw["outtime"], errors="coerce")
    los_days = ((t_out - t_in).dt.total_seconds() / 86400.0).clip(lower=1.0).fillna(1.0)

    cohort_df = pd.DataFrame(
        {
            "pseudonym": raw["pseudonym"],
            "patient_id": raw["patient_id"],
            "age": pd.to_numeric(raw["anchor_age"], errors="coerce"),
            "gender": raw["gender"].fillna("Unknown").astype(str),
            "admission_type": "EMERGENCY",
            "first_careunit": "Sepsis ICU (6.6k Cohort)",
            "los": los_days.round(2),
            "has_sepsis_label": True,
            "sepsis_onset_hour": 0.0,  # Trajectories are aligned to onsettime (t = 0)
            "baseline_sofa": pd.to_numeric(raw["sofa"], errors="coerce"),
        }
    )

    def _melt_group(col_map: dict[str, tuple[str, str]]) -> pd.DataFrame:
        parts = []
        for raw_prefix, (var_name, unit) in col_map.items():
            cols = [f"{raw_prefix}_{h}" for h in HOURS if f"{raw_prefix}_{h}" in raw.columns]
            if not cols:
                continue
            sub = raw[["pseudonym"] + cols].melt(id_vars="pseudonym", var_name="col", value_name="value")
            sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
            sub = sub.dropna(subset=["value"])
            if sub.empty:
                continue
            sub["hours_from_icu"] = sub["col"].str.rsplit("_", n=1).str[1].astype(np.float32)
            sub["variable"] = var_name
            sub["unit"] = unit
            parts.append(sub[["pseudonym", "hours_from_icu", "variable", "value", "unit"]])
        if not parts:
            return pd.DataFrame(columns=["pseudonym", "hours_from_icu", "variable", "value", "unit"])
        return pd.concat(parts, ignore_index=True)

    vitals_df = _melt_group(VITAL_MAP)
    labs_df = _melt_group(LAB_MAP)

    # Vasopressors (rate > 0)
    vaso_parts = []
    for raw_prefix, drug in VASO_MAP.items():
        cols = [f"{raw_prefix}_{h}" for h in HOURS if f"{raw_prefix}_{h}" in raw.columns]
        if not cols:
            continue
        sub = raw[["pseudonym"] + cols].melt(id_vars="pseudonym", var_name="col", value_name="rate")
        sub["rate"] = pd.to_numeric(sub["rate"], errors="coerce")
        sub = sub[sub["rate"] > 0]
        if sub.empty:
            continue
        sub["start_hour"] = sub["col"].str.rsplit("_", n=1).str[1].astype(np.float32)
        sub["end_hour"] = sub["start_hour"] + 1.0
        sub["drug"] = drug
        sub["rate_unit"] = "mcg/kg/min"
        vaso_parts.append(sub[["pseudonym", "start_hour", "end_hour", "drug", "rate", "rate_unit"]])
    vaso_df = (
        pd.concat(vaso_parts, ignore_index=True)
        if vaso_parts
        else pd.DataFrame(columns=["pseudonym", "start_hour", "end_hour", "drug", "rate", "rate_unit"])
    )

    # Urine output
    urine_cols = [f"urine_{h}" for h in HOURS if f"urine_{h}" in raw.columns]
    if urine_cols:
        u = raw[["pseudonym"] + urine_cols].melt(id_vars="pseudonym", var_name="col", value_name="volume_ml")
        u["volume_ml"] = pd.to_numeric(u["volume_ml"], errors="coerce")
        u = u[u["volume_ml"] >= 0]
        u["hours_from_icu"] = u["col"].str.rsplit("_", n=1).str[1].astype(np.float32)
        urine_df = u[["pseudonym", "hours_from_icu", "volume_ml"]].reset_index(drop=True)
    else:
        urine_df = pd.DataFrame(columns=["pseudonym", "hours_from_icu", "volume_ml"])

    # Comorbidities
    comorb_rows = []
    for c in COMORB_COLS:
        if c in raw.columns:
            pos = raw.loc[pd.to_numeric(raw[c], errors="coerce") == 1, "pseudonym"]
            if len(pos):
                comorb_rows.append(pd.DataFrame({"pseudonym": pos.values, "category": c}))
    comorb_df = (
        pd.concat(comorb_rows, ignore_index=True)
        if comorb_rows
        else pd.DataFrame(columns=["pseudonym", "category"])
    )

    empty_devices = pd.DataFrame(columns=["pseudonym", "device", "start_hour", "end_hour"])
    empty_micro = pd.DataFrame(columns=["pseudonym", "hours_from_icu", "specimen"])
    empty_abx = pd.DataFrame(columns=["pseudonym", "start_hour", "drug"])

    return Cohort(
        cohort=cohort_df,
        labs=labs_df,
        vitals=vitals_df,
        vasopressors=vaso_df,
        urine=urine_df,
        devices=empty_devices,
        micro=empty_micro,
        antimicrobials=empty_abx,
        comorbidities=comorb_df,
        issues=[
            "Sepsis 6.6k cohort measurements (_1.._24) are aligned to sepsis onsettime (post-onset hours 1..24); kept separate from pre-onset preventive training to prevent temporal leakage."
        ],
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    cohort = build_sepsis6k_cohort("/tmp/user_sepsis_db/download")
    out_dir = root / "data" / "processed_sepsis6k"
    write_cache(cohort, out_dir)
    print(
        f"Saved Sepsis 6.6k cohort to {out_dir}: "
        f"stays={len(cohort.cohort)}, vitals={len(cohort.vitals)}, "
        f"labs={len(cohort.labs)}, vaso={len(cohort.vasopressors)}, "
        f"urine={len(cohort.urine)}, comorbidities={len(cohort.comorbidities)}"
    )
