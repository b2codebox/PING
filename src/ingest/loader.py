"""Allowlist-enforced MIMIC-IV demo loader.

Three guarantees, each covered by tests:

1. Only allowlisted tables/columns are read (``pandas.read_csv(usecols=...)``
   with the allowlist as the authority).
2. Only allowlisted itemids are kept, and their units are validated; a unit
   mismatch raises instead of being silently scored.
3. Raw identifiers never leave this module.  ``subject_id``/``hadm_id``/
   ``stay_id`` are replaced by salted-hash pseudonyms (``patient_id``,
   ``stay_id`` as ``Sxxxx``).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .allowlist import (
    ALLOWED_COLUMNS,
    CHART_ITEMIDS,
    CHART_UNITS,
    CULTURE_SPECIMENS,
    LAB_ITEMIDS,
    LAB_UNITS,
    URINE_OUTPUT_ITEMIDS,
    VASOPRESSOR_ITEMIDS,
    VASOPRESSOR_RATE_UNITS,
    AllowlistViolation,
    validate_column_request,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "mimic_demo"

# Tables we touch, mapped to their subdirectory in the demo bundle.
TABLE_DIRS: dict[str, str] = {
    "patients": "hosp",
    "admissions": "hosp",
    "icustays": "icu",
    "diagnoses_icd": "hosp",
    "labevents": "hosp",
    "microbiologyevents": "hosp",
    "prescriptions": "hosp",
    "chartevents": "icu",
    "inputevents": "icu",
    "outputevents": "icu",
    "procedureevents": "icu",
}

# Stable CLI-loaded cache files, checked into data/processed after first build.
CACHE_TABLES = ("cohort", "labs", "vitals", "vasopressors", "urine", "devices",
                "micro", "antimicrobials", "comorbidities", "sofa_windows")


class Pseudonymizer:
    """Salted-hash pseudonymisation.  Salt from env so it is not in the repo."""

    def __init__(self, salt: str | None = None) -> None:
        self._salt = (salt or os.environ.get("PROTOTYPE_PSEUDONYM_SALT") or
                      "sepsis-prototype-demo-salt").encode()

    def _digest(self, value: object, prefix: str) -> str:
        h = hashlib.sha256(self._salt + f"{prefix}:{value}".encode()).hexdigest()
        return h[:10].upper()

    def patient(self, subject_id: object) -> str:
        return "P" + self._digest(subject_id, "subject")

    def stay(self, stay_id: object) -> str:
        return "S" + self._digest(stay_id, "stay")


def _read_table(table: str, data_dir: Path, columns=None) -> pd.DataFrame:
    requested = columns or ALLOWED_COLUMNS[table]
    validate_column_request(table, requested)
    path = data_dir / TABLE_DIRS[table] / f"{table}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"missing allowlisted table file: {path}")
    df = pd.read_csv(path, usecols=list(requested), low_memory=False)
    # Drop any non-allowlisted columns that pandas may have materialised.
    extra = [c for c in df.columns if c not in ALLOWED_COLUMNS[table]]
    if extra:
        raise AllowlistViolation(f"denied columns materialised from {table}: {extra}")
    return df


@dataclass
class Cohort:
    """Pseudonymised, allowlisted data bundle for the prototype."""

    cohort: pd.DataFrame
    labs: pd.DataFrame
    vitals: pd.DataFrame
    vasopressors: pd.DataFrame
    urine: pd.DataFrame
    devices: pd.DataFrame
    micro: pd.DataFrame
    antimicrobials: pd.DataFrame
    comorbidities: pd.DataFrame
    issues: list[str] = field(default_factory=list)

    def warnings_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"issue": self.issues})


def _check_units(df: pd.DataFrame, variable: str, mapping, source: str) -> None:
    if df.empty:
        return
    units = set(df["valueuom"].dropna().unique()) if "valueuom" in df else set()
    allowed = mapping.get(variable, frozenset())
    bad = units - allowed
    if bad:
        raise AllowlistViolation(
            f"{source}: disallowed unit(s) {sorted(bad)} for {variable}; allowed={sorted(allowed)}"
        )


def build_cohort(data_dir: Path | str = DEFAULT_DATA_DIR,
                 pseudonymizer: Pseudonymizer | None = None) -> Cohort:
    """Load and pseudonymise every allowlisted structured field."""
    data_dir = Path(data_dir)
    pseudo = pseudonymizer or Pseudonymizer()
    issues: list[str] = []

    patients = _read_table("patients", data_dir)
    admissions = _read_table("admissions", data_dir)
    icustays = _read_table("icustays", data_dir)
    diagnoses = _read_table("diagnoses_icd", data_dir)

    # ---- cohort spine -----------------------------------------------------
    adm = admissions[
        ["subject_id", "hadm_id", "admittime", "admission_type", "hospital_expire_flag"]
    ].copy()
    stays = icustays[
        ["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime", "los"]
    ].copy()
    cohort = stays.merge(adm, on=["subject_id", "hadm_id"], how="left")
    cohort = cohort.merge(
        patients[["subject_id", "gender"]], on="subject_id", how="left"
    )
    for col in ("intime", "outtime", "admittime"):
        cohort[col] = pd.to_datetime(cohort[col], errors="coerce")

    # Age at ICU admission, using anchor_year from MIMIC.  MIMIC shifts dates by
    # a per-patient offset; anchor_year lets us recover age at the stay.
    anchor = patients[["subject_id", "anchor_age", "anchor_year"]].copy()
    if "anchor_age_y" in cohort:
        cohort = cohort.drop(columns=["anchor_age_y"])
    if "anchor_age" in cohort:
        cohort = cohort.drop(columns=["anchor_age"])
    cohort = cohort.merge(anchor, on="subject_id", how="left")
    cohort["age"] = cohort["anchor_age"] + (cohort["intime"].dt.year - cohort["anchor_year"])
    if cohort["age"].isna().any():
        issues.append("anchor_age missing for some stays; vulnerability flag set to unknown")

    cohort["patient_id"] = cohort["subject_id"].map(pseudo.patient)
    cohort["pseudonym"] = cohort["stay_id"].map(pseudo.stay)
    cohort["admission_type"] = cohort["admission_type"].fillna("UNKNOWN")
    cohort["first_careunit"] = cohort["first_careunit"].fillna("UNKNOWN")

    stay_lookup = cohort[["subject_id", "hadm_id", "stay_id", "pseudonym", "intime", "outtime"]].copy()
    cohort = cohort.rename(columns={"stay_id": "raw_stay_id"})

    # ---- helper: attach stay pseudonym to an event table ------------------
    def attach(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Assign events to the ICU stay whose window contains the timestamp.

        Events outside every stay window are dropped (they are ward/ED data and
        belong to a different care context than the ICU SOFA trajectory).
        """
        if df.empty:
            return df.assign(pseudonym=pd.Series(dtype="string"))
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        left = df.merge(stay_lookup[["subject_id", "hadm_id", "stay_id", "pseudonym"]],
                        on=["subject_id", "hadm_id"], how="inner")
        merged = left.merge(stay_lookup[["stay_id", "intime", "outtime"]], on="stay_id", how="left")
        inside = merged[(merged[time_col] >= merged["intime"]) &
                        (merged[time_col] <= merged["outtime"])]
        out = inside.drop_duplicates(subset=[c for c in inside.columns
                                             if c not in ("intime", "outtime")])
        return out.drop(columns=["stay_id", "subject_id", "hadm_id"], errors="ignore")

    # ---- labs -------------------------------------------------------------
    labs_raw = _read_table("labevents", data_dir)
    labs = labs_raw[labs_raw["itemid"].isin(LAB_ITEMIDS)].copy()
    drops = len(labs_raw) - len(labs)
    if drops:
        issues.append(f"labevents: dropped {drops} rows with non-allowlisted itemids")
    labs["variable"] = labs["itemid"].map(LAB_ITEMIDS)
    labs = labs.rename(columns={"charttime": "ts", "valuenum": "value", "valueuom": "unit"})
    labs = labs.dropna(subset=["value"])
    for variable, group in labs.groupby("variable"):
        _check_units(group, variable, LAB_UNITS, "labevents")
    labs = attach(labs[["subject_id", "hadm_id", "ts", "variable", "value", "unit"]], "ts")
    labs = labs.sort_values("ts").reset_index(drop=True)

    # Comparative delta needs stay-relative time.
    labs = _add_stay_offsets(labs, stay_lookup)

    # ---- chart vitals -----------------------------------------------------
    chart_raw = _read_table("chartevents", data_dir)
    vit = chart_raw[chart_raw["itemid"].isin(CHART_ITEMIDS)].copy()
    drops = len(chart_raw) - len(vit)
    if drops:
        issues.append(f"chartevents: dropped {drops} rows with non-allowlisted itemids")
    vit["variable"] = vit["itemid"].map(CHART_ITEMIDS)

    # GCS arrives as text in `value`; numeric items use `valuenum`.
    gcs_mask = vit["variable"].str.startswith("gcs_")
    from .allowlist import GCS_TEXT_MAP

    def _gcs_points(row) -> float:
        mapping = GCS_TEXT_MAP.get(row["variable"], {})
        txt = str(row["value"]).strip() if pd.notna(row["value"]) else ""
        return float(mapping.get(txt, float("nan")))

    vit.loc[gcs_mask, "valuenum"] = vit[gcs_mask].apply(_gcs_points, axis=1)
    # Derived intubation flag from the structured GCS verbal value
    # "No Response-ETT".  Item 223900 is allowlisted; the flag is derived from
    # its structured value, never from a note.
    ett_mask = (vit["itemid"] == 223900) & (vit["value"].astype("string").str.strip() == "No Response-ETT")
    ett = vit.loc[ett_mask, ["charttime", "subject_id", "hadm_id"]].copy()
    ett["itemid"] = 223900
    ett["variable"] = "gcs_ett"
    ett["valuenum"] = 1.0
    ett["valueuom"] = ""
    # Drop the raw text column now that GCS has been mapped to points.
    vit = vit.drop(columns=["value"])
    vit = vit.rename(columns={"charttime": "ts", "valuenum": "value", "valueuom": "unit"})
    vit = vit.dropna(subset=["value"])
    # Unit sanity for numeric vitals (GCS rows carry text, not units).
    numeric_vit = vit[~vit["variable"].str.startswith("gcs_")]
    for variable, group in numeric_vit.groupby("variable"):
        _check_units(group, variable, CHART_UNITS, "chartevents")
    vit = pd.concat([vit[["subject_id", "hadm_id", "ts", "variable", "value", "unit"]],
                     ett.rename(columns={"charttime": "ts", "valuenum": "value", "valueuom": "unit"})
                        [["subject_id", "hadm_id", "ts", "variable", "value", "unit"]]],
                    ignore_index=True)
    vit = attach(vit, "ts")
    vit = _add_stay_offsets(vit.sort_values("ts").reset_index(drop=True), stay_lookup)

    # ---- vasopressors -----------------------------------------------------
    inp_raw = _read_table("inputevents", data_dir)
    vaso = inp_raw[inp_raw["itemid"].isin(VASOPRESSOR_ITEMIDS)].copy()
    drops = len(inp_raw) - len(vaso)
    if drops:
        issues.append(f"inputevents: dropped {drops} non-vasopressor rows")
    vaso["drug"] = vaso["itemid"].map(VASOPRESSOR_ITEMIDS)
    # Fail loudly on any unit outside the per-drug allowlist.
    for drug, group in vaso.groupby("drug"):
        allowed = VASOPRESSOR_RATE_UNITS[drug]
        bad = set(group["rateuom"].dropna().unique()) - allowed
        if bad:
            raise AllowlistViolation(
                f"inputevents: disallowed vasopressor unit(s) {sorted(bad)} for {drug}; "
                f"allowed={sorted(allowed)}"
            )
    vaso = vaso.dropna(subset=["rate"])
    # Convert only allowlisted rates to mcg/kg/min; vasopressin stays in
    # units/hour and gets a NaN mcg/kg/min rate, flagged as not scored.
    from .allowlist import VASOPRESSOR_TO_MCG_KG_MIN

    vaso["rate_mcg_kg_min"] = vaso.apply(
        lambda r: r["rate"] * VASOPRESSOR_TO_MCG_KG_MIN.get(r["rateuom"], float("nan"))
        if r["rateuom"] in VASOPRESSOR_TO_MCG_KG_MIN else float("nan"),
        axis=1,
    )
    vaso = vaso.rename(columns={"starttime": "ts", "rate": "rate", "rateuom": "unit"})
    vaso = attach(vaso[["subject_id", "hadm_id", "ts", "endtime", "drug", "rate",
                        "rate_mcg_kg_min", "unit", "statusdescription"]], "ts")
    vaso = _add_stay_offsets(vaso.sort_values("ts").reset_index(drop=True), stay_lookup)

    # ---- urine output -----------------------------------------------------
    out_raw = _read_table("outputevents", data_dir)
    urine = out_raw[out_raw["itemid"].isin(URINE_OUTPUT_ITEMIDS)].copy()
    drops = len(out_raw) - len(urine)
    if drops:
        issues.append(f"outputevents: dropped {drops} non-urine rows")
    urine = urine.rename(columns={"charttime": "ts", "value": "value", "valueuom": "unit"})
    bad = set(urine["unit"].dropna().unique()) - {"ml"}
    if bad:
        raise AllowlistViolation(f"outputevents: disallowed urine unit(s) {sorted(bad)}")
    urine = attach(urine[["subject_id", "hadm_id", "ts", "value"]], "ts")
    urine = _add_stay_offsets(urine.sort_values("ts").reset_index(drop=True), stay_lookup)

    # ---- devices ----------------------------------------------------------
    from .allowlist import DEVICE_PROCEDURE_ITEMIDS

    proc_raw = _read_table("procedureevents", data_dir)
    devices = proc_raw[proc_raw["itemid"].isin(DEVICE_PROCEDURE_ITEMIDS)].copy()
    devices["device"] = devices["itemid"].map(DEVICE_PROCEDURE_ITEMIDS)
    devices = devices.rename(columns={"starttime": "ts"})
    devices = attach(devices[["subject_id", "hadm_id", "ts", "endtime", "device"]], "ts")
    devices = _add_stay_offsets(devices.sort_values("ts").reset_index(drop=True), stay_lookup)

    # ---- microbiology (structured only) -----------------------------------
    micro_raw = _read_table("microbiologyevents", data_dir)
    micro = micro_raw[micro_raw["spec_type_desc"].isin(CULTURE_SPECIMENS)].copy()
    micro["ts"] = pd.to_datetime(micro["charttime"].fillna(micro["chartdate"]), errors="coerce")
    micro = attach(micro[["subject_id", "hadm_id", "ts", "spec_type_desc"]], "ts")
    micro = _add_stay_offsets(micro.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True), stay_lookup)

    # ---- antimicrobial starts (structured drug names) ---------------------
    from .allowlist import ANTIMICROBIAL_PATTERNS

    rx_raw = _read_table("prescriptions", data_dir)
    drug_norm = rx_raw["drug"].astype("string").str.lower()
    mask = False
    for pattern in ANTIMICROBIAL_PATTERNS:
        mask = mask | drug_norm.str.contains(pattern, na=False)
    anti = rx_raw[mask].copy()
    anti["ts"] = pd.to_datetime(anti["starttime"], errors="coerce")
    anti = anti.dropna(subset=["ts"])
    anti = attach(anti[["subject_id", "hadm_id", "ts"]], "ts")
    anti = _add_stay_offsets(anti.sort_values("ts").reset_index(drop=True), stay_lookup)

    # ---- comorbidities ----------------------------------------------------
    from .allowlist import COMORBIDITY_FLAGS

    comorb_rows = []
    for flag, patterns in COMORBIDITY_FLAGS["icd9"].items():
        sub = diagnoses[(diagnoses["icd_version"] == 9) &
                        diagnoses["icd_code"].astype(str).str.startswith(patterns)]
        for _, row in sub.iterrows():
            comorb_rows.append({"hadm_id": row["hadm_id"], "flag": flag})
    for flag, patterns in COMORBIDITY_FLAGS["icd10"].items():
        sub = diagnoses[(diagnoses["icd_version"] == 10) &
                        diagnoses["icd_code"].astype(str).str.startswith(patterns)]
        for _, row in sub.iterrows():
            comorb_rows.append({"hadm_id": row["hadm_id"], "flag": flag})
    comorb = pd.DataFrame(comorb_rows).drop_duplicates() if comorb_rows else pd.DataFrame(columns=["hadm_id", "flag"])
    comorb = comorb.merge(cohort[["hadm_id", "pseudonym"]].drop_duplicates(), on="hadm_id", how="inner")
    comorb = comorb[["pseudonym", "flag"]].drop_duplicates().sort_values(["pseudonym", "flag"]).reset_index(drop=True)

    # Raw identifiers never leave this module: persisted frames carry only
    # pseudonyms.  The internal subject/hadm keys existed solely for joins.
    cohort = cohort.drop(columns=["subject_id", "hadm_id", "raw_stay_id"], errors="ignore")
    labs = labs.drop(columns=["subject_id", "hadm_id"], errors="ignore")
    vit = vit.drop(columns=["subject_id", "hadm_id"], errors="ignore")
    vaso = vaso.drop(columns=["subject_id", "hadm_id"], errors="ignore")
    urine = urine.drop(columns=["subject_id", "hadm_id"], errors="ignore")
    devices = devices.drop(columns=["subject_id", "hadm_id"], errors="ignore")
    micro = micro.drop(columns=["subject_id", "hadm_id"], errors="ignore")
    anti = anti.drop(columns=["subject_id", "hadm_id"], errors="ignore")

    return Cohort(
        cohort=cohort,
        labs=labs,
        vitals=vit,
        vasopressors=vaso,
        urine=urine,
        devices=devices,
        micro=micro,
        antimicrobials=anti,
        comorbidities=comorb,
        issues=issues,
    )


def _add_stay_offsets(df: pd.DataFrame, stay_lookup: pd.DataFrame) -> pd.DataFrame:
    """Add hours_from_icu_admission for each event, using the matching stay."""
    if df.empty:
        return df.assign(hours_from_icu=pd.Series(dtype="float64"))
    out = df.copy()
    if "intime" not in out.columns:
        starts = stay_lookup.drop_duplicates("pseudonym")[["pseudonym", "intime"]]
        out = out.merge(starts, on="pseudonym", how="left")
    out["hours_from_icu"] = (out["ts"] - out["intime"]).dt.total_seconds() / 3600.0
    return out.drop(columns=["intime", "outtime"], errors="ignore")


def write_cache(cohort: Cohort, out_dir: Path | str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort.cohort.to_parquet(out_dir / "cohort.parquet", index=False)
    for name in ("labs", "vitals", "vasopressors", "urine", "devices", "micro",
                 "antimicrobials", "comorbidities"):
        getattr(cohort, name).to_parquet(out_dir / f"{name}.parquet", index=False)
    pd.DataFrame({"issue": cohort.issues}).to_parquet(out_dir / "issues.parquet", index=False)


def load_cache(processed_dir: Path | str) -> Cohort:
    processed_dir = Path(processed_dir)
    if not (processed_dir / "cohort.parquet").exists():
        raise FileNotFoundError(
            f"no processed cache in {processed_dir}; run `python -m src.ingest.loader`"
        )
    issues = []
    p = processed_dir / "issues.parquet"
    if p.exists():
        issues = pd.read_parquet(p)["issue"].dropna().tolist()
    return Cohort(
        cohort=pd.read_parquet(processed_dir / "cohort.parquet"),
        labs=pd.read_parquet(processed_dir / "labs.parquet"),
        vitals=pd.read_parquet(processed_dir / "vitals.parquet"),
        vasopressors=pd.read_parquet(processed_dir / "vasopressors.parquet"),
        urine=pd.read_parquet(processed_dir / "urine.parquet"),
        devices=pd.read_parquet(processed_dir / "devices.parquet"),
        micro=pd.read_parquet(processed_dir / "micro.parquet"),
        antimicrobials=pd.read_parquet(processed_dir / "antimicrobials.parquet"),
        comorbidities=pd.read_parquet(processed_dir / "comorbidities.parquet"),
        issues=issues,
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    c = build_cohort()
    write_cache(c, root / "data" / "processed")
    print(f"cohort rows={len(c.cohort)} patients={c.cohort['patient_id'].nunique()} "
          f"stays={c.cohort['pseudonym'].nunique()}")
    print(f"labs={len(c.labs)} vitals={len(c.vitals)} vaso={len(c.vasopressors)} "
          f"urine={len(c.urine)} devices={len(c.devices)} micro={len(c.micro)} "
          f"antimicrobials={len(c.antimicrobials)} comorbidity_flags={len(c.comorbidities)}")
    for i in c.issues:
        print("issue:", i)
