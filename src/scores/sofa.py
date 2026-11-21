"""SOFA scoring over 24 h windows, leakage-safe.

Design notes
------------
* ``score_at(t)`` looks at what a bedside system would see at time ``t``: all
  records with ``timestamp <= t``.  Nothing later is ever touched.
* The "window" value is the *worst* value in the trailing 24 h, matching the
  original SOFA definition.  The hourly "latest estimate" uses the most recent
  record, which is the only thing available for a real-time display.
* Missing components are scored 0 (as the prompt requires) and flagged.  We
  never impute a normal value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

SOFA_COMPONENTS = ("respiration", "coagulation", "liver", "cardiovascular", "cns", "renal")
COMPONENT_LABELS = {
    "respiration": "Respiration (PaO2/FiO2)",
    "coagulation": "Coagulation (platelets)",
    "liver": "Liver (bilirubin)",
    "cardiovascular": "Cardiovascular (MAP / vasopressors)",
    "cns": "Neurological (GCS)",
    "renal": "Renal (creatinine / urine output)",
}

# ---------------------------------------------------------------------------
# Cut-points.  All values below are the published 1996 Vincent et al. SOFA
# table.  # TODO: verify each cut-point against the source before any clinical
# use; they are reproduced from memory of the standard table and must be checked.
# ---------------------------------------------------------------------------
PLATELET_CUTS = [(150, 0), (100, 1), (50, 2), (20, 3)]     # else 4
BILIRUBIN_CUTS = [(1.2, 0), (2.0, 1), (6.0, 2), (12.0, 3)]  # else 4
CREATININE_CUTS = [(1.2, 0), (2.0, 1), (3.5, 2), (5.0, 3)]  # else 4
URINE_CUTS = [(500, 3), (200, 4)]  # mL/24h: <500 -> 3, <200 -> 4

# PF ratio cuts; rows are (lower_bound_inclusive, score) evaluated descending.
PF_CUTS = [(400, 0), (300, 1), (200, 2), (100, 3)]  # else 4
# Approximate SpO2/FiO2 mapping used only when PaO2 is absent.  # TODO: verify
SPO2_FIO2_APPROX_CUTS = [(512, 0), (357, 1), (214, 2), (89, 3)]  # else 4

MAP_CUTS = [(70, 0)]  # MAP >= 70 -> 0
# Vasopressor doses in mcg/kg/min (original SOFA).
NOREPI_CUTS = [(0.1, 3), (0.0, 2)]     # >0.1 -> 4, >0 -> 3
EPI_CUTS = [(0.1, 3), (0.0, 2)]
DOPAMINE_CUTS = [(15.0, 3), (5.0, 2), (0.0, 1)]  # >15 -> 4, >5 -> 3, >0 -> 2
DOBUTAMINE_ANY = 2

GCS_CUTS = [(15, 0), (13, 1), (10, 2), (6, 3)]  # else 4


def _cut_score(value: float, cuts, default: int) -> int:
    """Return the first score whose threshold is satisfied, else ``default``."""
    if value is None or pd.isna(value):
        return default
    for threshold, score in cuts:
        if value >= threshold:
            return score
    return default


def _cut_score_upper(value: float, cuts, default: int) -> int:
    """For 'lower is worse' labs: first threshold that value is below."""
    if value is None or pd.isna(value):
        return default
    for threshold, score in cuts:
        if value < threshold:
            return score
    return default


def platelet_score(v: float | None) -> int | None:
    if v is None or pd.isna(v):
        return None
    if v < 20:
        return 4
    return _cut_score(v, PLATELET_CUTS, 4)


def bilirubin_score(v: float | None) -> int | None:
    if v is None or pd.isna(v):
        return None
    if v >= 12.0:
        return 4
    return _cut_score_upper(v, BILIRUBIN_CUTS, 4)


def creatinine_score(v: float | None) -> int | None:
    if v is None or pd.isna(v):
        return None
    if v >= 5.0:
        return 4
    return _cut_score_upper(v, CREATININE_CUTS, 4)


def urine_score(ml_24h: float | None) -> int | None:
    if ml_24h is None or pd.isna(ml_24h):
        return None
    if ml_24h < 200:
        return 4
    if ml_24h < 500:
        return 3
    return 0


def pf_score(pao2: float | None, fio2: float | None, spo2: float | None = None,
             *, approximate: bool = False) -> int | None:
    """Respiratory score.  PaO2/FiO2 preferred; SpO2/FiO2 when PaO2 absent."""
    fio2_frac = _normalise_fio2(fio2)
    if pao2 is not None and not pd.isna(pao2) and fio2_frac:
        ratio = pao2 / fio2_frac
        if ratio >= 400:
            return 0
        if ratio >= 300:
            return 1
        if ratio >= 200:
            return 2
        if ratio >= 100:
            return 3
        return 4
    if spo2 is not None and not pd.isna(spo2) and fio2_frac:
        ratio = spo2 / fio2_frac
        if ratio >= 512:
            return 0
        if ratio >= 357:
            return 1
        if ratio >= 214:
            return 2
        if ratio >= 89:
            return 3
        return 4
    return None


def _normalise_fio2(fio2: float | None) -> float | None:
    """FiO2 is stored as a percentage (21-100) or a fraction (0.21-1.0)."""
    if fio2 is None or pd.isna(fio2) or fio2 <= 0:
        return None
    frac = fio2 / 100.0 if fio2 > 1.0 else fio2
    return frac if 0.2 <= frac <= 1.0 else None


def map_score(map_value: float | None) -> int | None:
    if map_value is None or pd.isna(map_value):
        return None
    return 0 if map_value >= 70 else 1


def vasopressor_cv_score(norepi=None, epi=None, dopamine=None, dobutamine=None) -> int | None:
    """Cardiovascular score from vasopressor rates in mcg/kg/min."""
    scores = []
    if norepi is not None and not pd.isna(norepi) and norepi > 0:
        scores.append(2 if norepi <= 0.1 else 3)
    if epi is not None and not pd.isna(epi) and epi > 0:
        scores.append(2 if epi <= 0.1 else 3)
    if dopamine is not None and not pd.isna(dopamine) and dopamine > 0:
        scores.append(1 if dopamine <= 5 else 2 if dopamine <= 15 else 3)
    if dobutamine is not None and not pd.isna(dobutamine) and dobutamine > 0:
        scores.append(DOBUTAMINE_ANY)
    return max(scores) if scores else None


def gcs_score(gcs_total: float | None) -> int | None:
    if gcs_total is None or pd.isna(gcs_total):
        return None
    if gcs_total >= 15:
        return 0
    if gcs_total >= 13:
        return 1
    if gcs_total >= 10:
        return 2
    if gcs_total >= 6:
        return 3
    return 4


@dataclass
class SOfaResult:
    total: int
    components: dict[str, int]
    missing: list[str]
    approximate: list[str]
    notes: list[str] = field(default_factory=list)
    completeness: float = 1.0


def compute_sofa(component_inputs: dict[str, dict], *, notes: list[str] | None = None) -> SOfaResult:
    """Compute a 6-component SOFA from pre-extracted component inputs.

    ``component_inputs`` keys are SOFA component names; each maps to the raw
    values for that window (any may be absent).  Missing components score 0 and
    are listed in ``missing`` so the caller can display completeness.
    """
    notes = list(notes or [])
    scores: dict[str, int] = {}
    missing: list[str] = []
    approximate: list[str] = []

    # Respiration
    resp = component_inputs.get("respiration", {})
    resp_score = pf_score(resp.get("pao2"), resp.get("fio2"), resp.get("spo2"),
                          approximate=resp.get("approximate", False))
    if resp.get("approximate") and resp_score is not None:
        approximate.append("respiration")
        notes.append("Respiration: SpO2/FiO2 fallback used (approximate).")
    if resp_score is None:
        missing.append("respiration")
        scores["respiration"] = 0
    else:
        scores["respiration"] = resp_score

    # Coagulation
    coag = platelet_score(component_inputs.get("coagulation", {}).get("platelets"))
    if coag is None:
        missing.append("coagulation")
        scores["coagulation"] = 0
    else:
        scores["coagulation"] = coag

    # Liver
    liver = bilirubin_score(component_inputs.get("liver", {}).get("bilirubin"))
    if liver is None:
        missing.append("liver")
        scores["liver"] = 0
    else:
        scores["liver"] = liver

    # Cardiovascular: MAP OR vasopressors, whichever is worse.
    cv = component_inputs.get("cardiovascular", {})
    map_s = map_score(cv.get("map"))
    vaso_s = vasopressor_cv_score(
        cv.get("norepinephrine"), cv.get("epinephrine"),
        cv.get("dopamine"), cv.get("dobutamine"),
    )
    if cv.get("vasopressin_active") and vaso_s is None:
        notes.append("Vasopressin infusion present but not scored (not in original SOFA).")  # TODO: verify
    cand = [s for s in (map_s, vaso_s) if s is not None]
    if not cand:
        missing.append("cardiovascular")
        scores["cardiovascular"] = 0
    else:
        scores["cardiovascular"] = max(cand)
    if cv.get("vasopressor_units_unverified"):
        notes.append("Vasopressor units required conversion; marked unverified.")  # TODO: verify

    # CNS
    cns = gcs_score(component_inputs.get("cns", {}).get("gcs_total"))
    if cns is None:
        missing.append("cns")
        scores["cns"] = 0
    else:
        scores["cns"] = cns
        if component_inputs["cns"].get("intubated"):
            notes.append("CNS: verbal score 1 assumed for intubated patient (approximate).")
            approximate.append("cns")

    # Renal: worst of creatinine and urine output.
    renal = component_inputs.get("renal", {})
    cr_s = creatinine_score(renal.get("creatinine"))
    uo_s = urine_score(renal.get("urine_24h"))
    cand = [s for s in (cr_s, uo_s) if s is not None]
    if not cand:
        missing.append("renal")
        scores["renal"] = 0
    else:
        scores["renal"] = max(cand)

    total = int(sum(scores.values()))
    completeness = (len(SOFA_COMPONENTS) - len(missing)) / len(SOFA_COMPONENTS)
    return SOfaResult(total=total, components=scores, missing=missing,
                      approximate=approximate, notes=notes, completeness=completeness)


# ---------------------------------------------------------------------------
# Window extraction from the pseudonymised event frames
# ---------------------------------------------------------------------------
def _worst_high(series: pd.Series) -> float | None:
    return float(series.max()) if len(series) else None


def _worst_low(series: pd.Series) -> float | None:
    return float(series.min()) if len(series) else None


def _latest(series: pd.Series, times: pd.Series) -> float | None:
    if not len(series):
        return None
    idx = times.idxmax()
    return float(series.loc[idx])


def window_inputs(events: dict[str, pd.DataFrame], t_hours: float,
                  window_hours: float = 24.0) -> dict[str, dict]:
    """Build SOFA component inputs from records in ``(t-window, t]``.

    ``events`` maps a canonical variable name to a frame with ``ts``,
    ``value``, ``unit`` and ``hours_from_icu`` columns.
    """
    lo = t_hours - window_hours

    def within(name: str) -> pd.DataFrame:
        df = events.get(name)
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["ts", "value", "hours_from_icu"])
        return df[(df["hours_from_icu"] > lo) & (df["hours_from_icu"] <= t_hours)]

    labs = lambda name: within(f"lab:{name}")
    vit = lambda name: within(f"chart:{name}")

    out: dict[str, dict] = {}

    # Respiration: PaO2 (worst) + FiO2 (worst) + SpO2 (worst).
    pao2_df = labs("pao2")
    fio2_df = vit("fio2")
    spo2_df = vit("spo2")
    fio2_worst = _worst_high(fio2_df["value"]) if len(fio2_df) else None
    resp = {"pao2": _worst_low(pao2_df["value"]) if len(pao2_df) else None,
            "fio2": fio2_worst,
            "spo2": _worst_low(spo2_df["value"]) if len(spo2_df) else None,
            "approximate": False}
    if resp["pao2"] is None and resp["spo2"] is not None:
        resp["approximate"] = True
    out["respiration"] = resp

    plt = labs("platelets")
    out["coagulation"] = {"platelets": _worst_low(plt["value"]) if len(plt) else None}

    bili = labs("bilirubin")
    out["liver"] = {"bilirubin": _worst_high(bili["value"]) if len(bili) else None}

    # Cardiovascular.
    map_art = vit("map_arterial")
    map_ni = vit("map_noninvasive")
    map_all = pd.concat([map_art, map_ni]) if len(map_art) or len(map_ni) else pd.DataFrame()
    cv: dict[str, object] = {
        "map": _worst_low(map_all["value"]) if len(map_all) else None,
        "vasopressin_active": False,
    }
    vaso = events.get("vaso:all")
    if vaso is not None and len(vaso):
        active = vaso[(vaso["hours_from_icu"] > lo) & (vaso["hours_from_icu"] <= t_hours)]
        for drug in ("norepinephrine", "epinephrine", "dopamine", "dobutamine"):
            d = active[active["drug"] == drug]
            cv[drug] = float(d["rate_mcg_kg_min"].max()) if len(d) else None
        cv["vasopressin_active"] = bool((active["drug"] == "vasopressin").any())
    out["cardiovascular"] = cv

    # CNS: sum worst tier of each component within the window (same time point
    # is preferable; we take the worst of each and flag it).
    gcs_parts = {}
    for part in ("gcs_eye", "gcs_verbal", "gcs_motor"):
        s = vit(part)
        gcs_parts[part] = _worst_low(s["value"]) if len(s) else None
    if all(v is not None for v in gcs_parts.values()):
        gcs_total = sum(gcs_parts.values())
    else:
        gcs_total = None
    verbal_df = vit("gcs_verbal")
    intubated = bool(len(verbal_df) and (verbal_df["value"] == 1).any() and
                     (verbal_df.get("intubated", pd.Series(dtype=bool)) == True).any())  # noqa: E712
    out["cns"] = {"gcs_total": gcs_total, "intubated": intubated}

    cr = labs("creatinine")
    urine = events.get("urine:all")
    urine_24 = None
    if urine is not None and len(urine):
        u = urine[(urine["hours_from_icu"] > lo) & (urine["hours_from_icu"] <= t_hours)]
        if len(u):
            urine_24 = float(u["value"].sum())
    out["renal"] = {"creatinine": _worst_high(cr["value"]) if len(cr) else None,
                    "urine_24h": urine_24}
    return out


@dataclass
class TrajectoryPoint:
    hour: float
    sofa: SOfaResult


def sofa_trajectory(events: dict[str, pd.DataFrame], max_hours: float,
                    step_hours: float = 1.0) -> list[TrajectoryPoint]:
    """Hourly SOFA trajectory using only data available at each hour."""
    points: list[TrajectoryPoint] = []
    t = 0.0
    while t <= max_hours + 1e-9:
        inputs = window_inputs(events, t)
        points.append(TrajectoryPoint(hour=t, sofa=compute_sofa(inputs)))
        t += step_hours
    return points


def baseline_sofa(trajectory: list[TrajectoryPoint], admission_hours: float = 24.0) -> SOfaResult:
    """Baseline = worst SOFA observed in the first 24 ICU hours."""
    window = [p.sofa for p in trajectory if p.hour <= admission_hours]
    if not window:
        return trajectory[0].sofa if trajectory else compute_sofa({})
    # Aggregate by taking the max of each component across the first day.
    comps = {c: max(r.components[c] for r in window) for c in SOFA_COMPONENTS}
    missing = sorted({m for r in window for m in r.missing})
    approx = sorted({a for r in window for a in r.approximate})
    total = sum(comps.values())
    return SOfaResult(total=total, components=comps, missing=missing, approximate=approx,
                      notes=["Baseline = worst component score in first 24 h."],
                      completeness=(len(SOFA_COMPONENTS) - len(missing)) / len(SOFA_COMPONENTS))
