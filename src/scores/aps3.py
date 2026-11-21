"""Prototype APS III (Acute Physiology Score III) estimator.

Design notes
------------
* Never labeled as APACHE II or full APACHE III (we do not add age/chronic
  health points or guess mortality equations).
* Uses MIMIC-derived APS III if available; otherwise computes a prototype
  estimate only when required variables are present, or returns
  ``status="unavailable"`` with the list of missing variables.

# TODO: verify  Every point weight and cut-point below is a simplified
# TODO: verify  prototype approximation of Knaus et al. (1991) APS III and MUST
# TODO: verify  be verified against the published APS III table before any
# TODO: verify  clinical or research use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_APS3_VARS = (
    "temp_c",
    "map",
    "heart_rate",
    "respiratory_rate",
    "oxygenation",
    "ph",
    "sodium",
    "hematocrit",
    "wbc",
    "bun",
    "glucose",
    "albumin",
    "bilirubin",
    "creatinine",
    "urine_24h",
    "gcs_total",
)

# Minimum core variables required to emit a prototype estimate instead of "unavailable".
# TODO: verify  Minimum completeness threshold for prototype APS III.
MIN_CORE_VARS = ("heart_rate", "respiratory_rate", "map", "temp_c", "gcs_total")


@dataclass
class APS3Result:
    score: int | None
    status: str  # "mimic_derived" | "prototype_estimate" | "unavailable"
    source: str
    missing_variables: list[str] = field(default_factory=list)
    components: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _pts_hr(v: float) -> int:
    # TODO: verify APS III heart rate weights
    if v < 40:
        return 8
    if v < 50:
        return 5
    if v < 100:
        return 0
    if v < 110:
        return 1
    if v < 120:
        return 5
    if v < 140:
        return 7
    if v < 155:
        return 13
    return 17


def _pts_map(v: float) -> int:
    # TODO: verify APS III MAP weights
    if v < 40:
        return 23
    if v < 60:
        return 15
    if v < 70:
        return 7
    if v < 80:
        return 6
    if v < 100:
        return 0
    if v < 120:
        return 4
    if v < 130:
        return 7
    if v < 140:
        return 9
    return 10


def _pts_temp(v: float) -> int:
    # TODO: verify APS III temperature (C) weights
    if v < 33.0:
        return 20
    if v < 33.5:
        return 16
    if v < 34.0:
        return 13
    if v < 35.0:
        return 8
    if v < 36.0:
        return 2
    if v < 40.0:
        return 0
    return 4


def _pts_rr(v: float) -> int:
    # TODO: verify APS III respiratory rate weights
    if v <= 5:
        return 17
    if v <= 11:
        return 8
    if v <= 13:
        return 7
    if v <= 24:
        return 0
    if v <= 34:
        return 6
    if v <= 39:
        return 9
    if v <= 49:
        return 11
    return 18


def _pts_pao2(v: float) -> int:
    # TODO: verify APS III PaO2 weights (assumes FiO2 < 0.5 simplified)
    if v < 50:
        return 15
    if v < 70:
        return 5
    if v < 80:
        return 2
    return 0


def _pts_hct(v: float) -> int:
    # TODO: verify APS III hematocrit weights
    if v < 41.0:
        return 3
    if v < 50.0:
        return 0
    return 3


def _pts_wbc(v: float) -> int:
    # TODO: verify APS III WBC weights
    if v < 1.0:
        return 19
    if v < 3.0:
        return 5
    if v < 20.0:
        return 0
    if v < 25.0:
        return 1
    return 5


def _pts_creatinine(v: float) -> int:
    # TODO: verify APS III creatinine weights (without ARF modifier)
    if v < 0.5:
        return 3
    if v < 1.5:
        return 0
    if v < 1.95:
        return 4
    return 7


def _pts_urine(v: float) -> int:
    # TODO: verify APS III 24h urine output weights
    if v < 400:
        return 15
    if v < 600:
        return 8
    if v < 900:
        return 7
    if v < 1500:
        return 5
    if v < 2000:
        return 4
    if v < 4000:
        return 0
    return 1


def _pts_bun(v: float) -> int:
    # TODO: verify APS III BUN weights
    if v < 17:
        return 0
    if v < 20:
        return 2
    if v < 40:
        return 7
    if v < 80:
        return 11
    return 12


def _pts_sodium(v: float) -> int:
    # TODO: verify APS III sodium weights
    if v < 120:
        return 3
    if v < 135:
        return 2
    if v < 155:
        return 0
    return 4


def _pts_albumin(v: float) -> int:
    # TODO: verify APS III albumin weights
    if v < 2.0:
        return 11
    if v < 2.5:
        return 6
    if v < 4.5:
        return 0
    return 4


def _pts_bilirubin(v: float) -> int:
    # TODO: verify APS III bilirubin weights
    if v < 2.0:
        return 0
    if v < 3.0:
        return 5
    if v < 5.0:
        return 6
    if v < 8.0:
        return 8
    return 16


def _pts_glucose(v: float) -> int:
    # TODO: verify APS III glucose weights
    if v < 40:
        return 8
    if v < 60:
        return 9
    if v < 200:
        return 0
    if v < 350:
        return 3
    return 5


def _pts_ph(v: float) -> int:
    # TODO: verify APS III pH weights (simplified without PaCO2 interaction)
    if v < 7.20:
        return 12
    if v < 7.35:
        return 3
    if v <= 7.45:
        return 0
    if v <= 7.55:
        return 2
    return 5


def _pts_gcs(v: float) -> int:
    # TODO: verify APS III GCS weights (simplified total-GCS mapping)
    if v >= 15:
        return 0
    if v >= 13:
        return 4
    if v >= 10:
        return 10
    if v >= 6:
        return 20
    return 33


_SCORERS = {
    "temp_c": _pts_temp,
    "map": _pts_map,
    "heart_rate": _pts_hr,
    "respiratory_rate": _pts_rr,
    "oxygenation": _pts_pao2,
    "ph": _pts_ph,
    "sodium": _pts_sodium,
    "hematocrit": _pts_hct,
    "wbc": _pts_wbc,
    "bun": _pts_bun,
    "glucose": _pts_glucose,
    "albumin": _pts_albumin,
    "bilirubin": _pts_bilirubin,
    "creatinine": _pts_creatinine,
    "urine_24h": _pts_urine,
    "gcs_total": _pts_gcs,
}


def compute_aps3(
    values: dict[str, float | None],
    *,
    mimic_derived_aps3: int | None = None,
    require_all: bool = False,
) -> APS3Result:
    """Compute APS III or mark unavailable.

    Never labels output as APACHE II or full APACHE III.
    """
    if mimic_derived_aps3 is not None and not pd.isna(mimic_derived_aps3):
        return APS3Result(
            score=int(mimic_derived_aps3),
            status="mimic_derived",
            source="mimic_derived",
            notes=["Using MIMIC-derived APS III (physiology only; not full APACHE III)."],
        )

    missing = [
        k for k in REQUIRED_APS3_VARS if values.get(k) is None or pd.isna(values.get(k))
    ]
    core_missing = [k for k in MIN_CORE_VARS if k in missing]

    if require_all and missing:
        return APS3Result(
            score=None,
            status="unavailable",
            source="unavailable",
            missing_variables=missing,
            notes=["APS III unavailable: missing required physiological variables."],
        )

    if core_missing:
        return APS3Result(
            score=None,
            status="unavailable",
            source="unavailable",
            missing_variables=missing,
            notes=[
                f"APS III unavailable: missing core variables ({', '.join(core_missing)})."
            ],
        )

    components: dict[str, int] = {}
    for var, fn in _SCORERS.items():
        val = values.get(var)
        if val is not None and not pd.isna(val):
            components[var] = int(fn(float(val)))

    notes = [
        "Prototype APS III physiology estimate only (not APACHE II / full APACHE III; no mortality equation)."
    ]
    if missing:
        notes.append(
            f"Missing {len(missing)}/{len(REQUIRED_APS3_VARS)} APS III variables scored as 0 (not imputed): {', '.join(missing)}."
        )

    return APS3Result(
        score=int(sum(components.values())),
        status="prototype_estimate",
        source="prototype_estimate",
        missing_variables=missing,
        components=components,
        notes=notes,
    )


def extract_aps3_inputs(
    events: dict[str, pd.DataFrame], t_hours: float, window_hours: float = 24.0
) -> dict[str, float | None]:
    """Extract leakage-safe (<= t_hours) worst/latest inputs for APS III."""
    lo = t_hours - window_hours

    def within(key: str) -> pd.DataFrame:
        df = events.get(key)
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["value", "hours_from_icu"])
        return df[(df["hours_from_icu"] > lo) & (df["hours_from_icu"] <= t_hours)]

    def latest(key: str) -> float | None:
        df = within(key)
        if len(df) == 0:
            return None
        return float(df.loc[df["hours_from_icu"].idxmax(), "value"])

    # Temperature: prefer Celsius, convert Fahrenheit if needed.
    temp_c = latest("chart:temp_celsius")
    if temp_c is None:
        temp_f = latest("chart:temp_fahrenheit")
        if temp_f is not None:
            temp_c = (temp_f - 32.0) * 5.0 / 9.0

    map_art = within("chart:map_arterial")
    map_ni = within("chart:map_noninvasive")
    map_all = pd.concat([map_art, map_ni]) if len(map_art) or len(map_ni) else pd.DataFrame()
    map_val = float(map_all["value"].min()) if len(map_all) else None

    gcs_parts = [latest(f"chart:{p}") for p in ("gcs_eye", "gcs_verbal", "gcs_motor")]
    gcs_total = sum(gcs_parts) if all(v is not None for v in gcs_parts) else None

    urine_df = within("urine:all")
    urine_24h = float(urine_df["value"].sum()) if len(urine_df) else None

    return {
        "temp_c": temp_c,
        "map": map_val,
        "heart_rate": latest("chart:heart_rate"),
        "respiratory_rate": latest("chart:respiratory_rate"),
        "oxygenation": latest("lab:pao2"),
        "ph": latest("lab:ph"),
        "sodium": latest("lab:sodium"),
        "hematocrit": latest("lab:hematocrit"),
        "wbc": latest("lab:wbc"),
        "bun": latest("lab:bun"),
        "glucose": latest("lab:glucose"),
        "albumin": latest("lab:albumin"),
        "bilirubin": latest("lab:bilirubin"),
        "creatinine": latest("lab:creatinine"),
        "urine_24h": urine_24h,
        "gcs_total": gcs_total,
    }
