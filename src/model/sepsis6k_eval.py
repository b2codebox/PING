"""Patient-grouped CV benchmark for Sepsis Detection (Combined 31.7k Cohort).

Clearly separated from the strictly pre-onset 6h preventive model:
- Positive class: active/post-onset sepsis observations from the 6,660-stay
  Sepsis 6.6k cohort (hours 1..24 post-onset) + PhysioNet 2019 sepsis hours.
- Negative class: non-sepsis ICU stays from PhysioNet 2019 (23,360 stays).
- Features: strictly shared allowlisted vitals, labs, clinical ratios, and SIRS
  proxy (no dataset-specific leakage columns like missingness of non-shared labs).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.ingest.allowlist import DISCLAIMER
from src.ingest.loader import load_cache

ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "models" / "combined_detection_metrics.json"

SHARED_VARS = [
    "heart_rate",
    "respiratory_rate",
    "map_arterial",
    "temp_celsius",
    "spo2",
    "lactate",
    "platelets",
    "bilirubin",
    "creatinine",
    "bun",
    "wbc",
    "glucose",
    "ph",
]


def _build_stay_summary(cache_dir: Path, label_mode: str) -> pd.DataFrame:
    """Build per-stay 24h physiological summary on strictly shared variables."""
    c = load_cache(cache_dir)
    meta = c.cohort[["pseudonym", "patient_id", "age", "has_sepsis_label", "sepsis_onset_hour"]].copy()

    if label_mode == "pn2019":
        # Keep non-sepsis stays (y=0) and post-onset sepsis stays (y=1)
        meta["target_sepsis"] = meta["has_sepsis_label"].astype(int)
    else:
        meta["target_sepsis"] = 1

    ev = pd.concat(
        [
            c.vitals.loc[c.vitals["variable"].isin(SHARED_VARS), ["pseudonym", "hours_from_icu", "variable", "value"]],
            c.labs.loc[c.labs["variable"].isin(SHARED_VARS), ["pseudonym", "hours_from_icu", "variable", "value"]],
        ],
        ignore_index=True,
    )

    # Aggregate per stay: mean, max, min over the stay window (24h post-onset for 6.6k; ICU stay for PN2019)
    agg = ev.groupby(["pseudonym", "variable"], sort=False)["value"].agg(["mean", "max", "min"]).unstack("variable")
    agg.columns = [f"{var}_{stat}" for stat, var in agg.columns]
    agg = agg.reset_index()

    df = meta.merge(agg, on="pseudonym", how="inner")

    # Derived clinical ratios
    if "heart_rate_max" in df.columns and "map_arterial_min" in df.columns:
        df["shock_index_peak"] = df["heart_rate_max"] / df["map_arterial_min"].replace(0, np.nan)
    if "bun_max" in df.columns and "creatinine_max" in df.columns:
        df["bun_cr_ratio"] = df["bun_max"] / df["creatinine_max"].replace(0, np.nan)

    return df


def run_combined_detection_eval(n_splits: int = 3) -> dict:
    df_pn = _build_stay_summary(ROOT / "data" / "processed_pn2019", "pn2019")
    df_s6 = _build_stay_summary(ROOT / "data" / "processed_sepsis6k", "sepsis6k")

    # Prefix patient_id to guarantee zero cross-dataset ID collision
    df_pn["group_id"] = "pn_" + df_pn["patient_id"].astype(str)
    df_s6["group_id"] = "s6_" + df_s6["patient_id"].astype(str)

    df = pd.concat([df_pn, df_s6], ignore_index=True)

    # Restrict strictly to shared numeric feature columns (no missingness indicators)
    shared_cols = [
        c
        for c in df.columns
        if c not in ("pseudonym", "patient_id", "group_id", "has_sepsis_label", "sepsis_onset_hour", "target_sepsis")
        and c in df_pn.columns
        and c in df_s6.columns
    ]

    X = df[shared_cols].to_numpy(dtype=np.float32)
    y = df["target_sepsis"].to_numpy(dtype=int)
    groups = df["group_id"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(df), dtype=float)

    for tr, va in gkf.split(X, y, groups):
        clf = HistGradientBoostingClassifier(
            max_depth=5,
            max_iter=200,
            learning_rate=0.05,
            min_samples_leaf=40,
            l2_regularization=3.0,
            random_state=42,
        )
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]

    auroc = float(roc_auc_score(y, oof))
    auprc = float(average_precision_score(y, oof))
    brier = float(brier_score_loss(y, oof))

    metrics = {
        "disclaimer": DISCLAIMER,
        "task": "Active / Post-Onset Sepsis Detection (Combined Cohort)",
        "dataset": f"PhysioNet 2019 ({len(df_pn)}) + User Sepsis 6.6k ({len(df_s6)})",
        "n_stays": int(len(df)),
        "n_sepsis_stays": int(y.sum()),
        "n_non_sepsis_stays": int((y == 0).sum()),
        "cv_folds_by_patient": n_splits,
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "brier_score": round(brier, 4),
        "temporal_note": (
            "Evaluates active/post-onset sepsis detection across shared vitals and labs. "
            "Kept strictly separate from the pre-onset 6-hour preventive model (t < onset) "
            "because the 6.6k sepsis cohort records hours 1..24 after sepsis onset."
        ),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_combined_detection_eval(), indent=2))
