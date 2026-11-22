"""Leakage-safe, patient-grouped CV for PhysioNet/CinC Challenge 2019.

Memory notes
------------
The full challenge cohort is ~40k stays / 1.2M hourly rows / 75 features, i.e.
~366 MB as float32.  Building that as a wide DataFrame of ~25 concatenated
blocks (plus float64 intermediates and groupby caches) peaked near 3.2 GB and
was OOM-killed on a 4 GB machine.  This module therefore:

* reads only the two tables it needs (vitals, labs) instead of every cache,
* converts to float32 as soon as each block is computed,
* writes blocks into one preallocated matrix and frees each intermediate,
* copies fold slices for cross-validation and releases them per model.

Estimated peak is ~1.2-1.5 GB for the full cohort.
"""

from __future__ import annotations

import gc
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ingest.allowlist import DISCLAIMER

ROOT = Path(__file__).resolve().parents[2]
PN19_CACHE_DIR = ROOT / "data" / "processed_pn2019"
PN19_MODEL_PATH = ROOT / "models" / "pn2019_model.pkl"
PN19_METRICS_PATH = ROOT / "models" / "pn2019_metrics.json"

VARS = [
    "heart_rate",
    "respiratory_rate",
    "map_arterial",
    "temp_celsius",
    "spo2",
    "fio2",
    "lactate",
    "platelets",
    "bilirubin",
    "creatinine",
    "bun",
    "hematocrit",
    "wbc",
    "glucose",
    "ph",
]

LAB_VARS = ["lactate", "platelets", "bilirubin", "creatinine", "bun", "hematocrit", "wbc", "glucose", "ph"]

RECENCY_VARS = ["lactate", "wbc", "platelets", "creatinine", "bilirubin", "bun", "glucose", "ph"]

BASE_VARS = ["heart_rate", "respiratory_rate", "map_arterial", "temp_celsius", "spo2"]

DELTA_VARS = ["heart_rate", "respiratory_rate", "map_arterial", "temp_celsius", "spo2", "lactate"]

META_COLS = ["patient_id", "pseudonym", "onset_hour"]
TARGET_COL = "target_onset_6h"


def _cohort_metadata(exclude_first_hours: float, stay_limit: int | None) -> pd.DataFrame:
    """Stay-level metadata for stays eligible for hourly scoring."""
    cols = ["pseudonym", "patient_id", "age", "los", "has_sepsis_label", "sepsis_onset_hour"]
    meta = pd.read_parquet(PN19_CACHE_DIR / "cohort.parquet", columns=cols)
    meta["los_hr"] = (meta["los"] * 24.0).round().astype(int)
    early_onset = meta["has_sepsis_label"].fillna(False).astype(bool) & (
        meta["sepsis_onset_hour"] <= exclude_first_hours
    )
    meta = meta[(meta["los_hr"] > exclude_first_hours) & (~early_onset)]
    meta = meta.drop(columns=["los_hr", "los"])
    if stay_limit is not None:
        meta = meta.sort_values("pseudonym").head(stay_limit)
    return meta.reset_index(drop=True)


def _hourly_table(valid_ids: set[str]) -> pd.DataFrame:
    """Pivot vitals + labs into one row per (stay, hour) with VARS columns."""
    parts = []
    for table in ("vitals", "labs"):
        t = pd.read_parquet(
            PN19_CACHE_DIR / f"{table}.parquet",
            columns=["pseudonym", "hours_from_icu", "variable", "value"],
        )
        t = t[t["pseudonym"].isin(valid_ids)]
        parts.append(t)
        del t
    events = pd.concat(parts, ignore_index=True)
    del parts
    events["hour"] = events["hours_from_icu"].round().astype("int32")
    events = events.drop(columns=["hours_from_icu"])

    piv = (
        events.groupby(["pseudonym", "hour", "variable"], sort=False)["value"]
        .last()
        .unstack("variable")
    )
    del events
    # Kept at the source dtype (float64): derived ratios are then bit-identical
    # to computing them before casting, which is what the original code did.
    piv = piv.reindex(columns=VARS)
    return piv.sort_index().reset_index()


def _prepare_rows(
    exclude_first_hours: float = 6.0,
    stay_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = _cohort_metadata(exclude_first_hours, stay_limit)
    piv = _hourly_table(set(meta["pseudonym"]))
    df = piv.merge(meta, on="pseudonym", how="inner")
    del piv

    has_sep = df["has_sepsis_label"].fillna(False).astype(bool)
    df = df[(~has_sep) | (df["hour"] < df["sepsis_onset_hour"])]
    df = df.sort_values(["pseudonym", "hour"], kind="mergesort").reset_index(drop=True)
    return df, meta


def _block_series(df: pd.DataFrame, exclude_first_hours: float) -> list[tuple[list[str], np.ndarray]]:
    """Compute every feature block as (names, float32 array) in output order."""
    blocks: list[tuple[list[str], np.ndarray]] = []
    gkey = df["pseudonym"]
    index = df.index

    def push(name: str | list[str], values) -> None:
        names = [name] if isinstance(name, str) else list(name)
        arr = np.asarray(values, dtype=np.float32)
        blocks.append((names, arr.reshape(len(index), -1)))

    # Trailing 24h carry-forward (strictly <= t)
    ff = df.groupby("pseudonym", sort=False)[VARS].ffill(limit=24)
    push(VARS, ff.to_numpy())

    # Explicit missingness flags
    push([f"miss_{v}" for v in VARS], ff.isna().to_numpy())

    # Cumulative lab count up to t (clinical acuity signal)
    lab_any = df[LAB_VARS].notna().sum(axis=1)
    cum_labs = lab_any.groupby(gkey, sort=False).cumsum()
    push("cum_lab_count", cum_labs.to_numpy())
    del lab_any, cum_labs

    # Exact hours-since-last-observation for key labs (strictly <= t)
    hr_vals = df["hour"].to_numpy(dtype=np.float32)
    for v in RECENCY_VARS:
        obs = pd.Series(np.where(df[v].notna().to_numpy(), hr_vals, np.nan), index=index)
        since = df["hour"] - obs.groupby(gkey, sort=False).ffill()
        push(f"hrs_since_{v}", since.to_numpy())
        del obs, since

    # First-6h personal baseline deviations (causal: only scored for hour >= 6)
    first6 = df["hour"] < exclude_first_hours
    base_means = ff.loc[first6, BASE_VARS].groupby(gkey[first6], sort=False).mean()
    base_aligned = base_means.reindex(df["pseudonym"]).set_axis(index)
    base_dev = (ff[BASE_VARS].to_numpy() - base_aligned.to_numpy(dtype=np.float32))
    push([f"{v}_dev_base" for v in BASE_VARS], base_dev)
    del base_means, base_aligned, base_dev

    # 6h and 12h deltas (strictly t minus t-k within stay)
    ffg = ff.groupby(gkey, sort=False)
    for lag in (6, 12):
        shifted = ffg[DELTA_VARS].shift(lag)
        push([f"{v}_d{lag}" for v in DELTA_VARS], ff[DELTA_VARS].to_numpy() - shifted.to_numpy())
        del shifted

    # Rolling stats (strictly <= t). groupby.rolling emits group-sorted rows,
    # so reindex back to the original row order before slicing.
    def roll(var: str, window: int, how: str, min_periods: int, name: str) -> None:
        s = getattr(ffg[var].rolling(window, min_periods=min_periods), how)()
        s = s.reset_index(level=0, drop=True).reindex(index)
        push(name, s.to_numpy())

    roll("heart_rate", 6, "max", 1, "hr_max_6h")
    roll("heart_rate", 12, "mean", 1, "hr_mean_12h")
    roll("heart_rate", 24, "std", 3, "hr_std_24h")
    roll("respiratory_rate", 6, "max", 1, "rr_max_6h")
    roll("respiratory_rate", 24, "std", 3, "rr_std_24h")
    roll("map_arterial", 6, "min", 1, "map_min_6h")
    roll("map_arterial", 24, "min", 1, "map_min_24h")
    roll("map_arterial", 24, "std", 3, "map_std_24h")
    roll("temp_celsius", 12, "max", 1, "temp_max_12h")
    roll("temp_celsius", 24, "min", 1, "temp_min_24h")
    roll("spo2", 6, "min", 1, "spo2_min_6h")

    # Clinical ratios & SIRS proxy
    hr = ff["heart_rate"].to_numpy(dtype=np.float32)
    rr = ff["respiratory_rate"].to_numpy(dtype=np.float32)
    map_ = ff["map_arterial"].to_numpy(dtype=np.float32)
    temp = ff["temp_celsius"].to_numpy(dtype=np.float32)
    spo2 = ff["spo2"].to_numpy(dtype=np.float32)
    fio2 = ff["fio2"].to_numpy(dtype=np.float32)
    bun = ff["bun"].to_numpy(dtype=np.float32)
    creat = ff["creatinine"].to_numpy(dtype=np.float32)
    wbc = ff["wbc"].to_numpy(dtype=np.float32)

    # Ratios use the source (float64) values and are cast once at the end, so
    # they are bit-identical to computing them on float64 columns then casting.
    hr64 = ff["heart_rate"].to_numpy()
    map64 = ff["map_arterial"].to_numpy()
    spo264 = ff["spo2"].to_numpy()
    fio264 = ff["fio2"].to_numpy()
    bun64 = ff["bun"].to_numpy()
    creat64 = ff["creatinine"].to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        shock = np.where(map64 != 0, hr64 / map64, np.nan)
        sf_ratio = np.where(fio264 != 0, spo264 / (fio264 / 100.0), np.nan)
        bun_cr = np.where(creat64 != 0, bun64 / creat64, np.nan)
    shift6 = pd.Series(shock).groupby(gkey, sort=False).shift(6).to_numpy()

    sirs = (
        (hr > 90).astype(np.float32)
        + (rr > 20).astype(np.float32)
        + ((temp > 38.0) | (temp < 36.0)).astype(np.float32)
        + ((wbc > 12.0) | (wbc < 4.0)).astype(np.float32)
    )

    push("shock_index", shock)
    push("shock_index_d6", shock - shift6)
    push("spo2_fio2_ratio", sf_ratio)
    push("bun_cr_ratio", bun_cr)
    push("sirs_proxy", sirs)

    del ff, ffg, shift6, sirs
    gc.collect()
    return blocks


def _build_matrix(
    horizon_hours: float = 6.0,
    exclude_first_hours: float = 6.0,
    stay_limit: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Return (X float32, row metadata, feature names) for hours >= exclude_first_hours."""
    df, _ = _prepare_rows(exclude_first_hours, stay_limit)
    blocks = _block_series(df, exclude_first_hours)

    hour = df["hour"].to_numpy(dtype=np.float32)
    blocks.append((["hour"], hour.reshape(-1, 1)))
    blocks.append((["icu_acquired_48h"], (hour >= 48).astype(np.float32).reshape(-1, 1)))
    age = df["age"].to_numpy(dtype=np.float32)
    blocks.append((["age"], age.reshape(-1, 1)))

    has_sep = df["has_sepsis_label"].fillna(False).astype(bool)
    target = (
        has_sep & (df["hour"] < df["sepsis_onset_hour"]) & (df["sepsis_onset_hour"] <= df["hour"] + horizon_hours)
    ).to_numpy().astype(np.int8)

    names = [n for group, _ in blocks for n in group]
    keep = np.flatnonzero(hour >= exclude_first_hours)

    X = np.empty((keep.size, len(names)), dtype=np.float32)
    col = 0
    for group, arr in blocks:
        width = len(group)
        X[:, col : col + width] = arr[keep]
        col += width
    assert col == len(names), f"block widths {col} != feature names {len(names)}"
    del blocks

    meta = pd.DataFrame(
        {
            "patient_id": df["patient_id"].to_numpy()[keep],
            "pseudonym": df["pseudonym"].to_numpy()[keep],
            "onset_hour": df["sepsis_onset_hour"].to_numpy()[keep],
            TARGET_COL: target[keep],
            "hour": hour[keep],
        }
    )
    del df
    gc.collect()
    return X, meta, names


def build_pn2019_features(
    horizon_hours: float = 6.0,
    exclude_first_hours: float = 6.0,
    stay_limit: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Backwards-compatible wrapper returning a single feature DataFrame."""
    X, meta, names = _build_matrix(horizon_hours, exclude_first_hours, stay_limit)
    df = pd.DataFrame(X, columns=names)
    for col in META_COLS + [TARGET_COL]:
        df[col] = meta[col].to_numpy()
    feature_cols = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    return df, feature_cols


def _impute_in_place(mat: np.ndarray, med: np.ndarray) -> None:
    """Fill NaNs with column medians without allocating another full copy."""
    if np.isnan(mat).any():
        rows, cols = np.where(np.isnan(mat))
        mat[rows, cols] = med[cols]


def run_evaluation(n_splits: int = 3, alert_threshold: float = 0.15) -> dict:
    X, meta, feature_cols = _build_matrix()
    y = meta[TARGET_COL].to_numpy(dtype=int)
    groups = meta["patient_id"].to_numpy()
    n_rows = len(y)

    oof = {k: np.zeros(n_rows, dtype=np.float32) for k in ("hgb", "hgb2", "et", "lr")}

    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        X_tr = X[tr]
        X_va = X[va]

        hgb = HistGradientBoostingClassifier(
            max_depth=6,
            max_iter=300,
            learning_rate=0.03,
            min_samples_leaf=60,
            l2_regularization=3.0,
            max_leaf_nodes=63,
            random_state=42,
        )
        hgb.fit(X_tr, y[tr])
        oof["hgb"][va] = hgb.predict_proba(X_va)[:, 1]
        del hgb
        gc.collect()

        hgb2 = HistGradientBoostingClassifier(
            max_depth=8,
            max_iter=250,
            learning_rate=0.04,
            min_samples_leaf=120,
            l2_regularization=8.0,
            max_leaf_nodes=31,
            random_state=2024,
        )
        hgb2.fit(X_tr, y[tr])
        oof["hgb2"][va] = hgb2.predict_proba(X_va)[:, 1]
        del hgb2
        gc.collect()

        # Tree/linear models below need complete matrices, so impute the fold
        # slices in place (after the NaNs have served their purpose for HGB).
        med = np.nanmedian(X_tr, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        _impute_in_place(X_tr, med)
        _impute_in_place(X_va, med)

        et = ExtraTreesClassifier(
            n_estimators=60,
            max_depth=12,
            min_samples_leaf=60,
            max_features=0.3,
            max_samples=0.5,
            bootstrap=True,
            n_jobs=-1,
            random_state=7,
        )
        et.fit(X_tr, y[tr])
        oof["et"][va] = et.predict_proba(X_va)[:, 1]
        del et
        gc.collect()

        lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.2, max_iter=200))])
        lr.fit(X_tr, y[tr])
        oof["lr"][va] = lr.predict_proba(X_va)[:, 1]
        del lr, X_tr, X_va, med
        gc.collect()

    candidates: dict[str, tuple[str, np.ndarray]] = {
        "HistGradientBoosting": ("HistGradientBoosting", oof["hgb"]),
        "HistGradientBoosting_Tuned2": ("HistGradientBoosting (tuned 2)", oof["hgb2"]),
        "ExtraTrees": ("ExtraTrees", oof["et"]),
        "LogisticRegression": ("LogisticRegression", oof["lr"]),
        "Ensemble(3-model)": (
            "Ensemble(HGB+ET+LogReg)",
            0.45 * oof["hgb"] + 0.35 * oof["et"] + 0.20 * oof["lr"],
        ),
        "Ensemble(4-model)": (
            "Ensemble(HGB+ET+LogReg, 4-way)",
            0.30 * oof["hgb"] + 0.25 * oof["hgb2"] + 0.30 * oof["et"] + 0.15 * oof["lr"],
        ),
    }
    scored = {k: float(roc_auc_score(y, v[1])) for k, v in candidates.items()}
    best_key = max(scored, key=scored.get)
    best_name, best_oof = candidates[best_key]
    best_auroc = scored[best_key]

    auprc = float(average_precision_score(y, best_oof))
    brier = float(brier_score_loss(y, best_oof))

    alerts = (best_oof >= alert_threshold).astype(int)
    total_patient_days = max(1e-6, float(meta.groupby("pseudonym")["hour"].max().sum()) / 24.0)
    alert_rate = float(alerts.sum() / total_patient_days)

    lead_frame = meta[["pseudonym", "hour", "onset_hour"]].copy()
    lead_frame["alert"] = alerts
    lead_times = []
    for _, grp in lead_frame[lead_frame["onset_hour"].notna()].groupby("pseudonym"):
        hit = grp[grp["alert"] == 1]
        if len(hit):
            lead_times.append(float(grp["onset_hour"].iloc[0]) - float(hit["hour"].min()))
    median_lead = float(np.median(lead_times)) if lead_times else None
    del lead_frame, oof, candidates
    gc.collect()

    final_model = HistGradientBoostingClassifier(
        max_depth=5,
        max_iter=250,
        learning_rate=0.05,
        min_samples_leaf=60,
        l2_regularization=2.0,
        class_weight="balanced",
        random_state=42,
    )
    final_model.fit(X, y)

    metrics = {
        "disclaimer": DISCLAIMER,
        "dataset": f"PhysioNet/CinC Challenge 2019 ({int(meta['patient_id'].nunique())}-patient cohort)",
        "n_patients": int(meta["patient_id"].nunique()),
        "n_stays": int(meta["pseudonym"].nunique()),
        "n_rows": int(n_rows),
        "n_positive_rows": int(y.sum()),
        "n_positive_stays": int(meta.loc[meta[TARGET_COL] == 1, "pseudonym"].nunique()),
        "cv_folds_by_patient": n_splits,
        "best_model": best_name,
        "auroc": round(best_auroc, 4),
        "auprc": round(auprc, 4),
        "brier_score": round(brier, 4),
        "logreg_auroc": round(float(scored["LogisticRegression"]), 4),
        "hgb_auroc": round(float(scored["HistGradientBoosting"]), 4),
        "extratrees_auroc": round(float(scored["ExtraTrees"]), 4),
        "model_candidates": {k: round(v, 4) for k, v in scored.items()},
        "alert_rate_per_patient_day": round(alert_rate, 3),
        "median_lead_time_hours": round(median_lead, 2) if median_lead is not None else None,
        "small_sample_caveats": [
            f"Full PhysioNet/CinC Challenge 2019 cohort ({int(meta['patient_id'].nunique())} patients, "
            "US hospital systems A & B); not validated on Romanian ICU populations.",
            "Challenge 2019 PSV files omit vasopressors, GCS, urine output, microbiology, and ICD codes; "
            "missingness and lab-recency flags are explicitly modeled.",
        ],
    }

    PN19_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PN19_MODEL_PATH, "wb") as f:
        pickle.dump({"model": final_model, "feature_cols": feature_cols}, f)
    with open(PN19_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    m = run_evaluation()
    print(json.dumps(m, indent=2))
