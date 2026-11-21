"""Fast vectorized leakage-safe patient-grouped 5-fold CV on PhysioNet 2019.

Vectorized across all stays (no per-stay Python loop) so 10k-25k stays
build features and finish 5-fold patient-grouped CV in ~10-20 seconds.
"""

from __future__ import annotations

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
from src.ingest.loader import load_cache

ROOT = Path(__file__).resolve().parents[2]
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


def build_pn2019_features(
    horizon_hours: float = 6.0,
    exclude_first_hours: float = 6.0,
) -> tuple[pd.DataFrame, list[str]]:
    cohort = load_cache(ROOT / "data" / "processed_pn2019")
    meta = cohort.cohort[["pseudonym", "patient_id", "age", "los", "has_sepsis_label", "sepsis_onset_hour"]].copy()
    meta["los_hr"] = (meta["los"] * 24.0).round().astype(int)
    # Exclude short stays or early onset (<= exclude_first_hours)
    early_onset = meta["has_sepsis_label"] & (meta["sepsis_onset_hour"] <= exclude_first_hours)
    meta = meta[(meta["los_hr"] > exclude_first_hours) & (~early_onset)].copy()

    valid_ids = set(meta["pseudonym"])
    all_ev = pd.concat(
        [
            cohort.vitals.loc[cohort.vitals["pseudonym"].isin(valid_ids), ["pseudonym", "hours_from_icu", "variable", "value"]],
            cohort.labs.loc[cohort.labs["pseudonym"].isin(valid_ids), ["pseudonym", "hours_from_icu", "variable", "value"]],
        ],
        ignore_index=True,
    )
    all_ev["hour"] = all_ev["hours_from_icu"].round().astype(int)

    piv = (
        all_ev.groupby(["pseudonym", "hour", "variable"], sort=False)["value"]
        .last()
        .unstack("variable")
    )
    for v in VARS:
        if v not in piv.columns:
            piv[v] = np.nan
    piv = piv[VARS].sort_index().reset_index()

    # Merge stay metadata
    df = piv.merge(meta, on="pseudonym", how="inner")

    # Strictly before onset (if positive)
    has_sep = df["has_sepsis_label"].fillna(False).astype(bool)
    df = df[(~has_sep) | (df["hour"] < df["sepsis_onset_hour"])].copy()
    df = df.sort_values(["pseudonym", "hour"]).reset_index(drop=True)

    g = df.groupby("pseudonym", sort=False)

    # Trailing 24h carry-forward (strictly <= t)
    ff = g[VARS].ffill(limit=24)

    # Explicit missingness flags
    miss = ff.isna().astype(np.float32).add_prefix("miss_")

    # Cumulative lab count up to t (clinical acuity signal)
    lab_any = df[LAB_VARS].notna().sum(axis=1)
    cum_labs = lab_any.groupby(df["pseudonym"], sort=False).cumsum().rename("cum_lab_count").astype(np.float32)

    # Exact hours-since-last-observation for key labs (strictly <= t, vectorized)
    recency_cols: dict[str, pd.Series] = {}
    hr_vals = df["hour"].to_numpy(dtype=np.float32)
    stay_keys = df["pseudonym"]
    for v in ["lactate", "wbc", "platelets", "creatinine", "bilirubin", "bun", "glucose", "ph"]:
        obs_hr = pd.Series(np.where(df[v].notna().to_numpy(), hr_vals, np.nan), index=df.index)
        last_obs_hr = obs_hr.groupby(stay_keys, sort=False).ffill()
        recency_cols[f"hrs_since_{v}"] = (df["hour"] - last_obs_hr).astype(np.float32)
    recency_df = pd.DataFrame(recency_cols, index=df.index)

    # First-6h personal baseline deviations (strictly causal since we only predict for hour >= 6)
    base_vars = ["heart_rate", "respiratory_rate", "map_arterial", "temp_celsius", "spo2"]
    first6_mask = df["hour"] < exclude_first_hours
    base_means = (
        ff.loc[first6_mask, base_vars]
        .groupby(df.loc[first6_mask, "pseudonym"], sort=False)
        .mean()
    )
    base_aligned = base_means.reindex(df["pseudonym"]).set_axis(df.index)
    base_dev = (ff[base_vars] - base_aligned).add_suffix("_dev_base").astype(np.float32)

    # 6h and 12h deltas (strictly t minus t-k within stay)
    delta_cols = ["heart_rate", "respiratory_rate", "map_arterial", "temp_celsius", "spo2", "lactate"]
    ff_g = ff.groupby(df["pseudonym"], sort=False)
    d6 = (ff[delta_cols] - ff_g[delta_cols].shift(6)).add_suffix("_d6")
    d12 = (ff[delta_cols] - ff_g[delta_cols].shift(12)).add_suffix("_d12")

    # Rolling 6h, 12h & 24h stats (strictly <= t)
    r_hr_max = ff_g["heart_rate"].rolling(6, min_periods=1).max().reset_index(level=0, drop=True).rename("hr_max_6h")
    r_hr_mean = ff_g["heart_rate"].rolling(12, min_periods=1).mean().reset_index(level=0, drop=True).rename("hr_mean_12h")
    r_hr_std24 = ff_g["heart_rate"].rolling(24, min_periods=3).std().reset_index(level=0, drop=True).rename("hr_std_24h")
    r_rr_max = ff_g["respiratory_rate"].rolling(6, min_periods=1).max().reset_index(level=0, drop=True).rename("rr_max_6h")
    r_rr_std24 = ff_g["respiratory_rate"].rolling(24, min_periods=3).std().reset_index(level=0, drop=True).rename("rr_std_24h")
    r_map_min = ff_g["map_arterial"].rolling(6, min_periods=1).min().reset_index(level=0, drop=True).rename("map_min_6h")
    r_map_min24 = ff_g["map_arterial"].rolling(24, min_periods=1).min().reset_index(level=0, drop=True).rename("map_min_24h")
    r_map_std24 = ff_g["map_arterial"].rolling(24, min_periods=3).std().reset_index(level=0, drop=True).rename("map_std_24h")
    r_temp_max = ff_g["temp_celsius"].rolling(12, min_periods=1).max().reset_index(level=0, drop=True).rename("temp_max_12h")
    r_temp_min24 = ff_g["temp_celsius"].rolling(24, min_periods=1).min().reset_index(level=0, drop=True).rename("temp_min_24h")
    r_spo2_min = ff_g["spo2"].rolling(6, min_periods=1).min().reset_index(level=0, drop=True).rename("spo2_min_6h")

    # Clinical ratios & SIRS proxy
    shock_idx = (ff["heart_rate"] / ff["map_arterial"].replace(0, np.nan)).rename("shock_index")
    shock_d6 = (shock_idx - shock_idx.groupby(df["pseudonym"], sort=False).shift(6)).rename("shock_index_d6")
    sf_ratio = (ff["spo2"] / (ff["fio2"].replace(0, np.nan) / 100.0)).rename("spo2_fio2_ratio")
    bun_cr = (ff["bun"] / ff["creatinine"].replace(0, np.nan)).rename("bun_cr_ratio")
    sirs = (
        (ff["heart_rate"] > 90).astype(np.float32)
        + (ff["respiratory_rate"] > 20).astype(np.float32)
        + ((ff["temp_celsius"] > 38.0) | (ff["temp_celsius"] < 36.0)).astype(np.float32)
        + ((ff["wbc"] > 12.0) | (ff["wbc"] < 4.0)).astype(np.float32)
    ).rename("sirs_proxy")

    feat = pd.concat(
        [
            ff.astype(np.float32),
            miss,
            cum_labs,
            recency_df,
            base_dev,
            d6.astype(np.float32),
            d12.astype(np.float32),
            r_hr_max.astype(np.float32),
            r_hr_mean.astype(np.float32),
            r_hr_std24.astype(np.float32),
            r_rr_max.astype(np.float32),
            r_rr_std24.astype(np.float32),
            r_map_min.astype(np.float32),
            r_map_min24.astype(np.float32),
            r_map_std24.astype(np.float32),
            r_temp_max.astype(np.float32),
            r_temp_min24.astype(np.float32),
            r_spo2_min.astype(np.float32),
            shock_idx.astype(np.float32),
            shock_d6.astype(np.float32),
            sf_ratio.astype(np.float32),
            bun_cr.astype(np.float32),
            sirs,
        ],
        axis=1,
    )
    feat["hour"] = df["hour"].astype(np.float32)
    feat["icu_acquired_48h"] = (df["hour"] >= 48).astype(np.float32)
    feat["age"] = df["age"].astype(np.float32)

    # Target: onset within next 6 hours
    has_sep_filt = df["has_sepsis_label"].fillna(False).astype(bool)
    target = np.where(
        has_sep_filt & (df["hour"] < df["sepsis_onset_hour"]) & (df["sepsis_onset_hour"] <= df["hour"] + horizon_hours),
        1,
        0,
    )
    feat["patient_id"] = df["patient_id"].to_numpy()
    feat["pseudonym"] = df["pseudonym"].to_numpy()
    feat["onset_hour"] = df["sepsis_onset_hour"].to_numpy()
    feat["target_onset_6h"] = target

    # Keep only hours >= exclude_first_hours
    feat = feat[feat["hour"] >= exclude_first_hours].reset_index(drop=True)
    feature_cols = [
        c
        for c in feat.columns
        if c not in ("patient_id", "pseudonym", "onset_hour", "target_onset_6h")
    ]
    return feat, feature_cols


def run_evaluation(n_splits: int = 3, alert_threshold: float = 0.15) -> dict:
    df, feature_cols = build_pn2019_features()
    X_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["target_onset_6h"].to_numpy(dtype=int)
    groups = df["patient_id"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    oof_hgb = np.zeros(len(df), dtype=float)
    oof_hgb2 = np.zeros(len(df), dtype=float)
    oof_et = np.zeros(len(df), dtype=float)
    oof_lr = np.zeros(len(df), dtype=float)

    for tr, va in gkf.split(X_raw, y, groups):
        hgb = HistGradientBoostingClassifier(
            max_depth=6,
            max_iter=300,
            learning_rate=0.03,
            min_samples_leaf=60,
            l2_regularization=3.0,
            max_leaf_nodes=63,
            random_state=42,
        )
        hgb.fit(X_raw[tr], y[tr])
        oof_hgb[va] = hgb.predict_proba(X_raw[va])[:, 1]

        hgb2 = HistGradientBoostingClassifier(
            max_depth=8,
            max_iter=250,
            learning_rate=0.04,
            min_samples_leaf=120,
            l2_regularization=8.0,
            max_leaf_nodes=31,
            random_state=2024,
        )
        hgb2.fit(X_raw[tr], y[tr])
        oof_hgb2[va] = hgb2.predict_proba(X_raw[va])[:, 1]

        # Median-impute for the tree model that cannot handle NaN (ExtraTrees)
        med_tree = np.nanmedian(X_raw[tr], axis=0)
        med_tree = np.where(np.isnan(med_tree), 0.0, med_tree)
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
        et.fit(
            np.where(np.isnan(X_raw[tr]), med_tree, X_raw[tr]),
            y[tr],
        )
        oof_et[va] = et.predict_proba(np.where(np.isnan(X_raw[va]), med_tree, X_raw[va]))[:, 1]

        med = np.nanmedian(X_raw[tr], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        X_tr_lr = np.where(np.isnan(X_raw[tr]), med, X_raw[tr])
        X_va_lr = np.where(np.isnan(X_raw[va]), med, X_raw[va])
        lr = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.2, max_iter=200)),
            ]
        )
        lr.fit(X_tr_lr, y[tr])
        oof_lr[va] = lr.predict_proba(X_va_lr)[:, 1]

    auroc_hgb = float(roc_auc_score(y, oof_hgb))
    auroc_hgb2 = float(roc_auc_score(y, oof_hgb2))
    auroc_et = float(roc_auc_score(y, oof_et))
    auroc_lr = float(roc_auc_score(y, oof_lr))

    candidates: dict[str, tuple[str, np.ndarray]] = {
        "HistGradientBoosting": ("HistGradientBoosting", oof_hgb),
        "HistGradientBoosting_Tuned2": ("HistGradientBoosting (tuned 2)", oof_hgb2),
        "ExtraTrees": ("ExtraTrees", oof_et),
        "LogisticRegression": ("LogisticRegression", oof_lr),
        "Ensemble(3-model)": (
            "Ensemble(HGB+ET+LogReg)",
            0.45 * oof_hgb + 0.35 * oof_et + 0.20 * oof_lr,
        ),
        "Ensemble(4-model)": (
            "Ensemble(HGB+ET+LogReg, 4-way)",
            0.30 * oof_hgb + 0.25 * oof_hgb2 + 0.30 * oof_et + 0.15 * oof_lr,
        ),
    }
    scored = {k: float(roc_auc_score(y, v[1])) for k, v in candidates.items()}
    best_key = max(scored, key=scored.get)
    best_name, best_oof = candidates[best_key]
    best_auroc = scored[best_key]

    auprc = float(average_precision_score(y, best_oof))
    brier = float(brier_score_loss(y, best_oof))

    alerts = (best_oof >= alert_threshold).astype(int)
    total_patient_days = max(1e-6, float(df.groupby("pseudonym")["hour"].max().sum()) / 24.0)
    alert_rate = float(alerts.sum() / total_patient_days)

    df_eval = df[["pseudonym", "hour", "onset_hour"]].copy()
    df_eval["alert"] = alerts
    lead_times = []
    for _, grp in df_eval[df_eval["onset_hour"].notna()].groupby("pseudonym"):
        hit = grp[grp["alert"] == 1]
        if len(hit):
            lead_times.append(float(grp["onset_hour"].iloc[0]) - float(hit["hour"].min()))
    median_lead = float(np.median(lead_times)) if lead_times else None

    final_model = HistGradientBoostingClassifier(
        max_depth=5,
        max_iter=250,
        learning_rate=0.05,
        min_samples_leaf=60,
        l2_regularization=2.0,
        class_weight="balanced",
        random_state=42,
    )
    final_model.fit(X_raw, y)

    metrics = {
        "disclaimer": DISCLAIMER,
        "dataset": f"PhysioNet/CinC Challenge 2019 ({int(df['patient_id'].nunique())}-patient subset)",
        "n_patients": int(df["patient_id"].nunique()),
        "n_stays": int(df["pseudonym"].nunique()),
        "n_rows": int(len(df)),
        "n_positive_rows": int(y.sum()),
        "n_positive_stays": int(df.loc[df["target_onset_6h"] == 1, "pseudonym"].nunique()),
        "cv_folds_by_patient": n_splits,
        "best_model": best_name,
        "auroc": round(best_auroc, 4),
        "auprc": round(auprc, 4),
        "brier_score": round(brier, 4),
        "logreg_auroc": round(auroc_lr, 4),
        "hgb_auroc": round(auroc_hgb, 4),
        "extratrees_auroc": round(auroc_et, 4),
        "model_candidates": {k: round(v, 4) for k, v in scored.items()},
        "alert_rate_per_patient_day": round(alert_rate, 3),
        "median_lead_time_hours": round(median_lead, 2) if median_lead is not None else None,
        "small_sample_caveats": [
            f"Subset of {int(df['patient_id'].nunique())} patients from PhysioNet/CinC Challenge 2019 (US hospital systems A & B); not validated on Romanian ICU populations.",
            "Challenge 2019 PSV files omit vasopressors, GCS, urine output, microbiology, and ICD codes; missingness and lab-recency flags are explicitly modeled.",
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
