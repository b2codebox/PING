"""Compact Logistic Regression baseline with patient-grouped cross-validation.

Small-cohort aware (n ~= 100 patients in MIMIC-IV Demo):
* Uses L2-regularized LogisticRegression on compact score-level + EWS features
  to limit overfitting.
* Splits strictly by ``patient_id`` via ``GroupKFold`` so no patient appears
  in both train and validation folds.
* Reports AUROC, AUPRC, Brier score, alert rate per patient-day, median lead
  time, and explicit small-sample caveats.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.builder import FEATURE_COLUMNS, build_feature_table
from src.ingest.allowlist import DISCLAIMER
from src.ingest.loader import load_cache

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "baseline_logreg.pkl"
METRICS_PATH = ROOT / "models" / "baseline_metrics.json"


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=500)),
        ]
    )


def evaluate_and_train(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    alert_threshold: float = 0.5,
) -> tuple[Pipeline, dict]:
    """Run patient-grouped CV, fit final compact model, and return metrics."""
    X = df[list(FEATURE_COLUMNS)].fillna(0.0).to_numpy(dtype=float)
    y = df["target_onset_6h"].to_numpy(dtype=int)
    groups = df["patient_id"].to_numpy()

    n_patients = int(df["patient_id"].nunique())
    n_stays = int(df["pseudonym"].nunique())
    n_pos_rows = int(y.sum())
    n_pos_stays = int(df.loc[df["target_onset_6h"] == 1, "pseudonym"].nunique())

    oof_probs = np.zeros(len(df), dtype=float)
    actual_splits = min(n_splits, max(2, n_patients))
    gkf = GroupKFold(n_splits=actual_splits)

    for train_idx, val_idx in gkf.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            oof_probs[val_idx] = float(y[train_idx].mean())
            continue
        pipe = _make_pipeline()
        pipe.fit(X[train_idx], y[train_idx])
        oof_probs[val_idx] = pipe.predict_proba(X[val_idx])[:, 1]

    has_both = len(np.unique(y)) == 2
    auroc = float(roc_auc_score(y, oof_probs)) if has_both else None
    auprc = float(average_precision_score(y, oof_probs)) if has_both else None
    brier = float(brier_score_loss(y, oof_probs)) if len(y) else None

    # Alert rate per patient-day (assuming 4h step -> 6 windows/day, or computed from hours).
    alerts = (oof_probs >= alert_threshold).astype(int)
    total_patient_days = max(1e-6, float(df.groupby("pseudonym")["hour"].max().sum()) / 24.0)
    alert_rate_per_patient_day = float(alerts.sum() / total_patient_days)

    # Median lead time among stays with an onset that received an alert before onset.
    df_eval = df[["pseudonym", "hour", "onset_hour"]].copy()
    df_eval["alert"] = alerts
    lead_times: list[float] = []
    for _, grp in df_eval[df_eval["onset_hour"].notna()].groupby("pseudonym"):
        hit = grp[grp["alert"] == 1]
        if len(hit):
            first_alert_hr = float(hit["hour"].min())
            onset_hr = float(grp["onset_hour"].iloc[0])
            if onset_hr >= first_alert_hr:
                lead_times.append(onset_hr - first_alert_hr)
    median_lead_hours = float(np.median(lead_times)) if lead_times else None

    # Fit final model on all rows for prototype serving.
    final_pipe = _make_pipeline()
    if has_both:
        final_pipe.fit(X, y)
    else:
        # Fallback if synthetic test has single class.
        final_pipe = None

    caveats = [
        f"Small-cohort prototype (n={n_patients} patients, {n_stays} ICU stays, {n_pos_stays} stays with onset). High variance expected.",
        "Not externally validated; MIMIC-IV Demo is US tertiary-ICU data and does not represent Romanian hospital populations.",
        "Prototype thresholds and APS III approximation require clinician verification.",
    ]

    metrics = {
        "disclaimer": DISCLAIMER,
        "n_patients": n_patients,
        "n_stays": n_stays,
        "n_rows": int(len(df)),
        "n_positive_rows": n_pos_rows,
        "n_positive_stays": n_pos_stays,
        "cv_folds_by_patient": actual_splits,
        "alert_threshold": alert_threshold,
        "auroc": round(auroc, 4) if auroc is not None else None,
        "auprc": round(auprc, 4) if auprc is not None else None,
        "brier_score": round(brier, 4) if brier is not None else None,
        "alert_rate_per_patient_day": round(alert_rate_per_patient_day, 3),
        "median_lead_time_hours": round(median_lead_hours, 2) if median_lead_hours is not None else None,
        "small_sample_caveats": caveats,
    }
    return final_pipe, metrics


def predict_with_drivers(pipe: Pipeline | None, features: dict[str, float]) -> tuple[float, list[dict]]:
    """Return risk probability and top linear feature drivers."""
    if pipe is None:
        return 0.0, []
    vec = np.array([[float(features.get(c, 0.0)) for c in FEATURE_COLUMNS]], dtype=float)
    prob = float(pipe.predict_proba(vec)[0, 1])

    scaler: StandardScaler = pipe.named_steps["scaler"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    z = scaler.transform(vec)[0]
    contrib = z * clf.coef_[0]
    ranked = sorted(
        zip(FEATURE_COLUMNS, contrib, vec[0]),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    drivers = [
        {
            "feature": name,
            "value": round(float(val), 3),
            "contribution": round(float(c), 4),
            "direction": "increases_risk" if c > 0 else "decreases_risk",
        }
        for name, c, val in ranked[:5]
    ]
    return round(prob, 4), drivers


def save_artifacts(pipe: Pipeline | None, metrics: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def load_artifacts() -> tuple[Pipeline | None, dict]:
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        cohort = load_cache(ROOT / "data" / "processed")
        df = build_feature_table(cohort)
        pipe, metrics = evaluate_and_train(df)
        save_artifacts(pipe, metrics)
        return pipe, metrics
    with open(MODEL_PATH, "rb") as f:
        pipe = pickle.load(f)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return pipe, metrics


if __name__ == "__main__":
    cohort = load_cache(ROOT / "data" / "processed")
    df = build_feature_table(cohort)
    pipe, metrics = evaluate_and_train(df)
    save_artifacts(pipe, metrics)
    print(json.dumps(metrics, indent=2))
