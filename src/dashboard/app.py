"""Streamlit interface for the Sepsis Onset, Organ Dysfunction & Early Warning prototype.

Displays:
* Mandatory clinical disclaimer on every view (no emojis).
* Color-coded alert banners (green = low, amber = early warning,
  orange = organ dysfunction, red = possible sepsis).
* Dataset switcher: MIMIC-IV Demo vs. PhysioNet/CinC Challenge 2019.
* EN / RO label toggle (keeping clinical terms in English).
* Patient SOFA trajectory, APS III, 6h risk score, top drivers, and EWS
  breakdown (vulnerability, early deterioration, pre-existing vs ICU-acquired
  infection context).
* Interactive clinician `/assess` simulator.
* Small-cohort validation metrics & Romanian hospital integration stub checklist.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.ews.alerts import evaluate_ews
from src.features.builder import snapshot_at, stay_events
from src.ingest.allowlist import DISCLAIMER
from src.ingest.loader import load_cache
from src.model.baseline import load_artifacts, predict_with_drivers
from src.scores.aps3 import compute_aps3
from src.scores.sofa import SOFA_COMPONENTS, compute_sofa, window_inputs

st.set_page_config(
    page_title="Sepsis & Early Warning Prototype",
    layout="wide",
)

ALERT_STYLES = {
    "none": ("#e8f5e9", "#2e7d32", "#1b5e20", "LOW RISK"),
    "early_warning": ("#fff8e1", "#f9a825", "#8d5500", "EARLY WARNING"),
    "organ_dysfunction": ("#fff3e0", "#ef6c00", "#bf360c", "ORGAN DYSFUNCTION"),
    "possible_sepsis": ("#ffebee", "#c62828", "#8e0000", "POSSIBLE SEPSIS"),
}


def render_alert_banner(tier: str, risk_label: str, summary: str) -> None:
    bg, border, text, badge = ALERT_STYLES.get(tier, ALERT_STYLES["none"])
    st.markdown(
        f"""
        <div style="background:{bg};border-left:6px solid {border};padding:14px 18px;
                    border-radius:var(--ui-radius);margin:10px 0 16px 0;color:{text};">
          <div style="font-size:12px;font-weight:700;letter-spacing:0.06em;margin-bottom:4px;">
            {badge}
          </div>
          <div style="font-size:16px;font-weight:600;margin-bottom:4px;">{risk_label}</div>
          <div style="font-size:13px;opacity:0.9;">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        f"""
        <div style="background:#f5f5f5;border:1px solid #bdbdbd;border-left:5px solid #424242;
                    padding:10px 14px;border-radius:var(--ui-radius);margin:0 0 14px 0;color:#212121;
                    font-size:13px;font-weight:600;">
          {DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_layout_css() -> None:
    """Apply a single spacing scale so every vertical gap in the app matches.

    Streamlit's defaults mix very large title margins with tight chart margins,
    which reads as uneven down a long page. These rules set one rhythm:
    14px between blocks, 18px above a section heading, 8px below it.
    """
    st.markdown(
        """
        <style>
        /* --- One corner radius everywhere ---------------------------------
           The theme sets 8px for Streamlit's own widgets. These rules give the
           hand-written HTML the same value and catch the few components that
           do not read the theme (code blocks, dataframe cells, chart frames).
        */
        :root {
            --ui-radius: 8px;
        }

        /* --- Page frame ------------------------------------------------- */
        .block-container {
            padding-top: 2.75rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1320px;
        }

        /* --- Title block: large heading, tight to its caption ----------- */
        .block-container h1 {
            font-size: 1.85rem !important;
            line-height: 1.22 !important;
            letter-spacing: -0.015em;
            margin: 0 0 0.35rem 0 !important;
            padding: 0 !important;
        }
        .block-container [data-testid="stCaptionContainer"] {
            margin: 0 0 0.9rem 0 !important;
        }
        .block-container [data-testid="stCaptionContainer"] p {
            font-size: 0.82rem;
            line-height: 1.45;
        }

        /* --- Tabs: even gap above, single rule below -------------------- */
        .block-container [data-testid="stTabs"] {
            margin: 0 0 0.1rem 0;
        }
        .block-container [data-testid="stTabs"] > div:first-child {
            margin-top: -0.55rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.75rem;
            border-bottom: 1px solid #e6e6e6;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 0 0.65rem 0;
            margin: 0;
        }

        /* --- Section headings: same gap above and below everywhere ------ */
        .block-container [data-testid="stHeading"] {
            margin: 1.15rem 0 0.5rem 0 !important;
            padding: 0 !important;
        }
        .block-container [data-testid="stHeading"] h2,
        .block-container [data-testid="stHeading"] h3 {
            font-size: 1.12rem !important;
            line-height: 1.35 !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        /* First heading inside a tab does not need the full top gap. */
        .block-container [data-testid="stTabs"] [data-testid="stHeading"]:first-child {
            margin-top: 0.35rem !important;
        }

        /* --- One rhythm for every stacked element ----------------------- */
        .block-container [data-testid="stVerticalBlock"] {
            gap: 0.85rem;
        }
        /* Inside a row of columns, keep the same rhythm. */
        .block-container [data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"] {
            gap: 0.6rem;
        }

        /* --- Charts, tables and metrics sit flush with their heading ---- */
        .block-container [data-testid="stArrowVegaLiteChart"],
        .block-container [data-testid="stDataFrame"],
        .block-container [data-testid="stTable"],
        .block-container [data-testid="stAltairChart"] {
            margin: 0 !important;
        }
        .block-container [data-testid="stMetric"] {
            padding: 0.6rem 0.75rem;
            border: 1px solid #e8e8e8;
            border-radius: var(--ui-radius);
        }
        .block-container [data-testid="stMetric"] label p {
            font-size: 0.78rem;
        }

        /* --- Sliders, inputs and buttons share one row height ----------- */
        .block-container [data-testid="stSlider"],
        .block-container [data-testid="stNumberInput"],
        .block-container [data-testid="stButton"] {
            margin: 0 !important;
        }
        .block-container [data-testid="stWidgetLabel"] {
            margin-bottom: 0.2rem;
        }

        /* --- Lists inside the EWS breakdown do not add their own gaps --- */
        .block-container [data-testid="stMarkdownContainer"] ul {
            margin: 0.15rem 0 0.15rem 0;
            padding-left: 1.1rem;
        }
        .block-container [data-testid="stMarkdownContainer"] li {
            margin: 0.1rem 0;
        }

        /* --- Banners align to the same rhythm --------------------------- */
        .block-container [data-testid="stMarkdownContainer"] > div[style*="border-left"] {
            margin-top: 0 !important;
        }

        /* --- Sidebar: tighter, consistent stack ------------------------- */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.7rem;
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
            margin-bottom: 0.15rem;
        }
        [data-testid="stSidebar"] h3 {
            font-size: 1rem !important;
            margin: 0.15rem 0 0.55rem 0 !important;
            padding: 0 !important;
        }
        [data-testid="stSidebar"] hr {
            margin: 0.9rem 0 !important;
        }

        /* --- Radius catch-all for components outside the theme ---------- */
        .block-container [data-testid="stCode"],
        .block-container [data-testid="stCodeBlock"],
        .block-container [data-testid="stExpander"] details,
        .block-container [data-testid="stAlert"],
        .block-container [data-testid="stNotification"],
        .block-container [data-testid="stDataFrame"] > div,
        .block-container [data-testid="stDataFrame"] [role="grid"],
        .block-container [data-testid="stTable"] table,
        .block-container [data-testid="stMetric"],
        .block-container [data-testid="stMultiSelect"] [data-baseweb="tag"],
        .block-container [data-baseweb="select"] > div,
        .block-container [data-baseweb="input"] > div,
        .block-container [data-baseweb="textarea"] > div,
        .block-container [data-baseweb="popover"] [role="listbox"],
        .block-container [data-baseweb="menu"] ul,
        .block-container [data-testid="stButton"] button,
        .block-container [data-testid="stBaseButton-secondary"],
        .block-container [data-testid="stDownloadButton"] button,
        .block-container [data-testid="stFileUploaderDropzone"],
        .block-container [data-testid="stVegaLiteChart"],
        .block-container [data-testid="stArrowVegaLiteChart"],
        .block-container [data-testid="stAltairChart"],
        .block-container [data-testid="stTabs"] [data-baseweb="tab-highlight"],
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div,
        [data-testid="stSidebar"] [data-baseweb="popover"] [role="listbox"],
        [data-testid="stSidebar"] [data-baseweb="menu"] ul,
        [data-testid="stSidebar"] [data-baseweb="tag"],
        [data-testid="stSidebar"] [data-testid="stButton"] button,
        [data-testid="stSidebar"] [data-testid="stAlert"] {
            border-radius: var(--ui-radius) !important;
        }

        /* --- Banners and cards written as raw HTML --------------------- */
        .block-container [data-testid="stMarkdownContainer"] > div[style*="border-left"] {
            border-radius: var(--ui-radius) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


LABELS = {
    "EN": {
        "title": "Sepsis Onset, Organ Dysfunction & Early Warning Prototype",
        "subtitle": "Structured ICU Cohort Prototype — MIMIC-IV Demo & PhysioNet/CinC Challenge 2019",
        "tab_patient": "Patient Trajectory & EWS",
        "tab_markers": "Hourly Markers & Sepsis Breakdown",
        "tab_assess": "Clinician /assess Simulator",
        "tab_model": "Model Validation & RO Checklist",
        "select_specialty": "ICU Specialty",
        "all_specialties": "All ICU Specialties",
        "select_stay": "Select Pseudonymised ICU Stay",
        "eval_hour": "Evaluation Hour (t, leakage-safe <= t)",
        "sofa_traj": "SOFA Trajectory (24h trailing window)",
        "ews_breakdown": "Early Warning System (EWS) Breakdown",
        "vuln": "1. Baseline Vulnerability",
        "deterioration": "2. Early Physiological & Lab Deterioration",
        "infection": "3. Pre-existing (<48h) vs. ICU-acquired (>=48h) Infection Context",
        "drivers": "Top 6-Hour Risk Model Drivers",
        "assess_btn": "Run Prototype Assessment",
    },
    "RO": {
        "title": "Prototip Debut Sepsis, Disfunctie de Organ & Avertizare Timpurie",
        "subtitle": "Prototip Cohorta ICU Structurata — MIMIC-IV Demo & PhysioNet/CinC Challenge 2019",
        "tab_patient": "Traiectorie Pacient & EWS",
        "tab_markers": "Markeri Orari & Sepsis Existent",
        "tab_assess": "Simulator Clinic /assess",
        "tab_model": "Validare Model & Lista Spital RO",
        "select_specialty": "Specialitate ICU",
        "all_specialties": "Toate Specialitatile ICU",
        "select_stay": "Selecteaza Internarea ICU Pseudonimizata",
        "eval_hour": "Ora Evaluarii (t, fara date viitoare <= t)",
        "sofa_traj": "Traiectorie SOFA (fereastra 24h)",
        "ews_breakdown": "Detaliere Early Warning System (EWS)",
        "vuln": "1. Vulnerabilitate de Baza",
        "deterioration": "2. Deteriorare Fiziologica & Lab Timpurie",
        "infection": "3. Context Infectie Pre-existenta (<48h) vs. Dobandita ICU (>=48h)",
        "drivers": "Factori Principali Risc 6 Ore",
        "assess_btn": "Ruleaza Evaluarea Prototip",
    },
}


@st.cache_resource
def get_cohort_by_name(dataset_name: str):
    if dataset_name == "All Datasets Combined":
        from src.ingest.loader import Cohort

        parts = [
            ("M", load_cache(ROOT / "data" / "processed")),
            ("P", load_cache(ROOT / "data" / "processed_pn2019")),
            ("S", load_cache(ROOT / "data" / "processed_sepsis6k")),
        ]

        def _prefix_df(df: pd.DataFrame, tag: str) -> pd.DataFrame:
            if df.empty or "pseudonym" not in df.columns:
                return df
            out = df.copy()
            out["pseudonym"] = f"{tag}_" + out["pseudonym"].astype(str)
            if "patient_id" in out.columns:
                out["patient_id"] = f"{tag}_" + out["patient_id"].astype(str)
            return out

        def _concat_attr(attr: str) -> pd.DataFrame:
            frames = [_prefix_df(getattr(c, attr), tag) for tag, c in parts if len(getattr(c, attr))]
            if not frames:
                return getattr(parts[0][1], attr).iloc[0:0].copy()
            return pd.concat(frames, ignore_index=True)

        return Cohort(
            cohort=_concat_attr("cohort"),
            labs=_concat_attr("labs"),
            vitals=_concat_attr("vitals"),
            vasopressors=_concat_attr("vasopressors"),
            urine=_concat_attr("urine"),
            devices=_concat_attr("devices"),
            micro=_concat_attr("micro"),
            antimicrobials=_concat_attr("antimicrobials"),
            comorbidities=_concat_attr("comorbidities"),
            issues=[issue for _, c in parts for issue in c.issues],
        )
    if dataset_name == "PhysioNet/CinC Challenge 2019":
        return load_cache(ROOT / "data" / "processed_pn2019")
    if dataset_name == "Sepsis 6.6k Cohort (Post-Onset)":
        return load_cache(ROOT / "data" / "processed_sepsis6k")
    return load_cache(ROOT / "data" / "processed")


@st.cache_resource
def get_model():
    return load_artifacts()


# Sidebar: Dataset & Language toggle
dataset_choice = st.sidebar.selectbox(
    "Cohort Dataset",
    [
        "All Datasets Combined",
        "MIMIC-IV Clinical Database Demo",
        "PhysioNet/CinC Challenge 2019",
        "Sepsis 6.6k Cohort (Post-Onset)",
    ],
)
lang = st.sidebar.radio(
    "UI Language / Limba (clinical terms stay in EN)", ["EN", "RO"], horizontal=True
)
t = LABELS[lang]

cohort = get_cohort_by_name(dataset_choice)
pipe, metrics = get_model()

# --- Sidebar: Global Medication, Vasopressor & Antimicrobial Filters -------
st.sidebar.markdown("---")
st.sidebar.subheader("Medication & Antimicrobial Filters")

vaso_stay_ids = set(cohort.vasopressors["pseudonym"].unique()) if len(cohort.vasopressors) else set()
abx_stay_ids = set(cohort.antimicrobials["pseudonym"].unique()) if len(cohort.antimicrobials) else set()

med_preset = st.sidebar.selectbox(
    "Premade Medication Filters",
    [
        "All Patients",
        "On Vasopressors (MAP Support)",
        "No Vasopressors Recorded",
        "Received Antimicrobials / Antibiotics",
        "No Antimicrobials Recorded",
        "Both Vasopressors & Antimicrobials",
    ],
)

vaso_drugs = (
    sorted(cohort.vasopressors["drug"].dropna().astype(str).unique().tolist())
    if len(cohort.vasopressors) and "drug" in cohort.vasopressors.columns
    else []
)
abx_drugs = (
    sorted(cohort.antimicrobials["drug"].dropna().astype(str).unique().tolist())
    if len(cohort.antimicrobials) and "drug" in cohort.antimicrobials.columns
    else []
)
med_options = [f"[Vasopressor] {d}" for d in vaso_drugs] + [f"[Antimicrobial] {d}" for d in abx_drugs]

selected_meds = st.sidebar.multiselect(
    "Filter by Specific Medication / Antibiotic (type to search)",
    options=med_options,
    default=[],
    placeholder="Search norepinephrine, vancomycin, cefepime...",
)
if not med_options:
    st.sidebar.caption("No medication or antimicrobial records in this dataset.")

inject_layout_css()
render_disclaimer()
st.title(t["title"])
st.caption(f"{t['subtitle']} | Active dataset: {dataset_choice} ({len(cohort.cohort)} stays)")

tab1, tab_markers, tab2, tab3 = st.tabs([t["tab_patient"], t["tab_markers"], t["tab_assess"], t["tab_model"]])

# ===========================================================================
# TAB 1: Patient Trajectory & EWS
# ===========================================================================
with tab1:
    all_stays_df = cohort.cohort.sort_values("los", ascending=False).reset_index(drop=True)
    has_pn_label = "has_sepsis_label" in all_stays_df.columns

    # --- Selection Section with ICU Specialty Subheading -------------------
    st.subheader(t["select_specialty"])
    unit_counts = all_stays_df["first_careunit"].value_counts()
    # Put specific named clinical ICU subspecialties first (e.g. CCU, Neuro SICU, CVICU, TSICU),
    # followed by general cohort ICU units.
    generic_units = {"MICU (Unit1)", "SICU (Unit2)", "ICU (A)", "ICU (B)", "Sepsis ICU (6.6k Cohort)"}
    specific_list = sorted([u for u in unit_counts.index if u not in generic_units])
    generic_list = [u for u in unit_counts.index if u in generic_units]
    ordered_units = specific_list + generic_list

    specialty_options = [t["all_specialties"]] + [
        f"{u} ({int(unit_counts[u])} stays)" for u in ordered_units
    ]
    selected_spec = st.selectbox(
        t["select_specialty"],
        specialty_options,
        label_visibility="collapsed",
    )
    if selected_spec == t["all_specialties"]:
        stays_df = all_stays_df
    else:
        unit_name = selected_spec.rsplit(" (", 1)[0]
        stays_df = all_stays_df[all_stays_df["first_careunit"] == unit_name]

    # Apply global Medication & Antimicrobial preset filter
    if med_preset == "On Vasopressors (MAP Support)":
        stays_df = stays_df[stays_df["pseudonym"].isin(vaso_stay_ids)]
    elif med_preset == "No Vasopressors Recorded":
        stays_df = stays_df[~stays_df["pseudonym"].isin(vaso_stay_ids)]
    elif med_preset == "Received Antimicrobials / Antibiotics":
        stays_df = stays_df[stays_df["pseudonym"].isin(abx_stay_ids)]
    elif med_preset == "No Antimicrobials Recorded":
        stays_df = stays_df[~stays_df["pseudonym"].isin(abx_stay_ids)]
    elif med_preset == "Both Vasopressors & Antimicrobials":
        stays_df = stays_df[stays_df["pseudonym"].isin(vaso_stay_ids & abx_stay_ids)]

    # Apply searchable specific medication / antimicrobial multi-select
    if selected_meds:
        sel_vaso = [m.replace("[Vasopressor] ", "", 1) for m in selected_meds if m.startswith("[Vasopressor] ")]
        sel_abx = [m.replace("[Antimicrobial] ", "", 1) for m in selected_meds if m.startswith("[Antimicrobial] ")]
        matched_ids: set[str] = set()
        if sel_vaso and len(cohort.vasopressors) and "drug" in cohort.vasopressors.columns:
            matched_ids |= set(cohort.vasopressors.loc[cohort.vasopressors["drug"].isin(sel_vaso), "pseudonym"])
        if sel_abx and len(cohort.antimicrobials) and "drug" in cohort.antimicrobials.columns:
            matched_ids |= set(cohort.antimicrobials.loc[cohort.antimicrobials["drug"].isin(sel_abx), "pseudonym"])
        stays_df = stays_df[stays_df["pseudonym"].isin(matched_ids)]

    stays_df = stays_df.reset_index(drop=True)
    if stays_df.empty:
        st.info(
            f"No stays match the current filters (Specialty: {selected_spec}, Medication Preset: {med_preset}). "
            "Showing all stays in this specialty instead."
        )
        stays_df = (
            all_stays_df
            if selected_spec == t["all_specialties"]
            else all_stays_df[all_stays_df["first_careunit"] == selected_spec.rsplit(" (", 1)[0]]
        ).reset_index(drop=True)

    def _fmt_stay(i: int) -> str:
        r = stays_df.iloc[i]
        return f"{r['pseudonym']} | {r['first_careunit']} | LOS {r['los']*24:.0f}h | {r['admission_type']}"

    sel_idx = st.selectbox(t["select_stay"], range(len(stays_df)), format_func=_fmt_stay)
    stay_row = stays_df.iloc[sel_idx]
    stay_id = stay_row["pseudonym"]
    max_hr = max(1.0, round(float(stay_row["los"]) * 24.0, 1))
    default_hr = round(min(24.0, max_hr), 1)

    def _set_eval_hour(v: float) -> None:
        clamped_v = round(max(0.0, min(float(v), max_hr)), 1)
        st.session_state["shared_eval_hour"] = clamped_v
        st.session_state["tab1_eval_hour"] = clamped_v
        st.session_state["tab1_eval_num"] = clamped_v
        st.session_state["markers_eval_hour_slider"] = clamped_v
        st.session_state["markers_eval_num"] = clamped_v

    # Keep a single synchronized evaluation hour across both tabs for the active stay.
    if st.session_state.get("_active_stay_id") != stay_id:
        st.session_state["_active_stay_id"] = stay_id
        _set_eval_hour(default_hr)
    else:
        _set_eval_hour(float(st.session_state.get("shared_eval_hour", default_hr)))

    def _sync_from(src_key: str) -> None:
        _set_eval_hour(float(st.session_state[src_key]))

    def _step_hour(delta: float) -> None:
        _set_eval_hour(float(st.session_state.get("shared_eval_hour", default_hr)) + delta)

    sl_col, dec_col, num_col, inc_col = st.columns([6, 0.7, 1.8, 0.7], vertical_alignment="bottom")
    with sl_col:
        st.slider(
            t["eval_hour"],
            min_value=0.0,
            max_value=max_hr,
            step=1.0,
            key="tab1_eval_hour",
            on_change=_sync_from,
            args=("tab1_eval_hour",),
        )
    with dec_col:
        st.button("◀", key="tab1_dec_hr", help="Decrease by 1 hour", on_click=_step_hour, args=(-1.0,), use_container_width=True)
    with num_col:
        st.number_input(
            "Hour",
            min_value=0.0,
            max_value=max_hr,
            step=1.0,
            key="tab1_eval_num",
            label_visibility="collapsed",
            on_change=_sync_from,
            args=("tab1_eval_num",),
        )
    with inc_col:
        st.button("▶", key="tab1_inc_hr", help="Increase by 1 hour", on_click=_step_hour, args=(1.0,), use_container_width=True)

    eval_hr = float(st.session_state["shared_eval_hour"])

    ev = stay_events(cohort, stay_id)
    snap = snapshot_at(cohort, stay_id, eval_hr, events=ev)
    prob, drivers = predict_with_drivers(pipe, snap["features"])
    ews = snap["ews"]
    sofa = snap["sofa"]
    aps3 = snap["aps3"]

    render_alert_banner(ews.alert_tier, ews.risk_label, ews.summary)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SOFA Total", f"{sofa.total}", delta=f"{snap['sofa_delta']:+d} vs baseline ({snap['sofa_baseline']})")
    c2.metric("SOFA Completeness", f"{int(sofa.completeness * 100)}%", delta=f"{len(sofa.missing)} missing")
    c3.metric("APS III (Physiology)", str(aps3.score) if aps3.score is not None else "Unavailable", delta=aps3.status)
    c4.metric("6h Onset Risk (Prototype)", f"{prob * 100:.1f}%")

    # Shared hourly trajectory (1h step, hours 0..eval_hr) used by both Tab 1 and Tab 1B.
    hourly_rows = []
    for hr in range(0, int(eval_hr) + 1):
        w = window_inputs(ev, float(hr), window_hours=24.0)
        s = compute_sofa(w)
        hourly_rows.append(
            {
                "Hour": hr,
                "SOFA Total": s.total,
                "Resp (pts)": s.components["respiration"],
                "Coag (pts)": s.components["coagulation"],
                "Liver (pts)": s.components["liver"],
                "CV (pts)": s.components["cardiovascular"],
                "CNS (pts)": s.components["cns"],
                "Renal (pts)": s.components["renal"],
                "MAP (mmHg)": round(w["cardiovascular"]["map"], 1) if w["cardiovascular"].get("map") is not None else None,
                "Platelets": round(w["coagulation"]["platelets"], 1) if w["coagulation"].get("platelets") is not None else None,
                "Bilirubin": round(w["liver"]["bilirubin"], 2) if w["liver"].get("bilirubin") is not None else None,
                "Creatinine": round(w["renal"]["creatinine"], 2) if w["renal"].get("creatinine") is not None else None,
                "GCS": w["cns"].get("gcs_total"),
            }
        )
    hourly_df = pd.DataFrame(hourly_rows)
    st.subheader(t["sofa_traj"])
    import altair as alt

    base_line = (
        alt.Chart(hourly_df)
        .mark_line(color="#1f77b4", strokeWidth=2.5, point=False)
        .encode(
            x=alt.X("Hour:Q", title="ICU Hour (<= t)"),
            y=alt.Y("SOFA Total:Q", title="SOFA Total (0-24)", scale=alt.Scale(zero=True)),
            tooltip=["Hour", "SOFA Total"],
        )
    )
    has_ref_onset = (
        has_pn_label
        and pd.notna(stay_row["has_sepsis_label"])
        and bool(stay_row["has_sepsis_label"])
        and pd.notna(stay_row["sepsis_onset_hour"])
    )
    if has_ref_onset and float(stay_row["sepsis_onset_hour"]) <= eval_hr:
        onset_h = float(stay_row["sepsis_onset_hour"])
        rule_df = pd.DataFrame({"Hour": [onset_h], "Label": [f"SepsisLabel=1 (hr {onset_h:.0f})"]})
        onset_rule = (
            alt.Chart(rule_df)
            .mark_rule(color="#b91c1c", strokeDash=[5, 4], strokeWidth=2)
            .encode(x="Hour:Q", tooltip=["Label"])
        )
        onset_text = (
            alt.Chart(rule_df)
            .mark_text(align="left", dx=5, dy=-8, color="#b91c1c", fontWeight="bold")
            .encode(x="Hour:Q", y=alt.value(14), text="Label:N")
        )
        st.altair_chart((base_line + onset_rule + onset_text).properties(height=280), use_container_width=True)
    else:
        st.altair_chart(base_line.properties(height=280), use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("SOFA Components (<= t)")
        comp_df = pd.DataFrame(
            [
                {
                    "Organ System": c,
                    "Score": sofa.components[c],
                    "Missing": "Yes (scored 0)" if c in sofa.missing else "No",
                    "Approximate": "Yes" if c in sofa.approximate else "No",
                }
                for c in SOFA_COMPONENTS
            ]
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        if aps3.notes:
            st.caption("APS III notes: " + " ".join(aps3.notes))

    with col_right:
        st.subheader(t["drivers"])
        if drivers:
            st.dataframe(pd.DataFrame(drivers), use_container_width=True, hide_index=True)
        else:
            st.caption("Model drivers unavailable.")

    # EWS 3-pillar breakdown
    st.subheader(t["ews_breakdown"])
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown(f"**{t['vuln']}** (`{ews.vulnerability_tier}`)")
        if ews.vulnerability_factors:
            for item in ews.vulnerability_factors:
                st.write(f"- {item}")
        else:
            st.caption("No elevated vulnerability flags recorded.")
    with e2:
        st.markdown(f"**{t['deterioration']}** (`{ews.deterioration_count}` signals)")
        if ews.deterioration_signals:
            for item in ews.deterioration_signals:
                st.write(f"- {item}")
        else:
            st.caption("No acute deterioration signals at hour t.")
    with e3:
        st.markdown(f"**{t['infection']}**")
        st.write(f"Window: `{ews.infection_window}`")
        st.write(f"Suspected infection: `{ews.suspected_infection}`")
        if ews.infection_factors:
            for item in ews.infection_factors:
                st.write(f"- {item}")
        else:
            st.caption("No structured infection or >=48h device flags.")

# ===========================================================================
# TAB 1B: Hourly Markers, SOFA Impact & Existing Sepsis Breakdown
# ===========================================================================
with tab_markers:
    st.caption(f"Active stay: {stay_id} ({stay_row['first_careunit']}, LOS {stay_row['los']*24:.0f}h)")
    msl_col, mdec_col, mnum_col, minc_col = st.columns([6, 0.7, 1.8, 0.7], vertical_alignment="bottom")
    with msl_col:
        st.slider(
            t["eval_hour"],
            min_value=0.0,
            max_value=max_hr,
            step=1.0,
            key="markers_eval_hour_slider",
            on_change=_sync_from,
            args=("markers_eval_hour_slider",),
        )
    with mdec_col:
        st.button("◀", key="markers_dec_hr", help="Decrease by 1 hour", on_click=_step_hour, args=(-1.0,), use_container_width=True)
    with mnum_col:
        st.number_input(
            "Hour",
            min_value=0.0,
            max_value=max_hr,
            step=1.0,
            key="markers_eval_num",
            label_visibility="collapsed",
            on_change=_sync_from,
            args=("markers_eval_num",),
        )
    with minc_col:
        st.button("▶", key="markers_inc_hr", help="Increase by 1 hour", on_click=_step_hour, args=(1.0,), use_container_width=True)

    marker_hr = eval_hr

    # Reuse the exact same snapshot & hourly table computed for the synchronized hour.
    m_snap = snap
    m_sofa = sofa
    m_ews = ews
    m_win = window_inputs(ev, eval_hr, window_hours=24.0)

    # --- Separate Category: Existing Sepsis Status -------------------------
    st.subheader("Existing Sepsis Status (<= t)")
    if m_ews.possible_sepsis_alert:
        sepsis_bg, sepsis_border, sepsis_txt = "#ffebee", "#c62828", "#8e0000"
        sepsis_headline = "Existing Possible Sepsis Criteria Met (SOFA delta >= 2 + Suspected Infection)"
    elif m_ews.organ_dysfunction_alert:
        sepsis_bg, sepsis_border, sepsis_txt = "#fff3e0", "#ef6c00", "#bf360c"
        sepsis_headline = "Organ Dysfunction Present (SOFA delta >= 2), Without Confirmed Infection Flag"
    elif m_ews.suspected_infection:
        sepsis_bg, sepsis_border, sepsis_txt = "#fff8e1", "#f9a825", "#8d5500"
        sepsis_headline = "Suspected Infection Present, Below SOFA Delta >= 2 Threshold"
    else:
        sepsis_bg, sepsis_border, sepsis_txt = "#e8f5e9", "#2e7d32", "#1b5e20"
        sepsis_headline = "No Existing Sepsis Criteria Met at Hour t"

    st.markdown(
        f"""
        <div style="background:{sepsis_bg};border-left:6px solid {sepsis_border};padding:12px 16px;
                    border-radius:var(--ui-radius);margin:6px 0 14px 0;color:{sepsis_txt};">
          <div style="font-size:15px;font-weight:600;">{sepsis_headline}</div>
          <div style="font-size:13px;margin-top:4px;">
            Infection Window: <b>{m_ews.infection_window}</b> |
            SOFA: <b>{m_sofa.total}</b> (baseline {m_snap['sofa_baseline']}, delta {m_snap['sofa_delta']:+d}) |
            Suspected Infection: <b>{'Yes' if m_ews.suspected_infection else 'No'}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Existing Sepsis Alert", "Yes" if m_ews.possible_sepsis_alert else "No")
    s2.metric("Organ Dysfunction (delta >= 2)", "Yes" if m_ews.organ_dysfunction_alert else "No")
    s3.metric("Suspected Infection", "Yes" if m_ews.suspected_infection else "No")
    if has_pn_label and pd.notna(stay_row["has_sepsis_label"]):
        pn_pos = bool(stay_row["has_sepsis_label"]) and (marker_hr >= float(stay_row["sepsis_onset_hour"]) - 1.0)
        s4.metric("PhysioNet SepsisLabel (<= t)", "1 (Positive)" if pn_pos else "0 (Negative)")
    else:
        s4.metric("Infection Window", "ICU-acquired (>=48h)" if marker_hr >= 48.0 else "Pre-existing (<48h)")

    if m_ews.infection_factors:
        st.write("Recorded infection context factors:", ", ".join(m_ews.infection_factors))

    # --- Marker-to-SOFA Impact at Hour t -----------------------------------
    st.subheader(f"Markers & SOFA Impact at Hour {marker_hr:.0f}")
    total_pts = max(1, m_sofa.total)

    def _fmt(v, unit=""):
        if v is None or pd.isna(v):
            return "Missing"
        return f"{v:.2f} {unit}".strip()

    resp = m_win["respiration"]
    cv = m_win["cardiovascular"]
    vaso_active = [
        f"{d}={cv.get(d):.3f}"
        for d in ("norepinephrine", "epinephrine", "dopamine", "dobutamine")
        if cv.get(d) is not None
    ]
    if cv.get("vasopressin_active"):
        vaso_active.append("vasopressin (active, unscored)")

    impact_rows = [
        {
            "Organ System": "Respiration",
            "Observed Marker(s) (24h worst)": (
                f"PaO2={_fmt(resp.get('pao2'), 'mmHg')}, SpO2={_fmt(resp.get('spo2'), '%')}, FiO2={_fmt(resp.get('fio2'), '%')}"
            ),
            "SOFA Points (0-4)": m_sofa.components["respiration"],
            "Share of SOFA Total": f"{(m_sofa.components['respiration'] / total_pts) * 100:.0f}%" if m_sofa.total > 0 else "0%",
            "Data Status": "Approximate (SpO2/FiO2)" if "respiration" in m_sofa.approximate else ("Missing (scored 0)" if "respiration" in m_sofa.missing else "Measured"),
        },
        {
            "Organ System": "Coagulation",
            "Observed Marker(s) (24h worst)": f"Platelets={_fmt(m_win['coagulation'].get('platelets'), 'K/uL')}",
            "SOFA Points (0-4)": m_sofa.components["coagulation"],
            "Share of SOFA Total": f"{(m_sofa.components['coagulation'] / total_pts) * 100:.0f}%" if m_sofa.total > 0 else "0%",
            "Data Status": "Missing (scored 0)" if "coagulation" in m_sofa.missing else "Measured",
        },
        {
            "Organ System": "Liver",
            "Observed Marker(s) (24h worst)": f"Bilirubin={_fmt(m_win['liver'].get('bilirubin'), 'mg/dL')}",
            "SOFA Points (0-4)": m_sofa.components["liver"],
            "Share of SOFA Total": f"{(m_sofa.components['liver'] / total_pts) * 100:.0f}%" if m_sofa.total > 0 else "0%",
            "Data Status": "Missing (scored 0)" if "liver" in m_sofa.missing else "Measured",
        },
        {
            "Organ System": "Cardiovascular",
            "Observed Marker(s) (24h worst)": f"MAP={_fmt(cv.get('map'), 'mmHg')}" + (f" | Vaso: {', '.join(vaso_active)}" if vaso_active else ""),
            "SOFA Points (0-4)": m_sofa.components["cardiovascular"],
            "Share of SOFA Total": f"{(m_sofa.components['cardiovascular'] / total_pts) * 100:.0f}%" if m_sofa.total > 0 else "0%",
            "Data Status": "Missing (scored 0)" if "cardiovascular" in m_sofa.missing else "Measured",
        },
        {
            "Organ System": "Neurological (CNS)",
            "Observed Marker(s) (24h worst)": f"GCS={_fmt(m_win['cns'].get('gcs_total'))}",
            "SOFA Points (0-4)": m_sofa.components["cns"],
            "Share of SOFA Total": f"{(m_sofa.components['cns'] / total_pts) * 100:.0f}%" if m_sofa.total > 0 else "0%",
            "Data Status": "Approximate (intubated)" if "cns" in m_sofa.approximate else ("Missing (scored 0)" if "cns" in m_sofa.missing else "Measured"),
        },
        {
            "Organ System": "Renal",
            "Observed Marker(s) (24h worst)": f"Creatinine={_fmt(m_win['renal'].get('creatinine'), 'mg/dL')}, Urine 24h={_fmt(m_win['renal'].get('urine_24h'), 'mL')}",
            "SOFA Points (0-4)": m_sofa.components["renal"],
            "Share of SOFA Total": f"{(m_sofa.components['renal'] / total_pts) * 100:.0f}%" if m_sofa.total > 0 else "0%",
            "Data Status": "Missing (scored 0)" if "renal" in m_sofa.missing else "Measured",
        },
    ]
    st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)

    # --- Hourly Marker & SOFA Contribution Table (Hour 0..t) ---------------
    st.subheader(f"Hour-by-Hour Markers & SOFA Contributions (Hours 0 to {eval_hr:.0f})")
    st.bar_chart(
        hourly_df.set_index("Hour")[
            ["Resp (pts)", "Coag (pts)", "Liver (pts)", "CV (pts)", "CNS (pts)", "Renal (pts)"]
        ]
    )
    st.dataframe(hourly_df, use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 2: Interactive Clinician /assess Simulator
# ===========================================================================
with tab2:
    with st.form("assess_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            icu_hr = st.number_input("ICU Hours Elapsed", min_value=0.0, max_value=500.0, value=52.0)
            age = st.number_input("Age (years)", min_value=18.0, max_value=110.0, value=72.0)
            adm_type = st.selectbox("Admission Type", ["EMERGENCY", "URGENT", "ELECTIVE"])
            sofa_base = st.number_input("Baseline SOFA (first 24h)", min_value=0, max_value=24, value=1)
            comorbs = st.multiselect(
                "Comorbidities / Immunosuppression",
                ["immunosuppression", "malignancy", "chronic_kidney_disease", "liver_disease", "diabetes", "copd", "heart_failure"],
                default=["immunosuppression"],
            )
        with f2:
            hr = st.number_input("Heart Rate (bpm)", min_value=20.0, max_value=250.0, value=118.0)
            rr = st.number_input("Respiratory Rate (/min)", min_value=4.0, max_value=60.0, value=26.0)
            temp_c = st.number_input("Temperature (C)", min_value=30.0, max_value=43.0, value=38.7)
            map_val = st.number_input("MAP (mmHg)", min_value=20.0, max_value=180.0, value=64.0)
            vaso_sim_drug = st.selectbox(
                "Vasopressor / MAP Support",
                ["None", "norepinephrine", "epinephrine", "dopamine", "dobutamine", "phenylephrine", "vasopressin"],
            )
            vaso_sim_rate = st.number_input("Vasopressor Rate (mcg/kg/min)", min_value=0.0, max_value=50.0, value=0.0, step=0.05)
            lactate = st.number_input("Lactate (mmol/L)", min_value=0.1, max_value=25.0, value=2.9)
            lactate_d6 = st.number_input("Lactate 6h Delta (mmol/L)", min_value=-10.0, max_value=15.0, value=0.9)
        with f3:
            spo2 = st.number_input("SpO2 (%)", min_value=50.0, max_value=100.0, value=93.0)
            fio2 = st.number_input("FiO2 (%)", min_value=21.0, max_value=100.0, value=40.0)
            platelets = st.number_input("Platelets (K/uL)", min_value=5.0, max_value=1000.0, value=120.0)
            creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=20.0, value=1.8)
            gcs = st.number_input("GCS Total", min_value=3.0, max_value=15.0, value=14.0)
            devices = st.multiselect("Invasive Devices", ["central_line", "urinary_catheter", "invasive_ventilation"], default=["central_line"])
            suspected_inf = st.checkbox("Clinician Suspected Infection", value=True)

        submitted = st.form_submit_button(t["assess_btn"])

    if submitted:
        sim_vasos = (
            {vaso_sim_drug: float(vaso_sim_rate)}
            if vaso_sim_drug != "None" and vaso_sim_rate > 0
            else {}
        )
        sofa_sim = compute_sofa(
            {
                "respiration": {"pao2": None, "fio2": fio2, "spo2": spo2, "approximate": True},
                "coagulation": {"platelets": platelets},
                "liver": {"bilirubin": None},
                "cardiovascular": {"map": map_val, "vasopressors": sim_vasos},
                "cns": {"gcs_total": gcs},
                "renal": {"creatinine": creatinine},
            }
        )
        delta_sim = sofa_sim.total - int(sofa_base)
        aps3_sim = compute_aps3(
            {
                "temp_c": temp_c,
                "map": map_val,
                "heart_rate": hr,
                "respiratory_rate": rr,
                "gcs_total": gcs,
                "creatinine": creatinine,
            }
        )
        ews_sim = evaluate_ews(
            hour=icu_hr,
            sofa_total=sofa_sim.total,
            sofa_delta=delta_sim,
            age=age,
            admission_type=adm_type,
            comorbidities=comorbs,
            lactate=lactate,
            lactate_delta_6h=lactate_d6,
            respiratory_rate=rr,
            heart_rate=hr,
            temp_c=temp_c,
            map_val=map_val,
            device_exposures=devices,
            clinician_suspected_infection=suspected_inf,
        )
        render_alert_banner(ews_sim.alert_tier, ews_sim.risk_label, ews_sim.summary)
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("SOFA Total", sofa_sim.total, delta=f"{delta_sim:+d} vs baseline")
        sc2.metric("APS III", str(aps3_sim.score) if aps3_sim.score is not None else "Unavailable", delta=aps3_sim.status)
        sc3.metric("Deterioration Signals", ews_sim.deterioration_count)
        st.write("Signals:", ews_sim.deterioration_signals)
        st.write("Infection Context:", ews_sim.infection_window, ews_sim.infection_factors)

# ===========================================================================
# TAB 3: Model Validation & Romanian Hospital Stub Checklist
# ===========================================================================
with tab3:
    st.subheader("Patient-Grouped Cross-Validation (MIMIC-IV Demo, n=99)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AUROC", str(metrics.get("auroc")))
    m2.metric("AUPRC", str(metrics.get("auprc")))
    m3.metric("Brier Score", str(metrics.get("brier_score")))
    m4.metric("Alerts / Patient-Day", str(metrics.get("alert_rate_per_patient_day")))
    m5.metric("Median Lead Time (h)", str(metrics.get("median_lead_time_hours")))

    pn19_metrics_path = ROOT / "models" / "pn2019_metrics.json"
    if pn19_metrics_path.exists():
        import json
        pn_m = json.loads(pn19_metrics_path.read_text())
        st.subheader(f"Patient-Grouped Cross-Validation ({pn_m.get('dataset')})")
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("AUROC (Best)", str(pn_m.get("auroc")), delta=f"LogReg {pn_m.get('logreg_auroc')}")
        p2.metric("AUPRC", str(pn_m.get("auprc")))
        p3.metric("Brier Score", str(pn_m.get("brier_score")))
        p4.metric("Alerts / Patient-Day", str(pn_m.get("alert_rate_per_patient_day")))
        p5.metric("Median Lead Time (h)", str(pn_m.get("median_lead_time_hours")))
        st.caption(
            f"Evaluated on {pn_m.get('n_patients')} eligible stays ({pn_m.get('n_positive_stays')} sepsis-positive stays, "
            f"{pn_m.get('n_rows')} hourly windows) via {pn_m.get('cv_folds_by_patient')}-fold patient-grouped CV."
        )

    comb_metrics_path = ROOT / "models" / "combined_detection_metrics.json"
    if comb_metrics_path.exists():
        import json
        cm = json.loads(comb_metrics_path.read_text())
        st.subheader(f"Active / Post-Onset Sepsis Detection ({cm.get('dataset')})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detection AUROC", str(cm.get("auroc")))
        c2.metric("Detection AUPRC", str(cm.get("auprc")))
        c3.metric("Brier Score", str(cm.get("brier_score")))
        c4.metric("Total Stays", f"{cm.get('n_stays'):,} ({cm.get('n_sepsis_stays'):,} sepsis)")
        st.caption(cm.get("temporal_note", ""))

    st.markdown("#### Small-Sample & Clinical Caveats")
    for c in metrics.get("small_sample_caveats", []):
        st.write(f"- {c}")

    st.markdown("#### Romanian Hospital Integration Stub Checklist (Not Connected)")
    st.markdown(
        """
- [ ] Hospital Ethics Committee & Institutional Review Board approval
- [ ] GDPR Data Protection Impact Assessment (DPIA) & Data Processing Agreement (DPA)
- [ ] Local ICU clinician sign-off on SOFA/EWS thresholds & APS III assumptions
- [ ] Validated HL7 FHIR R4 / HL7v2 mapping against hospital EHR (strict unit & LOINC audit)
- [ ] Security review, role-based access control, and audit logging
- [ ] Romanian/English UI clinical terminology review
        """
    )
