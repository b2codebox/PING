# Sepsis Onset, Organ Dysfunction & Early Warning Prototype

> **Decision support only. Not validated for clinical use. Requires clinician review.**

Research prototype for sepsis onset detection, organ dysfunction scoring (SOFA / APS III),
and tiered early-warning alerts on structured ICU data. (Not a medical device, not clinically
validated, and not connected to any live hospital system.)

## What it does

- **SOFA** (6 organ systems), **APACHE III** prototype estimate, tiered **EWS** alerts.
- **Leakage-safe** feature building: at evaluation hour `t`, only records with `timestamp <= t`.
- **Pre-onset 6-hour risk model** trained only on hours strictly before sepsis onset.
- Streamlit dashboard with pseudonymised stays, ICU specialty filter, medication filters,
  synchronized evaluation-hour control, and EN/RO labels (clinical terms kept in English).


MIMIC-IV requires PhysioNet credentialing and a data use agreement. Do not redistribute
MIMIC-IV or PhysioNet data. Patient-level caches must never be committed.

## Privacy

Identifiers are pseudonymised. Only explicitly allowlisted structured fields are used;
free-text notes and raw identifiers are rejected. Missing values are recorded as missing
rather than imputed as normal.

## Model results (patient-grouped CV, leakage-safe)

| Cohort | Task | AUROC | Brier |
|---|---|---|---|
| MIMIC-IV Demo (117 stays) | pre-onset 6h | 0.7202 | 0.1559 |
| PhysioNet 2019 (19,338 stays) | pre-onset 6h | 0.8400 | 0.0087 |


See [`docs/PITCH_SCRIPT_TEMPLATE.md`](docs/PITCH_SCRIPT_TEMPLATE.md) for the
fill-in-the-blanks speaking script (2:48 measured, with swappable modules),
[`docs/PITCH_3MIN.md`](docs/PITCH_3MIN.md) for the 3-minute judge pitch,
[`docs/PITCH_CAPABILITIES.md`](docs/PITCH_CAPABILITIES.md) for the capabilities and
scaling pitch, and
[`docs/MASTER_PROMPT_AND_PITCH.md`](docs/MASTER_PROMPT_AND_PITCH.md) for the
reusable build specification, the pitch, and objection handling.

## Limitations

- Small demo cohort; US hospital data; **not validated on Romanian ICU populations**.
- PhysioNet 2019 files omit vasopressors, GCS, urine output, microbiology, and ICD codes.
- Prototype thresholds and APS III approximation require clinician verification.
- Not a medical device. No clinical use.
