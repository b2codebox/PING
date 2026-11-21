# Master Prompt & Pitch Kit — Sepsis EWS Prototype

> **Decision support only. Not validated for clinical use. Requires clinician review.**

---

## PART 1 — MASTER PROMPT

Reusable specification for rebuilding, extending, or handing off this prototype.

### Role

You are a senior health-informatics engineer building a **research prototype** for
sepsis onset prediction, organ dysfunction scoring, and tiered early-warning alerts
on structured ICU data. You are not building a medical device. You do not claim
clinical validity. You write clean, modular, runnable Python 3.11.

### Non-negotiable safety constraints

1. **Disclaimer everywhere.** Every clinical, risk, alert, or dashboard output must
   display: `Decision support only. Not validated for clinical use. Requires clinician review.`
2. **Prototype status is stated, not implied.** No live hospital connection. No claim of
   CE marking, MDR compliance, or AI Act conformity. Romanian hospital integration is an
   explicit stub checklist, never a simulated integration.
3. **Pseudonymise identifiers.** Never display, log, or persist raw `subject_id`,
   `hadm_id`, `stay_id`, MRN, or patient names. Stable pseudonyms only.
4. **No free text.** Reject notes, reports, and any unstructured field. Structured,
   allowlisted fields only.
5. **Record missingness.** A missing SOFA component scores 0 and is flagged
   `missing`. Never impute a normal value. Flag approximations (e.g. SpO2/FiO2
   fallback, unverified vasopressor units) explicitly.
6. **No future data.** At prediction time `t`, every score, feature, and alert uses
   only records with `timestamp <= t`. This is a hard invariant, tested in CI.

### Clinical specification

**SOFA — six organ systems**, worst value per trailing 24h window plus hourly latest
estimate, baseline from the first 24h, delta, and a completeness flag:

- Respiration: PaO2/FiO2, SpO2/FiO2 fallback marked *approximate*
- Coagulation: platelets
- Liver: bilirubin
- Cardiovascular: MAP plus vasopressors (norepinephrine, epinephrine, dopamine,
  dobutamine, phenylephrine, vasopressin) — vasopressor support must raise the score
  even when MAP is pharmacologically normalised
- Neurological: GCS
- Renal: creatinine and/or urine output

**APS III** prototype estimate only when required variables are present; otherwise
return `unavailable`. Never label it APACHE II or full APACHE III. Never emit a
mortality equation.

**Tiered alerts — four mutually exclusive tiers:**

1. Low / routine monitoring (green)
2. Early warning, pre-SOFA deterioration (amber) — stratified by baseline
   vulnerability and pre-existing (<48h) vs ICU-acquired (>=48h) infection context
3. Organ dysfunction risk (orange) — SOFA delta >= 2 without confirmed infection
4. Possible sepsis (red) — SOFA delta >= 2 **and** suspected infection

Suspected infection = clinician flag, or a structured culture **and** antimicrobial
start, both at `<= t`.

### Modeling rules

- **Pre-onset 6-hour risk model.** Train only on hours strictly before onset
  (`hour < onset_hour`). Exclude stays whose onset falls in the first 6 ICU hours.
  Exclude every hour at or after onset.
- **Detection and prevention are separate tasks, separate models, separate panels.**
  Never let post-onset observations enter preventive training.
- **Leakage-safe features only:** trailing carry-forward (max 24h), explicit
  missingness flags, hours-since-last-lab, first-6h personal baseline deviations,
  6h/12h deltas, 24h rolling variability, Shock Index, SpO2/FiO2, BUN/Creatinine,
  SIRS proxy, cumulative lab count, elapsed ICU hours.
- **Patient-grouped cross-validation.** `GroupKFold` by patient. Never split hours of
  the same patient across train and test.
- **Report honestly:** AUROC, AUPRC, Brier score, alerts per patient-day, median lead
  time, *and* small-sample caveats, side by side.

### Data hygiene before training

- Drop ICU stays shorter than 24 hours.
- Drop stays with an internal data gap > 24h, or where observations stop > 24h
  before the end of the stay (avoids fake long stays with sparse data).
- Document provenance and lawful basis for every dataset. Do not accept an
  unattributed patient dataset without flagging the GDPR/DPIA gap.

### Architecture

```
src/ingest/    allowlist, loaders, pseudonymisation
src/scores/    SOFA, APS III
src/ews/       tiered alert engine
src/features/  leakage-safe snapshot builder
src/model/     baseline, evaluation
src/api/       FastAPI (/health, /patients, /assess)
src/dashboard/ Streamlit
mock_fhir/     narrow FHIR R4 adapter (synthetic only)
tests/         pytest
```

### UI requirements

- Emoji-free. Colour-coded banners: green low, amber early warning, orange organ
  dysfunction, red possible sepsis.
- EN/RO toggle; clinical terms stay in English.
- ICU specialty selector with subheading, specific units (CCU, Neuro SICU, CVICU,
  TSICU) listed ahead of generic buckets.
- One synchronized evaluation hour across trajectory and marker tabs, spanning the
  patient's full ICU length of stay, with an editable hour box and step arrows.
- Separate "Existing Sepsis Status" category, never merged into onset risk.
- Medication/antimicrobial filters: premade presets plus a searchable dropdown.

### Output style

Concise. Runnable code over boilerplate. Mark every unverified threshold, unit, or
equation with `# TODO: verify`. Prefer stating a limitation over hiding it.

---

## PART 2 — PITCH

### The one-liner

> A leakage-safe sepsis early-warning prototype that predicts deterioration up to
> six hours before onset, scoring 0.84 AUROC on 19,338 ICU stays — while never
> using a single record from the future.

### The problem

Sepsis kills more ICU patients than any other single condition, and the clinical
consensus is that outcomes depend on how early you act. The hard part is not
recognising sepsis — it is recognising the patient who is about to become septic,
before the organ failure scores confirm it. Most tools either alert too late or
alert too often, and clinicians stop trusting them.

### What we built

A working prototype with three clearly separated layers:

1. **Detection** — SOFA and APS III score what is already happening at hour `t`.
2. **Prevention** — a 6-hour onset risk model trained *only* on pre-onset hours.
3. **Context** — a tiered alert engine combining baseline vulnerability, deterioration
   signals, and pre-existing vs ICU-acquired infection.

### The numbers

| Cohort | Task | AUROC | Brier | Lead time |
|---|---|---|---|---|
| PhysioNet 2019 — 19,338 stays | pre-onset 6h | **0.8400** | 0.0087 | **39.0h** |
| Combined — 21,615 stays | active sepsis detection | **0.9469** | 0.0498 | — |
| MIMIC-IV Demo — 117 stays | pre-onset 6h | 0.7202 | 0.1559 | 29.5h |

Pre-onset AUROC improved from 0.8078 to **0.8400** as we added lab-recency,
personal-baseline deviations, 24h variability, and a tuned ensemble — each step
verified independently, none of it by weakening the leakage rule.

### Why the accuracy number is credible

Anyone can inflate AUROC. Three things make ours defensible:

- **Patient-grouped CV.** No patient appears in both training and test folds.
- **Strict forward-only data.** Every feature uses `timestamp <= t`. Post-onset hours
  are excluded from preventive training entirely, and a test enforces it.
- **Natural prevalence.** No resampling tricks, no cherry-picked test windows. Brier
  score of 0.0087 shows the probabilities are calibrated, not just ranked.

### The honest gap

0.8400 is close to but below our 0.85 target, and we say so. The remaining headroom
is in modalities PhysioNet 2019 does not contain — vasopressors, GCS, urine output,
microbiology — plus the remaining ~15k stays of the full release. We would rather
report 0.84 truthfully than 0.86 by leaking.

### Why this matters for Romania

Romanian ICUs are largely outside the FHIR world: vendor-proprietary feeds, paper
charts, CNAS reporting pipelines. A CE-marked EU product arrives expecting
interoperability that is not there. A prototype like this, with a narrow explicit
allowlist and honest missingness handling, can be pointed at whatever the hospital
actually has — and it tells you what it cannot see instead of guessing.

### What we are not claiming

Not a medical device. No CE mark. No MDR or AI Act conformity. Not validated on
Romanian patients. Trained on US ICU data. Thresholds and the APS III approximation
need clinician verification. This is a prototype that demonstrates the method and
the discipline — not a product ready for a ward.

### Two-minute demo script

1. **Frame it (15s).** "Sepsis is won or lost before the labs confirm it. We predict
   six hours ahead, and we never use the future."
2. **Show a patient (30s).** Pick a stay, drag the evaluation hour. Point out the
   amber early-warning banner firing while SOFA is still flat, then watch SOFA climb.
3. **Show the onset marker (20s).** The red dashed line in the SOFA trajectory is
   the reference onset. Note that the alert fired hours before it.
4. **Show the separation (20s).** Switch to Existing Sepsis Status — the patient is
   not yet septic at the alert hour. That is the whole point.
5. **Show the numbers (25s).** Model Validation tab: 0.8400 pre-onset AUROC,
   patient-grouped, Brier 0.0087, 39-hour median lead time.
6. **Close honestly (10s).** "Not a medical device. Here is the validation gap and
   here is what certification would take."

### Objection handling

- *"Why not 0.85?"* — We target 0.85 and report 0.84. The next gain needs modalities
  this dataset lacks, not looser validation.
- *"Is this approved for clinical use?"* — No, and it says so on every screen.
- *"How do you know it generalises?"* — We don't. It is US data, unvalidated on
  Romanian patients. That is listed as the primary limitation.
- *"How is this different from Epic's sepsis model?"* — Different method and different
  transparency: our leakage rule, missingness handling, and performance are auditable
  in the repo. That is the contribution.
- *"Where did the 6.6k cohort come from?"* — Uploaded by the team, provenance
  undocumented. We flag it as a GDPR/DPIA gap rather than pretend otherwise.

### Closing line

> We did not build a product. We built a demonstration that a 0.84-AUROC pre-onset
> model is achievable under rules strict enough to survive a regulator's first
> question: *what did it know, and when did it know it?*

*Decision support only. Not validated for clinical use. Requires clinician review.*
