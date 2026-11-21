# Pitch Script Template

> **Decision support only. Not validated for clinical use. Requires clinician review.**

A spoken script you fill in and expand. Slots are in `[BRACKETS]`. Modules are
swappable — pick the ones that fit your time and your audience.

**Rules for this document:** the spoken text is what you say out loud. The `*(cues)*`
are what you do. Nothing in a cue gets spoken.

---

## 1. The base script — 3:00

Speak this verbatim the first time you rehearse. Edit only the brackets.

### 0:00 — Open (32s)

*(Stand still. Do not touch the laptop yet. Look at the judges.)*

"We are [TEAM] from [SCHOOL]. We built a sepsis early-warning prototype for
[CONTEXT — e.g. Romanian intensive care].

Sepsis kills more ICU patients than any other condition, and the outcome is decided
before the lab confirms it. The patient about to become septic looks almost normal.

Hospitals buy early-warning software, then clinicians stop trusting it — it fires too
late, or constantly. We built the opposite: it predicts six hours ahead, and it is
honest about what it does not know."

### 0:32 — Demo (46s)

*(Switch to the dashboard. It is already open. Do not narrate the loading.)*

"This is live, on real ICU data — [N] thousand stays. I pick a patient and scrub
through their whole stay.

*(Drag the hour control slowly. Let them watch the banner change.)*

Watch the banner. Amber — early warning. Rising breathing rate, falling blood
pressure. The SOFA score is still flat. Nothing is diagnosed yet.

*(Drag forward. Pause.)*

Now SOFA climbs, and the red dashed line is when sepsis was recorded. Our alert fired
hours earlier.

*(Switch tab. Slow down here — this is the point.)*

At the alert hour, this patient is not yet septic. We keep detection and prediction
on separate panels, because mixing them is how these tools lose credibility."

### 1:18 — Data and accuracy (25s)

*(Gesture at the Model Validation tab or your slide. Do not read it out.)*

"Three real cohorts, [N] ICU stays. [NAME THE SOURCES — e.g. MIMIC-IV from Boston,
the PhysioNet Challenge 2019 set from two US hospital systems, and a sepsis cohort
provided to us].

Two numbers. Predicting sepsis six hours before onset: [X] AUROC on [N] stays.
Detecting it once it is present: [Y]. Median warning time, [Z] hours before onset.

And every prediction uses only data available at that moment. We never look ahead.
That is enforced in code, and it is tested."

### 1:43 — Why trust it (29s)

"What makes it trustworthy is not the model, it is the discipline around it.

Missing data is never filled in as normal — it is shown as missing. Every screen
carries the same disclaimer. Identifiers are pseudonymised. And we removed stays with
more than a day of silence, because a five-thousand-hour stay with two days of
readings is noise, not signal.

That is why we report [X] and not a nicer number."

### 2:12 — Feasibility (23s)

"Integration: [LOCAL CONTEXT — e.g. Romanian ICUs have vendor feeds and paper, not
FHIR]. So the design is a narrow, explicit data contract. It accepts a small
allowlisted set of fields, rejects anything else loudly, and tells you what it could
not see.

The path to a ward is certification and local validation, not a rewrite. We documented
that as a checklist instead of pretending it is done."

### 2:35 — Close (13s)

*(Stop moving. Say it slowly.)*

"We did not build a product. We built proof this accuracy is reachable under rules
strict enough to survive a regulator's first question: what did it know, and when did
it know it?"

---

## 2. Module library — swap these in

### Module A — The scaling argument (30s)

Use when judges care about the accuracy ceiling. This is your strongest unclaimed card.

"We have watched this number climb. At a hundred and seventeen stays it was 0.69. At
ten thousand, 0.78. Today, [X]. Every step came from more data or better features —
never from easier validation.

We are holding about half the available dataset. Doubling it, and adding the
modalities these files lack — vasopressors, GCS, urine output, microbiology — is
where the next points are."

### Module B — The honest control (20s)

Use when you sense scepticism about the headline number.

"We also report our weakest cohort: a hundred and seventeen stays, where the same
method scores 0.72. Small data does not flatter us. We are showing you that on
purpose."

### Module C — Detection versus prediction (15s)

Use if a judge looks confused about why you have two numbers.

"Detection answers 'is this patient septic now'. Prediction answers 'will they be'.
The first one is largely solved clinically. The second is the unsolved problem, which
is why we lead with the harder number."

### Module D — The clinician simulator (20s)

Use if there is time, or if judges are clinicians.

"There is also a what-if panel. You set the vitals, and it shows how organ
dysfunction scoring and the alert tier respond — including how vasopressor support
changes the cardiovascular score even when blood pressure looks normal."

### Module E — Data provenance, stated honestly (15s)

Use if you expect a governance question.

"One of our cohorts was provided to us with undocumented provenance. We flag that as a
governance gap rather than ignore it. For a real deployment, that would be resolved
before any pilot."

### Module F — The patient story (25s)

Use instead of Module A if your audience is non-technical.

"Imagine a seventy-year-old admitted with pneumonia. Hour twelve, they look stable —
normal blood pressure, normal temperature. Hour eighteen, their breathing rate has
crept up and their blood pressure has started to drift. Our system flags amber here,
six hours before the criteria for sepsis are met. That window is when treatment
works best — and that is the whole product."

---

## 3. Timing planner

Total budget: 3:00 = 450 words at a measured ~150 wpm. The base script is measured,
not estimated — it runs 2:48, leaving 12 seconds of buffer.

| Block | Words | Seconds | Starts at |
|---|---|---|---|
| Open | 79 | 32 | 0:00 |
| Demo | 114 | 46 | 0:32 |
| Data and accuracy | 63 | 25 | 1:18 |
| Why trust it | 73 | 29 | 1:43 |
| Feasibility | 57 | 23 | 2:12 |
| Close | 33 | 13 | 2:35 |
| **Total** | **419** | **~168** | **ends 2:48** |

Do not fill the last 12 seconds. A pause before you walk off reads as confidence.

If you add modules, cut in this order: **Feasibility → Why trust it → Demo detail.**
Never cut the demo and never cut the two accuracy numbers.

**How to check your own edits:** count only the quoted lines in section 1 and divide
by 150. If the total passes 430 words, you are over.

---

## 4. Numbers fill-in sheet

Fill this once, then never guess on stage.

| Slot | Your value | Where it comes from |
|---|---|---|
| Total ICU stays | 21,732 | `All Datasets Combined` in the dashboard caption |
| Pre-onset cohort | 19,338 | `models/pn2019_metrics.json` → `n_stays` |
| Pre-onset AUROC | 0.84 | same file → `auroc` |
| Pre-onset Brier | 0.0087 | same file → `brier_score` |
| Detection AUROC | 0.95 | `models/combined_detection_metrics.json` → `auroc` |
| Median lead time | 39 hours | `models/pn2019_metrics.json` → `median_lead_time_hours` |
| Weakest cohort | 117 stays, 0.72 | `models/baseline_metrics.json` |

**Say "before onset" every single time you say the 0.84 figure.** Not "0.84
accuracy". The distinction is the credibility.

---

## 5. Cue sheet — rehearse this, not the words

| Time | On screen | You do |
|---|---|---|
| 0:00 | Dashboard, not yet visible | Stand still, no laptop contact |
| 0:32 | All Datasets Combined | Switch over, let it load silently |
| 0:38 | A chosen patient | Drag hour control slowly |
| 0:52 | Alert banner | Pause. Let them read it |
| 1:02 | SOFA trajectory | Drag to onset line |
| 1:09 | Existing Sepsis Status tab | Slow down, lower your voice |
| 1:18 | Model Validation tab | Gesture, do not read |
| 2:35 | Return to dashboard | Stop moving entirely |

**Pre-load the dashboard before you walk on.** First load takes 30–60 seconds on a
free host. If it is cold, start from the Model Validation tab and demo second.

---

## 6. Q&A — one-line answers

**"Is 0.84 good?"** Close to published state of the art for pre-onset, and achieved
without using future data.

**"Why not 0.85?"** We targeted it and report 0.84 honestly. The next gain needs
modalities this dataset lacks.

**"Is this approved for clinical use?"** No. Not a medical device, no CE mark, stated
on every screen.

**"Would it work in a Romanian hospital?"** The method would. It needs site adapters
and Romanian validation first — that is on our checklist, not done.

**"What would deployment take?"** MDR certification, a quality management system,
local clinical validation, and integration work. Months, not weeks.

**"How is this different from Epic's sepsis model?"** Different method, and fully
auditable — the leakage rule, missingness handling and performance are all
inspectable.

**"What if more data does not help?"** Then we report that too. So far the curve says
it does.

---

## 7. Delivery notes

- **Never read the screen.** If you look at the laptop, the judges look at the laptop.
- **Say the disclaimer once, early, in your own words.** It buys credibility with
  clinical judges instead of costing it.
- **Pause after the amber banner.** That silence is the most persuasive second in the
  pitch.
- **Do not apologise for the accuracy.** It is a floor you can defend, not a ceiling
  you missed.
- **If the demo breaks:** switch to Module F (the patient story) and keep going. Never
  debug on stage.

*Decision support only. Not validated for clinical use. Requires clinician review.*
