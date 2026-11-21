# Capabilities & Scaling Pitch

> **Decision support only. Not validated for clinical use. Requires clinician review.**

A pitch built around what the prototype *does*, why its accuracy is credible, and why
the accuracy is a floor rather than a ceiling.

---

## PART 1 — What the prototype actually does

| Capability | What a judge sees |
|---|---|
| **Cohort browser** | 21,732 ICU stays across three real cohorts, searchable by ICU specialty — CCU, Neuro SICU, CVICU, TSICU, MICU, SICU |
| **Per-patient trajectory** | Scrub any patient hour by hour across their entire ICU stay, up to 5,000+ hours |
| **Six-organ SOFA** | Respiration, coagulation, liver, cardiovascular, CNS, renal — worst value per 24h window, baseline from first 24h, delta, completeness |
| **APS III** | Prototype physiology estimate, or an explicit "unavailable" rather than a guess |
| **Four-tier alerting** | Green routine, amber early warning, orange organ dysfunction, red possible sepsis — mutually exclusive |
| **Detection vs prevention, separated** | "Existing Sepsis Status" on its own panel, never merged into onset risk |
| **Leakage-safe risk score** | 6-hour onset probability computed only from data available at hour `t` |
| **Top risk drivers** | Ranked contributors per patient, not a black box |
| **Medication filters** | Vasopressor and antimicrobial filtering, preset and searchable — MAP support is visible, not hidden |
| **Onset visualisation** | Reference onset plotted inside the SOFA trajectory |
| **Bilingual UI** | EN/RO toggle, clinical terms kept in English |
| **Clinician simulator** | `/assess` endpoint plus interactive form for hypothetical patients, including vasopressor dose effects on cardiovascular SOFA |

### What makes it different from a model in a notebook

Three things you can point at on screen:

1. **The alert fires before the score moves.** Amber banner while SOFA is flat, then the
   climb. That is the product.
2. **Missing data looks missing.** A component with no measurement scores 0 and is
   labelled missing. It is never quietly imputed as normal.
3. **The number is unflattering and visible.** 0.84 is on the Model Validation tab, with
   the caveats beside it, in the same app the judges just used.

---

## PART 2 — Why the accuracy is credible

Anyone can produce a high AUROC. These four properties are why ours survives inspection.

| Guard | Effect |
|---|---|
| **Patient-grouped CV** | `GroupKFold` by patient — no patient's hours appear in both training and test |
| **Strict forward-only data** | Every feature uses `timestamp <= t`; enforced by a test |
| **Pre-onset training only** | Hours at or after onset are excluded from preventive training entirely |
| **Natural prevalence** | No resampling tricks; Brier 0.0087 shows probabilities are calibrated, not just ranked |

### The two tasks, stated separately

| Task | Cohort | AUROC | Brier | Lead time |
|---|---|---|---|---|
| **Pre-onset 6h prediction** | 19,338 stays | **0.8400** | 0.0087 | **39.0h** |
| **Active sepsis detection** | 21,615 stays (3,385 septic) | **0.9469** | 0.0498 | — |
| Pre-onset, small cohort | 117 stays | 0.7202 | 0.1559 | 29.5h |

The third row is the honest control: on 117 stays, the same method scores 0.72. Small
data does not flatter us — which is exactly the point of Part 3.

---

## PART 3 — Why more data makes it more accurate

This is not a hope. It is the measured trajectory of this prototype.

### The observed scaling curve

| Stage | Stays | AUROC | What changed |
|---|---|---|---|
| MIMIC-IV Demo | 117 | 0.6945 | Baseline, score features only |
| PhysioNet 2019, 1k | 982 | 0.7567 | Real hourly vitals and labs |
| PhysioNet 2019, 10k | 9,780 | 0.7799 | 10× trajectories |
| PhysioNet 2019, 25k | 24,673 | 0.8078 | 2.5× more, tuned boosting |
| Plus feature work | 19,338 | 0.8390 | Lab recency, personal baselines, 24h variability |
| Plus ensemble | 19,338 | **0.8400** | 4-model blend, best of 6 candidates |

Every point on that curve came from either more stays or better causal features. None
came from loosening validation. That is the argument: the method responds to data.

### Where the next gain comes from

**1. Finish the dataset.** We hold ~19.3k of the ~40.3k stays in the PhysioNet 2019
release. The remaining ~21k stays roughly double the cohort and add hundreds more
pre-onset septic trajectories — the rare pattern the model currently under-learns.

**2. Add the missing clinical modalities.** PhysioNet 2019 omits vasopressors, GCS,
urine output, invasive lines, microbiology and comorbidities. Those are exactly the
signals that precede deterioration. A cohort recording all six SOFA organs before onset
— full MIMIC-IV, eICU, or a hospital's own data — closes the largest remaining gap.

**3. Then, and only then, Romanian data.** Different case mix, resistance patterns and
documentation practice. Local validation is what converts a US-trained model into a
deployable one.

### Honest projection

If doubling the cohort and adding the missing modalities moves us from 0.84 to the
0.87–0.90 range, that would place the prototype alongside published pre-onset sepsis
results. We are not claiming that number — we are claiming the curve supports it, the
pipeline is built to absorb it, and the validation guardrails will still be in place
when it arrives.

---

## PART 4 — The 90-second version (spoken)

"We built a working dashboard, not a model in a notebook. Twenty-one thousand ICU
stays, three real cohorts. Pick any patient, scrub their whole stay hour by hour, and
watch the amber alert fire while their SOFA score is still flat. Then watch it climb.
That gap is the product.

Two numbers. Six hours before onset: 0.84 AUROC. Once sepsis is present: 0.95. Median
warning time, 39 hours.

The reason to believe 0.84 is that we report it under rules that make it hard to
cheat. No patient's data appears in both training and test. No prediction uses a
record from the future. Missing measurements are shown as missing, never filled in as
normal. And on our smallest cohort, the same method scores 0.72 — small data does not
flatter us.

Which is the real point. We have watched this number climb from 0.69 to 0.84 as the
data grew from 117 stays to over nineteen thousand. Every step came from more data or
better features, never from easier validation.

We are holding about half the available dataset. Doubling it, and adding the
modalities these files lack — vasopressors, GCS, urine output, microbiology — is where
the next points are. The pipeline is built to absorb them and the guardrails stay on.

Not a medical device. Not validated on Romanian patients. But the method scales, and
we can show you the curve."

---

## Judge Q&A

**"Why should I believe 0.84 will improve?"**
Because we can show the curve: 0.69 at 117 stays, 0.78 at 10k, 0.84 now. Same
method, no validation shortcuts. The pipeline is built for more data.

**"What is stopping you from just getting more data?"**
Nothing technical. The remaining PhysioNet stays are a download problem; the missing
modalities need a credentialed dataset like full MIMIC-IV or a hospital partnership.

**"Isn't 0.95 the number you should lead with?"**
No. Detection is the easier task and it was already solved clinically. Six-hour
pre-onset prediction is the unsolved problem, so 0.84 is the honest headline.

**"What if more data doesn't help?"**
Then we will report that too. The scaling curve so far says it does, but we report
what we measure, not what we hope.

**"Could this run on our hospital's data?"**
The pipeline, yes — with a site-specific adapter and Romanian validation first. The
data contract is narrow and explicit by design, so it works with vendor feeds rather
than requiring FHIR.

*Decision support only. Not validated for clinical use. Requires clinician review.*
