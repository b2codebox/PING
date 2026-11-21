# 3-Minute Pitch — Judge-Facing

> **Decision support only. Not validated for clinical use. Requires clinician review.**

Scoring targets, and where each is answered:

| Criterion | Where it lands | Time |
|---|---|---|
| Problem & importance | Opening | 0:00–0:30 |
| Demo / prototype | Live dashboard | 0:30–1:15 |
| Realistic data, volume, accuracy | Data slide | 1:15–1:50 |
| Execution quality | What is different | 1:50–2:25 |
| Feasibility & integration | Romania | 2:25–2:50 |
| Close | One line | 2:50–3:00 |

---

## The script (~430 words)

### 0:00 — Problem (30s)

"Sepsis kills more ICU patients than anything else, and the outcome is decided
before the lab result confirms it. The patient who is *about to* become septic looks
almost normal on the chart.

So hospitals buy early-warning software. Then clinicians stop trusting it — it fires
too late, or it fires constantly. We built the opposite: it predicts six hours *ahead*,
and it is honest about what it does not know."

### 0:30 — Demo (45s)

*[Open dashboard. All Datasets Combined, 21,732 ICU stays.]*

"This is live, on real ICU data. I pick a patient and scrub through their stay hour
by hour.

Watch the banner: amber, **early warning** — rising respiratory rate, falling blood
pressure. The SOFA score is still flat. Nothing has been diagnosed yet.

*[Drag slider forward.]* Now SOFA climbs, and the red dashed line is when sepsis was
actually recorded. Our alert fired **hours earlier**.

*[Switch tab.]* This is the part that matters: at the alert hour, the patient is **not
yet septic**. We keep detection and prediction on separate panels — mixing them is how
these tools lose credibility."

### 1:15 — Data & accuracy (35s)

"Three real cohorts, 21,732 ICU stays: MIMIC-IV from Boston, the PhysioNet Challenge
2019 set from two US hospital systems, and a sepsis cohort we were given.

Two numbers. Predicting sepsis six hours before onset: **0.84 AUROC** on 19,338
stays. Detecting it once present: **0.95**. Median warning time, **39 hours** before
onset.

And every prediction uses only data available at that moment. We never look ahead.
That is enforced in code and tested."

### 1:50 — Execution (35s)

"What makes it trustworthy is not the model, it is the discipline around it.

Missing data is never filled in as normal — it is shown as missing. Every screen
carries the same disclaimer. IDs are pseudonymised. And we threw out stays with more
than a day of silence, because a 5,000-hour stay with two days of readings is noise,
not signal.

That is why we report 0.84 and not a nicer number."

### 2:25 — Feasibility (25s)

"Integration: in Romania, ICUs do not have FHIR. They have vendor feeds and paper.

So the design is a narrow, explicit data contract — it accepts a small allowlisted set
of fields, rejects anything else loudly, and tells you what it could not see. That
means it can be pointed at whatever a hospital actually has.

The path to a ward is certification and local validation, not a rewrite. We have
documented that as a checklist rather than pretending it is done."

### 2:50 — Close (10s)

"We did not build a product. We built proof that this accuracy is reachable under
rules strict enough to survive the first question a regulator asks: *what did it know,
and when did it know it?*"

---

## Delivery notes

- **Do not** say AUROC without the word "before onset" — it is the distinction judges
  will probe.
- **Do not** read the numbers; point at them on screen.
- The demo is the centrepiece. If the dashboard is slow to load, the data slide is the
  fallback — rehearse starting from that tab.
- Say "prototype, not a medical device" once, early, in your own words. It buys
  credibility rather than costing it.

---

## Judge Q&A cheat sheet

**"Is 0.84 good?"**
It is close to published state of the art for pre-onset sepsis prediction, and it is
achieved without using future data. Detection — the easier task — is 0.95.

**"Why not 0.85?"**
We targeted it and report 0.84 honestly. The next gain needs data we do not have:
vasopressors, GCS, urine output, microbiology. Loosening validation would get the
number up and the credibility down.

**"Is this approved for clinical use?"**
No. Not a medical device, no CE mark, stated on every screen.

**"Would it work in a Romanian hospital?"**
The method would. The data contract would need site-specific adapters, and it would
need validation on Romanian patients first. That is on our checklist, not done.

**"Where did the 6.6k cohort come from?"**
Provided to us, provenance undocumented. We flag that as a governance gap rather
than ignore it.

**"How is this different from Epic's sepsis model?"**
Different method, and fully auditable: the leakage rule, the missingness handling,
and the performance are inspectable in the repo.

**"What would it take to deploy?"**
Certification under MDR, a quality management system, local clinical validation, and
hospital integration work. Months, not weeks — and we say so.

---

## If you get 2 minutes instead

Cut the Execution section to one sentence — *"missing data is shown as missing, never
imputed as normal"* — and keep the demo and the two accuracy numbers intact.

*Decision support only. Not validated for clinical use. Requires clinician review.*
