# Data: Source, Nature, and Collection Methodology

*Prepared in response to reviewer feedback on data provenance.*

---

## 1. Statement of the problem

No public corpus of labelled Non-Delivery Report (NDR) responses exists. Customer
replies to failed-delivery notifications contain personally identifiable
information and commercially sensitive operational data, so no third-party
logistics provider or shipping aggregator publishes them.

Our pipeline was therefore initially developed on a **synthetically generated
corpus of 4,606 code-mixed messages** across 13 intent classes. We state plainly
that this data is not evidence of real-world performance. It was used to validate
that the architecture runs end to end, and for nothing else. Published work on
code-mixed NLP supports this caution directly: real code-mixed data has been
shown to produce better results than synthetically generated datasets
(Nayak & Joshi, 2022, L3Cube-HingCorpus, LREC/WILDRE).

The sections below describe how real data is obtained.

---

## 2. Source: three tiers

### Tier 1 — Real customer text (primary training and evaluation data)

**Source.** Public product reviews of Indian courier consumer applications on the
Google Play Store (Delhivery, Shiprocket, DTDC, Blue Dart, Ecom Express),
supplemented by public complaint text on Trustpilot.

**Rationale.** These are real Indian customers describing real delivery failures,
in the exact linguistic register the system must process — code-mixed
Hindi-English, unnormalised spelling, SMS shorthand. Representative examples
observed in the source (paraphrased to avoid republication):

| Observed content | Maps to intent |
|---|---|
| Agent visible delivering nearby, called repeatedly, never came to the door | `FAKE_ATTEMPT_DISPUTE` |
| Agent recorded the customer as unavailable when they were at home all day | `FAKE_ATTEMPT_DISPUTE` |
| Customer asked agent to hold delivery; parcel left with the building guard | `ALREADY_RECEIVED` |
| Customer offered UPI payment, agent insisted on cash only | `PAYMENT_ISSUE` |
| Flat number and tower missing from the shipping label | `ADDRESS_INCORRECT` |

**Target volume.** 1,500–2,000 annotated messages, minimum 60 per intent class.

**Declared limitation.** Reviews are retrospective, longer, and negatively
skewed relative to prompted NDR replies. Section 3 describes the segmentation
step that mitigates the length and multi-topic mismatch; the sentiment skew
remains a limitation and is reported as such.

### Tier 2 — Code-mixed language resource (validation and baseline)

**L3Cube-HingCorpus** (Nayak & Joshi, 2022): 52.93M sentences, 1.04B tokens of
real Hindi-English code-mixed text in Roman script, released publicly at
`github.com/l3cube-pune/code-mixed-nlp` with the HingBERT / HingRoBERTa models.

Used to (a) validate the coverage of our ~560-entry Hinglish normalisation
lexicon against real spelling-variant distributions rather than our own
intuitions, and (b) provide a transformer baseline (HingRoBERTa fine-tuned) that
our lightweight TF-IDF pipeline must be compared against.

### Tier 3 — Synthetic augmentation (class balancing only)

The template generator is retained for minority classes where Tier 1 yields
insufficient examples. Class proportions are weighted to published NDR reason
distributions rather than chosen arbitrarily (customer refusal ≈37% of NDRs,
rescheduling ≈8.8%, out-of-delivery-area ≈2%; industry sources cited in the main
report). Synthetic rows are flagged in a `source` column and models are evaluated
with and without them.

---

## 3. Nature of the data

| Dimension | Specification |
|---|---|
| Unit of analysis | Single clause describing one delivery-failure reason |
| Segmentation | Reviews split on sentence and conjunction boundaries; clauses of 3–35 tokens retained; pure-sentiment clauses discarded |
| Target size | 1,500–2,000 labelled items |
| Class coverage | Minimum 60 items per intent, 13 intents |
| Sampling | 70% model-stratified for class coverage, **30% uniform random control** |
| Language mix | Measured and reported post-collection, not assumed |
| Annotation | 2 independent annotators, blind to each other and to model output |
| Adjudication | Third-party resolution of all disagreements |
| Reliability metric | **Cohen's κ, reported overall and per class** |
| Splits | 70/15/15 stratified train/validation/test, no segment from one review in two splits |
| Format | UTF-8 CSV: `item_id`, `text`, `intent`, `source`, `sample_stratum` |

### Sampling bias — declared

Using the existing classifier to bucket candidates ensures every intent is
represented, but biases the sample toward text the model already handles well.
We therefore draw 30% of the sample uniformly at random, ignoring the model.
**Accuracy on this control subset is reported separately as the unbiased
estimate**; the guided portion is used for training coverage.

Model predictions are never shown to annotators and are never treated as labels.

### Why inter-annotator agreement is reported before any model metric

Cohen's κ estimates the ceiling of the task. If two trained annotators agree at
κ = 0.65 on this taxonomy, a classifier reporting 95% accuracy against those
labels is measuring shared annotator habit, not task performance. Per-class κ
additionally identifies which intent pairs are genuinely ill-defined — for this
taxonomy we anticipate `REFUSED_NOT_NEEDED` vs `CANCEL_REQUEST` and
`CUSTOMER_UNAVAILABLE` vs `FAKE_ATTEMPT_DISPUTE` — and those are corrected in the
annotation guidelines, not in the model.

---

## 4. Ethics and compliance

- All Tier 1 text is public, user-published content, collected for
  non-commercial academic research.
- **Raw text is not republished.** The report contains only derived annotations,
  paraphrased examples, and aggregate statistics.
- Personal identifiers — phone numbers, email addresses, AWB/tracking numbers,
  URLs — are stripped programmatically at the point of collection, before any
  data is written to disk.
- No attempt is made to identify individual reviewers; reviewer names and IDs
  are discarded.
- Platform Terms of Service and institutional ethics guidance were reviewed prior
  to collection. *(Complete this line with the specific policies you checked.)*

---

## 5. Reproducibility

The collection pipeline is scripted and version-controlled:

| Stage | Script | Output |
|---|---|---|
| 1. Collect | `collect/fetch_reviews.py` | `data/raw_reviews.csv` (PII scrubbed) |
| 2. Segment & sample | `collect/prepare_annotation.py` | `data/annotate_{1,2}.csv`, `sample_manifest.csv` |
| 3. Measure agreement | `collect/agreement.py` | κ report, `data/adjudicate.csv` |
| 4. Adjudicate & merge | `collect/agreement.py` (re-run) | `data/gold_labels.csv` |
| 5. Train | `train.py --csv data/gold_labels.csv` | `ndr_model.joblib`, `metrics.json` |
| 6. Evaluate | `eval_challenge.py` | out-of-distribution report |

Random seeds are fixed throughout; the sample manifest records the stratum each
item was drawn from.

---

## 6. Known limitations

1. **Domain shift.** Review text is retrospective and negatively skewed;
   prompted NDR replies are shorter and more neutral. Segmentation addresses
   length but not sentiment skew.
2. **Selection bias.** Customers who write public reviews are not a random sample
   of customers who receive NDR messages; dissatisfied customers are
   over-represented.
3. **Class prevalence is not estimable** from this source. The relative frequency
   of intents in our corpus reflects who complains publicly, not the true NDR
   mix. Prevalence figures are taken from published industry sources instead.
4. **No outcome labels.** We can label what a customer said, not whether the
   resulting action succeeded. Closing that loop requires operator data.

These limitations bound the claims made: this work demonstrates that intents can
be recovered from real code-mixed customer text and mapped to operational
actions. It does not establish production accuracy, which requires labelled data
from a live NDR channel.
