# Project Portfolio

Welcome! This repository is a collection of projects I have worked on. Each
project has its own folder with source code, documentation, and instructions for
running it.

## Projects

### NDR Resolution System

An applied NLP system that classifies free-text WhatsApp replies to failed
delivery notifications, extracts useful delivery details, and recommends an
operational action. It combines Hinglish text normalization, word and character
TF-IDF features, logistic regression, rule-based guardrails, and confidence-based
human review.

- **Tools:** Python, scikit-learn, pandas, NumPy
- **Evaluation:** 62.5% accuracy on a hand-written, out-of-distribution test set
  of 64 messages. The project documentation explains the limitations and why
  synthetic held-out scores should not be treated as real-world performance.
- **Project files and setup:** [Open the NDR Resolution System](projects/ndr-resolution/README.md)
- **Source code:** [`projects/ndr-resolution/`](projects/ndr-resolution/)

To run its demo from the project directory:

```bash
cd projects/ndr-resolution
pip install -r requirements.txt
python predict.py --demo
```

### RFM Customer Intelligence

A customer analytics project that cleans online retail transactions, scores
recency, frequency, and monetary value, and explores customer segments in an
interactive standalone dashboard.

- **Tools:** Python, pandas, NumPy, HTML, CSS, JavaScript
- **Highlights:** Eight actionable customer segments, cross-filtered dashboard,
  cohort retention, revenue concentration, product and country analysis, and
  customer-level export.
- **Live dashboard:** [View the interactive RFM dashboard](https://monzoty.github.io/mjr-projects/projects/rfm-analysis/dashboard/)
- **Project files and setup:** [Open the RFM Customer Intelligence project](projects/rfm-analysis/README.md)
- **Source code:** [`projects/rfm-analysis/`](projects/rfm-analysis/)

## Repository layout

```text
projects/
├── ndr-resolution/    # NDR Resolution System
└── rfm-analysis/      # RFM Customer Intelligence
```

More projects can be added under `projects/`, with a short overview and link
added to this page.
