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

### NEESH Fragrance Chatbot

A branded conversational prototype that answers questions about NEESH perfumes
using a structured snapshot of fragrance notes, listed prices, and store
information. It uses a Flask backend and the Anthropic API; the API key stays
server-side and is supplied through environment variables.

- **Tools:** Python, Flask, Anthropic API, HTML, CSS, JavaScript
- **Highlights:** Responsive chat interface, perfume catalog context, and
  optional access-code protection for a hosted demo.
- **Project files and setup:** [Open the NEESH Fragrance Chatbot](projects/neesh-scent-guide/README.md)
- **Source code:** [`projects/neesh-scent-guide/`](projects/neesh-scent-guide/)
- **Live demo:** Not deployed yet. The project README includes instructions for
  running it locally and deploying it as a web service.

## Repository layout

```text
projects/
├── ndr-resolution/    # NDR Resolution System
├── rfm-analysis/      # RFM Customer Intelligence
└── neesh-scent-guide/ # NEESH Fragrance Chatbot
```

More projects can be added under `projects/`, with a short overview and link
added to this page.
