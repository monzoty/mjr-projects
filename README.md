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

## Repository layout

```text
projects/
└── ndr-resolution/    # NDR Resolution System
```

More projects can be added under `projects/`, with a short overview and link
added to this page.
