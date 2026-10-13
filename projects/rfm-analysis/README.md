# RFM Customer Intelligence

An end-to-end customer analytics project: clean retail transactions, calculate
recency, frequency, and monetary scores, segment customers, and explore the
results in a standalone interactive dashboard.

## Dashboard

The dashboard includes customer and revenue KPIs, an actionable segment
overview, a 5×5 recency/frequency matrix, monthly activity, revenue
concentration, cohort retention, customer distributions, country and product
breakdowns, and a searchable customer ledger with profiles and CSV export.
Country, segment, and score-cell selections interactively filter the views.

## Data and privacy

The analysis expects the column schema from the
[UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail).
Download its Excel workbook or prepare a CSV with the same column names:

`InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`

Place the file in the project root as `data.csv` or pass its path with `--data`.
The raw dataset and generated customer-level results are deliberately excluded
from Git: transaction files are large, and the outputs contain customer IDs
and purchase behaviour. Check the dataset's current terms before using or
redistributing it. Do not publish the raw data or generated output files unless
you have confirmed you are allowed to share them.

## Run locally

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python rfm_analysis.py
```

If using the UCI Excel workbook directly:

```powershell
python rfm_analysis.py --data "Online Retail.xlsx"
```

By default, the script reads `data.csv` beside `rfm_analysis.py` and writes:

- `outputs/rfm_dashboard.html` — open this file directly in a browser.
- `outputs/rfm_customers.csv` — customer-level RFM measures and segments.

Optional arguments:

```powershell
python rfm_analysis.py --data path\to\transactions.csv --output-dir path\to\results
python rfm_analysis.py --as-of 2011-12-09
```

The analysis date is inclusive. By default, it is the latest valid sales date
in the data. Recency is measured in whole days from midnight after the analysis
date.

## Method

Exact duplicates, cancellations, rows without a customer or valid invoice date,
and rows with nonpositive or invalid quantity or unit price are excluded.
Frequency counts distinct invoices, and monetary value is quantity multiplied
by unit price. Each R/F/M dimension receives a tied-safe 1–5 percentile score
(5 is best). The weighted overall score uses recency 15%, frequency 28%, and
monetary value 57%, scaled to 100. Segment labels use recency and frequency,
which keeps lapsed high-frequency customers visible as retention risks.

## Tests

Run the test suite without needing the source dataset:

```powershell
python -m unittest -v
```

## Repository contents

- `rfm_analysis.py` — data preparation, RFM scoring, and dashboard generation.
- `dashboard_template.html` — dashboard UI and visualizations.
- `test_rfm_analysis.py` — focused tests for cleaning, scoring, Excel input,
  and output generation.
- `requirements.txt` — runtime dependencies.
