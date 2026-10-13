"""Prepare customer RFM scores and generate a standalone dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}
SEGMENTS = [
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "New Customers",
    "Promising",
    "At Risk",
    "Cannot Lose Them",
    "Hibernating",
]
SEGMENT_ACTIONS = {
    "Champions": "Recent, frequent buyers. Reward them with early access and referral asks; they rarely need discounts.",
    "Loyal Customers": "Frequent buyers seen in the last few months. Upsell higher-value ranges and invite them into a loyalty scheme.",
    "Potential Loyalists": "Recent buyers with a few orders. Use personalised recommendations to turn them into regulars.",
    "New Customers": "First orders very recently. Run an onboarding sequence and prompt a second order within 30 days.",
    "Promising": "Middling recency with few orders. Build familiarity with limited-time offers and best-seller campaigns.",
    "At Risk": "Ordered several times but not lately. Send personalised win-back offers based on what they bought.",
    "Cannot Lose Them": "Former heavy buyers who have gone quiet. Contact them directly with the strongest win-back offer.",
    "Hibernating": "Long gone with few orders. Use low-cost reactivation only and keep them out of expensive campaigns.",
}
DASHBOARD_TEMPLATE = Path(__file__).with_name("dashboard_template.html")


def load_transactions(data_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load transactions, standardize their types, and record cleaning counts."""
    if data_path.suffix.lower() == ".csv":
        transactions = pd.read_csv(
            data_path,
            encoding="ISO-8859-1",
            dtype={"CustomerID": "string", "InvoiceNo": "string"},
        )
    elif data_path.suffix.lower() in {".xlsx", ".xlsm"}:
        transactions = pd.read_excel(
            data_path,
            dtype={"CustomerID": "string", "InvoiceNo": "string"},
        )
    else:
        raise ValueError(
            f"Unsupported input format '{data_path.suffix}'. Use a CSV or Excel "
            "(.xlsx, .xlsm) file."
        )
    missing_columns = REQUIRED_COLUMNS.difference(transactions.columns)
    if missing_columns:
        raise ValueError(
            "Input data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    audit = {"rows_read": len(transactions)}
    transactions = transactions.drop_duplicates().copy()
    audit["duplicate_rows_removed"] = audit["rows_read"] - len(transactions)

    transactions["CustomerID"] = transactions["CustomerID"].str.strip()
    transactions["InvoiceNo"] = transactions["InvoiceNo"].str.strip()
    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"], errors="coerce", format="mixed"
    )
    transactions["Quantity"] = pd.to_numeric(
        transactions["Quantity"], errors="coerce"
    )
    transactions["UnitPrice"] = pd.to_numeric(
        transactions["UnitPrice"], errors="coerce"
    )

    audit["invalid_dates_removed"] = int(transactions["InvoiceDate"].isna().sum())
    audit["missing_customer_rows_removed"] = int(
        transactions["CustomerID"].isna().sum()
        + transactions["CustomerID"].eq("").sum()
    )
    audit["cancelled_invoice_rows_removed"] = int(
        transactions["InvoiceNo"].str.upper().str.startswith("C", na=False).sum()
    )
    invalid_amounts = (
        transactions["Quantity"].isna()
        | transactions["UnitPrice"].isna()
        | transactions["Quantity"].le(0)
        | transactions["UnitPrice"].le(0)
    )
    audit["nonpositive_or_invalid_amount_rows_removed"] = int(invalid_amounts.sum())

    valid_customer = transactions["CustomerID"].notna() & transactions[
        "CustomerID"
    ].ne("")
    valid_invoice = ~transactions["InvoiceNo"].str.upper().str.startswith(
        "C", na=False
    )
    valid_amount = (
        transactions["Quantity"].gt(0) & transactions["UnitPrice"].gt(0)
    )
    transactions = transactions.loc[
        transactions["InvoiceDate"].notna()
        & valid_customer
        & valid_invoice
        & valid_amount
    ].copy()

    if transactions.empty:
        raise ValueError("No valid sales rows remain after cleaning the input data.")

    transactions["Revenue"] = transactions["Quantity"] * transactions["UnitPrice"]
    return transactions, audit


def _quintile_score(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    """Assign tied-safe 1-5 scores, with 5 representing the better outcome."""
    percentile = values.rank(
        method="average", ascending=higher_is_better, pct=True
    )
    scores = np.ceil(percentile * 5).clip(1, 5)
    return scores.astype(int)


def assign_segments(r_score: pd.Series, f_score: pd.Series) -> np.ndarray:
    """Map recency and frequency scores to named customer segments."""
    conditions = [
        (r_score >= 4) & (f_score >= 4),
        (r_score <= 2) & (f_score >= 4),
        (r_score <= 2) & (f_score >= 3),
        (f_score >= 4) & (r_score >= 3),
        (r_score == 5) & (f_score <= 2),
        (r_score >= 4) & (f_score >= 2),
        (r_score <= 2) & (f_score <= 2),
    ]
    return np.select(
        conditions,
        [
            "Champions",
            "Cannot Lose Them",
            "At Risk",
            "Loyal Customers",
            "New Customers",
            "Potential Loyalists",
            "Hibernating",
        ],
        default="Promising",
    )


def calculate_rfm(
    transactions: pd.DataFrame, as_of: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.Timestamp, pd.DataFrame]:
    """Aggregate valid sales into customer RFM metrics and segment labels."""
    latest_sale = transactions["InvoiceDate"].max()
    reference_date = (
        pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)
        if as_of is not None
        else latest_sale.normalize() + pd.Timedelta(days=1)
    )
    if reference_date <= latest_sale:
        raise ValueError(
            "The analysis date must be on or after the latest transaction date."
        )

    grouped = transactions.groupby("CustomerID", sort=False)
    rfm = grouped.agg(
        LastPurchaseDate=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        Country=("Country", lambda values: values.mode().iloc[0]),
    ).reset_index()
    rfm["RecencyDays"] = (
        reference_date - rfm["LastPurchaseDate"]
    ).dt.days.astype(int)

    rfm["R_Score"] = _quintile_score(rfm["RecencyDays"], higher_is_better=False)
    rfm["F_Score"] = _quintile_score(rfm["Frequency"], higher_is_better=True)
    rfm["M_Score"] = _quintile_score(rfm["Monetary"], higher_is_better=True)
    rfm["RFM_Score"] = (
        rfm["R_Score"] * 0.15
        + rfm["F_Score"] * 0.28
        + rfm["M_Score"] * 0.57
    ) * 20
    rfm["RFM_Score"] = rfm["RFM_Score"].round(1)

    rfm["Segment"] = assign_segments(rfm["R_Score"], rfm["F_Score"])

    rfm = rfm[
        [
            "CustomerID",
            "Country",
            "LastPurchaseDate",
            "RecencyDays",
            "Frequency",
            "Monetary",
            "R_Score",
            "F_Score",
            "M_Score",
            "RFM_Score",
            "Segment",
        ]
    ].sort_values(["Monetary", "Frequency"], ascending=False)

    monthly = (
        transactions.assign(Month=transactions["InvoiceDate"].dt.to_period("M"))
        .groupby(["Month", "Country"], as_index=False)["Revenue"]
        .sum()
    )
    monthly["Month"] = monthly["Month"].astype(str)
    return rfm.reset_index(drop=True), reference_date, monthly


def _json_for_html(value: Any) -> str:
    """Serialize data safely for an inline script element."""
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def build_dashboard_data(
    rfm: pd.DataFrame,
    transactions: pd.DataFrame,
    reference_date: pd.Timestamp,
    audit: dict[str, int],
) -> dict[str, Any]:
    """Assemble the compact data payload that the dashboard renders."""
    as_of = reference_date - pd.Timedelta(days=1)
    sales = transactions.assign(Month=transactions["InvoiceDate"].dt.to_period("M"))
    months = pd.period_range(sales["Month"].min(), sales["Month"].max(), freq="M")
    month_index = {month: i for i, month in enumerate(months)}
    countries = sorted(rfm["Country"].dropna().unique().tolist())
    country_index = {country: i for i, country in enumerate(countries)}
    segment_index = {segment: i for i, segment in enumerate(SEGMENTS)}

    history = (
        sales.groupby(["CustomerID", "Month"])
        .agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"))
        .reset_index()
    )
    history_by_customer: dict[str, list[float]] = {}
    for row in history.itertuples(index=False):
        history_by_customer.setdefault(row.CustomerID, []).extend(
            [month_index[row.Month], round(float(row.Revenue), 2), int(row.Orders)]
        )
    basket = sales.groupby("CustomerID").agg(
        Units=("Quantity", "sum"), Products=("StockCode", "nunique")
    )

    customers = []
    for row in rfm.join(basket, on="CustomerID").itertuples(index=False):
        customers.append(
            [
                row.CustomerID,
                country_index[row.Country],
                row.LastPurchaseDate.strftime("%Y-%m-%d"),
                int(row.RecencyDays),
                int(row.Frequency),
                round(float(row.Monetary), 2),
                int(row.R_Score),
                int(row.F_Score),
                int(row.M_Score),
                float(row.RFM_Score),
                segment_index[row.Segment],
                int(row.Units),
                int(row.Products),
                history_by_customer.get(row.CustomerID, []),
            ]
        )

    scores = pd.Series(range(1, 6))
    grid = [
        assign_segments(pd.Series([r] * 5), scores).tolist() for r in range(1, 6)
    ]

    product_sales = sales.merge(rfm[["CustomerID", "Segment"]], on="CustomerID")

    def top_products(frame: pd.DataFrame, limit: int = 10) -> list[list[Any]]:
        top = (
            frame.groupby("Description")
            .agg(
                Revenue=("Revenue", "sum"),
                Units=("Quantity", "sum"),
                Customers=("CustomerID", "nunique"),
            )
            .nlargest(limit, "Revenue")
        )
        return [
            [str(name).strip(), round(float(r.Revenue), 2), int(r.Units), int(r.Customers)]
            for name, r in top.iterrows()
        ]

    products = {"": top_products(product_sales)}
    for segment, frame in product_sales.groupby("Segment"):
        products[segment] = top_products(frame)

    return {
        "asOf": as_of.strftime("%Y-%m-%d"),
        "firstSale": transactions["InvoiceDate"].min().strftime("%Y-%m-%d"),
        "months": [str(month) for month in months],
        "lastMonthPartial": as_of.normalize() < months[-1].end_time.normalize(),
        "countries": countries,
        "segments": [
            {"name": segment, "action": SEGMENT_ACTIONS[segment]}
            for segment in SEGMENTS
        ],
        "grid": grid,
        "customers": customers,
        "products": products,
        "audit": {**audit, "rows_kept": len(transactions)},
    }


def build_dashboard(
    rfm: pd.DataFrame,
    transactions: pd.DataFrame,
    reference_date: pd.Timestamp,
    audit: dict[str, int],
) -> str:
    """Render the standalone HTML dashboard from the template."""
    payload = build_dashboard_data(rfm, transactions, reference_date, audit)
    template = DASHBOARD_TEMPLATE.read_text(encoding="utf-8")
    return template.replace("__DASHBOARD_DATA__", _json_for_html(payload))


def run_analysis(
    data_path: Path, output_dir: Path, as_of: pd.Timestamp | None = None
) -> tuple[Path, Path]:
    """Run the complete analysis and save the customer data and dashboard."""
    transactions, audit = load_transactions(data_path)
    rfm, reference_date, _ = calculate_rfm(transactions, as_of)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "rfm_customers.csv"
    html_path = output_dir / "rfm_dashboard.html"
    rfm.to_csv(csv_path, index=False, date_format="%Y-%m-%d")
    html_path.write_text(
        build_dashboard(rfm, transactions, reference_date, audit), encoding="utf-8"
    )
    return csv_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score customers using RFM and generate an interactive dashboard."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("data.csv"),
        help="Transaction CSV or Excel file (default: data.csv beside this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("outputs"),
        help="Folder for the customer CSV and dashboard (default: outputs).",
    )
    parser.add_argument(
        "--as-of",
        type=lambda value: pd.Timestamp(value),
        help="Optional analysis cutoff date in YYYY-MM-DD format.",
    )
    args = parser.parse_args()
    csv_path, html_path = run_analysis(args.data, args.output_dir, args.as_of)
    print(f"Customer RFM data saved to: {csv_path.resolve()}")
    print(f"Interactive dashboard saved to: {html_path.resolve()}")
    print("Open rfm_dashboard.html in a browser to explore segments and customers.")


if __name__ == "__main__":
    main()
