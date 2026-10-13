import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rfm_analysis import calculate_rfm, load_transactions, run_analysis


def make_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("100", "A", "2024-01-01 10:00", 1, 10, "United Kingdom"),
            ("100", "B", "2024-01-10 10:00", 2, 10, "United Kingdom"),
            ("200", "C", "2024-01-01 10:00", 1, 5, "France"),
            ("300", "D", "2024-01-09 10:00", 3, 10, "France"),
        ],
        columns=[
            "CustomerID",
            "InvoiceNo",
            "InvoiceDate",
            "Quantity",
            "UnitPrice",
            "Country",
        ],
    ).assign(
        InvoiceDate=lambda frame: pd.to_datetime(frame["InvoiceDate"]),
        Revenue=lambda frame: frame["Quantity"] * frame["UnitPrice"],
    )


class RfmAnalysisTests(unittest.TestCase):
    def test_calculate_rfm_scores_recent_and_frequent_customers_higher(self) -> None:
        rfm, reference_date, _ = calculate_rfm(make_transactions())
        customers = rfm.set_index("CustomerID")

        self.assertEqual(reference_date, pd.Timestamp("2024-01-11"))
        self.assertEqual(customers.loc["100", "RecencyDays"], 0)
        self.assertEqual(customers.loc["100", "Frequency"], 2)
        self.assertGreater(customers.loc["100", "R_Score"], customers.loc["200", "R_Score"])
        self.assertGreater(customers.loc["100", "F_Score"], customers.loc["200", "F_Score"])
        self.assertTrue(rfm["RFM_Score"].between(20, 100).all())
        self.assertEqual(len(rfm), 3)

    def test_loader_excludes_cancellations_missing_ids_and_invalid_lines(self) -> None:
        rows = [
            ["A", "SKU", "Item", 1, "2024-01-01 10:00", 10, "100", "UK"],
            ["C-A", "SKU", "Item", -1, "2024-01-02 10:00", 10, "100", "UK"],
            ["B", "SKU", "Item", 1, "2024-01-03 10:00", 10, "", "UK"],
            ["D", "SKU", "Item", 0, "2024-01-04 10:00", 10, "200", "UK"],
        ]
        columns = [
            "InvoiceNo",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "UnitPrice",
            "CustomerID",
            "Country",
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "transactions.csv"
            pd.DataFrame(rows, columns=columns).to_csv(source, index=False)
            cleaned, audit = load_transactions(source)

        self.assertEqual(cleaned["InvoiceNo"].tolist(), ["A"])
        self.assertEqual(audit["cancelled_invoice_rows_removed"], 1)
        self.assertEqual(audit["missing_customer_rows_removed"], 1)
        self.assertEqual(audit["nonpositive_or_invalid_amount_rows_removed"], 2)

    def test_loader_accepts_excel_workbooks(self) -> None:
        rows = [
            ["A", "SKU", "Item", 1, "2024-01-01 10:00", 10, "100", "UK"],
        ]
        columns = [
            "InvoiceNo",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "UnitPrice",
            "CustomerID",
            "Country",
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "transactions.xlsx"
            pd.DataFrame(rows, columns=columns).to_excel(source, index=False)
            cleaned, _ = load_transactions(source)

        self.assertEqual(cleaned["CustomerID"].tolist(), ["100"])
        self.assertEqual(cleaned["Revenue"].tolist(), [10])

    def test_run_analysis_creates_csv_and_self_contained_dashboard(self) -> None:
        columns = [
            "InvoiceNo",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "UnitPrice",
            "CustomerID",
            "Country",
        ]
        source_rows = [
            ["A", "SKU", "Item", 1, "2024-01-01 10:00", 10, "100", "UK"],
            ["B", "SKU", "Item", 2, "2024-01-02 10:00", 10, "200", "France"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "transactions.csv"
            pd.DataFrame(source_rows, columns=columns).to_csv(source, index=False)
            csv_path, html_path = run_analysis(source, root / "results")

            self.assertTrue(csv_path.is_file())
            self.assertTrue(html_path.is_file())
            self.assertEqual(len(pd.read_csv(csv_path)), 2)
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("Customer Intelligence", html)
            self.assertIn("Champions", html)
            self.assertNotIn("__CUSTOMERS__", html)


if __name__ == "__main__":
    unittest.main()
