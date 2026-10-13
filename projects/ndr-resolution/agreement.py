"""
Step 3: measure how much two humans agree, then build the gold-standard file.

This script produces the number that answers "what is the nature of your data".
Cohen's kappa is the headline: it corrects raw agreement for the agreement you
would expect by chance, so it is comparable across datasets with different class
balances.

    python collect/agreement.py data/annotate_1.csv data/annotate_2.csv

Why it matters more than any model metric: kappa is an estimate of your task's
ceiling. If two trained humans reach kappa 0.65 on your taxonomy, a classifier
reporting 95% accuracy on the same taxonomy is measuring the annotators' shared
habits, not the task. Report kappa BEFORE any model number.

Landis & Koch (1977) is the conventional reading, with the caveat that these
bands are rules of thumb and not a standard:
    < 0.20  slight       0.21-0.40  fair        0.41-0.60  moderate
    0.61-0.80 substantial     > 0.80  almost perfect

Below about 0.60, the fix is the taxonomy or the guidelines, not more annotators.
Per-class kappa tells you which pair of intents is doing the damage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ndr.taxonomy import INTENT_CODES  # noqa: E402


def load(path: str, who: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("item_id", "text", "label"):
        if col not in df.columns:
            sys.exit(f"{path} is missing the '{col}' column")
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    bad = set(df.loc[df.label.ne("") & ~df.label.isin(INTENT_CODES), "label"]) - {"NAN"}
    if bad:
        sys.exit(f"{path} has labels outside the taxonomy: {sorted(bad)}")
    return df[["item_id", "text", "label"]].rename(columns={"label": who})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs=2, help="Two completed annotation CSVs")
    ap.add_argument("--out", default="data/adjudicate.csv")
    ap.add_argument("--gold", default="data/gold_labels.csv")
    args = ap.parse_args()

    a = load(args.files[0], "ann_1")
    b = load(args.files[1], "ann_2")
    m = a.merge(b[["item_id", "ann_2"]], on="item_id", how="inner")
    m = m[(m.ann_1 != "") & (m.ann_2 != "") & (m.ann_1 != "NAN") & (m.ann_2 != "NAN")]
    if m.empty:
        sys.exit("No overlapping labelled items — has anyone actually labelled yet?")

    agree = (m.ann_1 == m.ann_2)
    kappa = cohen_kappa_score(m.ann_1, m.ann_2, labels=INTENT_CODES)

    print("=" * 70)
    print("INTER-ANNOTATOR AGREEMENT")
    print("=" * 70)
    print(f"Doubly-annotated items : {len(m)}")
    print(f"Raw agreement          : {agree.mean() * 100:.1f}%")
    print(f"Cohen's kappa          : {kappa:.3f}")
    band = ("almost perfect" if kappa > .80 else "substantial" if kappa > .60
            else "moderate" if kappa > .40 else "fair" if kappa > .20 else "slight")
    print(f"Interpretation         : {band} (Landis & Koch bands)")

    print("\nPer-class agreement (one-vs-rest kappa):")
    print(f"  {'intent':<24}{'n(a1)':>7}{'n(a2)':>7}{'kappa':>9}")
    for c in INTENT_CODES:
        n1, n2 = (m.ann_1 == c).sum(), (m.ann_2 == c).sum()
        if n1 == 0 and n2 == 0:
            continue
        k = cohen_kappa_score((m.ann_1 == c).astype(int), (m.ann_2 == c).astype(int))
        flag = "  <-- weak" if k < 0.60 else ""
        print(f"  {c:<24}{n1:>7}{n2:>7}{k:>9.3f}{flag}")

    present = [c for c in INTENT_CODES if ((m.ann_1 == c) | (m.ann_2 == c)).any()]
    cm = confusion_matrix(m.ann_1, m.ann_2, labels=present)
    pairs = sorted(
        ((present[i], present[j], cm[i][j]) for i in range(len(present))
         for j in range(len(present)) if i != j and cm[i][j] > 0),
        key=lambda x: -x[2],
    )
    if pairs:
        print("\nMost-confused pairs — fix the GUIDELINES for these, not the model:")
        for x, y, n in pairs[:8]:
            print(f"  {x:<24} vs {y:<24} {n}")

    disagree = m[~agree].copy()
    disagree["final_label"] = ""
    disagree.to_csv(args.out, index=False)
    print(f"\n{len(disagree)} disagreements -> {args.out}")
    print("A third person fills 'final_label', then re-run with --gold to merge.")

    gold = m[agree][["item_id", "text", "ann_1"]].rename(columns={"ann_1": "intent"})
    adj_path = Path(args.out)
    if adj_path.exists():
        adj = pd.read_csv(adj_path)
        if "final_label" in adj and adj["final_label"].notna().any():
            resolved = adj[adj.final_label.astype(str).str.strip() != ""]
            if len(resolved):
                extra = resolved[["item_id", "text", "final_label"]].rename(
                    columns={"final_label": "intent"})
                gold = pd.concat([gold, extra], ignore_index=True)
                print(f"Merged {len(extra)} adjudicated items")

    gold.to_csv(args.gold, index=False)
    print(f"Gold standard ({len(gold)} rows) -> {args.gold}")
    print(f"\nTrain on it:  python train.py --csv {args.gold}")


if __name__ == "__main__":
    main()
