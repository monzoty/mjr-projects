"""
Step 2: turn long reviews into NDR-reply-shaped units and sample them for
annotation.

Two problems this solves.

**Length mismatch.** A Play Store review runs 40-200 words and covers several
complaints. An NDR reply is 3-20 words about one thing. Feeding whole reviews to
a model trained for short replies is a distribution mismatch you would have to
declare and could not defend. So reviews are split into clauses, and only
clauses that look like a delivery-failure statement are kept.

**Class imbalance.** Purely random sampling of complaints yields mostly generic
rants and almost no PAYMENT_ISSUE or SELF_PICKUP. Annotating 1,500 rows to end
up with 9 examples of a class is wasted effort.

    python collect/prepare_annotation.py --n 1500 --annotators 2

METHODOLOGICAL WARNING — read before defending this in a viva.
The existing model is used ONLY to bucket candidates so every intent gets
covered. Its predictions are NOT labels and are NOT shown to annotators; the
output sheet has an empty label column. Even so, model-guided sampling biases
the set toward text the model already handles. That is why --control-frac of the
sample is drawn uniformly at random, ignoring the model entirely. Report accuracy
on the control portion separately: it is the unbiased estimate.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ndr.taxonomy import INTENT_CODES  # noqa: E402

# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #

_SPLIT = re.compile(r"(?<=[.!?;])\s+|\s*\n+\s*|\s+(?:but|and then|also|however)\s+",
                    re.I)

# A clause is kept only if it plausibly describes a delivery-failure event.
_RELEVANT = re.compile(
    r"\b(deliver|delivery|delivered|courier|parcel|packet|package|shipment|"
    r"address|attempt|reschedul|cancel|refus|cash|cod|payment|pickup|collect|"
    r"agent|boy|call|phone|contact|available|home|office|out of station|"
    r"received|receive|return|rto|otp|door|gate|guard)\w*\b", re.I)

# Discard clauses that are pure sentiment with no operational content.
_NOISE = re.compile(
    r"^\W*(worst|pathetic|bad|good|nice|thanks?|ok|useless|fraud|cheat|"
    r"very bad|excellent|poor|horrible|great|super|awesome|1 star|5 star)"
    r"[\s\W]*(service|app|experience|company|courier)?[\s\W]*$", re.I)


def segment(text: str) -> list[str]:
    out = []
    for part in _SPLIT.split(str(text)):
        part = re.sub(r"\s+", " ", part).strip(" .,-–—")
        if not (3 <= len(part.split()) <= 35):
            continue
        if _NOISE.match(part):
            continue
        if not _RELEVANT.search(part):
            continue
        out.append(part)
    return out


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def stratified_sample(df: pd.DataFrame, n: int, control_frac: float,
                      per_class_min: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_control = int(n * control_frac)
    n_guided = n - n_control

    control = df.sample(min(n_control, len(df)), random_state=seed)
    control = control.assign(sample_stratum="control_random")

    rest = df.drop(control.index)
    picks = []
    per_class = max(per_class_min, n_guided // max(len(INTENT_CODES), 1))
    for intent in INTENT_CODES:
        pool = rest[rest["_bucket"] == intent]
        if pool.empty:
            continue
        take = pool.sample(min(per_class, len(pool)), random_state=seed)
        picks.append(take.assign(sample_stratum=f"guided_{intent}"))
        rest = rest.drop(take.index)

    guided = pd.concat(picks) if picks else rest.head(0)
    if len(guided) > n_guided:
        guided = guided.sample(n_guided, random_state=seed)
    elif len(guided) < n_guided and len(rest):
        top_up = rest.sample(min(n_guided - len(guided), len(rest)), random_state=seed)
        guided = pd.concat([guided, top_up.assign(sample_stratum="guided_topup")])

    out = pd.concat([control, guided]).sample(frac=1.0, random_state=seed)
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", default="data/raw_reviews.csv")
    ap.add_argument("--model", default="ndr_model.joblib")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--annotators", type=int, default=2)
    ap.add_argument("--control-frac", type=float, default=0.30)
    ap.add_argument("--per-class-min", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    if not Path(args.reviews).exists():
        sys.exit(f"{args.reviews} not found — run collect/fetch_reviews.py first.")

    raw = pd.read_csv(args.reviews)
    print(f"Loaded {len(raw)} reviews")

    rows = []
    for _, r in raw.iterrows():
        for seg in segment(r["raw_text"]):
            rows.append({"text": seg, "source": r.get("source"),
                         "rating": r.get("rating")})
    seg_df = pd.DataFrame(rows)
    seg_df["_key"] = seg_df["text"].str.lower().str.replace(r"\W+", "", regex=True)
    seg_df = seg_df.drop_duplicates(subset=["_key"]).drop(columns=["_key"])
    print(f"Segmented -> {len(seg_df)} candidate messages "
          f"(median {int(seg_df.text.str.split().str.len().median())} words)")

    # Weak bucketing for coverage only — never used as a label.
    try:
        from ndr.model import NDRResolver
        resolver = NDRResolver.load(args.model)
        seg_df["_bucket"] = resolver.predict(seg_df["text"].tolist())
        print("Bucketed for stratification (predictions discarded, not labels)")
    except Exception as exc:  # noqa: BLE001
        print(f"No model available ({exc}); falling back to random sampling")
        seg_df["_bucket"] = "UNCLEAR"

    sample = stratified_sample(seg_df, args.n, args.control_frac,
                               args.per_class_min, args.seed)
    sample.insert(0, "item_id", [f"IT{i:05d}" for i in range(len(sample))])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # One blank sheet per annotator. No model output, no other annotator's
    # labels — independence is the whole point of measuring agreement.
    sheet = sample[["item_id", "text"]].copy()
    sheet["label"] = ""
    sheet["notes"] = ""
    for i in range(1, args.annotators + 1):
        p = outdir / f"annotate_{i}.csv"
        sheet.to_csv(p, index=False)
        print(f"  wrote {p}")

    sample.drop(columns=["_bucket"]).to_csv(outdir / "sample_manifest.csv", index=False)
    (outdir / "LABELS.txt").write_text(
        "Allowed labels (copy exactly):\n\n" + "\n".join(INTENT_CODES) +
        "\n\nUse UNCLEAR if the message does not state a delivery-failure reason.\n"
        "Leave blank ONLY if the text is unreadable; do not guess.\n"
    )
    print(f"  wrote {outdir/'sample_manifest.csv'} and {outdir/'LABELS.txt'}")
    print(f"\n{len(sample)} items ready "
          f"({(sample.sample_stratum=='control_random').sum()} unbiased control).")
    print("Next: two people label independently, then run collect/agreement.py")


if __name__ == "__main__":
    main()
