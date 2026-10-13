"""
Coverage audit for the synthetic corpus.

Coverage is a claim. This script is the evidence. It exits non-zero if any
required cell is empty, so a gap fails the build instead of quietly shipping.

    python verify_coverage.py                # generates and audits 20,000 rows
    python verify_coverage.py --csv data.csv # audits an existing corpus
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from ndr.frames import LANGS, frame_inventory
from ndr.synth import build_corpus, split_frame_disjoint
from ndr.taxonomy import ACTIONS, INTENT_BY_CODE, INTENT_CODES, decide

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'[ok]  ' if ok else '[FAIL]'} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-per-intent", type=int, default=300)
    args = ap.parse_args()

    df = pd.read_csv(args.csv) if args.csv else build_corpus(args.rows, seed=args.seed)
    print(f"\nCorpus: {len(df)} rows\n")

    # ---------------------------------------------------------------- 1
    print("1. Intent coverage")
    present = set(df.intent.unique())
    missing = [i for i in INTENT_CODES if i not in present]
    check("all 15 intents present", not missing, missing or f"{len(present)} found")
    counts = df.intent.value_counts()
    thin = counts[counts < args.min_per_intent]
    check(f"every intent >= {args.min_per_intent} rows", thin.empty,
          "min = %d (%s)" % (counts.min(), counts.idxmin()) if thin.empty
          else f"thin: {dict(thin)}")

    # ---------------------------------------------------------------- 2
    print("\n2. Action coverage")
    acts = {INTENT_BY_CODE[i].primary_action for i in present}
    acts |= {INTENT_BY_CODE[i].secondary_action for i in present} - {None}
    unreachable = sorted(set(ACTIONS) - acts)
    check("every action reachable from some intent", not unreachable,
          f"unreachable: {unreachable}" if unreachable else f"{len(acts)}/{len(ACTIONS)}")

    # ---------------------------------------------------------------- 3
    print("\n3. Language coverage")
    langs = set(df.lang.unique()) if "lang" in df else set()
    check("all 3 languages present", set(LANGS) <= langs, ", ".join(sorted(langs)))
    if "lang" in df:
        cross = df.groupby(["intent", "lang"]).size().unstack(fill_value=0)
        empty = [(i, l) for i in cross.index for l in cross.columns if cross.loc[i, l] == 0]
        check("every intent x language cell non-empty", not empty,
              f"{cross.size} cells, min {int(cross.values.min())} rows"
              if not empty else f"empty: {empty[:6]}")

    # ---------------------------------------------------------------- 4
    print("\n4. Frame coverage")
    if "frame_id" in df:
        used = set(df.frame_id.unique())
        required = {f"{i}:{k}" for i, k in frame_inventory()}
        gap = sorted(required - used)
        check("every declared frame used", not gap,
              f"{len(required)} frames" if not gap else f"unused: {gap[:6]}")

    # ---------------------------------------------------------------- 5
    print("\n5. Register and noise coverage")
    if "register" in df:
        check("all 4 registers present", df.register.nunique() >= 4,
              ", ".join(sorted(df.register.unique())))
    if "noised" in df:
        r = df.noised.mean()
        check("both noised and clean rows present", 0.3 < r < 0.9, f"{r:.0%} noised")

    # ---------------------------------------------------------------- 6
    print("\n6. Segment coverage")
    if "segment" in df:
        segs = set(df.segment.unique())
        need = {"core", "adversarial", "multi_intent", "clean_control"}
        check("all 4 segments present", need <= segs, ", ".join(sorted(segs)))

    # ---------------------------------------------------------------- 7
    print("\n7. Frame-disjoint split integrity")
    if "frame_id" in df:
        tr, va, te = split_frame_disjoint(df, seed=args.seed)
        leak = (set(tr.frame_id) & set(te.frame_id)) | (set(tr.frame_id) & set(va.frame_id))
        check("no frame appears in two splits", not leak,
              f"train {len(tr)} / val {len(va)} / test {len(te)}")
        check("all intents present in train", set(tr.intent) == set(INTENT_CODES),
              f"{tr.intent.nunique()}/15")
        check("all intents present in test", set(te.intent) == set(INTENT_CODES),
              f"{te.intent.nunique()}/15")

    # ---------------------------------------------------------------- 8
    print("\n8. Safety invariants")
    for code in ("CANCEL_REQUEST", "REFUSED_NOT_NEEDED", "NEEDS_HUMAN_REVIEW"):
        d = decide(code, 1.0, {})
        check(f"{code} never auto-executes", d.mode == "HUMAN_REQUIRED", d.mode)
    check("low confidence routes to human review",
          decide("RESCHEDULE_REQUEST", 0.10, {}).intent == "NEEDS_HUMAN_REVIEW")

    # ---------------------------------------------------------------- 9
    print("\n9. Duplication")
    dup = 1 - df.text.str.lower().nunique() / len(df)
    check("exact-duplicate rate under 15%", dup < 0.15, f"{dup:.1%}")

    # ------------------------------------------------------------ summary
    if "lang" in df and "intent" in df:
        print("\nIntent x language matrix:")
        print(df.groupby(["intent", "lang"]).size().unstack(fill_value=0).to_string())

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED: {', '.join(FAILS)}")
        sys.exit(1)
    print("All coverage checks passed.")


if __name__ == "__main__":
    main()
