"""Train the NDR resolver and report evaluation metrics."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from ndr.synth import build_corpus, split_frame_disjoint
from ndr.model import NDRResolver
from ndr.normalize import preprocess
from ndr.taxonomy import ABSTAIN_THRESHOLD, AUTO_EXECUTE_THRESHOLD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Labelled CSV with 'text' and 'intent' columns. "
                                  "Omit to use the synthetic corpus.")
    ap.add_argument("--out", default="ndr_model.joblib")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--random-split", action="store_true",
                    help="Use a random row split instead of frame-disjoint. "
                         "Inflates accuracy; for comparison only.")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        if not {"text", "intent"} <= set(df.columns):
            sys.exit("CSV must contain 'text' and 'intent' columns.")
        print(f"Loaded {len(df)} labelled rows from {args.csv}")
    else:
        df = build_corpus(args.rows, seed=args.seed)
        print(f"Built synthetic corpus: {len(df)} rows, "
              f"{df['intent'].nunique()} intents, "
              f"{df['frame_id'].nunique()} frames")

    # Frame-disjoint by default. Rows sharing a frame_id are paraphrases of the
    # same underlying sentence; splitting them at random leaks paraphrases across
    # the boundary and turns the score into a memorisation measure.
    if args.random_split or "frame_id" not in df.columns:
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"].tolist(), df["intent"].tolist(),
            test_size=0.20, random_state=args.seed, stratify=df["intent"],
        )
        split_note = "RANDOM ROW SPLIT (inflated — for comparison only)"
    else:
        tr, va, te = split_frame_disjoint(df, seed=args.seed)
        X_train, y_train = tr["text"].tolist(), tr["intent"].tolist()
        X_test, y_test = te["text"].tolist(), te["intent"].tolist()
        split_note = (f"FRAME-DISJOINT SPLIT (honest) — train {len(tr)}, "
                      f"val {len(va)}, test {len(te)}")
    print(split_note)

    print("\nTraining…")
    resolver = NDRResolver().fit(X_train, y_train)

    # ------------------------------------------------------------- held-out
    y_pred = resolver.predict(X_test)
    print("\n" + "=" * 72)
    print("HELD-OUT TEST SET — " + split_note)
    print("=" * 72)
    print(f"Accuracy      : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro F1      : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Weighted F1   : {f1_score(y_test, y_pred, average='weighted'):.4f}\n")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    # ------------------------------------------------- confusion (top errors)
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    errs = [(labels[i], labels[j], cm[i][j])
            for i in range(len(labels)) for j in range(len(labels))
            if i != j and cm[i][j] > 0]
    errs.sort(key=lambda x: -x[2])
    if errs:
        print("Most frequent confusions (true -> predicted, count):")
        for t, p, c in errs[:10]:
            print(f"  {t:<22} -> {p:<22} {c}")

    # ------------------------------------------------- abstention behaviour
    print("\n" + "=" * 72)
    print("CONFIDENCE / AUTOMATION TRADE-OFF")
    print("=" * 72)
    proba = resolver.predict_proba(X_test)
    conf = proba.max(axis=1)
    correct = np.array(y_pred) == np.array(y_test)
    print(f"{'threshold':>10} {'auto-handled %':>16} {'accuracy on those':>19}")
    for th in [0.0, 0.3, ABSTAIN_THRESHOLD, 0.6, AUTO_EXECUTE_THRESHOLD, 0.85, 0.9]:
        mask = conf >= th
        cov = mask.mean() * 100
        acc = correct[mask].mean() * 100 if mask.any() else float("nan")
        print(f"{th:>10.2f} {cov:>15.1f}% {acc:>18.2f}%")
    print("\nRead this as: the fraction of NDRs the system can close without a human,")
    print("and how often it is right on those. Everything below the threshold is")
    print("routed to the ops queue rather than guessed at.")

    # ------------------------------------------------------- explainability
    print("\n" + "=" * 72)
    print("TOP LEARNED FEATURES (sanity check — do these look like real signals?)")
    print("=" * 72)
    for intent in ["FAKE_ATTEMPT_DISPUTE", "PAYMENT_ISSUE", "NEEDS_HUMAN_REVIEW"]:
        feats = [f for f, _ in resolver.top_features(intent, n=8)]
        print(f"  {intent:<22} {', '.join(feats)}")

    resolver.save(args.out)
    print(f"\nSaved model -> {args.out}")

    metrics = {
        "n_rows": len(df),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "split": split_note,
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics -> metrics.json")


if __name__ == "__main__":
    main()
