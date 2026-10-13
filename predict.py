"""
Inference CLI.

  python predict.py --text "kl subhh 11 bje ke bad bhejna ghr pe koi ni h"
  python predict.py --csv replies.csv --out resolved.csv
  python predict.py --csv export.csv --text-col customer_message

Batch mode carries every source column through to the output, so the
recommendation stays attached to the order id, customer name and contact
number it belongs to.
  python predict.py --demo
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from ndr.model import NDRResolver

DEMO = [
    "kl subhh 11 bje ke bad bhejna ghr pe koi ni h abhi",
    "mai ghar pe hi tha koi nahi aaya jhoot bol rahe ho",
    "plzzz cancle kr do mujhe nhi chahiye 🙏",
    "address glt h sahi pincode 302001 h flat no 402",
    "मुझे कल शाम 6 बजे भेज दीजिए मेरा नंबर 9876543210 है",
    "cash nhi tha ghr pe online payment kr skta hu kya",
    "mai 10 din ke liye gaon gaya hu wapas aake bataunga",
    "dlvry boy ne call hi ni kiya aur wapas le gya",
    "mil gaya tha kal hi guard ke paas",
    "hmm ok",
]


def _print(decision, text: str) -> None:
    print(f"\n  reply     : {text}")
    print(f"  intent    : {decision.intent}  ({decision.confidence:.2f})")
    print(f"  action    : {decision.action} — {decision.action_description}")
    print(f"  mode      : {decision.mode}")
    if decision.secondary_action:
        print(f"  also      : {decision.secondary_action}")
    if decision.entities:
        print(f"  entities  : {decision.entities}")
    print(f"  why       : {decision.reason}")
    if decision.alternatives:
        alts = ", ".join(f"{a['intent']} {a['confidence']:.2f}"
                         for a in decision.alternatives[:2])
        print(f"  runner-up : {alts}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ndr_model.joblib")
    ap.add_argument("--text")
    ap.add_argument("--csv")
    ap.add_argument("--out", default="resolved.csv")
    ap.add_argument("--text-col", default="text",
                    help="Name of the column holding the customer reply.")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    resolver = NDRResolver.load(args.model)

    if args.text:
        d = resolver.resolve(args.text)
        if args.json:
            from dataclasses import asdict
            print(json.dumps(asdict(d), indent=2))
        else:
            _print(d, args.text)
        return

    if args.csv:
        df = pd.read_csv(args.csv)
        if args.text_col not in df.columns:
            raise SystemExit(
                f"CSV has no column named '{args.text_col}'. "
                f"Found: {', '.join(df.columns)}. "
                f"Use --text-col to point at the right one."
            )

        rows = resolver.resolve_batch(df[args.text_col].tolist())
        res = pd.DataFrame(rows)

        # Flatten the entity dict into columns ops can actually filter and act
        # on. An alternate phone number given inside the message is the single
        # most useful field here — it is often the *only* reachable number.
        ents = res["entities"]
        flat = pd.DataFrame({
            "intent": res["intent"],
            "confidence": res["confidence"].round(3),
            "action": res["action"],
            "mode": res["mode"],
            "secondary_action": res["secondary_action"],
            "alt_phone_in_message": [e.get("phone") for e in ents],
            "requested_date": [e.get("date") for e in ents],
            "requested_time": [e.get("time") for e in ents],
            "new_pincode": [e.get("pincode") for e in ents],
            "reason": res["reason"],
        })

        # Every original column is carried through untouched, so the output is
        # routable: order id, customer name, contact number, 3PL reason code and
        # anything else in the source file stay attached to the recommendation.
        out = pd.concat([df.reset_index(drop=True), flat], axis=1)
        out.to_csv(args.out, index=False)

        print(f"Resolved {len(out)} replies -> {args.out}")
        print(f"Carried through {len(df.columns)} source column(s): "
              f"{', '.join(df.columns)}")
        print("\nAction mix:")
        print(out["action"].value_counts().to_string())
        print("\nAutomation mix:")
        print(out["mode"].value_counts().to_string())

        n_alt = flat["alt_phone_in_message"].notna().sum()
        if n_alt:
            print(f"\n{n_alt} repl(ies) contained an alternate phone number "
                  f"— see 'alt_phone_in_message'.")

        queue = out[out["mode"] != "AUTO_EXECUTE"]
        if len(queue):
            print(f"{len(queue)} row(s) need a human: filter mode != AUTO_EXECUTE.")
        return

    if args.demo or True:
        print("=" * 78)
        print("NDR RESOLUTION DEMO")
        print("=" * 78)
        for t in DEMO:
            _print(resolver.resolve(t), t)
        print()


if __name__ == "__main__":
    main()
