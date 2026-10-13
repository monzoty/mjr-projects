"""
High-precision rule layer.

This is NOT a fallback classifier and it does not simply override the model.
It exists for three specific jobs:

  1.  **Cold start.** On day one there is no labelled data. Rules give a usable
      system while labels accumulate.
  2.  **Negation-scope cases the bag-of-ngrams model reliably gets wrong.**
      "koi nahi tha" (nobody was home -> customer's fault) and "koi nahi aaya"
      (nobody came -> courier's fault) share almost every token but demand
      opposite actions. A word-order-blind model cannot separate these; a
      three-token regex can.
  3.  **Disagreement detection.** When a strong rule and the model disagree,
      that is the single best signal available that the reply is genuinely
      ambiguous — so it gets routed to a human instead of being auto-executed.

Rules run on *normalized* text, so they only need to cover canonical spellings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    intent: str
    pattern: str
    strength: str = "strong"   # strong | weak
    note: str = ""

    def compiled(self):
        return re.compile(self.pattern)


RULES: list[Rule] = [
    # --- negation-scope pairs (the reason this layer exists) -----------------
    Rule("FAKE_ATTEMPT_DISPUTE", r"\bkoi\b(?:\s+\w+){0,2}\s+nahi\s+aaya\b",
         note="'nobody came' — courier fault, not customer fault"),
    Rule("FAKE_ATTEMPT_DISPUTE", r"\baaya\s+hi\s+nahi\b"),
    Rule("FAKE_ATTEMPT_DISPUTE", r"\bnobody\s+came\b|\bno\s+one\s+(came|visited)\b"),
    Rule("FAKE_ATTEMPT_DISPUTE", r"\b(jhooth|farzi|fake|fraud)\b"),
    Rule("CUSTOMER_UNAVAILABLE", r"\bkoi\b(?:\s+\w+){0,2}\s+nahi\s+(tha|hai|the)\b",
         strength="weak", note="'nobody was home' — customer unavailable"),

    # --- irreversible actions: demand explicit lexical evidence -------------
    Rule("CANCEL_REQUEST", r"\bcancel\b"),
    Rule("REFUSED_NOT_NEEDED",
         r"\bnahi\s+chahiye\b|\bdont?\s+want\b|\bnot\s+needed\b|\bwapas\s+le\s+jao\b"),

    # --- unambiguous single-token signals ------------------------------------
    # NOTE: an earlier version of this rule was just \b(milla|already)\b, which
    # fired on "market se sasta milla" ("found it cheaper elsewhere") and turned
    # a REFUSED_NOT_NEEDED into ALREADY_RECEIVED — i.e. it suppressed an RTO on
    # a parcel the customer had explicitly rejected. The object of the verb is
    # now required, because "milla" alone does not say *what* was received.
    Rule("ALREADY_RECEIVED",
         r"\b(parcel|delivery|courier)\b(?:\s+\w+){0,3}\s+milla\b"
         r"|\bmilla\s+(hai|hai\b|chuka)\b"
         r"|\balready\s+(delivered|received|milla)\b"
         r"|\bmilla\s+<?\w*>?\s*(hai)\b",
         strength="weak"),
    Rule("SELF_PICKUP", r"\b(khud|self)\b(?:\s+\w+){0,3}\s*(le|lelunga|pickup|collect)\b"),
    Rule("SELF_PICKUP", r"\bpickup\b|\bcollect\b"),
    Rule("PAYMENT_ISSUE", r"\bpaisa\s+nahi\b|\bcash\s+nahi\b|\bno\s+cash\b"),
    Rule("PAYMENT_ISSUE", r"\bonline\b(?:\s+\w+){0,2}\s*(karo|kar|payment|pay)\b", strength="weak"),
    Rule("ADDRESS_INCORRECT", r"\baddress\s+galat\b|\bgalat\s+address\b|\bgalat\s+pata\b"),
    Rule("ADDRESS_CHANGE", r"\baddress\s+(change|update)\b|\bchange\s+address\b"),
    Rule("OUT_OF_STATION", r"\bout\s+of\s+(station|city|town)\b"),
    Rule("COURIER_NO_CONTACT", r"\bcall\s+nahi\b|\bphone\s+nahi\s+karo?\b|\bno\s+call\b"),
]

_COMPILED = [(r, r.compiled()) for r in RULES]


def apply_rules(normalized_text: str) -> tuple[str | None, str, list[str]]:
    """
    Return (intent or None, strength, matched_patterns).

    If several rules fire, a 'strong' match wins over a 'weak' one; ties are
    broken by the order above, which is arranged most-specific-first.
    """
    hits: list[tuple[Rule, str]] = []
    for rule, rx in _COMPILED:
        if rx.search(normalized_text):
            hits.append((rule, rule.pattern))

    if not hits:
        return None, "none", []

    strong = [h for h in hits if h[0].strength == "strong"]
    chosen = (strong or hits)[0]
    return chosen[0].intent, chosen[0].strength, [p for _, p in hits]
