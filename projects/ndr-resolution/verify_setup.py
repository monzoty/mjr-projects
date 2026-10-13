"""
Fresh-install self-check.

Run this before train.py. It tests each layer in isolation so that if something
is broken you know *which* layer, instead of staring at a stack trace from the
middle of a sklearn pipeline.

    python verify_setup.py

Exits 0 if everything passes, 1 otherwise.
"""

from __future__ import annotations

import sys
import traceback

PASS, FAIL = "  [ok]  ", "  [FAIL]"
failures: list[str] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        print(f"{PASS} {name}" + (f" — {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} {name} — {type(exc).__name__}: {exc}")
        failures.append(name)
        if "-v" in sys.argv:
            traceback.print_exc()


# --------------------------------------------------------------------------- #
print("\n1. Dependencies")


def _deps():
    import joblib, numpy, pandas, sklearn  # noqa: F401
    return (f"sklearn {sklearn.__version__}, pandas {pandas.__version__}, "
            f"numpy {numpy.__version__}")


check("third-party packages import", _deps)

# --------------------------------------------------------------------------- #
print("\n2. Package layout")


def _pkg():
    import ndr
    from ndr import frames, model, normalize, rules, synth, taxonomy  # noqa: F401
    return f"ndr {ndr.__version__}, all 6 modules found"


check("ndr package imports", _pkg)

# --------------------------------------------------------------------------- #
print("\n3. Normalization layer")


def _norm_hinglish():
    from ndr.normalize import preprocess
    out, _ = preprocess("plzzz cancle kr do mujhe nhi chahiye")
    assert "cancel" in out, f"expected 'cancel' in {out!r}"
    assert "nahi" in out, f"expected 'nahi' in {out!r}"
    return out


def _norm_devanagari():
    from ndr.normalize import preprocess
    out, ents = preprocess("मुझे कल शाम 6 बजे भेज दीजिए")
    assert "kal" in out, f"Devanagari transliteration failed: {out!r}"
    assert ents.get("date") == "kal", f"date not extracted: {ents}"
    return f"{out}  ents={ents}"


def _entities():
    from ndr.normalize import preprocess
    _, ents = preprocess("mera number 9876543210 hai pincode 302001 kal 5 baje")
    assert ents.get("phone") == "9876543210", ents
    assert ents.get("pincode") == "302001", ents
    assert ents.get("date") == "kal", ents
    return str(ents)


check("slang folding (plzzz/cancle/nhi)", _norm_hinglish)
check("Devanagari transliteration", _norm_devanagari)
check("entity extraction (phone/pincode/date)", _entities)

# --------------------------------------------------------------------------- #
print("\n4. Rule layer")


def _rules():
    from ndr.normalize import preprocess
    from ndr.rules import apply_rules
    # The negation-scope pair that the rule layer exists for.
    a, _, _ = apply_rules(preprocess("mai ghar pe hi tha koi nahi aaya")[0])
    assert a == "FAKE_ATTEMPT_DISPUTE", f"got {a}"
    b, _, _ = apply_rules(preprocess("ghar pe koi nahi tha")[0])
    assert b == "CUSTOMER_UNAVAILABLE", f"got {b}"
    # Regression test for the 'sasta mil gaya' bug.
    c, _, _ = apply_rules(preprocess("market se sasta mil gaya isliye rehne do")[0])
    assert c != "ALREADY_RECEIVED", "ALREADY_RECEIVED rule is over-broad again"
    return "negation scope + 'sasta mil gaya' regression both pass"


check("rule precision", _rules)

# --------------------------------------------------------------------------- #
print("\n5. Corpus generation")


def _corpus():
    from ndr.synth import build_corpus
    df = build_corpus(3000, seed=1, multi_intent_rows=100,
                      adversarial_rows=200, clean_control_rows=100)
    assert len(df) > 2000, f"only {len(df)} rows"
    assert df["intent"].nunique() == 15, f"{df['intent'].nunique()} intents, expected 15"
    return f"{len(df)} rows, {df['intent'].nunique()} intents"


check("synthetic corpus builds", _corpus)

# --------------------------------------------------------------------------- #
print("\n6. Model train/predict round-trip (small, ~10s)")


def _model():
    import tempfile, os
    from ndr.synth import build_corpus
    from ndr.model import NDRResolver
    df = build_corpus(3000, seed=1, multi_intent_rows=100,
                      adversarial_rows=200, clean_control_rows=100)
    r = NDRResolver().fit(df["text"], df["intent"])
    d = r.resolve("plzzz cancle kr do")
    assert d.intent == "CANCEL_REQUEST", f"got {d.intent}"
    assert d.mode == "HUMAN_REQUIRED", f"cancel must be human-gated, got {d.mode}"
    # persistence round-trip
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.joblib")
        r.save(p)
        d2 = NDRResolver.load(p).resolve("plzzz cancle kr do")
    assert d2.intent == d.intent, "save/load changed predictions"
    return f"{d.intent} -> {d.action} [{d.mode}], save/load stable"


check("fit / resolve / save / load", _model)

# --------------------------------------------------------------------------- #
print("\n7. Safety invariant")


def _safety():
    from ndr.taxonomy import REQUIRES_HUMAN_APPROVAL, decide
    assert REQUIRES_HUMAN_APPROVAL == {
        "REFUSED_NOT_NEEDED", "CANCEL_REQUEST", "NEEDS_HUMAN_REVIEW",
    }, REQUIRES_HUMAN_APPROVAL
    # Even at total confidence, irreversible actions must not auto-execute.
    for code in REQUIRES_HUMAN_APPROVAL:
        d = decide(code, 1.0, {})
        assert d.mode == "HUMAN_REQUIRED", f"{code} auto-executed at conf=1.0"
    # Low confidence must always abstain.
    d = decide("RESCHEDULE_REQUEST", 0.10, {})
    assert d.action == "ESCALATE_OPS", d.action
    assert d.intent == "NEEDS_HUMAN_REVIEW", d.intent
    return "irreversible actions gated; low confidence abstains"


check("irreversible actions cannot auto-execute", _safety)

# --------------------------------------------------------------------------- #
print()
if failures:
    print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    print("Re-run with -v for full tracebacks.")
    sys.exit(1)
print("All checks passed. You can now run: python train.py")
sys.exit(0)
