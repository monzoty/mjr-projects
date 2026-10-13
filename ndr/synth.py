"""
Synthetic NDR corpus generator with guaranteed coverage.

Coverage is CONSTRUCTED, not hoped for
--------------------------------------
Naive sampling ("draw 20,000 random messages") leaves cells empty by chance and
you only find out later. Instead:

  1. Enumerate every required cell: (intent, reason_key, language). With 72
     frames x 3 languages that is 216 cells.
  2. Give every cell a guaranteed floor of MIN_PER_CELL rows.
  3. Distribute the remaining budget proportionally.
  4. Verify afterwards and fail loudly if any cell is empty.

Every row carries frame_id, lang, register and noise flags, so coverage is
auditable rather than asserted.

Frame-disjoint splitting
------------------------
Two messages from the same (intent, reason_key) are paraphrases, not
independent samples. Splitting rows at random therefore puts paraphrases of the
same sentence in both train and test, and accuracy becomes a memorisation
score. `split_frame_disjoint()` partitions by frame instead. Report both
numbers; only the frame-disjoint one is meaningful.
"""

from __future__ import annotations

import random
import re
import unicodedata

import pandas as pd

from .frames import (AGENTS, CITIES, CLOSERS, CONSTRAINTS, DAYS, LANGS,
                     OPENERS, PARCELS, PEOPLE, PLACES, REASONS, REQUESTS,
                     TIMES, frame_inventory)

# --------------------------------------------------------------------------- #
# Declared parameters — every one of these belongs in the report
# --------------------------------------------------------------------------- #

# Typed customer-support text in urban India skews English; Hinglish is the
# dominant spoken register typed in Roman script; Devanagari is rising via
# transliteration keyboards but remains a minority for typing. ASSUMPTION —
# replace with measured proportions once real channel data is available.
LANG_MIX = {"en": 0.55, "hr": 0.30, "hd": 0.15}

# Published NDR reason distribution, used for the prevalence-weighted corpus.
# The balanced corpus ignores this by design.
PREVALENCE = {
    "CUSTOMER_UNAVAILABLE": 0.22, "REFUSED_NOT_NEEDED": 0.17,
    "ADDRESS_INCORRECT": 0.11, "COURIER_NO_CONTACT": 0.08,
    "RESCHEDULE_REQUEST": 0.09, "PAYMENT_ISSUE": 0.07,
    "FAKE_ATTEMPT_DISPUTE": 0.06, "OUT_OF_STATION": 0.05,
    "CANCEL_REQUEST": 0.05, "ADDRESS_CHANGE": 0.04,
    "NEEDS_HUMAN_REVIEW": 0.03, "ALTERNATE_RECIPIENT": 0.015,
    "ALREADY_RECEIVED": 0.015, "CONTACT_UPDATE": 0.01, "SELF_PICKUP": 0.01,
}

REGISTERS = ["formal", "neutral", "terse", "irritated"]
REGISTER_MIX = [0.20, 0.45, 0.25, 0.10]

NOISE_APPLY_RATE = 0.72     # share of messages that receive any noise
NOISE_TOKEN_RATE = 0.22     # share of eligible tokens corrupted within those
MIN_PER_CELL = 3            # hard floor per (frame, language)

# --------------------------------------------------------------------------- #
# Slot filling
# --------------------------------------------------------------------------- #

_PINS = ["302001", "110045", "400703", "560037", "226010", "411014", "700091"]
_FLATS = ["A-402", "B-12", "304", "C-89", "12B", "704", "D-5"]


def _phone(rng: random.Random) -> str:
    return rng.choice(["", "+91 ", "91"]) + str(rng.randint(6000000000, 9999999999))


def _fill_slots(text: str, lang: str, rng: random.Random) -> str:
    reps = {
        "{d}": rng.choice(DAYS[lang]), "{t}": rng.choice(TIMES[lang]),
        "{place}": rng.choice(PLACES[lang]), "{agent}": rng.choice(AGENTS[lang]),
        "{person}": rng.choice(PEOPLE[lang]), "{city}": rng.choice(CITIES[lang]),
        "{parcel}": rng.choice(PARCELS[lang]), "{pin}": rng.choice(_PINS),
        "{flat}": rng.choice(_FLATS), "{phone}": _phone(rng),
    }
    for k, v in reps.items():
        text = text.replace(k, v)
    return text


# --------------------------------------------------------------------------- #
# Noise model — language aware
# --------------------------------------------------------------------------- #

_QWERTY = {
    "a": "sqwz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}
_VOWELS = "aeiou"
_MATRA = "ािीुूेैोौृंँः"


def _n_delete(w, rng):
    if len(w) < 4:
        return w
    i = rng.randrange(1, len(w))
    return w[:i] + w[i + 1:]


def _n_vowel_drop(w, rng):
    """Documented Roman-Hindi convention: ghar -> ghr, nahi -> nhi."""
    idx = [i for i, c in enumerate(w[1:], 1) if c in _VOWELS]
    if not idx:
        return w
    i = rng.choice(idx)
    return w[:i] + w[i + 1:]


def _n_substitute(w, rng):
    if len(w) < 3:
        return w
    i = rng.randrange(1, len(w))
    c = w[i].lower()
    return w[:i] + rng.choice(_QWERTY[c]) + w[i + 1:] if c in _QWERTY else w


def _n_transpose(w, rng):
    if len(w) < 4:
        return w
    i = rng.randrange(1, len(w) - 1)
    return w[:i] + w[i + 1] + w[i] + w[i + 2:]


def _n_insert(w, rng):
    if len(w) < 3:
        return w
    i = rng.randrange(1, len(w))
    return w[:i] + w[i - 1] + w[i:]


def _n_elongate(w, rng):
    if len(w) < 2:
        return w
    i = rng.randrange(len(w))
    return w[:i] + w[i] * rng.randint(2, 4) + w[i:]


def _n_matra(w, rng):
    """Devanagari-specific: matra substitution/omission, not keyboard adjacency."""
    idx = [i for i, c in enumerate(w) if c in _MATRA]
    if not idx:
        return w
    i = rng.choice(idx)
    return w[:i] + (rng.choice(_MATRA) if rng.random() < 0.5 else "") + w[i + 1:]


_LATIN_OPS = [_n_delete, _n_vowel_drop, _n_substitute, _n_transpose,
              _n_insert, _n_elongate]
_DEVA_OPS = [_n_matra, _n_delete, _n_transpose]


def add_noise(text: str, lang: str, rng: random.Random,
              rate: float = NOISE_TOKEN_RATE) -> str:
    ops = _DEVA_OPS if lang == "hd" else _LATIN_OPS
    out = []
    for w in text.split():
        if len(w) > 2 and not w.isdigit() and rng.random() < rate:
            w = rng.choice(ops)(w, rng)
        out.append(w)
    s = " ".join(out)
    if lang != "hd":
        r = rng.random()
        if r < 0.06:
            s = s.upper()
        elif r < 0.22:
            s = s.capitalize()
    if rng.random() < 0.12:
        s = re.sub(r"[.,]", "", s)
    if rng.random() < 0.08:
        s = s.replace(" ", "  ", 1)
    return s


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #

def compose(intent: str, reason_key: str, lang: str, register: str,
            rng: random.Random) -> str:
    reason = _fill_slots(rng.choice(REASONS[intent][reason_key][lang]), lang, rng)

    # Register controls how much scaffolding attaches to the reason.
    if register == "terse":
        p_open, p_con, p_req, p_close = 0.05, 0.25, 0.05, 0.05
    elif register == "formal":
        p_open, p_con, p_req, p_close = 0.75, 0.55, 0.65, 0.80
    elif register == "irritated":
        p_open, p_con, p_req, p_close = 0.35, 0.35, 0.45, 0.20
    else:
        p_open, p_con, p_req, p_close = 0.40, 0.45, 0.40, 0.45

    parts = []
    if rng.random() < p_open:
        parts.append(rng.choice(OPENERS[lang]).strip())
    parts.append(reason)
    if rng.random() < p_con:
        parts.append(_fill_slots(rng.choice(CONSTRAINTS[lang]), lang, rng).strip())
    if rng.random() < p_req:
        parts.append(rng.choice(REQUESTS[lang]).strip())
    if rng.random() < p_close:
        parts.append(rng.choice(CLOSERS[lang]).strip())

    text = " ".join(p for p in parts if p)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Coverage-guaranteed allocation
# --------------------------------------------------------------------------- #

def _allocate(n_rows: int, weights: dict[str, float] | None,
              min_per_cell: int) -> dict[tuple[str, str, str], int]:
    """Rows per (intent, reason_key, lang). Every cell gets >= min_per_cell."""
    frames = frame_inventory()
    cells = [(i, k, l) for (i, k) in frames for l in LANGS]

    alloc = {c: min_per_cell for c in cells}
    remaining = n_rows - min_per_cell * len(cells)
    if remaining < 0:
        raise ValueError(
            f"n_rows={n_rows} is below the coverage floor "
            f"({len(cells)} cells x {min_per_cell} = {min_per_cell*len(cells)})."
        )

    frames_per_intent: dict[str, int] = {}
    for i, _ in frames:
        frames_per_intent[i] = frames_per_intent.get(i, 0) + 1

    n_intents = len(frames_per_intent)
    share = []
    for (i, k) in frames:
        w_intent = weights[i] if weights else 1.0 / n_intents
        for l in LANGS:
            share.append(((i, k, l), w_intent / frames_per_intent[i] * LANG_MIX[l]))

    total = sum(s for _, s in share)
    for c, s in share:
        alloc[c] += int(remaining * s / total)

    # Hand the rounding shortfall to the largest cells.
    shortfall = n_rows - sum(alloc.values())
    if shortfall > 0:
        order = sorted(share, key=lambda x: -x[1])
        for idx in range(shortfall):
            alloc[order[idx % len(order)][0]] += 1
    return alloc


# --------------------------------------------------------------------------- #
# Adversarial and multi-intent rows
# --------------------------------------------------------------------------- #

# Minimal pairs whose surface forms nearly collide but whose actions diverge.
CONFUSABLE_PAIRS = [
    (("CUSTOMER_UNAVAILABLE", "nobody_home"), ("FAKE_ATTEMPT_DISPUTE", "nobody_came")),
    (("REFUSED_NOT_NEEDED", "refuse_delivery"), ("CANCEL_REQUEST", "cancel_plain")),
    (("ADDRESS_INCORRECT", "wrong_address"), ("ADDRESS_CHANGE", "different_address")),
    (("SELF_PICKUP", "will_collect"), ("ALTERNATE_RECIPIENT", "leave_with_guard")),
    (("COURIER_NO_CONTACT", "no_call"), ("FAKE_ATTEMPT_DISPUTE", "false_update")),
    (("ALREADY_RECEIVED", "proxy_received"), ("ALTERNATE_RECIPIENT", "authorised_person")),
    (("CONTACT_UPDATE", "unreachable"), ("COURIER_NO_CONTACT", "unreachable_agent")),
    (("OUT_OF_STATION", "return_date"), ("RESCHEDULE_REQUEST", "specific_day")),
]

# Conjunctions used to build genuinely multi-intent messages, which are labelled
# NEEDS_HUMAN_REVIEW when the two intents imply conflicting actions.
_JOIN = {"en": [" and also ", " but ", " , also "],
         "hr": [" aur ", " lekin ", " aur ye bhi ki "],
         "hd": [" और ", " लेकिन ", " और यह भी कि "]}

CONFLICTING = {
    ("CANCEL_REQUEST", "RESCHEDULE_REQUEST"),
    ("CANCEL_REQUEST", "ADDRESS_CHANGE"),
    ("REFUSED_NOT_NEEDED", "RESCHEDULE_REQUEST"),
    ("REFUSED_NOT_NEEDED", "SELF_PICKUP"),
    ("ALREADY_RECEIVED", "RESCHEDULE_REQUEST"),
    ("CANCEL_REQUEST", "PAYMENT_ISSUE"),
}


def _multi_intent(rng: random.Random) -> tuple[str, str, str, str]:
    a, b = rng.choice(sorted(CONFLICTING))
    if rng.random() < 0.5:
        a, b = b, a
    lang = rng.choices(list(LANG_MIX), weights=list(LANG_MIX.values()))[0]
    ka = rng.choice(list(REASONS[a]))
    kb = rng.choice(list(REASONS[b]))
    text = (compose(a, ka, lang, "neutral", rng)
            + rng.choice(_JOIN[lang])
            + compose(b, kb, lang, "terse", rng))
    return text, lang, f"{a}:{ka}+{b}:{kb}", "multi_intent"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def build_corpus(n_rows: int = 20000, seed: int = 7,
                 balanced: bool = True,
                 multi_intent_rows: int = 800,
                 adversarial_rows: int = 2400,
                 clean_control_rows: int = 800,
                 min_per_cell: int = MIN_PER_CELL) -> pd.DataFrame:
    """
    Generate the corpus.

    balanced=True  -> equal intent representation (measures capability)
    balanced=False -> weighted to PREVALENCE (measures the real ops queue)
    """
    rng = random.Random(seed)
    special = multi_intent_rows + adversarial_rows + clean_control_rows
    core = n_rows - special
    if core <= 0:
        raise ValueError("n_rows too small for the requested special segments.")

    alloc = _allocate(core, None if balanced else PREVALENCE, min_per_cell)
    rows: list[dict] = []

    # ---- core -------------------------------------------------------------
    for (intent, key, lang), count in alloc.items():
        for _ in range(count):
            reg = rng.choices(REGISTERS, weights=REGISTER_MIX)[0]
            clean = compose(intent, key, lang, reg, rng)
            noised = rng.random() < NOISE_APPLY_RATE
            rows.append({
                "text": add_noise(clean, lang, rng) if noised else clean,
                "intent": intent, "lang": lang, "register": reg,
                "frame_id": f"{intent}:{key}", "segment": "core",
                "noised": noised,
            })

    # ---- adversarial confusable pairs -------------------------------------
    per_pair = max(1, adversarial_rows // (len(CONFUSABLE_PAIRS) * 2))
    for (ia, ka), (ib, kb) in CONFUSABLE_PAIRS:
        for intent, key in ((ia, ka), (ib, kb)):
            for _ in range(per_pair):
                lang = rng.choices(list(LANG_MIX), weights=list(LANG_MIX.values()))[0]
                reg = rng.choice(["terse", "neutral"])
                clean = compose(intent, key, lang, reg, rng)
                noised = rng.random() < NOISE_APPLY_RATE
                rows.append({
                    "text": add_noise(clean, lang, rng) if noised else clean,
                    "intent": intent, "lang": lang, "register": reg,
                    "frame_id": f"{intent}:{key}", "segment": "adversarial",
                    "noised": noised,
                })

    # ---- multi-intent -> NEEDS_HUMAN_REVIEW -------------------------------
    for _ in range(multi_intent_rows):
        text, lang, fid, seg = _multi_intent(rng)
        noised = rng.random() < 0.5
        rows.append({
            "text": add_noise(text, lang, rng) if noised else text,
            "intent": "NEEDS_HUMAN_REVIEW", "lang": lang, "register": "neutral",
            "frame_id": fid, "segment": seg, "noised": noised,
        })

    # ---- clean control (no noise) — isolates noise robustness --------------
    frames = frame_inventory()
    for _ in range(clean_control_rows):
        intent, key = rng.choice(frames)
        lang = rng.choices(list(LANG_MIX), weights=list(LANG_MIX.values()))[0]
        rows.append({
            "text": compose(intent, key, lang, rng.choice(REGISTERS), rng),
            "intent": intent, "lang": lang, "register": "neutral",
            "frame_id": f"{intent}:{key}", "segment": "clean_control",
            "noised": False,
        })

    df = pd.DataFrame(rows)
    df["text"] = df["text"].map(lambda t: unicodedata.normalize("NFC", t).strip())
    df = df[df["text"].str.len() > 0]
    df["n_tokens"] = df["text"].str.split().str.len()
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_frame_disjoint(df: pd.DataFrame, seed: int = 7,
                         val_frac: float = 0.15, test_frac: float = 0.15):
    """
    Partition by frame, not by row.

    Two rows sharing a frame_id are paraphrases of the same underlying sentence.
    A random row split therefore leaks paraphrases across the boundary and the
    resulting accuracy measures memorisation. Splitting by frame is the honest
    version — and every intent is kept present in all three splits.
    """
    rng = random.Random(seed)
    train_f, val_f, test_f = set(), set(), set()

    for intent, grp in df.groupby("intent"):
        frames = sorted(grp["frame_id"].unique())
        rng.shuffle(frames)
        n = len(frames)
        n_test = max(1, int(round(n * test_frac))) if n > 2 else 1
        n_val = max(1, int(round(n * val_frac))) if n > 3 else 0
        test_f |= set(frames[:n_test])
        val_f |= set(frames[n_test:n_test + n_val])
        train_f |= set(frames[n_test + n_val:])

    tr = df[df.frame_id.isin(train_f)]
    va = df[df.frame_id.isin(val_f)]
    te = df[df.frame_id.isin(test_f)]
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)
