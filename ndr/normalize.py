"""
Normalization for noisy, code-mixed (Hindi-English) WhatsApp replies.

The whole difficulty of this problem lives in this file. A customer reply like

    "bhai kl subhh 11 bje ke bad bhejna ghr pe koi ni h abhi"

has to end up close enough to

    "bhai kal subah <TIME> ke baad bhejna ghar pe koi nahi hai abhi"

that a bag-of-ngrams model can see it as the same thing as fifty other people
saying the same thing differently.

Strategy, in order:
  1. Unicode + emoji cleanup
  2. Devanagari -> Roman transliteration (people mix scripts mid-sentence)
  3. Entity extraction (dates/times/phones/pincodes) on the readable text
  4. Elongation collapse ("plzzzz" -> "plz")
  5. Lexicon folding: ~400 slang/abbreviation/misspelling variants -> canonical
  6. Entity masking -> <DATE> / <TIME> / <PHONE> / <PINCODE> / <NUM>

Step 5 is the workhorse. Step 6 matters because the literal digits of a phone
number are noise to the classifier, but the *fact* that a phone number was given
is strong evidence of a "call me / different number" intent.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- #
# 1. Devanagari -> Roman
# --------------------------------------------------------------------------- #

_DEVA = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ए": "e",
    "ऐ": "ai", "ओ": "o", "औ": "au", "ऋ": "ri",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh",
    "स": "s", "ह": "h", "ळ": "l", "क्ष": "ksh", "त्र": "tr", "ज्ञ": "gy",
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai",
    "ो": "o", "ौ": "au", "ृ": "ri", "ं": "n", "ँ": "n", "ः": "h", "्": "",
    "़": "", "ॉ": "o", "ऑ": "o",
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}


def deva_to_roman(text: str) -> str:
    if not any("\u0900" <= ch <= "\u097F" for ch in text):
        return text
    return "".join(_DEVA.get(ch, ch) for ch in text)


# --------------------------------------------------------------------------- #
# 2. Lexicon: variant -> canonical
# --------------------------------------------------------------------------- #
# Written as canonical -> [variants] because that is how a human maintains it;
# inverted at import time into the lookup the normalizer actually uses.

_CANON: dict[str, list[str]] = {
    # --- negation / existence (critical: negation flips intent) ---
    "nahi": ["nhi", "nai", "nahin", "nahee", "nhii", "nahiii", "ni", "nhee", "naa", "nah"],
    "hai": ["h", "hain", "hy", "he", "hei", "hain", "hae"],
    "tha": ["thaa", "th"],
    "koi": ["koe", "koii", "kio", "coi"],
    "nobody": ["kohi nahi", "koi nhi"],

    # --- people / home ---
    "ghar": ["ghr", "gher", "gharr", "ghar", "home", "hme", "hom"],
    "office": ["ofc", "offc", "ofis", "officee", "off"],
    "main": ["mai", "mei", "me", "m", "mein"],
    "hum": ["hm", "hume", "humein", "hme"],
    "aap": ["ap", "aapp", "aapko", "apko"],
    "bhai": ["bhaiya", "bhaiyya", "bhaii", "bro", "brother", "bhy"],
    "mummy": ["mom", "maa", "mother", "mataji"],
    "papa": ["dad", "father", "pitaji"],
    "neighbour": ["padosi", "padoshi", "neighbor", "nghbr"],
    "guard": ["gaurd", "watchman", "chowkidar", "security"],

    # --- time words ---
    "kal": ["kl", "kaal", "tomorrow", "tommorow", "tommorrow", "tmrw", "tmw", "2morrow"],
    "aaj": ["aj", "ajj", "today", "tday", "aaz"],
    "parso": ["parsoo", "parson", "dayaftertomorrow"],
    "subah": ["subh", "subhah", "subhe", "sbh", "morning", "mrng", "mrning", "savere", "sawere"],
    "shaam": ["sham", "shm", "evening", "evng", "evning", "eve"],
    "dopahar": ["dopher", "afternoon", "noon", "lunch time"],
    "raat": ["rat", "night", "nite", "ngt"],
    "baje": ["bje", "bajey", "baze", "oclock", "o clock"],
    "baad": ["bad", "bd", "after", "afterr", "ke bad"],
    "pehle": ["pahle", "phle", "before", "b4"],
    "din": ["dn", "days", "day", "dino"],
    "hafta": ["hafte", "week", "wk", "saptah"],
    "somvar": ["monday", "mon", "sombar"],
    "mangalvar": ["tuesday", "tue", "tues"],
    "budhvar": ["wednesday", "wed"],
    "guruvar": ["thursday", "thu", "thurs", "brihaspativar"],
    "shukravar": ["friday", "fri"],
    "shanivar": ["saturday", "sat", "sanivar"],
    "ravivar": ["sunday", "sun", "itwar", "etwar"],
    "weekend": ["wkend", "week end"],

    # --- request / politeness ---
    "please": ["plz", "pls", "plss", "plzz", "pleez", "pleaseee", "krpya", "kripya", "kirpya"],
    "karo": ["kro", "kr", "karo", "kardo", "krdo", "kar do", "karna", "krna", "kariye"],
    "bhejo": ["bhej", "bhejna", "bhjo", "bhejdo", "bhej do", "send", "snd", "bhejiye", "bhejdena"],
    "dedo": ["de do", "dedo", "dedena", "de dena", "dena", "dedijiye", "dijiye", "dijie", "dijiye"],
    "lelunga": ["le lunga", "lelenge", "le lenge", "loonga"],
    "chahiye": ["chaiye", "chahie", "chhiye", "want", "need", "chahiy"],
    "sakta": ["skta", "sakte", "skte", "sakti"],

    # --- delivery domain ---
    "delivery": ["dlvry", "delivary", "deliverey", "delevery", "dilivery", "del", "dlivry",
                 "delivry", "deliveri", "dilevery"],
    "delivery boy": ["delivery agent", "courier boy", "dlvry boy", "delivry boy",
                     "delivery person", "courier guy", "delivry agent"],
    "courier": ["curior", "courrier", "corier", "kurier", "courier"],
    "parcel": ["parcell", "parsel", "pracel", "packet", "pkt", "package", "packge", "saman",
               "samaan", "order", "ordr", "odr", "item"],
    "address": ["addres", "adress", "addr", "adres", "addrs", "pata", "pta", "addressa"],
    "pincode": ["pin code", "pin", "zipcode", "zip", "postal code"],
    "phone": ["fone", "mobile", "mob", "number", "no", "num", "nmbr", "nnbr", "contact number", "cell"],
    "call": ["cal", "cll", "phone karo", "ring", "kall"],
    "attempt": ["atempt", "attemt", "atempted"],
    "aaya": ["aya", "ayaa", "came", "aayaa"],
    "aayega": ["ayega", "aaega", "ayga", "aayenge"],
    "milega": ["milga", "milge", "milenge"],
    "wapas": ["vapas", "wapis", "return", "rtrn", "back"],
    "cancel": ["cancle", "cencel", "canel", "cansel", "cnl", "cancell", "canccel", "radd"],
    "reschedule": ["re schedule", "reshedule", "reschdule", "rescedule", "re-schedule"],
    "paisa": ["paise", "paisa", "cash", "money", "rupee", "rupees", "rs", "amount"],
    "online": ["onlin", "prepaid", "pre paid", "upi", "gpay", "google pay", "phonepe",
               "phone pe", "paytm", "card", "netbanking"],
    "cod": ["c o d", "cash on delivery", "cashondelivery"],
    "chhutti": ["chutti", "chhuti", "holiday", "leave", "off"],
    "shaadi": ["shadi", "wedding", "marriage"],
    "bahar": ["bhar", "outside", "out", "bahr", "baahar"],
    "gaon": ["gaav", "village", "gav", "gaaon"],
    "sheher": ["shehar", "city", "shahar"],
    "travel": ["trip", "tour", "travelling", "traveling", "ghumne"],
    "hospital": ["hospitl", "hsptl", "clinic"],
    "flat": ["flt", "apartment", "aprtment", "appartment", "tower", "society", "socity"],
    "galat": ["glat", "wrong", "wrng", "incorrect", "galt", "glt"],
    "sahi": ["shi", "correct", "crct", "right", "thik", "theek", "thk"],
    "milla": ["mil gaya", "milgya", "mil gya", "received", "recieved", "recived", "mil chuka"],
    "pickup": ["pick up", "pikup", "collect", "khud le lunga"],
    "hub": ["warehouse", "godown", "office se", "branch", "center", "centre"],
    "wait": ["wt", "intezar", "intzar", "ruko", "rukiye"],
    "problem": ["problm", "prblm", "issue", "dikkat", "dikat", "pareshani"],
    "sorry": ["sry", "srry", "maaf", "maf"],
    "thanks": ["thnx", "thx", "tnx", "dhanyavad", "shukriya", "thank u", "thnks"],
    "ok": ["okay", "okk", "k", "thik hai", "theek hai", "acha", "achha", "haan ji"],
    "haan": ["han", "ha", "hn", "yes", "ys", "yeah", "yep", "ji"],
    "kyun": ["kyu", "q", "why", "kyo"],
    "kaun": ["kon", "who", "koun"],
    "kab": ["kb", "when", "kabb"],
    "kahan": ["kaha", "where", "khan"],
    "abhi": ["abi", "now", "rightnow", "abhii"],
    "jaldi": ["jldi", "fast", "quick", "urgent", "urgnt", "asap", "turant"],
    "phir": ["fir", "again", "agn", "dubara", "dobara", "dubra"],
    "jhooth": ["jhuth", "jhut", "lie", "false", "fake", "fraud", "farzi", "jhoot"],
    "aur": ["or", "and", "n", "&"],
    "lekin": ["but", "bt", "magar", "par"],
    "sirf": ["only", "srf", "bas", "bs"],
    "matlab": ["mtlb", "means", "meaning"],
    "mere": ["mre", "mera", "my", "mra", "meri"],
    "apna": ["apne", "apni", "own"],
    "naya": ["new", "nya", "nayi", "naye"],
    "purana": ["old", "purani", "prana"],
}

# English additions live in their own dict and are MERGED into _CANON below.
# They must not be extra keys inside _CANON itself: several of them ("nahi",
# "milla", "aaya"...) already exist there, and a duplicate key in a Python dict
# literal silently discards the earlier entry. That bug removed every Hinglish
# negation variant and was only visible as "ni" quietly failing to fold.
_CANON_EN: dict[str, list[str]] = {
    # --- English SMS shorthand ------------------------------------------------
    # Folded onto the SAME canonical tokens as their Hinglish equivalents where
    # the meaning matches. "no call came" and "call nahi aaya" then share the
    # tokens 'call' and 'nahi', so a single model covers both languages instead
    # of having to learn each intent twice from half the data.
    "you": ["u", "yu", "youu", "ya"],
    "your": ["ur", "yr", "youre"],
    "are": ["r", "aree"],
    "message": ["msg", "mesage", "mssg", "text", "sms", "whatsapp"],
    "nahi": ["not", "dont", "didnt", "doesnt", "wasnt", "werent", "isnt",
             "arent", "cant", "cannot", "wont", "couldnt", "wouldnt", "havent",
             "hasnt", "hadnt", "noone", "nobody", "none", "never", "neither",
             "without", "unable"],
    "please": ["kindly", "request u", "requesting"],
    "sakta": ["can", "could", "able"],
    "chahiye": ["require", "required", "requirement"],
    "milla": ["got it", "have it", "in my hand", "handed over", "collected",
              "picked up by", "taken by"],
    "aaya": ["arrived", "showed up", "turned up", "reached", "visited", "came by"],
    "bahar": ["stepped out", "gone out", "away"],
    "jaldi": ["asap", "immediately", "soon", "quickly", "hurry"],
    "abhi": ["currently", "at present", "right now", "at the moment"],
    "problem": ["trouble", "difficulty", "concern", "complaint"],
}

for _k, _v in _CANON_EN.items():
    _CANON.setdefault(_k, [])
    _CANON[_k] = list(dict.fromkeys(_CANON[_k] + _v))

VARIANT_TO_CANON: dict[str, str] = {}
for canon, variants in _CANON.items():
    for v in variants:
        VARIANT_TO_CANON.setdefault(v, canon)

# Multi-word variants must be replaced before tokenization.
_MULTIWORD = sorted(
    [v for v in VARIANT_TO_CANON if " " in v], key=len, reverse=True
)


# --------------------------------------------------------------------------- #
# 3. Entity extraction
# --------------------------------------------------------------------------- #

_PHONE_RE = re.compile(r"\b(?:\+?91[\-\s]?)?([6-9]\d{9})\b")
_PINCODE_RE = re.compile(r"\b([1-9]\d{5})\b")
_TIME_RE = re.compile(
    r"\b(\d{1,2})\s*(?::|\.)?\s*(\d{2})?\s*"
    r"(am|pm|a\.m|p\.m|baje|bje|bajey|o\s?clock)\b", re.I
)
_HOUR_RANGE_RE = re.compile(r"\b(\d{1,2})\s*(?:se|to|-)\s*(\d{1,2})\s*(?:baje|bje|pm|am)?\b", re.I)

_REL_DAY = {
    "aaj": 0, "today": 0,
    "kal": 1, "tomorrow": 1, "tmrw": 1,
    "parso": 2, "parson": 2,
}
_WEEKDAYS = ["somvar", "mangalvar", "budhvar", "guruvar", "shukravar", "shanivar", "ravivar"]
_DAYPARTS = {"subah": "morning", "shaam": "evening", "dopahar": "afternoon", "raat": "night"}

_N_DAYS_RE = re.compile(r"\b(\d{1,2})\s*(?:din|days|day)\b", re.I)
_N_WEEKS_RE = re.compile(r"\b(\d{1,2})\s*(?:hafta|hafte|week|weeks)\b", re.I)
_DATE_NUM_RE = re.compile(r"\b(\d{1,2})\s*(?:tarikh|tareekh|th|st|nd|rd)\b", re.I)


def extract_entities(text: str) -> dict:
    """Pull structured slots out of the (transliterated, lightly cleaned) reply."""
    ents: dict = {}
    low = text.lower()

    m = _PHONE_RE.search(low)
    if m:
        ents["phone"] = m.group(1)

    for m in _PINCODE_RE.finditer(low):
        if not (ents.get("phone") and m.group(1) in ents["phone"]):
            ents["pincode"] = m.group(1)
            break

    m = _TIME_RE.search(low)
    if m:
        hh, mm, mark = m.group(1), m.group(2) or "00", m.group(3).lower()
        ents["time"] = f"{int(hh)}:{mm} {'pm' if mark.startswith('p') else 'am' if mark.startswith('a') else ''}".strip()
    else:
        m = _HOUR_RANGE_RE.search(low)
        if m:
            ents["time"] = f"{int(m.group(1))}:00-{int(m.group(2))}:00"

    for part, label in _DAYPARTS.items():
        if part in low:
            ents["day_part"] = label
            ents.setdefault("time", label)
            break

    for word, offset in _REL_DAY.items():
        if re.search(rf"\b{word}\b", low):
            ents["date"] = word
            ents["date_offset_days"] = offset
            break

    if "date" not in ents:
        for wd in _WEEKDAYS:
            if wd in low:
                ents["date"] = wd
                break

    if "date" not in ents:
        m = _N_DAYS_RE.search(low)
        if m:
            ents["date"] = f"+{m.group(1)} days"
            ents["date_offset_days"] = int(m.group(1))
        else:
            m = _N_WEEKS_RE.search(low)
            if m:
                ents["date"] = f"+{int(m.group(1)) * 7} days"
                ents["date_offset_days"] = int(m.group(1)) * 7
            else:
                m = _DATE_NUM_RE.search(low)
                if m and 1 <= int(m.group(1)) <= 31:
                    ents["date"] = f"day {m.group(1)}"

    return ents


# --------------------------------------------------------------------------- #
# 4. Normalizer
# --------------------------------------------------------------------------- #

_EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF" "\u2640-\u2642"
    "\u2700-\u27BF" "\uFE0F" "]+", flags=re.UNICODE,
)
_ELONG_RE = re.compile(r"(.)\1{2,}")
_PUNCT_RE = re.compile(r"[^\w\s<>]")
_WS_RE = re.compile(r"\s+")

# Contractions must be expanded BEFORE punctuation is stripped, or "didn't"
# becomes the two useless tokens "didn" and "t" and the negation is lost. Since
# negation flips intent in this domain, losing it is expensive.
_CONTRACTIONS = [
    (r"\bcan\s*'?\s*t\b", " can not "),
    (r"\bwon\s*'?\s*t\b", " will not "),
    (r"\bn\s*'\s*t\b", " not "),
    (r"\b(\w+)n't\b", r"\1 not "),
    (r"\s*'\s*m\b", " am "),
    (r"\s*'\s*re\b", " are "),
    (r"\s*'\s*ve\b", " have "),
    (r"\s*'\s*ll\b", " will "),
    (r"\s*'\s*d\b", " would "),
    (r"\s*'\s*s\b", " is "),
]


def expand_contractions(text: str) -> str:
    for pat, rep in _CONTRACTIONS:
        text = re.sub(pat, rep, text)
    return text


def collapse_elongation(text: str) -> str:
    """plzzzzz -> plzz -> handled by lexicon; nahiiii -> nahii."""
    return _ELONG_RE.sub(r"\1\1", text)


def _fold_token(tok: str) -> str:
    if tok in VARIANT_TO_CANON:
        return VARIANT_TO_CANON[tok]
    # Try shaving a trailing repeated char ("plzz" -> "plz", "nahii" -> "nahi")
    if len(tok) > 2 and tok[-1] == tok[-2]:
        shaved = tok[:-1]
        if shaved in VARIANT_TO_CANON:
            return VARIANT_TO_CANON[shaved]
        if shaved in _CANON:
            return shaved
    return tok


def normalize(text: str, mask_entities: bool = True) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    t = unicodedata.normalize("NFKC", text)
    t = _EMOJI_RE.sub(" ", t)
    t = deva_to_roman(t)
    t = t.lower()
    t = expand_contractions(t)
    t = collapse_elongation(t)

    if mask_entities:
        t = _PHONE_RE.sub(" <PHONE> ", t)
        t = _PINCODE_RE.sub(" <PINCODE> ", t)
        t = _TIME_RE.sub(" <TIME> ", t)
        t = _HOUR_RANGE_RE.sub(" <TIME> ", t)

    t = t.replace("<", " <").replace(">", "> ")
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()

    # Multi-word slang folding
    for phrase in _MULTIWORD:
        if phrase in t:
            t = t.replace(phrase, VARIANT_TO_CANON[phrase])

    tokens = [_fold_token(tok) for tok in t.split()]

    if mask_entities:
        tokens = [
            "<NUM>" if tok.isdigit() and not tok.startswith("<") else tok
            for tok in tokens
        ]

    return " ".join(tokens)


def preprocess(text: str) -> tuple[str, dict]:
    """Return (normalized_text, entities) — the single entry point used elsewhere."""
    readable = deva_to_roman(unicodedata.normalize("NFKC", text or ""))
    readable = _EMOJI_RE.sub(" ", readable)
    readable = expand_contractions(readable.lower())
    # Fold slang before entity extraction so "kl subhh" is findable as "kal subah".
    folded = " ".join(_fold_token(tok) for tok in _WS_RE.sub(" ", readable.lower()).split())
    entities = extract_entities(folded + " " + readable.lower())
    return normalize(text), entities
