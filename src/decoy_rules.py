"""Decoy signals: extra words, legal-form conflicts and house-number shifts (v2, stress-tested).

VALIDATED (Kaggle kernel amz-decoy-ab2): mini replica of the team pipeline (TF-IDF top-10 blocking, team
pair_features, LightGBM, best S1 per record if p > thr), 30k S1 entities per country, true matches + realistic
decoys, word lists mined OUT OF SAMPLE, 2-fold by S1 entity. Macro F0.5:
                                 US      India
  team features (baseline)      0.9750  0.9695
  + house features only          0.9826  0.9718
  + decoy words/legal + margin   0.9816  0.9778
  + all (decoy_features + margin) 0.9877  0.9794     (+1.27 / +0.99 points)
  "margin" = blocking score minus the best other candidate's score (not in this file; one line in the pipeline).

HOUSE NUMBER (amz-housenum): decoys sit on the parent's street a few doors away. Same house number: true matches
71% (US) / 55% (India) vs decoys 1% / 16%; same street + shifted number: true 7% / 8% vs decoys 65% / 27%, median
shift 7-9. France (no labels): identical-name pairs WITH a legal-form conflict: same number 0.5%, shifted 48%;
WITHOUT conflict: same number 72%. Same fingerprint, so these features transfer to France.

FRANCE word lists are INFERRED (no labels): the 7 words form the same tight equal-count cluster the generator uses
for US/India decoy words (~6.5k each), their English equivalents are decoys in US/India (group, holdings,
international, india: P = 1.00), and their legal-conflict rate (6-8%) matches US/India decoy words, not other words.
"fils"/"associes" are left out on purpose (evidence conflicts).

Original US/India mining notes:


A decoy is an S2/S3 record that matches no S1 entity. The generator builds many decoys as <real S1 name> + <word>
or as the same name with a different legal form. Mined by Kaggle kernels amz-decoy-mining / amz-decoy-legal on
200k sampled decoys and 200k sampled true pairs (see decoy_words.tsv for counts and probabilities).

  STRONG_DECOY_WORDS: P(decoy | S2/S3 core = S1 core + this word) >= 0.95, seen in >= 20 decoys.
                      Found in 43% (US) / 54% (India) of decoys that have an S1 parent;
                      only 0.01% / 0.03% of true-match extra tokens are in the list.
  SAFE_EXTRA_WORDS:   P(decoy | extra word) <= 0.10, seen in >= 20 true matches (generator noise on true matches).
  legal_form_conflict(): both names have a legal form and neither set contains the other (Inc vs LLC,
                      LLP vs Private Limited). Flags 21% (US) / 13.5% (India) of decoys with a parent; among
                      identical-core true pairs mismatches are 0.7% (US, mostly PC/LP) and ~0 for India.
  Either signal: 63% (US) / 62% (India) of decoys that have an S1 parent.

Use as LightGBM features on candidate pairs, not as hard rules. Requires the patched features.py (core_name
transliterates Indic text; the India word lists were mined on transliterated names, e.g. "stors", "tredars").
"""
import re
from difflib import SequenceMatcher

from features import ADDR, ascii_fold, core_name, norm_name

try:
    from rapidfuzz.distance import Levenshtein as _lev
except ImportError:  # difflib fallback, same scale
    _lev = None

STRONG_DECOY_WORDS = {
    "US": {  # 74 words
        "ace", "advanced", "all", "apex", "associates", "brands", "brown", "care", "central", "choice",
        "city", "classic", "clinic", "coastal", "colonial", "custom", "diamond", "direct", "downtown", "dynamics",
        "east", "eastgate", "elite", "empire", "environmental", "first", "golden", "greater", "green", "group",
        "harbor", "health", "hernandez", "highland", "holdings", "horizon", "integrated", "interstate", "jones", "lakeside",
        "martinez", "metro", "midtown", "miller", "modern", "north", "northside", "optimal", "pacific", "peak",
        "physicians", "primary", "riverside", "select", "signature", "smart", "smith", "solutions", "south", "southside",
        "specialists", "strategic", "studios", "summit", "superior", "supreme", "union", "uptown", "valley", "value",
        "w", "west", "westgate", "works",
    },
    "India": {  # 139 words
        "agencies", "associates", "atomobails", "automobiles", "bakery", "bekari", "bekri", "care", "clinic", "constructions",
        "developers", "ejensij", "ejensis", "electronics", "elektraniks", "engineering", "estate", "exports", "farmasi", "farmesi",
        "farnichar", "finance", "foods", "foundation", "furniture", "garmentas", "garments", "general", "group", "hardaveyar",
        "hardver", "hardware", "healthcare", "holdings", "hospitality", "hotel", "ilektroniks", "impex", "india", "industries",
        "infotech", "infra", "infrastructure", "infratech", "investment", "investments", "janaral", "janral", "javuli", "jenaral",
        "jenarel", "jewellers", "jvelars", "jyuvelars", "logistics", "maa", "management", "marketing", "media", "medicals",
        "medikals", "metikkals", "modern", "motars", "motors", "mottars", "my", "new", "north", "om",
        "otomobails", "overseas", "perfect", "pharmacy", "pioneer", "power", "premier", "prime", "producer", "products",
        "projects", "properties", "provijan", "provishan", "provision", "public", "raj", "ram", "real", "red",
        "restaurant", "restorant", "restorent", "sai", "shakti", "shiva", "shivam", "shyam", "silver", "sky",
        "smart", "software", "solutions", "sons", "south", "southern", "star", "steel", "stil", "stores",
        "stors", "suitas", "sun", "sunrise", "super", "supreme", "surya", "svits", "swastik", "sweets",
        "systems", "tech", "technologies", "teksatailas", "teksatails", "tekstails", "textiles", "tirupati", "traders", "tredars",
        "unavakam", "unique", "united", "universal", "urban", "varttakarkal", "ventures", "vision", "white",
    },
    "France": {  # 7 words, INFERRED (see module docstring)
        "developpement", "distribution", "france", "groupe", "holding", "international", "participations",
    },
}

SAFE_EXTRA_WORDS = {
    "US": {  # 32 words
        "5ervice", "5ervices", "a", "aka", "as", "b", "business", "c0m", "center", "com",
        "d", "dba", "denta1", "doing", "earnosethroat", "enterprises", "f", "fka", "formerly", "hea1th",
        "k", "known", "l1c", "labs", "lnc", "nee", "one", "service", "services", "sys",
        "t", "trading",
    },
    "India": {  # 29 words
        "5ervices", "8rothers", "a", "aka", "as", "b", "business", "c0m", "center", "com",
        "d", "dba", "de1hi", "doing", "f", "fka", "formerly", "jay", "k", "known",
        "labs", "lakshmi", "lndia", "nee", "partners", "service", "services", "sys", "t",
    },
    "France": {  # INFERRED: alias markers + website debris, true-match noise in US/India
        "aka", "as", "business", "com", "dba", "doing", "fka", "formerly", "known", "nee", "services", "www",
    },
}

# canonical legal forms (US / India / France)
_FORM = {
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co",
    "company": "co", "ltd": "ltd", "limited": "ltd", "lp": "lp", "llp": "llp", "pllc": "pllc", "pc": "pc",
    "pa": "pa", "plc": "plc", "pvt": "pvt", "private": "pvt", "public": "public",
    "sarl": "sarl", "sarlu": "sarlu", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "snc": "snc", "scs": "scs", "sca": "sca", "ei": "ei", "eirl": "eirl", "earl": "earl", "gaec": "gaec",
    "selarl": "selarl", "selas": "selas", "scm": "scm", "scp": "scp", "sccv": "sccv", "scop": "scop", "gie": "gie",
}
_INITIALS = re.compile(r"\b(?:[a-z] ){1,}[a-z]\b")      # "s a r l" / "l l c" (dots already turned into spaces)
_NUM = re.compile(r"\d+")
_SPLIT = re.compile(r"[^\w]+")


def _core_tokens(name):
    """core_name tokens with dotted initialisms glued ("S.A.R.L." -> "sarl") and legal forms removed, so a
    spelled-out legal form never shows up as four one-letter "extra words"."""
    glued = _INITIALS.sub(lambda m: m.group(0).replace(" ", ""), norm_name(name))
    return {t for t in core_name(glued).split() if t not in _FORM}


def extra_words(q_name, s1_name):
    """Core-name tokens of the S2/S3 name that are not in the S1 core name."""
    return _core_tokens(q_name) - _core_tokens(s1_name)


def legal_form(name):
    """Set of canonical legal forms in a name ("Foo Pvt. Ltd." -> {"pvt", "ltd"}); Indic text is transliterated."""
    joined = _INITIALS.sub(lambda m: m.group(0).replace(" ", ""), norm_name(name))
    return frozenset(_FORM[t] for t in joined.split() if t in _FORM)


def legal_form_conflict(q_name, s1_name):
    """1 if both names carry a legal form and neither set contains the other (Inc vs LLC), else 0."""
    a, b = legal_form(q_name), legal_form(s1_name)
    return int(bool(a) and bool(b) and not (a <= b or b <= a))


def _street_components(addr):
    """[(house_number, frozenset(street words))] for each comma component that has both."""
    out = []
    for comp in ascii_fold(addr).split(","):
        toks = [t for t in _SPLIT.split(comp) if t]
        nums = [t for t in toks if _NUM.fullmatch(t)]
        words = frozenset(ADDR.get(t, t) for t in toks if not _NUM.search(t))
        if nums and words:
            out.append((int(nums[0]), words))
    return out


def house_features(q_addr, s1_addr):
    """House-number agreement on the best-matching street component of the two addresses.

    Decoys are placed a few doors down the parent's street: same street, house number shifted by a small amount.
    Returns house_same, house_small_shift (1..20), house_large_shift (>20), house_other_street, house_missing,
    house_shift (absolute difference, -1 when not comparable).
    """
    ca, cb = _street_components(q_addr), _street_components(s1_addr)
    if not ca or not cb:
        return {"house_same": 0, "house_small_shift": 0, "house_large_shift": 0, "house_other_street": 0,
                "house_missing": 1, "house_shift": -1}
    j, na, nb = max((len(wa & wb) / max(len(wa | wb), 1), na, nb) for na, wa in ca for nb, wb in cb)
    if j < 0.5:
        return {"house_same": 0, "house_small_shift": 0, "house_large_shift": 0, "house_other_street": 1,
                "house_missing": 0, "house_shift": -1}
    d = abs(na - nb)
    return {"house_same": int(d == 0), "house_small_shift": int(0 < d <= 20), "house_large_shift": int(d > 20),
            "house_other_street": 0, "house_missing": 0, "house_shift": d}


def _similar(a, b):
    """Normalised edit similarity in [0, 1]."""
    if _lev is not None:
        return _lev.normalized_similarity(a, b)
    return SequenceMatcher(None, a, b).ratio()


def decoy_features(q_name, s1_name, country, q_addr=None, s1_addr=None):
    """Pair features: n_decoy_words, n_safe_words, n_typo_extra, n_other_extra, n_missing, legal_form_conflict
    (+ house_* if both addresses are given).

    n_typo_extra: added words that are a close misspelling (edit similarity >= 0.6) of a word missing from the S2/S3
    name. On labelled train, a one-word substitution of this kind is a decoy only 9% (US) / 5% (India) of the time,
    vs 66% / 67% when the new word is another real word (kernel amz-decoy-subst). Not yet in the A/B replica.
    """
    qt, st = _core_tokens(q_name), _core_tokens(s1_name)
    extra, missing = qt - st, st - qt
    strong = STRONG_DECOY_WORDS.get(country, set())
    safe = SAFE_EXTRA_WORDS.get(country, set())
    n_d, n_s = len(extra & strong), len(extra & safe)
    rest = extra - strong - safe
    n_t = sum(1 for w in rest if any(_similar(w, m) >= 0.6 for m in missing))
    out = {"n_decoy_words": n_d, "n_safe_words": n_s, "n_typo_extra": n_t, "n_other_extra": len(rest) - n_t,
           "n_missing": len(missing), "legal_form_conflict": legal_form_conflict(q_name, s1_name)}
    if q_addr is not None and s1_addr is not None:
        out.update(house_features(q_addr, s1_addr))
    return out
