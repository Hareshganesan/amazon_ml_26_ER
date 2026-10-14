"""Text normalisation and pair features shared by blocking, training and inference."""
import re
import unicodedata

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
except ImportError:  # rapidfuzz is required for features; blocking still works without it
    fuzz = JaroWinkler = None

# Legal / filler tokens (US, India, France) removed from the "core" name.
LEGAL = {
    "inc", "incorporated", "llc", "llp", "ltd", "limited", "pvt", "private", "corp", "corporation",
    "co", "company", "plc", "lp", "pllc", "pc", "the", "and", "of",
    "sarl", "sas", "sasu", "sa", "sci", "eurl", "snc",
}
# Address abbreviations -> canonical form (applied token-wise after lowercasing).
ADDR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "hwy": "highway",
    "pkwy": "parkway", "cir": "circle", "trl": "trail", "ste": "suite", "apt": "apartment", "fl": "floor",
    "n": "north", "s": "south", "e": "east", "w": "west", "nr": "near", "opp": "opposite",
    "r": "rue", "ch": "chemin", "imp": "impasse", "all": "allee", "rte": "route",
}
_non_word = re.compile(r"[^\w]+")
_num = re.compile(r"\d+")

from transliterate import transliterate  # noqa: E402  (Venky: 9 Indian scripts -> Latin, no dependencies)

from french_rules import fr_core_name, glue_initials, is_domain_name, normalise_fr_addr, normalise_fr_name, squash_name  # noqa: E402

VOCAB = {}  # token -> number of S1 core names containing it (set with set_vocab)
NAME_CNT = {}  # "country|core name" -> number of S1 entities with that core name (set with set_name_counts)


def set_name_counts(s1):
    """Count S1 entities per (country, core name); adds s1["name_cnt"] and fills NAME_CNT for query lookups.

    French names are templated ("Bordeaux Club" x530), so a name match alone is weak evidence there;
    this lets the model trust the name when it is unique and fall back on the address when it is not.
    """
    key = s1["country"] + "|" + s1["c_name"]
    cnt = key.map(key.value_counts())
    s1["name_cnt"] = cnt.to_numpy(np.int32)
    NAME_CNT.clear()
    NAME_CNT.update(key.value_counts().to_dict())


def set_vocab(core_names):
    """Token document frequencies over S1 core names; used to tell typos (rare) from real words (common)."""
    from collections import Counter
    c = Counter()
    for n in core_names:
        c.update(set(n.split()))
    VOCAB.clear()
    VOCAB.update(c)
    VOCAB["__N__"] = sum(1 for _ in core_names) if hasattr(core_names, "__len__") else 0


def ascii_fold(s):
    """Transliterate Indian scripts to Latin (transliterate.py), then lowercase and strip accents.

    Without the transliteration step NFKD strips Indic vowel signs ("प्राइवेट" -> "पराइवट") and Indian-script
    names can never match their Latin S1 spelling. Latin text passes through transliterate() unchanged.
    """
    s = transliterate(s)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def norm_name(s):
    """Lowercase/fold, glue dotted initialisms (S.A.R.L. -> sarl), split on non-word characters."""
    return " ".join(t for t in _non_word.split(glue_initials(ascii_fold(s).replace("&", " and "))) if t)


def core_name(s):
    """Name without legal suffixes / fillers, tokens sorted (robust to reordering)."""
    return " ".join(sorted(t for t in norm_name(s).split() if t not in LEGAL))


def norm_addr(s):
    """Normalise an address: ASCII-fold/transliterate, split on non-word characters, expand abbreviations (St -> street)."""
    return " ".join(ADDR.get(t, t) for t in _non_word.split(ascii_fold(s)) if t)


def add_text_columns(df):
    """Add normalised columns used by blocking and features.

    French rows use the French rules (french_rules.py) INSTEAD of the US-shaped ones: the generic ADDR
    map turns "st"/"ste" (saint) into street/suite and "dr" (docteur) into drive.
    """
    df["n_name"] = df["business_name"].map(norm_name)
    df["c_name"] = df["business_name"].map(core_name)
    df["n_addr"] = df["business_address"].map(norm_addr)
    fr = (df["country"] == "France").to_numpy()
    if fr.any():
        df.loc[fr, "n_name"] = df.loc[fr, "business_name"].map(lambda x: normalise_fr_name(x, drop_legal=False))
        df.loc[fr, "c_name"] = df.loc[fr, "business_name"].map(fr_core_name)
        df.loc[fr, "n_addr"] = df.loc[fr, "business_address"].map(normalise_fr_addr)
    df["sq_name"] = df["business_name"].map(squash_name)
    df["is_domain"] = df["business_name"].map(is_domain_name).astype(np.int8)
    df["name_addr"] = df["c_name"] + " | " + df["n_addr"]
    nums = df["business_address"].map(lambda a: _num.findall(a))
    df["postcode"] = nums.map(lambda ns: next((n for n in reversed(ns) if len(n) in (5, 6)), ""))
    df["nums"] = nums.map(lambda ns: " ".join(sorted(set(ns))))
    df["non_latin"] = df["business_name"].map(lambda s: any(ord(c) > 0x24F for c in s)).astype(np.int8)
    return df


try:
    from rapidfuzz.process import cpdist
except ImportError:
    cpdist = None


def _pairwise(a, b, fn):
    """Element-wise similarity of two aligned string columns (multi-threaded rapidfuzz when possible)."""
    if cpdist is not None and fn is not _jacc:
        scorer = JaroWinkler.normalized_similarity if fn is JaroWinkler.similarity else fn
        return cpdist(list(a), list(b), scorer=scorer, workers=-1).astype(np.float32)
    return np.fromiter((fn(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))


def _jacc(x, y):
    """Token Jaccard similarity of two whitespace-separated strings; -1 if either is empty."""
    sx, sy = set(x.split()), set(y.split())
    if not sx or not sy:
        return -1.0
    return len(sx & sy) / len(sx | sy)


def _diff_token_freq(x, y):
    """(max, min) log10 document frequency of tokens present in only one of the two core names.

    Decoys are built as "<true name> + <a real word>" (Groupe, Parents, Fils), so the differing token is
    frequent; genuine duplicates differ by a typo, which is rare or unseen. -1 when no token differs.
    """
    d = set(x.split()) ^ set(y.split())
    if not d:
        return -1.0, -1.0
    f = [np.log10(1 + VOCAB.get(t, 0)) for t in d]
    return max(f), min(f)


def _containment(x, y):
    """(share of x's tokens found in y, share of y's tokens found in x); -1 when a side is empty.

    India/France S1 addresses carry an extra district/region that S2/S3 drop ("Kolkata, Kolkata, Howrah" vs
    "KOLKATA, KOLKATA"), and S2/S3 sometimes add a component; containment is not penalised by the extra part.
    """
    sx, sy = set(x.split()), set(y.split())
    if not sx or not sy:
        return -1.0, -1.0
    inter = len(sx & sy)
    return inter / len(sx), inter / len(sy)


from functools import lru_cache  # noqa: E402

CACHE_SIZE = 400_000  # per cache; 2M per cache pushed 10M-record test runs over Kaggle's RAM (v12 killed)


def clear_caches():
    """Empty the per-string LRU caches between test chunks to bound memory."""
    for fn in (_dr_core, _dr_legal, _dr_street, deleet, _street_raw):
        fn.cache_clear()


@lru_cache(maxsize=CACHE_SIZE)
def _dr_core(name):
    """Cached core-name tokens from decoy_rules (name without legal form and filler words)."""
    import decoy_rules as dr
    return frozenset(dr._core_tokens(name))


@lru_cache(maxsize=CACHE_SIZE)
def _dr_legal(name):
    """Cached legal form (LLC, Pvt Ltd, SARL, ...) detected by decoy_rules."""
    import decoy_rules as dr
    return dr.legal_form(name)


@lru_cache(maxsize=CACHE_SIZE)
def _dr_street(addr):
    """Cached (house number, street) components of an address parsed by decoy_rules."""
    import decoy_rules as dr
    return tuple(dr._street_components(addr))


_LEET = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "3": "e", "4": "a", "7": "t", "9": "g"})
_ALNUM_TOK = re.compile(r"\b(?=\w*[a-z])(?=\w*\d)\w+\b")


@lru_cache(maxsize=CACHE_SIZE)
def deleet(name):
    """Undo look-alike digit substitutions inside words ("tara0ra" -> "taraora", "pinnac1e" -> "pinnacle").

    Only tokens that mix letters and digits are changed, so house numbers and pure numbers are untouched.
    The generator applies this corruption to true matches (6.5% of v5's hesitant misses).
    """
    return _ALNUM_TOK.sub(lambda m: m.group(0).translate(_LEET), name)


_NUMTOK = re.compile(r"\d+")


@lru_cache(maxsize=CACHE_SIZE)
def _street_raw(addr):
    """[(house number string without leading zeros, frozenset(street words))] per comma component."""
    out = []
    for comp in ascii_fold(addr).split(","):
        toks = [t for t in _non_word.split(comp) if t]
        nums = [t for t in toks if t.isdigit()]
        words = frozenset(ADDR.get(t, t) for t in toks if not _NUMTOK.search(t))
        if nums and words:
            out.append((nums[0].lstrip("0") or "0", words))
    return tuple(out)


def _house_text(ca, cb):
    """House numbers compared as TEXT on the best-matching street component.

    The generator types true matches' house numbers with typos ("406"/"409", "224"/"2244", "10"/"410"),
    which numeric shift features read as "a different building". Returns (digit edit distance, one number
    contains the other, same length); (-1, -1, -1) when not comparable.
    """
    if not ca or not cb:
        return -1, -1, -1
    j, na, nb = max((len(wa & wb) / max(len(wa | wb), 1), na, nb) for na, wa in ca for nb, wb in cb)
    if j < 0.5:
        return -1, -1, -1
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(na, nb), int(na in nb or nb in na), int(len(na) == len(nb))


def _house(ca, cb):
    """Venky's house_features() on precomputed street components
    -> (same, small, large, other_street, missing, shift, street_jacc).

    street_jacc = word Jaccard of the best-matching "number + street" components. It compares only the street
    part, so it is not penalised by the administrative tail (district / region / departement) that sources
    include at different depths (Sanku, task 2).
    """
    if not ca or not cb:
        return 0, 0, 0, 0, 1, -1, -1.0
    j, na, nb = max((len(wa & wb) / max(len(wa | wb), 1), na, nb) for na, wa in ca for nb, wb in cb)
    if j < 0.5:
        return 0, 0, 0, 1, 0, -1, j
    d = abs(na - nb)
    return int(d == 0), int(0 < d <= 20), int(d > 20), 0, 0, d, j


def _decoy(qc, sc, ql, sl, country):
    """Venky's decoy_features() on precomputed core-token / legal-form sets:
    (n_decoy_words, n_safe_words, n_typo_extra, n_other_extra, n_missing, legal_form_conflict)."""
    import decoy_rules as dr
    extra, missing = qc - sc, sc - qc
    strong = dr.STRONG_DECOY_WORDS.get(country, set())
    safe = dr.SAFE_EXTRA_WORDS.get(country, set())
    n_d, n_s = len(extra & strong), len(extra & safe)
    rest = extra - strong - safe
    n_t = sum(1 for w in rest if any(dr._similar(w, m) >= 0.6 for m in missing))
    conflict = int(bool(ql) and bool(sl) and not (ql <= sl or sl <= ql))
    return n_d, n_s, n_t, len(rest) - n_t, len(missing), conflict


def _margin(pairs, col):
    """Score minus the best OTHER candidate's score for the same query (top candidate: best minus 2nd best).

    Vectorised: sort by (query, -score), take each query's 1st and 2nd values; queries with one candidate
    use -1 as the missing 2nd score.
    """
    q = pairs["q_idx"].to_numpy()
    v = pairs[col].to_numpy().astype(np.float32)
    order = np.lexsort((-v, q))
    qs, vs = q[order], v[order]
    first = np.r_[True, qs[1:] != qs[:-1]]
    grp = np.cumsum(first) - 1
    top1 = vs[first][grp]
    has2 = np.r_[~first[1:], False] & first  # position is a group's first and the next row is the same group
    second = np.full(first.sum(), -1.0, dtype=np.float32)
    second[grp[has2]] = vs[np.flatnonzero(has2) + 1]
    top2 = second[grp]
    m = np.where(vs >= top1, vs - top2, vs - top1)
    out = np.empty_like(m)
    out[order] = m
    return out


USE_V14 = False  # set by v5_full.py --v14 (adds v14_features columns)


def pair_features(q, s1, pairs):
    """pairs: DataFrame with q_idx, s1_idx (+ blocking scores). Returns feature DataFrame."""
    a = q.iloc[pairs["q_idx"].to_numpy()].reset_index(drop=True)
    b = s1.iloc[pairs["s1_idx"].to_numpy()].reset_index(drop=True)
    f = pd.DataFrame(index=pairs.index)
    for col in [c for c in pairs.columns if c not in ("q_idx", "s1_idx", "label")]:
        f[col] = pairs[col].to_numpy()
    f["name_ratio"] = _pairwise(a["n_name"], b["n_name"], fuzz.ratio)
    f["core_ratio"] = _pairwise(a["c_name"], b["c_name"], fuzz.ratio)
    f["core_partial"] = _pairwise(a["c_name"], b["c_name"], fuzz.partial_ratio)
    f["name_tset"] = _pairwise(a["n_name"], b["n_name"], fuzz.token_set_ratio)
    f["core_jw"] = _pairwise(a["c_name"], b["c_name"], JaroWinkler.similarity)
    f["core_tsort"] = _pairwise(a["c_name"], b["c_name"], fuzz.token_sort_ratio)
    f["addr_tset"] = _pairwise(a["n_addr"], b["n_addr"], fuzz.token_set_ratio)
    f["addr_ratio"] = _pairwise(a["n_addr"], b["n_addr"], fuzz.ratio)
    f["addr_tsort"] = _pairwise(a["n_addr"], b["n_addr"], fuzz.token_sort_ratio)
    f["nums_jacc"] = _pairwise(a["nums"], b["nums"], _jacc)
    pa, pb = a["postcode"].to_numpy(), b["postcode"].to_numpy()
    f["postcode_eq"] = np.where((pa == "") | (pb == ""), -1, (pa == pb).astype(np.int8)).astype(np.int8)
    f["addr_empty"] = ((a["n_addr"] == "").to_numpy() | (b["n_addr"] == "").to_numpy()).astype(np.int8)
    f["len_diff"] = np.abs(a["c_name"].str.len().to_numpy() - b["c_name"].str.len().to_numpy())
    f["q_non_latin"] = a["non_latin"].to_numpy()
    f["q_src3"] = a["entity_id"].str.startswith("S3").to_numpy().astype(np.int8)
    f["sq_ratio"] = _pairwise(a["sq_name"], b["sq_name"], fuzz.ratio)
    f["sq_partial"] = _pairwise(a["sq_name"], b["sq_name"], fuzz.partial_ratio)
    f["is_domain"] = (a["is_domain"].to_numpy() | b["is_domain"].to_numpy()).astype(np.int8)
    dff = np.array([_diff_token_freq(x, y) for x, y in zip(a["c_name"], b["c_name"])], dtype=np.float32).reshape(-1, 2)
    f["diff_tok_maxfreq"], f["diff_tok_minfreq"] = dff[:, 0], dff[:, 1]
    # --- name ambiguity: how many S1 entities / candidates share this core name ---
    f["core_eq"] = (a["c_name"].to_numpy() == b["c_name"].to_numpy()).astype(np.int8)
    if "name_cnt" in b:
        f["s1_name_cnt"] = np.log1p(b["name_cnt"].to_numpy()).astype(np.float32)
    qkey = a["country"] + "|" + a["c_name"]
    f["q_name_cnt"] = np.log1p(np.fromiter((NAME_CNT.get(k, 0) for k in qkey), dtype=np.float32, count=len(a)))
    g = pd.DataFrame({"q": pairs["q_idx"].to_numpy(), "n": pd.factorize(b["c_name"])[0],
                      "addr": f["addr_tset"].to_numpy()})
    grp = g.groupby(["q", "n"])["addr"]
    f["cands_same_name"] = grp.transform("size").to_numpy(np.int16)
    f["addr_rank_in_name"] = grp.rank(method="min", ascending=False).to_numpy(np.float32)
    f["addr_gap_in_name"] = (grp.transform("max") - g["addr"]).to_numpy(np.float32)
    hs = np.array([_house(_dr_street(x), _dr_street(y)) for x, y in zip(a["business_address"], b["business_address"])],
                  dtype=np.float32).reshape(-1, 7)
    for i, c in enumerate(["house_same", "house_small_shift", "house_large_shift", "house_other_street", "house_missing",
                           "house_shift", "street_jacc"]):
        f[c] = hs[:, i]
    ht = np.array([_house_text(_street_raw(x), _street_raw(y)) for x, y in zip(a["business_address"], b["business_address"])],
                  dtype=np.float32).reshape(-1, 3)
    f["house_digit_edit"], f["house_num_contains"], f["house_same_len"] = ht[:, 0], ht[:, 1], ht[:, 2]
    da = a["n_name"].map(deleet)
    db = b["n_name"].map(deleet)
    f["name_ratio_deleet"] = _pairwise(da, db, fuzz.ratio)
    f["name_tsort_deleet"] = _pairwise(da, db, fuzz.token_sort_ratio)
    f["has_leet"] = ((da != a["n_name"]).to_numpy() | (db != b["n_name"]).to_numpy()).astype(np.int8)
    dc = np.array([_decoy(_dr_core(x), _dr_core(y), _dr_legal(x), _dr_legal(y), c)
                   for x, y, c in zip(a["business_name"], b["business_name"], a["country"])], dtype=np.float32).reshape(-1, 6)
    for i, c in enumerate(["n_decoy_words", "n_safe_words", "n_typo_extra", "n_other_extra", "n_missing", "legal_form_conflict"]):
        f[c] = dc[:, i]
    for col in [c for c in ("cos_name_addr", "cos_e5", "p_prune") if c in pairs]:
        f[f"margin_{col}"] = _margin(pairs, col)
    ac = np.array([_containment(x, y) for x, y in zip(a["n_addr"], b["n_addr"])], dtype=np.float32).reshape(-1, 2)
    f["addr_q_in_s1"], f["addr_s1_in_q"] = ac[:, 0], ac[:, 1]
    nc = np.array([_containment(x, y) for x, y in zip(a["c_name"], b["c_name"])], dtype=np.float32).reshape(-1, 2)
    f["name_q_in_s1"], f["name_s1_in_q"] = nc[:, 0], nc[:, 1]
    f["n_diff_tok"] = np.fromiter((len(set(x.split()) ^ set(y.split())) for x, y in zip(a["c_name"], b["c_name"])),
                                  dtype=np.int16, count=len(a))
    if USE_V14:
        for c, v in v14_features(a, b).items():
            f[c] = v.to_numpy()
    return f


# ---------------- v14 features (from Sanku's error review, task 3) ----------------
ORDINALS = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th",
            "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th"}


def addr_norm_v14(n_addr, country):
    """Normalised address + ordinals as numbers (fifth -> 5th) + abbreviations learned from train pairs
    (gj -> gujarat, tx -> texas; addr_learned.py). Lets a city+state stub match the full address."""
    from addr_learned import ADDR_LEARNED
    return " ".join(ADDR_LEARNED.get((country, t), ORDINALS.get(t, t)) for t in n_addr.split())


def idf_jaccard(x, y):
    """Rarity-weighted Jaccard of core-name tokens: losing a distinctive word costs more than losing a generic one
    ("gurgaon academy" vs "gurgaon services"). Weights = log(1 + N / (1 + document frequency)) from VOCAB."""
    sx, sy = set(x.split()), set(y.split())
    if not sx or not sy:
        return -1.0
    n = max(VOCAB.get("__N__", 1), 1)
    w = lambda t: np.log1p(n / (1 + VOCAB.get(t, 0)))
    inter = sum(w(t) for t in sx & sy)
    union = sum(w(t) for t in sx | sy)
    return inter / union if union else -1.0


def v14_features(a, b):
    """New pair features (a = S2/S3 rows, b = S1 rows, aligned): learned-abbreviation address coverage both ways,
    ordinal/abbreviation-normalised address similarity, rarity-weighted name Jaccard, duplicated-word count."""
    qa = [addr_norm_v14(x, c) for x, c in zip(a["n_addr"], a["country"])]
    sa = [addr_norm_v14(x, c) for x, c in zip(b["n_addr"], b["country"])]
    cov = np.array([_containment(x, y) for x, y in zip(qa, sa)], dtype=np.float32).reshape(-1, 2)
    out = {"addr_cov_q_in_s1": cov[:, 0], "addr_cov_s1_in_q": cov[:, 1],
           "addr_tset_v14": _pairwise(pd.Series(qa), pd.Series(sa), fuzz.token_set_ratio),
           "name_idf_jacc": np.array([idf_jaccard(x, y) for x, y in zip(a["c_name"], b["c_name"])], dtype=np.float32),
           "dup_words_diff": np.array([(len(x.split()) - len(set(x.split()))) - (len(y.split()) - len(set(y.split())))
                                       for x, y in zip(a["n_name"], b["n_name"])], dtype=np.float32)}
    return pd.DataFrame(out)
