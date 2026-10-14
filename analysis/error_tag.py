"""Tag v5 validation errors with candidate explanations, for the Task 3 hand-read.

Produces a first pass only: every row still gets read by hand, but grouping 60 rows by a computed
signal makes the recurring patterns visible instead of hiding them in prose.

    python error_tag.py --n 60
"""
import argparse
import os
import re
import sys
import unicodedata
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_alnum = re.compile(r"[^a-z0-9]+")
_num = re.compile(r"\d+")
_word = re.compile(r"[^\w]+")
LOOKALIKE = str.maketrans({"0": "o", "1": "l", "5": "s", "3": "e", "8": "b", "4": "a"})
LEGAL = {"inc", "incorporated", "llc", "llp", "ltd", "limited", "pvt", "private", "corp",
         "corporation", "co", "company", "plc", "lp", "pllc", "pc", "and", "the", "of",
         "partners", "group", "holdings"}
US_STATES = {"al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in",
             "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv",
             "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn",
             "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy"}


def fold(s):
    """Lowercase, strip accents, keep non-Latin as-is."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def toks(s):
    """Split ASCII-folded text into lowercase word tokens."""
    return [t for t in _word.split(fold(s)) if t]


def core(s):
    """Name tokens without legal-form words."""
    return {t for t in toks(s) if t not in LEGAL}


def has_indic(s):
    """True if the text contains any Indic-script character (U+0900-U+0D7F)."""
    return any(0x0900 <= ord(c) <= 0x0D7F for c in s)


def house(addr):
    """First number in the address, as a string."""
    m = _num.search(addr)
    return m.group(0) if m else ""


def street_tokens(addr):
    """Address tokens without numbers and US state codes."""
    return {t for t in toks(addr) if not t.isdigit() and t not in US_STATES}


def tag(rec_name, rec_addr, s1_name, s1_addr):
    """Candidate explanations for why this pair was hard. Multi-label."""
    tags = []
    if not s1_name and not s1_addr:
        return ["no counterpart row"]
    rn, sn = core(rec_name), core(s1_name)
    shared = rn & sn

    # --- name side ---
    if has_indic(rec_name) or has_indic(s1_name):
        both = has_indic(rec_name) and has_indic(s1_name)
        latin_part = [t for t in toks(rec_name) if not has_indic(t)]
        tags.append("mixed-script name" if (not both and latin_part) else "non-Latin script")
    if re.search(r"\.(com|in|net|org)\b", rec_name + s1_name, re.I):
        tags.append("website in name")
    if any(ch.isdigit() for ch in rec_name) and \
            _alnum.sub("", fold(rec_name)).translate(LOOKALIKE) == _alnum.sub("", fold(s1_name)).translate(LOOKALIKE):
        tags.append("look-alike digit in name")
    if not shared and rn and sn:
        tags.append("no shared name word (trade name)")
    elif shared and (rn - sn) and not (sn - rn):
        tags.append("record name has extra word")
    elif shared and (sn - rn) and not (rn - sn):
        tags.append("record name missing a word")
    elif shared and (rn - sn) and (sn - rn):
        tags.append("name word substituted")

    # --- address side ---
    ra, sa = rec_addr.strip(), s1_addr.strip()
    if not ra or not sa:
        tags.append("empty address")
    else:
        hr, hs = house(ra), house(sa)
        st_r, st_s = street_tokens(ra), street_tokens(sa)
        j = len(st_r & st_s) / max(len(st_r | st_s), 1)
        if hr and hs and hr != hs:
            if hr.lstrip("0") == hs.lstrip("0") or hr in hs or hs in hr:
                tags.append("house number formatting")
            else:
                tags.append("house number differs")
        if j >= 0.6:
            tags.append("same street")
        elif j < 0.25:
            tags.append("address unrelated")
        if len(toks(ra)) <= 4 or len(toks(sa)) <= 4:
            tags.append("address truncated")
        if (st_r - st_s) and not (st_s - st_r):
            tags.append("record addr has extra detail")
        elif (st_s - st_r) and not (st_r - st_s):
            tags.append("record addr missing detail")
    return tags or ["no obvious signal"]


def main():
    """Sample n errors per type from errors.tsv, tag each with the error patterns, write the tagged sample."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", default="errors.tsv")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--types", default="FN_threshold,FP_decoy,FN_ranked_below")
    ap.add_argument("--out", default="error_sample_tagged.tsv")
    args = ap.parse_args()

    e = pd.read_csv(args.errors, sep="\t", dtype=str, keep_default_na=False)
    rows = []
    for t in args.types.split(","):
        d = e[(e.error_type == t) & (e.record_addr.str.strip() != "")]
        # FP rows compare against the predicted S1; FN rows against the true S1
        s1n, s1a = ("pred_s1_name", "pred_s1_addr") if t.startswith("FP") else ("true_s1_name", "true_s1_addr")
        d = d[d[s1n].str.strip() != ""]
        take = d.sample(min(args.n, len(d)), random_state=0)
        print("%-16s %d available, %d sampled" % (t, len(d), len(take)))
        for _, r in take.iterrows():
            tg = tag(r.record_name, r.record_addr, r[s1n], r[s1a])
            rows.append({"error_type": t, "country": r.country, "p_best": r.p_best, "p_true": r.p_true,
                         "record_name": r.record_name, "record_addr": r.record_addr,
                         "s1_name": r[s1n], "s1_addr": r[s1a], "tags": "|".join(tg),
                         "why": "", "recoverable": "", "signal": ""})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, sep="\t", index=False)
    print("\nwrote %s (%d rows)\n" % (args.out, len(out)))

    for t, g in out.groupby("error_type"):
        print("=== %s (n=%d) ===" % (t, len(g)))
        c = Counter(x for s in g.tags for x in s.split("|"))
        for k, v in c.most_common(12):
            print("   %-34s %3d  (%4.0f%%)" % (k, v, 100 * v / len(g)))
        print()


if __name__ == "__main__":
    main()
