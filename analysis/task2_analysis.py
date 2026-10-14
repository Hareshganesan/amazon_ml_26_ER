"""Task 2: name sharing in US/India, and how true matches get corrupted.

Part A -- how ambiguous are core names?
  * share of S1 entities whose core name is shared with another S1 entity of the same country
  * group-size distribution and the top-20 most-shared core names, per country and split
  * how ambiguous real matches are: for true train pairs whose S2/S3 core name equals its S1 core
    name, how many OTHER S1 entities carry that same core name

Part B -- how does the S2/S3 record differ from its true S1 record?
  * a multi-label classification of name corruptions and address corruptions on sampled true pairs
  * a review file so a human can confirm the labels on a subsample

Uses only the provided dataset. Normalisation comes from the team's features.py, so the numbers
describe the pipeline's actual view of the text.

Kaggle usage (CPU is enough)::

    !python task2_analysis.py --out /kaggle/working/task2_out
"""
import argparse
import os
import random
import re
from collections import Counter, defaultdict
from glob import glob

import pandas as pd

from features import ADDR, LEGAL, ascii_fold, core_name, norm_addr, norm_name

try:
    from rapidfuzz.distance import Levenshtein
    _edit = Levenshtein.distance
except ImportError:  # keep the typo label working without rapidfuzz (slower, same result)
    def _edit(x, y):
        """Levenshtein distance; stdlib fallback so labels do not change when rapidfuzz is absent."""
        prev = list(range(len(y) + 1))
        for i, cx in enumerate(x, 1):
            cur = [i]
            for j, cy in enumerate(y, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cx != cy)))
            prev = cur
        return prev[-1]


def _is_abbrev(short, long_):
    """True when `short` looks like an abbreviation of `long_` (prefix or in-order subsequence).

    Catches ctr/center, mgmt/management, intl/international, which a prefix test alone misses.
    """
    if len(short) >= len(long_) or len(short) < 2:
        return False
    if long_.startswith(short):
        return True
    it = iter(long_)
    return short[0] == long_[0] and all(c in it for c in short)

COLS = ["entity_id", "business_name", "business_address", "country"]
# Junk wrappers are markers like "<<", "--", "***", "|" -- NOT an ordinary trailing "." or ","
# (that is plain punctuation and belongs to the case-punctuation label).
_junk = re.compile(r"^\s*[<>*_|+]+|[<>*_|+]+\s*$|^\s*-{2,}|-{2,}\s*$")
_tld = re.compile(r"\.(?:com|fr|in|net|org|co\.in|eu|biz|info)\b", re.I)
_num = re.compile(r"\d+")
_alnum = re.compile(r"[^a-z0-9]+")
LANDMARKS = {"near", "nr", "opposite", "opp", "behind", "beside", "next", "above", "below"}
BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 20, "6-20"), (21, 100, "21-100"),
           (101, 10 ** 9, ">100")]


def find_dataset(explicit=None):
    """Locate the dataset root holding train/ and test/, on Kaggle or locally."""
    if explicit:
        return explicit
    hits = glob("/kaggle/input/**/dataset/train", recursive=True) or \
        glob("/kaggle/input/**/train", recursive=True)
    if not hits:
        raise SystemExit("dataset not found; pass --data <path containing train/ and test/>")
    return os.path.dirname(hits[0])


def read_source(path, usecols=COLS, chunksize=500_000, countries=None):
    """Stream one source TSV, optionally keeping only certain countries."""
    parts = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             usecols=usecols, chunksize=chunksize):
        parts.append(chunk if countries is None else chunk[chunk["country"].isin(countries)])
    return pd.concat(parts, ignore_index=True)


# --- Part A ---------------------------------------------------------------------------------------


def bucket(n):
    """Label the group-size bucket a count falls into."""
    for lo, hi, name in BUCKETS:
        if lo <= n <= hi:
            return name
    return ">100"


def name_sharing(s1, split, out):
    """Core-name sharing per country; writes the top-20 lists and returns summary rows."""
    rows, tops = [], []
    for country, grp in s1.groupby("country"):
        cores = Counter(grp["c_name"])
        total = len(grp)
        empty = cores.get("", 0)
        shared = sum(c for k, c in cores.items() if c > 1 and k != "")
        sizes = Counter()
        for k, c in cores.items():
            if k != "":
                sizes[bucket(c)] += c        # entities, not groups
        row = {"split": split, "country": country, "entities": total, "empty_core": empty,
               "shared_entities": shared, "shared_pct": round(100.0 * shared / max(total - empty, 1), 2)}
        row.update({"size_" + b[2]: sizes.get(b[2], 0) for b in BUCKETS})
        rows.append(row)
        for core, c in cores.most_common(25):
            if core:
                tops.append({"split": split, "country": country, "core_name": core, "entities": c})
                if sum(1 for t in tops if t["split"] == split and t["country"] == country) >= 20:
                    break
    pd.DataFrame(tops).to_csv(os.path.join(out, "partA_top_shared_%s.tsv" % split),
                              sep="\t", index=False)
    return rows


def match_ambiguity(pairs, s1, others, out, sample=200_000, seed=0):
    """For true pairs, how many OTHER S1 entities share the core name the S2/S3 record points at."""
    by_country = {c: Counter(g["c_name"]) for c, g in s1.groupby("country")}
    s1_core = dict(zip(s1["entity_id"], s1["c_name"]))
    s1_country = dict(zip(s1["entity_id"], s1["country"]))
    other_core = dict(zip(others["entity_id"], others["c_name"]))

    use = pairs.sample(min(sample, len(pairs)), random_state=seed)
    rows = []
    for s1_id, other_id in zip(use["source1_entity_id"], use["matched_id"]):
        c1, oc = s1_core.get(s1_id), other_core.get(other_id)
        if c1 is None or oc is None:
            continue
        country = s1_country[s1_id]
        rivals = by_country[country].get(oc, 0) - (1 if oc == c1 else 0)
        rows.append({"country": country, "core_equal": int(oc == c1), "rival_s1_entities": rivals})
    amb = pd.DataFrame(rows)
    amb.to_csv(os.path.join(out, "partA_match_ambiguity_raw.tsv"), sep="\t", index=False)

    summary = []
    for country, g in amb.groupby("country"):
        eq = g[g.core_equal == 1]
        summary.append({
            "country": country,
            "pairs_sampled": len(g),
            "core_name_equal_pct": round(100.0 * len(eq) / max(len(g), 1), 2),
            "of_those_unambiguous_pct": round(100.0 * (eq.rival_s1_entities == 0).mean(), 2) if len(eq) else 0,
            "median_rivals": int(eq.rival_s1_entities.median()) if len(eq) else 0,
            "mean_rivals": round(float(eq.rival_s1_entities.mean()), 2) if len(eq) else 0,
            "max_rivals": int(eq.rival_s1_entities.max()) if len(eq) else 0,
        })
    return summary


# --- Part B ---------------------------------------------------------------------------------------


def _toks(s):
    """Normalised name tokens."""
    return [t for t in norm_name(s).split() if t]


def _squash(s):
    """Bare alphanumerics with any TLD removed, so a domain can be compared with a spelled name."""
    return _alnum.sub("", _tld.sub("", ascii_fold(s)))


def classify_name(a, b):
    """Multi-label classification of how name b differs from name a."""
    labels = []
    if a == b:
        return ["identical"]
    na, nb = norm_name(a), norm_name(b)
    if na == nb:
        labels.append("case-punctuation only")
    if _junk.search(a) or _junk.search(b):
        labels.append("junk prefix-suffix")
    if _tld.search(a) or _tld.search(b):
        labels.append("website-domain")
    if any(ord(c) > 0x24F for c in a + b):
        labels.append("other script")

    # A domain or a script change already explains the whole difference when the squashed forms
    # contain one another; adding word-level labels on top would just be noise.
    qa, qb = _squash(a), _squash(b)
    if ("website-domain" in labels or "other script" in labels) and qa and qb \
            and (qa in qb or qb in qa):
        return labels
    # a name written wholly in another script leaves no Latin tokens to diff against
    if "other script" in labels and not (qa and qb):
        return labels

    ta, tb = _toks(a), _toks(b)
    sa, sb = set(ta), set(tb)
    if sa == sb and ta != tb:
        labels.append("word order swapped")
    if (sa ^ sb) and (sa ^ sb) <= LEGAL:
        labels.append("legal suffix added-dropped-moved")

    extra_a, extra_b = sa - sb - LEGAL, sb - sa - LEGAL
    # a token pair that is a near-miss is a typo or an abbreviation, not a drop plus an add
    typo = abbrev = False
    for x in sorted(extra_a):
        for y in sorted(extra_b):
            if x not in extra_a or y not in extra_b:
                continue
            if 0 < _edit(x, y) <= 2 and max(len(x), len(y)) >= 4:
                typo = True
            elif _is_abbrev(x, y) or _is_abbrev(y, x):
                abbrev = True
            else:
                continue
            extra_a.discard(x)
            extra_b.discard(y)
            break
    if typo:
        labels.append("typo (1-2 chars)")
    if abbrev:
        labels.append("abbreviation")
    if extra_a:
        labels.append("word dropped")
    if extra_b:
        labels.append("word added")

    ca, cb = set(core_name(a).split()), set(core_name(b).split())
    if ca and cb and not (ca & cb) and "other script" not in labels:
        # nothing shared once legal forms are gone, and neither squashed form contains the other
        if not (qa and qb and (qa in qb or qb in qa)):
            labels.append("completely different name")
    return labels or ["other"]


def _components(s):
    """Comma-separated address components, each normalised with its tokens sorted."""
    return [" ".join(sorted(norm_addr(c).split())) for c in s.split(",") if norm_addr(c).strip()]


def classify_address(a, b):
    """Multi-label classification of how address b differs from address a.

    Compared on the RAW text, not just the pipeline's normalised view: "Ave" vs "Avenue" is an
    abbreviation worth counting even though norm_addr() maps both to "avenue".
    """
    if not a.strip() or not b.strip():
        return ["component missing"]
    ra = _alnum.sub(" ", ascii_fold(a)).split()
    rb = _alnum.sub(" ", ascii_fold(b)).split()
    if ra == rb:
        return ["identical"]

    labels = []
    na, nb = norm_addr(a), norm_addr(b)
    # An abbreviation on one side whose expansion appears on the other ("Rd" vs "Road").
    # Requiring the whole normalised address to match would only catch pairs whose ONLY
    # difference is the abbreviation, which is why India scored 0% before this.
    sra, srb = set(ra), set(rb)
    if any(t in ADDR and ADDR[t] in srb and t not in srb for t in sra) or        any(t in ADDR and ADDR[t] in sra and t not in sra for t in srb):
        labels.append("abbreviation")

    ca, cb = _components(a), _components(b)
    if sorted(ca) == sorted(cb) and ca != cb:
        labels.append("component reordered")

    # Token-level, not whole-component: "Shop No.102, Citi Mall" vs "102, CITI MALL" drops tokens
    # without leaving any component string intact, so a set-of-components test misses it.
    ta, tb = set(na.split()), set(nb.split())
    jacc = len(ta & tb) / len(ta | tb) if (ta | tb) else 1.0
    if tb < ta:                      # the S2/S3 side lacks tokens the S1 side has
        labels.append("component missing")

    numa, numb = _num.findall(a), _num.findall(b)
    if numa and numb and numa[0] != numb[0]:
        labels.append("house number differs")
    if (set(rb) & LANDMARKS) - (set(ra) & LANDMARKS):
        labels.append("landmark added")

    if jacc < 0.5:
        labels.append("different")
        if "house number differs" in labels:
            labels.remove("house number differs")   # implied by a wholly different address
    elif not labels:
        # substantially the same address, but no single category describes the edit
        labels.append("component missing" if ta - tb else "other")
    return labels


def sample_pairs(pairs, s1_country, n_per_country=300, seed=0):
    """300 true pairs per country: 100 S2, 100 S3, the rest mixed."""
    rnd = random.Random(seed)
    pairs = pairs.copy()
    pairs["country"] = pairs["source1_entity_id"].map(s1_country)
    pairs = pairs[pairs["country"].notna()]
    pairs["src"] = pairs["matched_id"].str[:2]
    quota = n_per_country // 3          # 300 -> 100 S2, 100 S3, 100 mixed
    out = []
    for country, g in pairs.groupby("country"):
        idx = []
        for src in ("S2", "S3"):
            pool = g.index[g["src"] == src].tolist()
            idx += rnd.sample(pool, min(quota, len(pool)))
        rest = [i for i in g.index if i not in set(idx)]
        idx += rnd.sample(rest, max(0, min(n_per_country - len(idx), len(rest))))
        out.append(g.loc[idx])
    return pd.concat(out, ignore_index=True)


def label_table(df, col, out, name):
    """Percentage of pairs carrying each label, per country."""
    rows = []
    for country, g in df.groupby("country"):
        counts = Counter(l for labels in g[col] for l in labels.split("|"))
        for label, c in counts.most_common():
            rows.append({"country": country, "label": label, "pairs": c,
                         "pct": round(100.0 * c / len(g), 2)})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "partB_%s_labels.tsv" % name), sep="\t", index=False)
    return table


def examples(df, col, a_col, b_col, per_label=3):
    """Up to `per_label` example pairs for each label."""
    seen, rows = defaultdict(int), []
    for _, r in df.iterrows():
        for label in r[col].split("|"):
            key = (r["country"], label)
            if seen[key] < per_label:
                seen[key] += 1
                rows.append({"country": r["country"], "label": label,
                             "s1": r[a_col], "match": r[b_col], "match_id": r["matched_id"]})
    return pd.DataFrame(rows)


def main():
    """Task 2: name-sharing study and corruption taxonomy on sampled train pairs; writes tables and a hand-check file."""
    ap = argparse.ArgumentParser(description="Task 2: name sharing and corruption taxonomy")
    ap.add_argument("--data", help="dataset root containing train/ and test/")
    ap.add_argument("--out", default="task2_out")
    ap.add_argument("--countries", default="US,India")
    ap.add_argument("--n_pairs", type=int, default=300)
    ap.add_argument("--amb_sample", type=int, default=200_000)
    ap.add_argument("--review", type=int, default=50, help="rows to put in the hand-check file")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    countries = args.countries.split(",")
    data = find_dataset(args.data)

    # ---- Part A -----------------------------------------------------------------------------
    summary = []
    train_s1 = None
    for split in ("train", "test"):
        path = os.path.join(data, split, "%s_source1.tsv" % split)
        s1 = read_source(path, countries=countries)
        s1["c_name"] = s1["business_name"].map(core_name)
        print("%s S1: %s rows (%s)" % (split, format(len(s1), ","), ", ".join(countries)))
        summary += name_sharing(s1, split, args.out)
        if split == "train":
            train_s1 = s1
    part_a = pd.DataFrame(summary)
    part_a.to_csv(os.path.join(args.out, "partA_sharing.tsv"), sep="\t", index=False)
    print(part_a.to_string(index=False))

    # ---- true pairs -------------------------------------------------------------------------
    gt = pd.read_csv(os.path.join(data, "train", "train_ground_truth.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    gt = gt[gt["matched_entity_ids"] != ""]
    # reset_index is required: explode repeats the S1 row's index label for each matched id, and a
    # later .loc[labels] would then pull back every pair of that entity instead of the sampled one.
    pairs = (gt.assign(matched_id=gt["matched_entity_ids"].str.split(","))
             .explode("matched_id").reset_index(drop=True))
    pairs["matched_id"] = pairs["matched_id"].str.strip()
    s1_country = dict(zip(train_s1["entity_id"], train_s1["country"]))
    pairs = pairs[pairs["source1_entity_id"].isin(s1_country)]
    print("true pairs in %s: %s" % ("/".join(countries), format(len(pairs), ",")))

    # Only the rows we actually look at are loaded: the ambiguity sample plus the Part B sample.
    # Loading every matched S2/S3 row would be ~7.6M rows for no benefit.
    amb_pairs = pairs.sample(min(args.amb_sample, len(pairs)), random_state=0)
    sample = sample_pairs(pairs, s1_country, args.n_pairs)
    wanted = set(amb_pairs["matched_id"]) | set(sample["matched_id"])
    others = []
    for i in (2, 3):
        path = os.path.join(data, "train", "train_source%d.tsv" % i)
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                 usecols=COLS, chunksize=500_000):
            others.append(chunk[chunk["entity_id"].isin(wanted)])
    others = pd.concat(others, ignore_index=True)
    others["c_name"] = others["business_name"].map(core_name)
    print("S2/S3 rows loaded: %s (of %s true pairs)"
          % (format(len(others), ","), format(len(pairs), ",")))

    amb = match_ambiguity(amb_pairs, train_s1, others, args.out, len(amb_pairs))
    amb_df = pd.DataFrame(amb)
    amb_df.to_csv(os.path.join(args.out, "partA_match_ambiguity.tsv"), sep="\t", index=False)
    print(amb_df.to_string(index=False))

    # ---- Part B -----------------------------------------------------------------------------
    s1_by_id = train_s1.set_index("entity_id")
    o_by_id = others.set_index("entity_id")
    rows = []
    for _, r in sample.iterrows():
        if r["matched_id"] not in o_by_id.index:
            continue
        a, b = s1_by_id.loc[r["source1_entity_id"]], o_by_id.loc[r["matched_id"]]
        rows.append({
            "country": r["country"], "src": r["src"],
            "s1_id": r["source1_entity_id"], "matched_id": r["matched_id"],
            "s1_name": a["business_name"], "match_name": b["business_name"],
            "s1_addr": a["business_address"], "match_addr": b["business_address"],
            "name_labels": "|".join(classify_name(a["business_name"], b["business_name"])),
            "addr_labels": "|".join(classify_address(a["business_address"], b["business_address"])),
        })
    part_b = pd.DataFrame(rows)
    part_b.to_csv(os.path.join(args.out, "partB_pairs_labelled.tsv"), sep="\t", index=False)
    print("labelled %s true pairs" % format(len(part_b), ","))

    name_tbl = label_table(part_b, "name_labels", args.out, "name")
    addr_tbl = label_table(part_b, "addr_labels", args.out, "addr")
    print(name_tbl.to_string(index=False))
    print(addr_tbl.to_string(index=False))

    examples(part_b, "name_labels", "s1_name", "match_name").to_csv(
        os.path.join(args.out, "partB_name_examples.tsv"), sep="\t", index=False)
    examples(part_b, "addr_labels", "s1_addr", "match_addr").to_csv(
        os.path.join(args.out, "partB_addr_examples.tsv"), sep="\t", index=False)

    review = part_b.sample(min(args.review, len(part_b)), random_state=1).copy()
    review["name_labels_correct"] = ""
    review["addr_labels_correct"] = ""
    review[["country", "s1_name", "match_name", "name_labels", "name_labels_correct",
            "s1_addr", "match_addr", "addr_labels", "addr_labels_correct",
            "s1_id", "matched_id"]].to_csv(
        os.path.join(args.out, "REVIEW_task2.csv"), index=False, encoding="utf-8-sig")
    print("wrote %s/REVIEW_task2.csv (%d rows to hand-check)" % (args.out, len(review)))


if __name__ == "__main__":
    main()
