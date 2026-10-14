"""Blocking probe: how many true matches does TF-IDF nearest-neighbour blocking keep in the top K?

Samples N S1 entities from train and all their true S2/S3 matches. For every matched S2/S3
record we search the S1 sample (same country only) and report recall@K = share of true pairs whose S1 entity is in the top K.

Run on Kaggle CPU (~10 GB RAM):
    python blocking_probe.py --data /kaggle/input/amz-ml-2026/student_resource/dataset --n 50000
Note: a small S1 sample makes search easier than the real 1.7M index, so treat recall as optimistic
and compare variants against each other at the same --n.
"""
import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def norm(s):
    """Lowercase + strip accents + keep alphanumerics (non-Latin scripts survive NFKC-lowercase)."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^\w]+", " ", s).strip()


def read(p):
    """Read a competition TSV with every column as a string and empty cells kept as ''."""
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)


def topk_recall(s1, q, field, ks, chunk=1000):
    """TF-IDF char n-grams on `field`; per-country brute-force cosine top-K; return recall per K."""
    hits = {k: 0 for k in ks}
    total = 0
    for country, q_c in q.groupby("country"):
        s1_c = s1[s1["country"] == country].reset_index(drop=True)
        if s1_c.empty:
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        X = vec.fit_transform(s1_c[field]).T.tocsr()
        Q = vec.transform(q_c[field])
        true_idx = q_c["s1_id"].map(pd.Series(s1_c.index, index=s1_c["entity_id"])).to_numpy()
        kmax = min(max(ks), len(s1_c))
        for i in range(0, Q.shape[0], chunk):
            sims = (Q[i:i + chunk] @ X).toarray()
            top = np.argpartition(-sims, kmax - 1, axis=1)[:, :kmax]
            order = np.take_along_axis(sims, top, 1).argsort(1)[:, ::-1]
            top = np.take_along_axis(top, order, 1)
            t = true_idx[i:i + chunk]
            ok = ~np.isnan(t)
            for k in ks:
                hits[k] += int((top[ok, :k] == t[ok, None]).any(1).sum())
            total += int(ok.sum())
        print(f"  {field} {country}: S1={len(s1_c)} queries={len(q_c)}", flush=True)
    return {k: hits[k] / max(total, 1) for k in ks}


def main(data, n, seed):
    """Sample n ground-truth S1 entities, block their S2/S3 records against them per view and print recall@K."""
    d = Path(data) / "train"
    gt = read(d / "train_ground_truth.tsv").sample(n=n, random_state=seed)
    pair = {b: a for a, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"])
            for b in ids.split(",") if b}
    s1 = read(d / "train_source1.tsv")
    s1 = s1[s1["entity_id"].isin(set(gt["source1_entity_id"]))]

    parts = []
    for s in (2, 3):
        df = read(d / f"train_source{s}.tsv")
        df["s1_id"] = df["entity_id"].map(pair)
        # recall only needs the true matches of our sampled S1 entities (decoys never count for recall)
        parts.append(df[df["s1_id"].notna()])
        del df
    q = pd.concat(parts, ignore_index=True)
    print(f"S1 sample={len(s1)}  queries={len(q)}  (true pairs={q['s1_id'].notna().sum()})")

    for frame in (s1, q):
        frame["name"] = frame["business_name"].map(norm)
        frame["name_addr"] = frame["name"] + " " + frame["business_address"].map(norm)
    ks = [1, 3, 5, 10, 20, 50]
    for field in ("name", "name_addr"):
        r = topk_recall(s1, q, field, ks)
        print(f"{field:10s} " + "  ".join(f"R@{k}={v:.4f}" for k, v in r.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.data, a.n, a.seed)
