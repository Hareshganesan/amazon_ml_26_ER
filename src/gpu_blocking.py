"""Scalable candidate generation: TF-IDF char n-grams -> SVD (dense 256-d) -> GPU brute-force top-K.

Exact sparse search costs ~25 CPU-hours per 1M queries at full index size, so each record is embedded
as an L2-normalised SVD projection of its TF-IDF vector and searched with a dense matrix product on
GPU (torch). The exact sparse TF-IDF cosine is then recomputed for each candidate pair as a feature.
"""
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    torch, DEVICE = None, "cpu"

VIEWS = {  # field -> (analyzer, ngram_range)
    "c_name": ("char_wb", (2, 4)),
    "name_addr": ("char_wb", (3, 3)),
}
EXTRA_VIEWS = {  # optional views (v7): address only, for trade-name records whose name shares nothing with S1
    "n_addr": ("char_wb", (3, 3)),
}


class View:
    """TF-IDF + SVD encoder for one text field, fitted once on S1 text of all countries."""

    def __init__(self, field, analyzer, ngrams, dim=256, fit_rows=600_000, seed=0):
        """One retrieval view: TF-IDF on one text field (analyzer, n-gram range) projected to `dim` dimensions by SVD."""
        self.field, self.dim, self.fit_rows, self.seed = field, dim, fit_rows, seed
        self.vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, min_df=3, max_features=300_000,
                                   sublinear_tf=True, dtype=np.float32)

    def fit(self, texts):
        """Fit the TF-IDF vocabulary and the TruncatedSVD projection on a sample of the texts."""
        texts = pd.Series(texts)
        sample = texts.sample(n=min(self.fit_rows, len(texts)), random_state=self.seed)
        X = self.vec.fit_transform(sample)
        self.svd = TruncatedSVD(n_components=min(self.dim, X.shape[1] - 1), random_state=self.seed).fit(X)
        return self

    def sparse(self, texts):
        """Sparse TF-IDF vectors of the texts (used for exact-cosine re-ranking)."""
        return self.vec.transform(texts)

    def dense(self, X):
        """Project TF-IDF vectors with the SVD and L2-normalise; fp16 output for the GPU inner-product search."""
        E = (X @ self.svd.components_.T).astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
        return E.astype(np.float16)


def dense_topk(Eq, Es, k, chunk=4096):
    """Top-k inner products of Eq rows against Es rows (GPU if available). Returns (idx, sim)."""
    k = min(k, len(Es))
    if torch is None:
        s = Eq @ Es.T
        idx = np.argsort(-s, 1)[:, :k]
        return idx, np.take_along_axis(s, idx, 1)
    dt = torch.float16 if DEVICE == "cuda" else torch.float32
    S = torch.from_numpy(Es).to(DEVICE, dt)
    out_i, out_s = [], []
    step = max(256, min(chunk, int(6e8 // max(len(Es), 1))))  # cap the score block at ~1.2 GB fp16
    for i in range(0, len(Eq), step):
        q = torch.from_numpy(Eq[i:i + step]).to(DEVICE, dt)
        v, ix = torch.topk(q @ S.T, k, dim=1)
        out_i.append(ix.cpu().numpy())
        out_s.append(v.float().cpu().numpy())
    return np.vstack(out_i), np.vstack(out_s)


def e5_encode(model, texts, bs=512):
    """L2-normalised multilingual-e5 embeddings (fp16). e5 wants a 'query: ' prefix for symmetric matching."""
    E = model.encode(["query: " + t for t in texts], batch_size=bs, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
    return E.astype(np.float16)


def rowwise_dot(A, B, ia, ib, chunk=2_000_000):
    """Dot product of dense rows A[ia[j]] and B[ib[j]] (cosine for normalised embeddings)."""
    out = np.empty(len(ia), dtype=np.float32)
    for i in range(0, len(ia), chunk):
        out[i:i + chunk] = (A[ia[i:i + chunk]].astype(np.float32) * B[ib[i:i + chunk]].astype(np.float32)).sum(1)
    return out


def rowwise_cos(A, B, ia, ib, chunk=2_000_000):
    """Exact cosine between sparse rows A[ia[j]] and B[ib[j]] (rows already L2-normalised)."""
    out = np.empty(len(ia), dtype=np.float32)
    for i in range(0, len(ia), chunk):
        out[i:i + chunk] = np.asarray(A[ia[i:i + chunk]].multiply(B[ib[i:i + chunk]]).sum(1)).ravel()
    return out


def candidates(views, s1, q, k, s1_cache=None, verbose=True, fetch=None, e5=None, block_fields=None):
    """Per-country candidate pairs [q_idx, s1_idx, cos_*, rank_*, gap_*, n_cand] (positional indexes).

    fetch: if set, take the top-`fetch` per view by (approximate) SVD similarity, re-score them with
    the exact sparse TF-IDF cosine and keep the best k per view (recovers most of the SVD recall loss).
    e5: optional (E_s1, E_q) normalised embeddings of all s1 rows / all q rows; adds an e5 top-k view.
    block_fields: TF-IDF views used for candidate generation (default all); the others only give features.

    s1_cache: optional dict field -> (sparse, dense) for all of s1, to avoid re-encoding the index.
    """
    if s1_cache is None:
        s1_cache = {}
        for f, v in views.items():
            Xs = v.sparse(s1[f])
            s1_cache[f] = (Xs, v.dense(Xs))
    q_enc = {}
    for f, v in views.items():
        Xq = v.sparse(q[f])
        q_enc[f] = (Xq, v.dense(Xq))
    parts = []
    s1_country = s1["country"].to_numpy()
    for country, q_c in q.groupby("country", sort=False):
        s1_pos = np.flatnonzero(s1_country == country)
        if len(s1_pos) == 0:
            continue
        qpos = q_c.index.to_numpy()
        per = []
        if e5 is not None:
            t = time.time()
            idx, _ = dense_topk(e5[1][qpos], e5[0][s1_pos], k)
            kk = idx.shape[1]
            per.append(pd.DataFrame({"q_idx": np.repeat(qpos, kk), "s1_idx": s1_pos[idx.ravel()],
                                     "rank_e5": np.tile(np.arange(kk, dtype=np.int16), len(qpos))}))
            if verbose:
                print(f"  gpu-block {country} e5: S1={len(s1_pos)} q={len(qpos)} {time.time() - t:.0f}s", flush=True)
        for spec in (block_fields or list(views)):
            f, kf = (spec.split(":")[0], int(spec.split(":")[1])) if ":" in spec else (spec, k)
            t = time.time()
            idx, _ = dense_topk(q_enc[f][1][qpos], s1_cache[f][1][s1_pos], max(fetch or kf, kf))
            cand = s1_pos[idx]  # (nq, n_fetched) global S1 positions
            if fetch and cand.shape[1] > kf:
                exact = rowwise_cos(q_enc[f][0], s1_cache[f][0], np.repeat(qpos, cand.shape[1]), cand.ravel())
                exact = exact.reshape(cand.shape)
                top = np.argsort(-exact, axis=1)[:, :kf]  # exact-cosine re-rank
                cand = np.take_along_axis(cand, top, 1)
            kk = cand.shape[1]
            per.append(pd.DataFrame({"q_idx": np.repeat(qpos, kk), "s1_idx": cand.ravel(),
                                     f"rank_{f}": np.tile(np.arange(kk, dtype=np.int16), len(qpos))}))
            if verbose:
                print(f"  gpu-block {country} {f}: S1={len(s1_pos)} q={len(qpos)} {time.time() - t:.0f}s", flush=True)
        m = per[0]
        for p in per[1:]:
            m = m.merge(p, on=["q_idx", "s1_idx"], how="outer")
        parts.append(m)
    pairs = pd.concat(parts, ignore_index=True)
    qi, si = pairs["q_idx"].to_numpy(), pairs["s1_idx"].to_numpy()
    if e5 is not None:
        pairs["rank_e5"] = pairs["rank_e5"].fillna(k).astype(np.int16)
        pairs["cos_e5"] = rowwise_dot(e5[1], e5[0], qi, si)
        pairs["gap_e5"] = pairs.groupby("q_idx")["cos_e5"].transform("max") - pairs["cos_e5"]
    for f in views:
        if f"rank_{f}" in pairs:
            pairs[f"rank_{f}"] = pairs[f"rank_{f}"].fillna(k).astype(np.int16)
        pairs[f"cos_{f}"] = rowwise_cos(q_enc[f][0], s1_cache[f][0], qi, si)
        pairs[f"gap_{f}"] = pairs.groupby("q_idx")[f"cos_{f}"].transform("max") - pairs[f"cos_{f}"]
    pairs["n_cand"] = pairs.groupby("q_idx")["s1_idx"].transform("size").astype(np.int16)
    return pairs
