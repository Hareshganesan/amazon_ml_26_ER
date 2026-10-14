"""v1 full pipeline: train on a train sample searched against the FULL train S1 index, predict test.

1. Fit TF-IDF+SVD views on S1 text (train + test S1, unsupervised).
2. Train: sample N S1 entities; queries = their true S2/S3 matches + same-rate decoys; candidates are
   searched in the full train S1 index (realistic hard negatives). 2-fold OOF (grouped by S1) picks the
   decision threshold on macro F0.5; a final LightGBM is trained on all sampled pairs.
3. Test: candidates for every S2/S3 record (chunked), features, probabilities, one-to-one decision.
4. Write output/matching_results.tsv and output/candidate_pairs.tsv.

    python src/v1_full.py --data $DATA --out output --n_train 100000 --k 10 --fetch 30
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import add_text_columns, pair_features, set_name_counts, set_vocab  # noqa: E402
from gpu_blocking import DEVICE, VIEWS, View, candidates, e5_encode  # noqa: E402
from metric import score  # noqa: E402

T0 = time.time()
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)


def log(msg):
    """Print a progress message prefixed with the seconds elapsed since the script started."""
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def read(p, cols=None):
    """Read a competition TSV (optionally only some columns) with every column as a string."""
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, usecols=cols)


def best_per_query(q_idx, s1_idx, prob, t):
    """One-to-one decision: each query keeps its highest-probability S1 candidate if prob > t."""
    order = np.lexsort((-prob, q_idx))
    q_s, s_s, p_s = q_idx[order], s1_idx[order], prob[order]
    first = np.r_[True, q_s[1:] != q_s[:-1]]
    keep = first & (p_s > t)
    return q_s[keep], s_s[keep]


def e5_text(df):
    """Raw name + address text for the multilingual e5 encoder (handles Indian scripts and French)."""
    return (df["business_name"] + ", " + df["business_address"]).tolist()


def encode_s1(views, s1):
    """Encode the S1 index once per view: sparse TF-IDF and dense SVD vectors, reused by every query chunk."""
    cache = {}
    for f, v in views.items():
        Xs = v.sparse(s1[f])
        cache[f] = (Xs, v.dense(Xs))
    return cache


BG_RAW = None  # --dense_bg: all background records (raw columns), consumed by v5_full.main


def train_stage(a, views):
    """Build the training sample: S1 entities, their S2/S3 records and no-match queries.

    Samples n_train S1 entities. With orphan_frac, drops that share of entities from the index while keeping their
    records as no-match queries (test-like orphans). With dense_bg, keeps every decoy/orphan record that retrieves a
    sampled entity (test-like decoy density). Returns (S1 frame, query frame, ground truth).
    """
    d = Path(a.data) / "train"
    gt = read(d / "train_ground_truth.tsv")
    matched_all = {x for ids in gt["matched_entity_ids"] for x in ids.split(",") if x}
    orphan_frac = getattr(a, "orphan_frac", 0.0)
    s1 = read(d / "train_source1.tsv")
    orphans = set()
    if orphan_frac > 0:
        # Test-like index: the test has the train's number of S2/S3 records but ~21% fewer S1 entities, i.e. their
        # records stay behind as "orphans" (real businesses whose S1 is absent). Drop that share of S1 entities from
        # the index and keep a proportional sample of their records as no-match queries.
        drop = gt.sample(frac=orphan_frac, random_state=a.seed + 17)
        gt = gt.drop(drop.index)
        s1 = s1[~s1["entity_id"].isin(set(drop["source1_entity_id"]))].reset_index(drop=True)
    frac = a.n_train / len(s1)
    dense = getattr(a, "dense_bg", False)
    if orphan_frac > 0:
        # dense_bg: ALL orphan records (v5_full keeps those landing on sampled entities); else a proportional sample
        orphans = {x for ids in (drop if dense else drop.sample(frac=frac, random_state=a.seed))["matched_entity_ids"]
                   for x in ids.split(",") if x}
        del drop
    gt = gt.sample(n=a.n_train, random_state=a.seed)
    pair = {b: s for s, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]) for b in ids.split(",") if b}
    truth = {s: {x for x in ids.split(",") if x} for s, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}
    parts = []
    for s in (2, 3):
        df = read(d / f"train_source{s}.tsv")
        parts.append(df[df["entity_id"].isin(pair)])
        un = df[~df["entity_id"].isin(matched_all)]
        parts.append(un if dense else un.sample(frac=frac, random_state=a.seed))  # dense_bg: all decoys, filtered later
        if orphans:
            parts.append(df[df["entity_id"].isin(orphans)])
        del df
    del matched_all
    q = pd.concat(parts, ignore_index=True)
    q["true_s1"] = q["entity_id"].map(pair).fillna("")
    q["bg"] = q["true_s1"] == ""  # background = decoys + orphan records (no true S1 in the index)
    if dense:  # raw text only: v5_full normalises it in chunks and keeps the records landing on sampled entities
        global BG_RAW
        BG_RAW = q[q["bg"]].reset_index(drop=True)
        q = q[~q["bg"]].reset_index(drop=True)
    add_text_columns(s1)
    add_text_columns(q)
    log(f"train: S1 index={len(s1)} queries={len(q)} (orphan records={int(q['entity_id'].isin(orphans).sum())})")
    return s1, q, truth


def main(a):
    # ---------- text + views ----------
    """v1 pipeline: fit views, block, build features, train LightGBM, pick the OOF threshold, score the test, write outputs."""
    tr_s1, tr_q, truth = train_stage(a, None)
    te_s1 = read(Path(a.data) / "test" / "test_source1.tsv")
    add_text_columns(te_s1)
    log(f"text ready; device={DEVICE}")
    views = {}
    for f, (an, ng) in VIEWS.items():
        views[f] = View(f, an, ng).fit(pd.concat([tr_s1[f], te_s1[f]], ignore_index=True))
        log(f"view {f} fitted: vocab={len(views[f].vec.vocabulary_)}")

    set_vocab(pd.concat([tr_s1["c_name"], te_s1["c_name"]], ignore_index=True))
    model_e5, tr_e5 = None, None
    if a.e5:
        from sentence_transformers import SentenceTransformer
        model_e5 = SentenceTransformer(a.e5, device=DEVICE)
        model_e5.max_seq_length = 64
        if DEVICE == "cuda":
            model_e5.half()
        t = time.time()
        tr_e5 = (e5_encode(model_e5, e5_text(tr_s1)), e5_encode(model_e5, e5_text(tr_q)))
        log(f"e5 encoded train S1+queries ({len(tr_s1) + len(tr_q)} texts, {(len(tr_s1) + len(tr_q)) / (time.time() - t):.0f}/s)")

    set_name_counts(tr_s1)

    # ---------- training pairs (full train S1 index) ----------
    pairs = candidates(views, tr_s1, tr_q, a.k, fetch=a.fetch, e5=tr_e5, block_fields=a.block_fields)
    s1_id = tr_s1["entity_id"].to_numpy()
    y = (s1_id[pairs["s1_idx"].to_numpy()] == tr_q["true_s1"].to_numpy()[pairs["q_idx"].to_numpy()]).astype(np.int8)
    n_true = int((tr_q["true_s1"] != "").sum())
    log(f"train pairs={len(pairs)} ({len(pairs) / len(tr_q):.1f}/q) blocking recall={y.sum() / n_true:.4f}")
    X = pair_features(tr_q, tr_s1, pairs)
    log(f"train features done: {X.shape}")

    rng = np.random.default_rng(a.seed)
    s1_fold = dict(zip(truth.keys(), rng.integers(0, 2, len(truth))))
    q_fold = np.where(tr_q["true_s1"] != "", tr_q["true_s1"].map(s1_fold).fillna(0), rng.integers(0, 2, len(tr_q))).astype(int)
    fold = q_fold[pairs["q_idx"].to_numpy()]
    oof = np.zeros(len(pairs), dtype=np.float32)
    iters = []
    for k in (0, 1):
        tr, va = fold != k, fold == k
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), 2000, valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        log(f"fold {k}: best_iter={m.best_iteration}")
    qi, si = pairs["q_idx"].to_numpy(), pairs["s1_idx"].to_numpy()
    q_ids = tr_q["entity_id"].to_numpy()
    sample_ids = np.array(list(truth.keys()))
    in_sample = set(sample_ids)
    res, res_naive = {}, {}
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        bq, bs = best_per_query(qi, si, oof, t)
        pred, pred_naive = {}, {}
        rng_fp = np.random.default_rng(1)
        for q_i, s_i in zip(bq, bs):
            s_id = s1_id[s_i]
            pred_naive.setdefault(s_id, set()).add(q_ids[q_i])
            # realistic evaluation: a false match always lands on SOME S1 entity at test time. Our sample's
            # wrong matches mostly land outside the sample, where they would be invisible; move them onto a
            # random sampled entity so the FP inflow per entity matches the full-data rate.
            if s_id not in in_sample:
                s_id = sample_ids[rng_fp.integers(len(sample_ids))]
            pred.setdefault(s_id, set()).add(q_ids[q_i])
        key = round(float(t), 2)
        res[key], res_naive[key] = score(pred, truth), score(pred_naive, truth)
        log(f"OOF threshold {t:.2f}: realistic F0.5 = {res[key]:.5f}   (naive {res_naive[key]:.5f})")
    best_t = max(res, key=res.get)
    log(f"BEST OOF: t={best_t} F0.5={res[best_t]:.5f}")
    model = lgb.train(PARAMS, lgb.Dataset(X, y), int(np.mean(iters) * 1.1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "lgb_v1_full.txt"))
    json.dump({"oof_f05": res, "oof_f05_naive": res_naive, "best_t": best_t, "blocking_recall": float(y.sum() / n_true), "k": a.k, "fetch": a.fetch, "e5": a.e5, "block_fields": a.block_fields,
               "n_train": a.n_train}, open(out / "v1_full_results.json", "w"), indent=2)
    feat_cols = list(X.columns)
    del pairs, X, y, oof, tr_s1, tr_q, tr_e5

    # ---------- test inference ----------
    set_name_counts(te_s1)  # test-side ambiguity counts (same definition, test S1 index)
    te_cache = encode_s1(views, te_s1)
    te_e5_s1 = e5_encode(model_e5, e5_text(te_s1)) if model_e5 is not None else None
    te_ids = te_s1["entity_id"].to_numpy()
    # compact storage: pairs as (s1 position, global query code) int arrays; code indexes all_qids
    all_qids, offset = [], 0
    b_s1, b_q, b_p, c_s1, c_q = [], [], [], [], []
    for s in (2, 3):
        src = read(Path(a.data) / "test" / f"test_source{s}.tsv")
        all_qids.append(src["entity_id"].to_numpy())
        for i in range(0, len(src), a.chunk):
            q = src.iloc[i:i + a.chunk].reset_index(drop=True)
            add_text_columns(q)
            e5 = (te_e5_s1, e5_encode(model_e5, e5_text(q))) if model_e5 is not None else None
            p = candidates(views, te_s1, q, a.k, s1_cache=te_cache, verbose=False, fetch=a.fetch, e5=e5,
                           block_fields=a.block_fields)
            Xt = pair_features(q, te_s1, p)[feat_cols]
            prob = model.predict(Xt).astype(np.float32)
            qi, si = p["q_idx"].to_numpy(), p["s1_idx"].to_numpy()
            c_s1.append(si.astype(np.int32))
            c_q.append((qi + offset + i).astype(np.int64))
            bq, bs = best_per_query(qi, si, prob, -1.0)  # best candidate of every query, threshold applied later
            best_p = pd.Series(prob).groupby(qi).max().reindex(bq).to_numpy(np.float32)
            b_s1.append(bs.astype(np.int32))
            b_q.append((bq + offset + i).astype(np.int64))
            b_p.append(best_p)
            log(f"test source{s} rows {i}-{i + len(q)}: pairs={len(p)} matched@t={int((best_p > best_t).sum())}")
            del q, p, Xt, prob
        offset += len(src)
        del src
    all_qids = np.concatenate(all_qids)

    def write(fname, col, s1_parts, q_parts):
        """Write one output TSV (one row per S1 entity, comma-separated S2/S3 ids); returns the number of non-empty rows."""
        s1a, qa = np.concatenate(s1_parts), np.concatenate(q_parts)
        order = np.lexsort((qa, s1a))
        s1a, qa = s1a[order], qa[order]
        keep = np.r_[True, (s1a[1:] != s1a[:-1]) | (qa[1:] != qa[:-1])]  # drop duplicate pairs
        s1a, qa = s1a[keep], qa[keep]
        bounds = np.searchsorted(s1a, np.arange(len(te_ids) + 1))
        with open(out / fname, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"source1_entity_id\t{col}\n")
            for j, e in enumerate(te_ids):
                fh.write(f"{e}\t{','.join(all_qids[qa[bounds[j]:bounds[j + 1]]])}\n")
        return int((np.diff(bounds) > 0).sum())

    n_c = write("candidate_pairs.tsv", "candidate_entity_ids", c_s1, c_q)
    b_s1, b_q, b_p = np.concatenate(b_s1), np.concatenate(b_q), np.concatenate(b_p)
    np.savez_compressed(out / "test_best.npz", s1_pos=b_s1, q_code=b_q, prob=b_p)  # for re-thresholding later
    n_m = write("matching_results.tsv", "matched_entity_ids", [b_s1[b_p > best_t]], [b_q[b_p > best_t]])
    log(f"S1 with candidates={n_c}, with matches={n_m} of {len(te_ids)} (t={best_t})")
    for t in a.extra_t:
        keep = b_p > t
        n = write(f"matching_results_t{t}.tsv", "matched_entity_ids", [b_s1[keep]], [b_q[keep]])
        log(f"extra file t={t}: S1 with matches={n}, matched queries={int(keep.sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--n_train", type=int, default=100000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--fetch", type=int, default=30, help="SVD candidates per view re-ranked by exact cosine")
    ap.add_argument("--chunk", type=int, default=500000)
    ap.add_argument("--e5", default="", help="sentence-transformers model for an extra embedding view, e.g. intfloat/multilingual-e5-small")
    ap.add_argument("--block_fields", nargs="*", default=None, help="TF-IDF views used for blocking (default: all)")
    ap.add_argument("--extra_t", type=float, nargs="*", default=[0.5, 0.7, 0.85],
                    help="also write matching_results_t<t>.tsv for these thresholds")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
