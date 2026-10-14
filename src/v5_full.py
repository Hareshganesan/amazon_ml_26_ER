"""v5 full pipeline: wide blocking -> cheap pruner (cascade) -> rich pair features -> LightGBM -> F0.5 decision.

Stage 1 (blocking): per country, TF-IDF name+address (SVD top-`fetch` re-ranked by exact cosine, keep k) and
    multilingual-e5 name+address top-k. About 18 candidates per S2/S3 record, ~98% recall.
Stage 2 (pruner = the LAST candidate stage): a small LightGBM on blocking similarities only keeps the top
    `prune_top` candidates per record with probability >= `prune_p`. This set is written to candidate_pairs.tsv
    (organisers rank smaller candidate sets higher; Mithilesh's probe: ~7 per S1 entity, -0.4 pt recall).
Stage 3 (matcher): full pair features (string, containment, name ambiguity, Venky's decoy/house/legal features,
    margins) -> LightGBM -> each S2/S3 record keeps its best S1 candidate if p > t (t chosen on realistic OOF F0.5).

Training pairs: N sampled train S1 entities; queries = their true matches + same-rate true decoys, searched
against the FULL train S1 index. Pruner and matcher use 2-fold OOF grouped by S1 entity, so the matcher is
trained on pruner outputs that look like test-time outputs.

    python src/v5_full.py --data $DATA --out output --n_train 200000 --e5 intfloat/multilingual-e5-small
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
from gpu_blocking import DEVICE, EXTRA_VIEWS, VIEWS, View, candidates, dense_topk, e5_encode  # noqa: E402
from metric import score  # noqa: E402
from v1_full import best_per_query, e5_text, encode_s1, read, train_stage  # noqa: E402

T0 = time.time()
MATCH_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
SIB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
PRUNE_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                    verbose=-1, num_threads=4)


def log(msg):
    """Print a progress message prefixed with the seconds elapsed since the script started."""
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def prune_cols(pairs):
    """Blocking-only columns used by the stage-2 pruner (cheap: no string features)."""
    return [c for c in pairs.columns if c.startswith(("cos_", "rank_", "gap_")) or c == "n_cand"]


def prune(pairs, prob, top, pmin):
    """Keep the top `top` candidates per query with probability >= pmin; returns the kept pairs (+ p_prune)."""
    rank = pd.Series(prob).groupby(pairs["q_idx"].to_numpy()).rank(ascending=False, method="first").to_numpy()
    keep = (rank <= top) & (prob >= pmin)
    out = pairs.loc[keep].reset_index(drop=True)
    out["p_prune"] = prob[keep].astype(np.float32)
    out["n_cand_pruned"] = out.groupby("q_idx")["s1_idx"].transform("size").astype(np.int16)
    return out


DENSE_BG = None  # --dense_bg: set of background query ids already present at full test-like density


def realistic_eval(qi, si, prob, t, s1_id, q_ids, truth, sample_ids, in_sample, seed=1):
    """Macro F0.5 on the sampled S1 entities, moving out-of-sample false matches onto random sampled
    entities so the false-positive inflow per entity matches the full test set (see Documentation §5).
    With --dense_bg every background record that lands on a sampled entity is already a query, so background
    false matches on out-of-sample entities are dropped instead of moved (moving them would double-count)."""
    bq, bs = best_per_query(qi, si, prob, t)
    rng = np.random.default_rng(seed)
    pred = {}
    for q_i, s_i in zip(bq, bs):
        s_id = s1_id[s_i]
        if s_id not in in_sample:
            if DENSE_BG is not None and q_ids[q_i] in DENSE_BG:
                continue
            s_id = sample_ids[rng.integers(len(sample_ids))]
        pred.setdefault(s_id, set()).add(q_ids[q_i])
    return score(pred, truth)


def oof_lgb(params, X, y, fold, rounds, early=True):
    """2-fold out-of-fold predictions; returns (oof, best iterations)."""
    oof = np.zeros(len(y), dtype=np.float32)
    iters = []
    for k in (0, 1):
        tr, va = fold != k, fold == k
        if early:
            m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), rounds, valid_sets=[lgb.Dataset(X[va], y[va])],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            iters.append(m.best_iteration)
            oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        else:
            m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), rounds)
            iters.append(rounds)
            oof[va] = m.predict(X[va])
    return oof, iters


def dump_errors(out, s1, q, pairs, oof, t, y, stage1_hit, n_max=4000):
    """Write errors.tsv: one row per wrong decision on the OOF sample, with the reason and both records.

    error_type: FP_decoy (a no-match record was matched), FP_wrong_entity (matched to the wrong S1),
    FN_blocking (true S1 never a candidate), FN_pruned (lost in the cascade), FN_threshold (true S1 was the
    best candidate but p <= t), FN_ranked_below (another candidate scored higher and was rejected/accepted).
    """
    s1_id = s1["entity_id"].to_numpy()
    df = pd.DataFrame({"q": pairs["q_idx"].to_numpy(), "s": pairs["s1_idx"].to_numpy(), "p": oof, "y": y})
    df = df.sort_values(["q", "p"], ascending=[True, False])
    best = df.drop_duplicates("q").set_index("q")
    true_row = df[df["y"] == 1].drop_duplicates("q").set_index("q")
    rows = []
    for qi in range(len(q)):
        t_s1 = q["true_s1"].iat[qi]
        b = best.loc[qi] if qi in best.index else None
        pred = s1_id[int(b["s"])] if b is not None and b["p"] > t else ""
        if pred == t_s1:
            continue
        if t_s1 == "":
            etype = "FP_decoy"
        elif pred != "":
            etype = "FP_wrong_entity"
        elif qi not in stage1_hit:
            etype = "FN_blocking"
        elif qi not in true_row.index:
            etype = "FN_pruned"
        elif b is not None and int(b["s"]) == int(true_row.loc[qi, "s"]):
            etype = "FN_threshold"
        else:
            etype = "FN_ranked_below"
        best_s1 = s1_id[int(b["s"])] if b is not None else ""  # top candidate even when rejected (Sanku: FN_ranked_below)
        rows.append((etype, qi, best_s1, t_s1, float(b["p"]) if b is not None else -1.0,
                     float(true_row.loc[qi, "p"]) if qi in true_row.index else -1.0))
    err = pd.DataFrame(rows, columns=["error_type", "qi", "pred_s1", "true_s1", "p_best", "p_true"])
    if err.empty:
        print("ERROR COUNTS: no errors on the OOF sample", flush=True)
        return
    err["qi"] = err["qi"].astype(np.int64)
    counts = err.groupby(["error_type"]).size()
    by_country = err.assign(country=q["country"].to_numpy()[err["qi"]]).groupby(["country", "error_type"]).size()
    print(f"ERROR COUNTS (OOF, t={t}):", counts.to_string(), by_country.to_string(), sep=chr(10), flush=True)
    idx = pd.Series(np.arange(len(s1)), index=s1_id)
    sample = err.groupby("error_type", group_keys=False).apply(lambda g: g.sample(min(len(g), n_max // 6), random_state=0))
    def rec(frame, pos):
        """Return (business_name, business_address) arrays of a frame at the given row positions."""
        return frame["business_name"].to_numpy()[pos], frame["business_address"].to_numpy()[pos]
    qn, qa = rec(q, sample["qi"].to_numpy())
    pp = sample["pred_s1"].map(idx).fillna(-1).astype(int).to_numpy()
    tp = sample["true_s1"].map(idx).fillna(-1).astype(int).to_numpy()
    pn = np.where(pp >= 0, s1["business_name"].to_numpy()[pp], "")
    pa = np.where(pp >= 0, s1["business_address"].to_numpy()[pp], "")
    tn = np.where(tp >= 0, s1["business_name"].to_numpy()[tp], "")
    ta = np.where(tp >= 0, s1["business_address"].to_numpy()[tp], "")
    pd.DataFrame({"error_type": sample["error_type"].to_numpy(), "country": q["country"].to_numpy()[sample["qi"]],
                  "p_best": sample["p_best"].round(3).to_numpy(), "p_true": sample["p_true"].round(3).to_numpy(),
                  "record_name": qn, "record_addr": qa, "pred_s1_name": pn, "pred_s1_addr": pa,
                  "true_s1_name": tn, "true_s1_addr": ta}).to_csv(out / "errors.tsv", sep=chr(9), index=False)
    counts.to_csv(out / "error_counts.tsv", sep=chr(9))


def load_pseudo(a, te_s1):
    """French pseudo-labels from a previous run's test_best.npz (best S1 + probability per test S2/S3 record).

    Positives: French records whose best candidate had p >= pseudo_pos (label = that S1 entity).
    Decoys: French records whose best candidate had p < pseudo_neg (label = no match).
    France has no training labels; this lets the matcher see French text (legal forms, rue/bd, templated
    names) with the same kind of labels it gets for US/India. Self-training on test is allowed (clarifications).
    """
    z = np.load(a.pseudo_npz)
    d = Path(a.data) / "test"
    src = pd.concat([read(d / f"test_source{s}.tsv") for s in (2, 3)], ignore_index=True)  # ~10M rows, freed below
    te_country = te_s1["country"].to_numpy()
    s1_pos, q_code, prob = z["s1_pos"], z["q_code"], z["prob"]
    fr = (src["country"].to_numpy()[q_code] == "France") & (te_country[s1_pos] == "France")
    rng = np.random.default_rng(a.seed)
    pos = np.flatnonzero(fr & (prob >= a.pseudo_pos))
    neg = np.flatnonzero(fr & (prob < a.pseudo_neg))
    pos = rng.choice(pos, min(len(pos), a.pseudo_n), replace=False)
    neg = rng.choice(neg, min(len(neg), a.pseudo_n // 3), replace=False)
    sel = np.r_[pos, neg]
    q = src.iloc[q_code[sel]].reset_index(drop=True)
    del src
    q["true_s1"] = np.r_[te_s1["entity_id"].to_numpy()[s1_pos[pos]], np.full(len(neg), "", dtype=object)]
    log(f"pseudo-labels: {len(pos)} French positives (p>={a.pseudo_pos}), {len(neg)} French decoys (p<{a.pseudo_neg})")
    return q


def rss():
    """Resident memory of this process (Linux), for the test-phase logs."""
    try:
        kb = int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1])
        return f"RSS {kb / 2**20:.1f} GB"
    except (OSError, StopIteration):
        return "RSS n/a"


def main(a):
    """Full pipeline (entry point).

    Training: build the orphan/dense training sample, fit the retrieval views and e5 embeddings, stage-1 wide blocking,
    stage-2 LightGBM pruner (out-of-fold), stage-3 pair features and LightGBM matcher, OOF threshold sweep (optional
    stage 4). Test: block and score in chunks on GPU, save pass-1 pairs and features (--save_test) for CPU re-scoring,
    and write matching_results.tsv and candidate_pairs.tsv.
    """
    import features
    features.USE_V14 = a.v14
    MATCH_PARAMS.update(learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=a.min_leaf, feature_fraction=a.ff)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr_s1, tr_q, truth = train_stage(a, None)
    te_s1 = read(Path(a.data) / "test" / "test_source1.tsv")
    add_text_columns(te_s1)
    log(f"text ready; device={DEVICE}")
    view_cfg = dict(VIEWS, **{f: EXTRA_VIEWS[f] for f in a.extra_views})
    views = {f: View(f, an, ng).fit(pd.concat([tr_s1[f], te_s1[f]], ignore_index=True)) for f, (an, ng) in view_cfg.items()}
    set_vocab(pd.concat([tr_s1["c_name"], te_s1["c_name"]], ignore_index=True))
    set_name_counts(tr_s1)
    if a.dense_bg:
        # Test-like group density: keep EVERY background record (decoy / orphan) whose top-2 name+address
        # candidate is a sampled entity. With a proportional sample the sampled entities saw only ~7% of the
        # decoys the test gives them, so sibling groups were unrealistically clean (LB 0.964 on 26 Sep).
        global DENSE_BG
        import gc
        import v1_full
        v = views["name_addr"]
        s_dense = v.dense(v.sparse(tr_s1["name_addr"]))
        sampled = tr_s1["entity_id"].isin(set(truth)).to_numpy()
        s_country = tr_s1["country"].to_numpy()
        bg_raw, v1_full.BG_RAW = v1_full.BG_RAW, None
        n_bg, kept = len(bg_raw), []
        for i in range(0, n_bg, 400_000):  # normalise text chunk-wise; keep only records landing on sampled S1
            ch = bg_raw.iloc[i:i + 400_000].reset_index(drop=True).copy()
            add_text_columns(ch)
            eq = v.dense(v.sparse(ch["name_addr"]))
            hit = np.zeros(len(ch), bool)
            for c, idx_c in ch.groupby("country").indices.items():
                sp = np.flatnonzero(s_country == c)
                if len(sp):
                    top, _ = dense_topk(eq[idx_c], s_dense[sp], 2)
                    hit[idx_c] = sampled[sp[top]].any(axis=1)
            kept.append(ch[hit].copy())
            del ch, eq
        del bg_raw, s_dense
        bgk = pd.concat(kept, ignore_index=True)
        tr_q = pd.concat([tr_q, bgk], ignore_index=True)
        DENSE_BG = set(bgk["entity_id"])
        log(f"dense_bg: kept {len(bgk)} of {n_bg} background records landing on sampled entities; "
            f"queries now {len(tr_q)} ({rss()})")
        del bgk, kept
        gc.collect()
        try:  # hand freed heap back to the OS (glibc) before the big blocking step
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
    model_e5 = tr_e5 = None
    if a.e5:
        from sentence_transformers import SentenceTransformer
        model_e5 = SentenceTransformer(a.e5, device=DEVICE)
        model_e5.max_seq_length = 64
        if DEVICE == "cuda":
            model_e5.half()
        t = time.time()
        tr_e5 = (e5_encode(model_e5, e5_text(tr_s1)), e5_encode(model_e5, e5_text(tr_q)))
        log(f"e5 train encoded ({(len(tr_s1) + len(tr_q)) / (time.time() - t):.0f} texts/s)")

    te_cache = te_e5_s1 = None  # test-side encodings: built only after the big train blocking frames are freed

    # ---------- stage 1: wide blocking on train ----------
    pairs = candidates(views, tr_s1, tr_q, a.k, fetch=a.fetch, e5=tr_e5, block_fields=a.block_fields)
    del tr_e5
    tr_e5 = None
    import gc
    gc.collect()
    s1_id = tr_s1["entity_id"].to_numpy()
    q_ids = tr_q["entity_id"].to_numpy()
    y1 = (s1_id[pairs["s1_idx"].to_numpy()] == tr_q["true_s1"].to_numpy()[pairs["q_idx"].to_numpy()]).astype(np.int8)
    n_true = int((tr_q["true_s1"] != "").sum())
    res = {"stage1_recall": float(y1.sum() / n_true), "stage1_cands_per_query": len(pairs) / len(tr_q)}
    log(f"stage1: pairs={len(pairs)} ({res['stage1_cands_per_query']:.1f}/q) recall={res['stage1_recall']:.4f}")

    rng = np.random.default_rng(a.seed)
    s1_fold = dict(zip(truth.keys(), rng.integers(0, 2, len(truth))))
    q_fold = np.where(tr_q["true_s1"] != "", tr_q["true_s1"].map(s1_fold).fillna(0), rng.integers(0, 2, len(tr_q))).astype(int)

    # ---------- stage 2: pruner (OOF so the matcher sees realistic pruned sets) ----------
    pcols = prune_cols(pairs)
    Xp = pairs[pcols].to_numpy(np.float32)
    p_oof, _ = oof_lgb(PRUNE_PARAMS, Xp, y1, q_fold[pairs["q_idx"].to_numpy()], 300, early=False)
    pruner = lgb.train(PRUNE_PARAMS, lgb.Dataset(Xp, y1), 300)
    pruner.save_model(str(out / "lgb_v5_pruner.txt"))
    stage1_hit = set(pairs["q_idx"].to_numpy()[y1 == 1])  # queries whose true S1 survived stage 1
    pairs = prune(pairs, p_oof, a.prune_top, a.prune_p)
    y = (s1_id[pairs["s1_idx"].to_numpy()] == tr_q["true_s1"].to_numpy()[pairs["q_idx"].to_numpy()]).astype(np.int8)
    res.update(stage2_recall=float(y.sum() / n_true), stage2_cands_per_query=len(pairs) / len(tr_q))
    log(f"stage2 (top{a.prune_top}, p>={a.prune_p}): pairs={len(pairs)} ({res['stage2_cands_per_query']:.2f}/q) "
        f"recall={res['stage2_recall']:.4f}")
    del Xp, p_oof

    # ---------- stage 3: matcher ----------
    X = pair_features(tr_q, tr_s1, pairs)
    log(f"train features: {X.shape}")
    fold = q_fold[pairs["q_idx"].to_numpy()]
    n_tr = len(X)
    if a.pseudo_npz:
        import gc
        gc.collect()
        te_cache = encode_s1(views, te_s1)
        te_e5_s1 = e5_encode(model_e5, e5_text(te_s1)) if model_e5 is not None else None
        fr_q = load_pseudo(a, te_s1)
        add_text_columns(fr_q)
        set_name_counts(te_s1)  # French pairs are scored against the test S1 index
        fr_e5 = (te_e5_s1, e5_encode(model_e5, e5_text(fr_q))) if model_e5 is not None else None
        fp = candidates(views, te_s1, fr_q, a.k, s1_cache=te_cache, verbose=False, fetch=a.fetch, e5=fr_e5,
                        block_fields=a.block_fields)
        fp = prune(fp, pruner.predict(fp[pcols].to_numpy(np.float32)).astype(np.float32), a.prune_top, a.prune_p)
        te_id = te_s1["entity_id"].to_numpy()
        yf = (te_id[fp["s1_idx"].to_numpy()] == fr_q["true_s1"].to_numpy()[fp["q_idx"].to_numpy()]).astype(np.int8)
        Xf = pair_features(fr_q, te_s1, fp)[list(X.columns)]
        ff = np.random.default_rng(a.seed + 1).integers(0, 2, len(fr_q))[fp["q_idx"].to_numpy()]
        log(f"pseudo pairs: {len(Xf)} (positives kept after cascade: {int(yf.sum())})")
        X = pd.concat([X, Xf], ignore_index=True)
        y = np.r_[y, yf]
        fold = np.r_[fold, ff]
        del fp, Xf, yf, ff, fr_q, fr_e5
    oof_all, iters = oof_lgb(MATCH_PARAMS, X, y, fold, a.max_rounds)
    oof = oof_all[:n_tr]  # realistic evaluation uses the labelled train sample only
    log(f"matcher best iters={iters}")
    qi, si = pairs["q_idx"].to_numpy(), pairs["s1_idx"].to_numpy()
    sample_ids = np.array(list(truth.keys()))
    in_sample = set(sample_ids)
    sweep = {}
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        sweep[t] = realistic_eval(qi, si, oof, t, s1_id, q_ids, truth, sample_ids, in_sample)
        log(f"OOF threshold {t:.2f}: realistic F0.5 = {sweep[t]:.5f}")
    best_t = max(sweep, key=sweep.get)
    log(f"BEST OOF: t={best_t} F0.5={sweep[best_t]:.5f}")
    # per-country realistic sweep (US / India; France has no labels)
    s1_country = dict(zip(s1_id, tr_s1["country"].to_numpy()))
    per_country = {}
    for c in sorted(set(tr_s1["country"])):
        tc = {k: v for k, v in truth.items() if s1_country.get(k) == c}
        ids_c = np.array(list(tc.keys()))
        if len(ids_c) == 0:
            continue
        m = tr_s1["country"].to_numpy()[si] == c  # only candidates pointing at this country's S1 entities
        per_country[c] = {t: realistic_eval(qi[m], si[m], oof[m], t, s1_id, q_ids, tc, ids_c, set(ids_c))
                          for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)}
        bc = max(per_country[c], key=per_country[c].get)
        log(f"per-country {c}: best t={bc} F0.5={per_country[c][bc]:.5f}  " + " ".join(f"{t}:{v:.4f}" for t, v in per_country[c].items()))
    if a.save_train:
        # everything needed for CPU-only model experiments (tuning, other models, ensembles, thresholds)
        meta = pd.DataFrame({"q_id": q_ids[qi], "s1_id": s1_id[si], "country": tr_s1["country"].to_numpy()[si],
                             "y": y[:n_tr], "fold": fold[:n_tr], "oof": oof})
        X.iloc[:n_tr].assign(**{c: meta[c].to_numpy() for c in meta}).to_parquet(out / "train_pairs.parquet", index=False)
        pd.DataFrame({"s1_id": list(truth.keys()), "matched": [",".join(sorted(v)) for v in truth.values()],
                      "country": [s1_country.get(k, "") for k in truth]}).to_parquet(out / "train_truth.parquet", index=False)
        log(f"saved train_pairs.parquet ({n_tr} rows, {X.shape[1]} features) + train_truth.parquet")
    if a.dump_errors:
        dump_errors(out, tr_s1, tr_q, pairs, oof, best_t, y[:n_tr], stage1_hit)
    model = lgb.train(MATCH_PARAMS, lgb.Dataset(X, y), int(np.mean(iters) * 1.1))
    model.save_model(str(out / "lgb_v5_matcher.txt"))
    model2 = None
    if a.siblings:
        # stage 4: sibling-consistency features from the stage-3 OOF probabilities (train sample only)
        from features import _street_raw
        from siblings import sibling_stats, sibling_stats_v2
        qh = tr_q["business_address"].map(lambda x: (_street_raw(x)[0][0] if _street_raw(x) else "")).to_numpy()
        if a.sib_v2:
            S = sibling_stats_v2(qi, si, oof, qh, tr_q["c_name"].to_numpy(), tr_q["n_addr"].to_numpy(),
                                 tr_q["entity_id"].str.startswith("S3").to_numpy(), tr_s1["c_name"].to_numpy())
        else:
            S = sibling_stats(qi, si, oof, qh, tr_q["c_name"].to_numpy(), tr_q["n_addr"].to_numpy())
        X2 = np.hstack([X.iloc[:n_tr].to_numpy(np.float32), S, oof[:, None]])
        oof2, it2 = oof_lgb(SIB_PARAMS, X2, y[:n_tr], fold[:n_tr], 4000)
        sweep2 = {t: realistic_eval(qi, si, oof2, t, s1_id, q_ids, truth, sample_ids, in_sample)
                  for t in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)}
        for t, v in sweep2.items():
            log(f"STAGE4 OOF threshold {t:.2f}: realistic F0.5 = {v:.5f}")
        best_t = max(sweep2, key=sweep2.get)
        log(f"STAGE4 BEST: t={best_t} F0.5={sweep2[best_t]:.5f} (stage 3 was {sweep[max(sweep, key=sweep.get)]:.5f})")
        model2 = lgb.train(SIB_PARAMS, lgb.Dataset(X2, y[:n_tr]), int(np.mean(it2) * 1.1))
        model2.save_model(str(out / "lgb_v5_stage4.txt"))
        res["stage4_oof_f05"] = sweep2
        del X2, S, oof2
    imp = pd.Series(model.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print("feature importance (gain, top 25):\n" + (imp / imp.sum()).head(25).round(4).to_string(), flush=True)
    res.update(oof_f05=sweep, per_country=per_country, best_t=best_t, siblings=a.siblings, n_train=a.n_train, orphan_frac=a.orphan_frac, dense_bg=a.dense_bg, sib_v2=a.sib_v2, v14=a.v14, pseudo_npz=a.pseudo_npz, pseudo_pos=a.pseudo_pos, k=a.k, fetch=a.fetch, e5=a.e5,
               prune_top=a.prune_top, prune_p=a.prune_p, importance=(imp / imp.sum()).round(5).to_dict())
    json.dump(res, open(out / "v5_results.json", "w"), indent=2)
    feat_cols = list(X.columns)
    if a.skip_test:
        log("skip_test: done after training")
        return
    del pairs, X, y, oof, oof_all, tr_s1, tr_q, tr_e5

    # ---------- test ----------
    set_name_counts(te_s1)
    if te_cache is None:
        te_cache = encode_s1(views, te_s1)
    if te_e5_s1 is None and model_e5 is not None:
        te_e5_s1 = e5_encode(model_e5, e5_text(te_s1))
    te_ids = te_s1["entity_id"].to_numpy()
    all_qids, offset = [], 0
    b_s1, b_q, b_p, c_s1, c_q = [], [], [], [], []
    n_stage1 = 0
    if model2 is not None:
        import tempfile
        from features import _street_raw
        # --save_test keeps pass-1 features + record texts in the output, so stage 4 can be retrained/re-applied
        # later on a CPU session without redoing the 4-hour test blocking
        tmp = out / "test_parts" if a.save_test else Path(tempfile.mkdtemp(prefix="v5_stage4_"))
        tmp.mkdir(parents=True, exist_ok=True)
        chunk_files, text_files = [], []
    for s in (2, 3):
        # read each source in chunks: holding a whole 5M-row source next to the sibling data ran out of RAM (v13)
        n_src = 0
        reader = pd.read_csv(Path(a.data) / "test" / f"test_source{s}.tsv", sep="\t", dtype=str,
                             keep_default_na=False, chunksize=a.chunk)
        for q in reader:
            i = n_src
            q = q.reset_index(drop=True)
            n_src += len(q)
            all_qids.append(q["entity_id"].to_numpy())
            add_text_columns(q)
            if model2 is not None:  # per-record texts for stage 4 go to disk, not RAM
                f_ = tmp / f"text_{len(text_files)}.parquet"
                pd.DataFrame({"h": q["business_address"].map(lambda x: (_street_raw(x)[0][0] if _street_raw(x) else "")),
                              "n": q["c_name"], "a": q["n_addr"], "s3": s == 3}).to_parquet(f_, index=False)
                text_files.append(f_)
            e5 = (te_e5_s1, e5_encode(model_e5, e5_text(q))) if model_e5 is not None else None
            p = candidates(views, te_s1, q, a.k, s1_cache=te_cache, verbose=False, fetch=a.fetch, e5=e5,
                           block_fields=a.block_fields)
            n_stage1 += len(p)
            p = prune(p, pruner.predict(p[pcols].to_numpy(np.float32)).astype(np.float32), a.prune_top, a.prune_p)
            if len(p) == 0:  # every candidate pruned (e.g. a chunk of pure decoys): nothing to score
                log(f"test source{s} rows {i}-{i + len(q)}: no candidates after pruning")
                continue
            Xt = pair_features(q, te_s1, p)[feat_cols].to_numpy(np.float32)
            prob = model.predict(Xt).astype(np.float32)
            qi, si = p["q_idx"].to_numpy(), p["s1_idx"].to_numpy()
            c_s1.append(si.astype(np.int32))
            c_q.append((qi + offset + i).astype(np.int64))
            if model2 is not None:  # stage 4 needs every chunk's pairs before sibling groups are known
                f_ = tmp / f"chunk_{len(chunk_files)}.npz"
                np.savez(f_, X=Xt, q=(qi + offset + i).astype(np.int64), s=si.astype(np.int32), p=prob)
                chunk_files.append(f_)
                log(f"test source{s} rows {i}-{i + len(q)}: pass-1 pairs={len(p)} ({rss()})")
                del q, p, prob, Xt
                continue
            bq, bs = best_per_query(qi, si, prob, -1.0)
            best_p = pd.Series(prob).groupby(qi).max().reindex(bq).to_numpy(np.float32)
            b_s1.append(bs.astype(np.int32))
            b_q.append((bq + offset + i).astype(np.int64))
            b_p.append(best_p)
            log(f"test source{s} rows {i}-{i + len(q)}: pruned pairs={len(p)} matched@t={int((best_p > best_t).sum())} ({rss()})")
            del q, p, prob, Xt
        offset += n_src
    all_qids = np.concatenate(all_qids)
    if model2 is not None:
        # stage 4 test pass: sibling groups from pass-1 probabilities over ALL test records, then re-score
        # blocking indexes and text caches are no longer needed: free them before the sibling features
        import gc
        import features
        del te_cache, te_e5_s1, model_e5, views
        features.clear_caches()
        gc.collect()
        log(f"freed blocking indexes before stage 4 ({rss()})")
        from siblings import sibling_stats, sibling_stats_v2
        parts = [np.load(f_) for f_ in chunk_files]
        PQ = np.concatenate([z["q"] for z in parts])
        PS = np.concatenate([z["s"] for z in parts])
        P1 = np.concatenate([z["p"] for z in parts])
        del parts
        txt = pd.concat([pd.read_parquet(f_) for f_ in text_files], ignore_index=True)
        assert len(txt) == len(all_qids), (len(txt), len(all_qids))
        qh, qn, qa, q3 = (txt[c].to_numpy() for c in ("h", "n", "a", "s3"))
        del txt
        log(f"stage 4: record texts loaded ({rss()})")
        if a.sib_v2:
            S = sibling_stats_v2(PQ, PS, P1, qh, qn, qa, q3, te_s1["c_name"].to_numpy())
        else:
            S = sibling_stats(PQ, PS, P1, qh, qn, qa)
        log(f"stage 4: sibling features for {len(PQ)} test pairs")
        np.savez_compressed(out / "test_best_stage3.npz", **dict(zip(("s1_pos", "q_code", "prob"),
                            (lambda bq, bs: (bs, bq, pd.Series(P1).groupby(PQ).max().reindex(bq).to_numpy(np.float32)))(
                                *best_per_query(PQ, PS, P1, -1.0)))))
        P2, row = np.empty(len(PQ), np.float32), 0
        for f_ in chunk_files:
            z = np.load(f_)
            n = len(z["q"])
            P2[row:row + n] = model2.predict(np.hstack([z["X"], S[row:row + n], z["p"][:, None]]))
            row += n
        bq, bs = best_per_query(PQ, PS, P2, -1.0)
        b_q = [bq.astype(np.int64)]
        b_s1 = [bs.astype(np.int32)]
        b_p = [pd.Series(P2).groupby(PQ).max().reindex(bq).to_numpy(np.float32)]
        log(f"stage 4 done: matched@t={int((b_p[0] > best_t).sum())}")
        # pair-level probabilities for proper ensembles across runs (same query may pick different S1 per run)
        np.savez_compressed(out / "test_pairs.npz", q=PQ, s=PS.astype(np.int32), p1=P1, p2=P2)

    def write(fname, col, s1a, qa):
        """Write one output TSV (one row per S1 entity, comma-separated S2/S3 ids); returns (non-empty rows, pairs)."""
        order = np.lexsort((qa, s1a))
        s1a, qa = s1a[order], qa[order]
        keep = np.r_[True, (s1a[1:] != s1a[:-1]) | (qa[1:] != qa[:-1])]
        s1a, qa = s1a[keep], qa[keep]
        bounds = np.searchsorted(s1a, np.arange(len(te_ids) + 1))
        with open(out / fname, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"source1_entity_id\t{col}\n")
            for j, e in enumerate(te_ids):
                fh.write(f"{e}\t{','.join(all_qids[qa[bounds[j]:bounds[j + 1]]])}\n")
        return int((np.diff(bounds) > 0).sum()), len(s1a)

    n_c, n_pairs = write("candidate_pairs.tsv", "candidate_entity_ids", np.concatenate(c_s1), np.concatenate(c_q))
    log(f"candidate_pairs: {n_pairs} pairs = {n_pairs / len(te_ids):.2f} per S1 entity "
        f"(stage 1 had {n_stage1 / len(te_ids):.1f}); S1 with candidates={n_c}")
    b_s1, b_q, b_p = np.concatenate(b_s1), np.concatenate(b_q), np.concatenate(b_p)
    np.savez_compressed(out / "test_best.npz", s1_pos=b_s1, q_code=b_q, prob=b_p)
    n_m, _ = write("matching_results.tsv", "matched_entity_ids", b_s1[b_p > best_t], b_q[b_p > best_t])
    log(f"matching_results: S1 with matches={n_m} of {len(te_ids)} (t={best_t})")
    for t in a.extra_t:
        keep = b_p > t
        n, _ = write(f"matching_results_t{t}.tsv", "matched_entity_ids", b_s1[keep], b_q[keep])
        log(f"extra file t={t}: S1 with matches={n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--n_train", type=int, default=200000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--fetch", type=int, default=30)
    ap.add_argument("--e5", default="intfloat/multilingual-e5-small")
    ap.add_argument("--block_fields", nargs="*", default=["name_addr"], help='views used for blocking, "field" or "field:k"')
    ap.add_argument("--extra_views", nargs="*", default=[], help="optional extra TF-IDF views, e.g. n_addr")
    ap.add_argument("--prune_top", type=int, default=5)
    ap.add_argument("--prune_p", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pseudo_npz", default="", help="test_best.npz of a previous run -> French pseudo-labels")
    ap.add_argument("--pseudo_pos", type=float, default=0.95)
    ap.add_argument("--pseudo_neg", type=float, default=0.02)
    ap.add_argument("--pseudo_n", type=int, default=150000, help="max French pseudo-positive records")
    ap.add_argument("--lr", type=float, default=0.1, help="matcher LightGBM learning rate")
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--min_leaf", type=int, default=100)
    ap.add_argument("--ff", type=float, default=0.8, help="feature_fraction")
    ap.add_argument("--max_rounds", type=int, default=2000)
    ap.add_argument("--siblings", action="store_true", help="stage 4: sibling-consistency re-scoring")
    ap.add_argument("--sib_v2", action="store_true", help="stage 4 with the richer sibling_stats_v2 features")
    ap.add_argument("--orphan_frac", type=float, default=0.0,
                    help="drop this share of train S1 from the index and keep their records as no-match queries "
                         "(test has ~21%% fewer S1 than train for the same number of records)")
    ap.add_argument("--dense_bg", action="store_true",
                    help="train with ALL decoy/orphan records that land on sampled entities (test-like group density)")
    ap.add_argument("--save_test", action="store_true", help="keep test pass-1 features + texts in output/test_parts")
    ap.add_argument("--v14", action="store_true", help="add v14 pair features (learned address abbreviations etc.)")
    ap.add_argument("--save_train", action="store_true", help="write train_pairs.parquet (features+labels+OOF) for CPU experiments")
    ap.add_argument("--skip_test", action="store_true", help="stop after training (no test inference)")
    ap.add_argument("--dump_errors", action="store_true", help="write errors.tsv / error_counts.tsv from the OOF sample")
    ap.add_argument("--extra_t", type=float, nargs="*", default=[0.5, 0.6, 0.8, 0.85, 0.9])
    main(ap.parse_args())
