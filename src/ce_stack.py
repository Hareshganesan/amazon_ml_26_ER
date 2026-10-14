"""Cross-encoder score as an extra stage-3 feature (GPU), trained and judged on DENSE (test-like) data.

A small multilingual transformer (intfloat/multilingual-e5-small, MIT) reads the record text and the S1 text
together ("core name | address" on both sides) and predicts "same business?". Hand-made features compare strings
with fixed rules; a cross-encoder can learn which differences are typos and which are decoy words.

Steps (all on one run's output with --save_train and --save_test, e.g. v18):
  1. 2-fold out-of-fold cross-encoder scores on the dense training pairs (same folds as the pipeline).
  2. Stage-3 LightGBM with and without the CE feature, 2-fold OOF, dense evaluation -> the honest gain.
  3. If --test: a CE model trained on all training pairs scores the saved test pairs; the final LightGBM (+CE)
     re-scores them -> output/test_best_ce.npz (best S1 per record) for the usual local submission build.

    python src/ce_stack.py --run <dir of a --save_train/--save_test run> --data $DATA [--test] [--mock]
"""
import argparse
import gc
import re
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import add_text_columns  # noqa: E402
from fopt_probe import best_per_record, eval_keep  # noqa: E402

T0 = time.time()
META = ["q_id", "s1_id", "country", "y", "fold", "oof"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)


def log(m):
    """Print a progress message prefixed with the seconds elapsed since the script started."""
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


class CE:
    """Cross-encoder wrapper (mock=True: random scores, to test the plumbing without torch)."""

    def __init__(self, a):
        """Load the tokenizer and model class (skipped in --mock mode, which returns random scores for dry runs)."""
        self.a, self.mock = a, a.mock
        if not self.mock:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.torch = torch
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.tok = AutoTokenizer.from_pretrained(a.model)
            self.Auto = AutoModelForSequenceClassification

    def train(self, ta, tb, y):
        """Fine-tune the model on text pairs (ta, tb) with labels y: AdamW, linear warm-up, fp16 on GPU; returns the model."""
        if self.mock:
            return None
        torch, a = self.torch, self.a
        from transformers import get_linear_schedule_with_warmup
        model = self.Auto.from_pretrained(a.model, num_labels=1).to(self.dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        n = len(y)
        steps = a.epochs * ((n + a.bs - 1) // a.bs)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        scaler = torch.cuda.amp.GradScaler(enabled=self.dev == "cuda")
        rng = np.random.default_rng(0)
        model.train()
        yt = torch.tensor(y, dtype=torch.float32)
        for ep in range(a.epochs):
            perm = rng.permutation(n)
            for i in range(0, n, a.bs):
                idx = perm[i:i + a.bs]
                enc = self.tok([ta[j] for j in idx], [tb[j] for j in idx], truncation=True, max_length=a.max_len,
                               padding=True, return_tensors="pt").to(self.dev)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
                    logits = model(**enc).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), yt[idx].to(self.dev))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                if (i // a.bs) % 500 == 0:
                    log(f"  CE epoch {ep} step {i // a.bs}/{steps // a.epochs} loss {loss.item():.4f}")
        model.eval()
        return model

    def predict(self, model, ta, tb):
        """Score text pairs in batches; returns sigmoid probabilities as float32."""
        if self.mock:
            return np.random.default_rng(len(ta)).random(len(ta)).astype(np.float32)
        torch, a = self.torch, self.a
        out = np.empty(len(ta), np.float32)
        with torch.no_grad():
            for i in range(0, len(ta), a.pbs):
                enc = self.tok(ta[i:i + a.pbs], tb[i:i + a.pbs], truncation=True, max_length=a.max_len,
                               padding=True, return_tensors="pt").to(self.dev)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
                    out[i:i + len(enc["input_ids"])] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
        return out


TEXT_MODE = "norm"  # set from --text_mode: norm = "core name | address" (normalised), raw = original strings


FR_SUFFIX = ["& Fils", "& Associés", "Et Fils", "(France)", "& Cie", "Cie"]
LEGAL_WORDS = {"llc", "inc", "ltd", "pvt", "corp", "co", "llp", "limited", "private", "plc", "lp"}


def fr_noise(t, rng):
    """French-style generator noise seen in the (unlabelled) French test records, applied to a raw training text
    "name | address": drop a non-first name word, append a family/partner suffix, move the legal form to the front,
    drop the last address component (region), shuffle the address components."""
    name, _, addr = t.partition(" | ")
    w = name.split()
    r = rng.random(5)
    if r[0] < 0.2 and len(w) > 2:
        del w[int(rng.integers(1, len(w)))]
    if r[1] < 0.2:
        w += FR_SUFFIX[int(rng.integers(len(FR_SUFFIX)))].split()
    if r[2] < 0.1 and len(w) > 1 and w[-1].lower().strip(".") in LEGAL_WORDS:
        w = [w[-1]] + w[:-1]
    parts = [x.strip() for x in addr.split(",") if x.strip()]
    if r[3] < 0.3 and len(parts) > 1:
        parts = parts[:-1]
    if r[4] < 0.3 and len(parts) > 1:
        parts = [parts[i] for i in rng.permutation(len(parts))]
    return " ".join(w) + " | " + ", ".join(parts)


def text(df):
    """Cross-encoder input text 'name | address' per row: original strings in raw mode, normalised core name and address otherwise."""
    if TEXT_MODE == "raw":
        return (df["business_name"] + " | " + df["business_address"]).tolist()
    return (df["c_name"] + " | " + df["n_addr"]).tolist()


def oof_lgb(X, y, fold):
    """2-fold out-of-fold LightGBM probabilities: train on one fold (early stopping on the other), predict it, both ways."""
    o = np.zeros(len(y), np.float32)
    for k in (0, 1):
        tr, va = fold != k, fold == k
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), 4000, valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        o[va] = m.predict(X[va], num_iteration=m.best_iteration)
    return o


def dense_scores(df, p, truth, rec2ent, ids):
    """Dense-validation macro F0.5 of the one-to-one decision (best S1 per record, p > t) for a grid of thresholds."""
    bq, bs, bp = best_per_record(df["q_id"].to_numpy(), df["s1_id"].to_numpy(), p)
    return {t: eval_keep(bq, bs, bp > t, truth, rec2ent, ids) for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)}


def main(a):
    """Run the cross-encoder stage on one pipeline run.

    Loads the dense training pairs, fine-tunes the cross-encoder (2-fold OOF, or --single_fold), stacks its score as an
    extra LightGBM feature, reports dense F0.5 with and without it, and with --test re-scores the saved test pairs and
    writes output/test_best_ce.npz (best S1 per record and its probability).
    """
    global TEXT_MODE
    TEXT_MODE = a.text_mode
    run = Path(a.run)
    tp = next(run.rglob("train_pairs.parquet"))
    df = pd.read_parquet(tp)
    tt = pd.read_parquet(tp.parent / "train_truth.parquet")
    truth = {s: set(m.split(",")) - {""} for s, m in zip(tt["s1_id"], tt["matched"])}
    rec2ent = {r: s for s, v in truth.items() for r in v}
    ids = np.array(list(truth.keys()))
    if a.limit:
        keep_q = set(df["q_id"].drop_duplicates().sample(n=min(a.limit, df["q_id"].nunique()), random_state=0))
        df = df[df["q_id"].isin(keep_q)].reset_index(drop=True)
    tr = Path(a.data) / "train"
    need_q, need_s = set(df["q_id"]), set(df["s1_id"])
    recs = pd.concat([pd.read_csv(tr / f"train_source{s}.tsv", sep="\t", dtype=str, keep_default_na=False)
                      .query("entity_id in @need_q") for s in (2, 3)], ignore_index=True)
    s1 = pd.read_csv(tr / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False).query("entity_id in @need_s")
    add_text_columns(recs)
    add_text_columns(s1)
    rt = dict(zip(recs["entity_id"], text(recs)))
    st = dict(zip(s1["entity_id"], text(s1)))
    ta, tb = [rt[x] for x in df["q_id"]], [st[x] for x in df["s1_id"]]
    if a.fr_aug:  # French-style noise on the record side of every training pair (label-free augmentation)
        arng = np.random.default_rng(7)
        ta = [fr_noise(t, arng) for t in ta]
        log(f"fr_aug: record texts noised, e.g. {ta[0]!r} / {ta[1]!r}")
    y, fold = df["y"].to_numpy(), df["fold"].to_numpy()
    log(f"{len(df)} training pairs, texts ready, positives {y.mean():.3f}")
    ce = CE(a)
    rng = np.random.default_rng(1)
    if a.single_fold:
        # Cheap test run (GPU budget): ONE CE model, trained on fold 0, scores fold 1 (-> training rows of the
        # fused LightGBM, whose CE feature is therefore out-of-sample) and then the test pairs. ~1/2 the CE cost.
        tr_idx = np.flatnonzero(fold == a.sf_fold)
        if len(tr_idx) > a.max_train:
            tr_idx = rng.choice(tr_idx, a.max_train, replace=False)
        m_all = ce.train([ta[i] for i in tr_idx], [tb[i] for i in tr_idx], y[tr_idx])
        va = np.flatnonzero(fold != a.sf_fold)
        ce_va = ce.predict(m_all, [ta[i] for i in va], [tb[i] for i in va])
        log(f"single-fold CE: trained on {len(tr_idx)}, scored {len(va)}; mean pos {ce_va[y[va] == 1].mean():.3f} "
            f"neg {ce_va[y[va] == 0].mean():.3f}")
        X0 = df[[c for c in df.columns if c not in META]].to_numpy(np.float32)
        X1v, yv = np.hstack([X0[va], ce_va[:, None]]), y[va]
        inner = pd.factorize(df["q_id"].to_numpy()[va])[0] % 5 == 0  # 20% of records for early stopping
        mm = lgb.train(PARAMS, lgb.Dataset(X1v[~inner], yv[~inner]), 4000,
                       valid_sets=[lgb.Dataset(X1v[inner], yv[inner])], callbacks=[lgb.early_stopping(50, verbose=False)])
        final = lgb.train(PARAMS, lgb.Dataset(X1v, yv), int(mm.best_iteration * 1.1))
        log(f"fused LightGBM on {len(va)} fold-1 pairs, {int(mm.best_iteration * 1.1)} rounds")
        # free the training data before the test pass (free Colab has 12.7 GB RAM)
        del df, recs, s1, rt, st, ta, tb, X0, X1v, mm
        gc.collect()
        score_test(a, run, ce, m_all, final)
        return
    ce_oof = np.zeros(len(df), np.float32)
    for k in (0, 1):
        tr_idx = np.flatnonzero(fold != k)
        if len(tr_idx) > a.max_train:
            tr_idx = rng.choice(tr_idx, a.max_train, replace=False)
        m = ce.train([ta[i] for i in tr_idx], [tb[i] for i in tr_idx], y[tr_idx])
        va = np.flatnonzero(fold == k)
        ce_oof[va] = ce.predict(m, [ta[i] for i in va], [tb[i] for i in va])
        log(f"CE fold {k}: trained on {len(tr_idx)}, scored {len(va)}; mean score pos {ce_oof[va][y[va] == 1].mean():.3f} "
            f"neg {ce_oof[va][y[va] == 0].mean():.3f}")
        del m
    X0 = df[[c for c in df.columns if c not in META]].to_numpy(np.float32)
    p_base = oof_lgb(X0, y, fold)
    base = dense_scores(df, p_base, truth, rec2ent, ids)
    X1 = np.hstack([X0, ce_oof[:, None]])
    p_plus = oof_lgb(X1, y, fold)
    plus = dense_scores(df, p_plus, truth, rec2ent, ids)
    Path("output").mkdir(exist_ok=True)
    np.savez_compressed("output/ce_oof.npz", q=df["q_id"].to_numpy(), s=df["s1_id"].to_numpy(), y=y, fold=fold,
                        p_base=p_base, p_plus=p_plus, ce=ce_oof)
    # consistency reference for the test gate: records accepted by +CE relative to the baseline, per threshold
    qq, ss = df["q_id"].to_numpy(), df["s1_id"].to_numpy()
    _, bs0, bp0 = best_per_record(qq, ss, p_base)
    _, bs1, bp1 = best_per_record(qq, ss, p_plus)
    for t in (0.5, 0.55, 0.6, 0.65, 0.7):
        n0, n1 = int((bp0 > t).sum()), int((bp1 > t).sum())
        log(f"accepted records @t={t}: baseline {n0}, +CE {n1} ({n1 / n0 - 1:+.2%}); entities covered "
            f"{len(np.unique(bs0[bp0 > t]))} -> {len(np.unique(bs1[bp1 > t]))}")
    bq1, bs1_, bp1_ = best_per_record(qq, ss, p_plus)
    rank = pd.DataFrame({"s": bs1_, "p": bp1_}).groupby("s")["p"].rank(ascending=False, method="first").to_numpy() - 1
    for t1 in (0.55, 0.6, 0.65, 0.7):
        row = []
        for t2 in (0.4, 0.45, 0.5, 0.55, 0.6):
            keep = np.where(rank == 0, bp1_ > t1, bp1_ > t2)
            row.append(f"{t2}:{eval_keep(bq1, bs1_, keep, truth, rec2ent, ids):.5f}")
        log(f"+CE first>{t1} extra: " + " ".join(row))
    alone = dense_scores(df, ce_oof, truth, rec2ent, ids)
    for name, r in (("stage3 baseline", base), ("stage3 + CE feature", plus), ("CE alone", alone)):
        b = max(r, key=r.get)
        log(f"DENSE {name:22s} best {r[b]:.5f} @ t={b}   " + " ".join(f"{t}:{v:.5f}" for t, v in r.items()))
    gain = max(plus.values()) - max(base.values())
    log(f"CE GAIN on dense validation: {gain:+.5f}")
    if not a.test:
        return
    # ---------- test: CE on all training pairs, final LightGBM (+CE), re-score the saved test pairs ----------
    idx = np.arange(len(df))
    if len(idx) > a.max_train_final:
        idx = rng.choice(idx, a.max_train_final, replace=False)
    m_all = ce.train([ta[i] for i in idx], [tb[i] for i in idx], y[idx])
    it = []
    for k in (0, 1):
        trk = fold != k
        mm = lgb.train(PARAMS, lgb.Dataset(X1[trk], y[trk]), 4000, valid_sets=[lgb.Dataset(X1[~trk], y[~trk])],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        it.append(mm.best_iteration)
    final = lgb.train(PARAMS, lgb.Dataset(X1, y), int(np.mean(it) * 1.1))
    score_test(a, run, ce, m_all, final)


def score_test(a, run, ce, m_all, final):
    """CE model m_all + fused LightGBM `final` re-score the run's saved test pairs -> output/test_best_ce.npz."""
    parts = next(run.rglob("test_parts"))
    num = lambda f: int(re.search(r"_(\d+)\.", f.name).group(1))
    chunks = sorted(parts.glob("chunk_*.npz"), key=num)
    if TEXT_MODE == "raw":  # original strings, in the pipeline's record order (source 2 then source 3)
        src = pd.concat([pd.read_csv(Path(a.data) / "test" / f"test_source{k}.tsv", sep="\t", dtype=str,
                                     keep_default_na=False, usecols=["business_name", "business_address"])
                         for k in (2, 3)], ignore_index=True)
        rec_text = (src["business_name"] + " | " + src["business_address"]).to_numpy()
        del src
    else:
        txt = pd.concat([pd.read_parquet(f) for f in sorted(parts.glob("text_*.parquet"), key=num)], ignore_index=True)
        rec_text = (txt["n"] + " | " + txt["a"]).to_numpy()
        del txt
    te_s1 = pd.read_csv(Path(a.data) / "test" / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    add_text_columns(te_s1)
    s1_text = np.array(text(te_s1))
    s1_ctry = te_s1["country"].to_numpy()
    del te_s1
    gc.collect()
    log(f"test: {len(chunks)} chunks, {len(rec_text)} records, CE on pairs with stage-3 p > {a.min_p}"
        + (f", only S1 in {a.test_country}" if a.test_country else ""))
    Q, S, P = [], [], []
    for n, f in enumerate(chunks[:a.test_chunks] if a.test_chunks else chunks):
        z = dict(np.load(f))
        if a.test_country:  # score only the pairs of one country (records never cross countries)
            keep = s1_ctry[z["s"]] == a.test_country
            z = {k: v[keep] for k, v in z.items()}
            if not len(z["q"]):
                log(f"  test chunk {n}: no {a.test_country} pairs")
                continue
        cs = np.zeros(len(z["q"]), np.float32)
        sel = np.flatnonzero(z["p"] > a.min_p)
        cs[sel] = ce.predict(m_all, rec_text[z["q"][sel]].tolist(), s1_text[z["s"][sel]].tolist())
        pn = final.predict(np.hstack([z["X"], cs[:, None]])).astype(np.float32)
        Q.append(z["q"]); S.append(z["s"]); P.append(pn)
        log(f"  test chunk {n}: {len(z['q'])} pairs, CE on {len(sel)}")
    Q, S, P = np.concatenate(Q), np.concatenate(S), np.concatenate(P)
    bq, bs, bp = best_per_record(Q, S, P)
    Path("output").mkdir(exist_ok=True)
    np.savez_compressed("output/test_best_ce.npz", s1_pos=bs.astype(np.int32), q_code=bq.astype(np.int64), prob=bp)
    log(f"wrote output/test_best_ce.npz: {len(bq)} records, accepted@0.6 {int((bp > 0.6).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--pbs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--max_train", type=int, default=400_000)
    ap.add_argument("--max_train_final", type=int, default=600_000)
    ap.add_argument("--min_p", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only this many training records")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--test_chunks", type=int, default=0, help="smoke test: only the first N test chunks")
    ap.add_argument("--mock", action="store_true", help="random CE scores (plumbing test without torch)")
    ap.add_argument("--text_mode", choices=["norm", "raw"], default="norm")
    ap.add_argument("--single_fold", action="store_true", help="cheap test run: one CE model (see main)")
    ap.add_argument("--sf_fold", type=int, default=0, help="single_fold: train the CE on this fold, stack on the other")
    ap.add_argument("--test_country", default="", help="score only test pairs whose S1 is in this country")
    ap.add_argument("--fr_aug", action="store_true", help="French-style noise on training record texts (fr_noise)")
    main(ap.parse_args())
