"""Local (CPU): on the raw e5-base dense-validation OOF (ce_oof.npz) test
(1) mixing the no-CE stage-3 probability into the CE-stacked one: p = (1-w) * p_plus + w * p_base, and
(2) a finer first/extra(/third) threshold grid, each scored with the dense macro F0.5 on both halves of the entities.

    python src/mix_rule_local.py runs/amz-ce-raw-base/output/ce_oof.npz runs/v18t_dl/output/train_truth.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fopt_probe import best_per_record, eval_keep  # noqa: E402

z = np.load(sys.argv[1], allow_pickle=True)
tt = pd.read_parquet(sys.argv[2])
truth_all = {s: set(m.split(",")) - {""} for s, m in zip(tt["s1_id"], tt["matched"])}
rec2ent = {r: s for s, v in truth_all.items() for r in v}
q, s, y, fold = z["q"], z["s"], z["y"], z["fold"]
ent_fold = pd.Series(fold[y == 1], index=s[y == 1]).groupby(level=0).first()
halves = []
for k in (0, 1):
    tk = {e: v for e, v in truth_all.items() if ent_fold.get(e, int(e.split("-")[-1]) % 2) == k}
    halves.append((tk, np.array(list(tk.keys()))))


def prep(p):
    """Best candidate per record, per-half truth masks and each record's rank within its S1 entity for probabilities p."""
    bq, bs, bp = best_per_record(q, s, p)
    te = np.array([rec2ent.get(x, "") for x in bq])
    hm = [np.array([(t == "") or (t in tk) for t in te]) for tk, _ in halves]
    rank = pd.DataFrame({"s": bs, "p": bp}).groupby("s")["p"].rank(ascending=False, method="first").to_numpy() - 1
    return bq, bs, bp, hm, rank


def score(bq, bs, hm, keep):
    """Dense macro F0.5 of a keep-mask on each of the two entity halves."""
    return [eval_keep(bq, bs, keep & m, tk, rec2ent, ids) for (tk, ids), m in zip(halves, hm)]


def rule(bp, rank, t1, t2, t3=None):
    """First/second/third+ thresholds by the record's rank within its S1 entity; returns the keep-mask."""
    t3 = t2 if t3 is None else t3
    return np.where(rank == 0, bp > t1, np.where(rank == 1, bp > t2, bp > t3))


# (1) mixing weight, with the current rule 0.75 / 0.35
ref = None
for w in (0.0, 0.1, 0.2, 0.3, 0.5):
    p = ((1 - w) * z["p_plus"] + w * z["p_base"]).astype(np.float32)
    bq, bs, bp, hm, rank = prep(p)
    r = score(bq, bs, hm, rule(bp, rank, 0.75, 0.35))
    ref = ref or r
    print(f"MIX w={w:.1f}: halves {r[0]:.5f} / {r[1]:.5f}   delta vs w=0: {r[0]-ref[0]:+.5f} / {r[1]-ref[1]:+.5f}", flush=True)

# (2) finer rule grid on the CE-stacked probability (w = 0)
bq, bs, bp, hm, rank = prep(z["p_plus"].astype(np.float32))
base = score(bq, bs, hm, rule(bp, rank, 0.75, 0.35))
res = []
for t1 in (0.65, 0.7, 0.75, 0.8, 0.85):
    for t2 in (0.25, 0.3, 0.35, 0.4, 0.45):
        for t3 in (None, t2 + 0.1, t2 + 0.2):
            r = score(bq, bs, hm, rule(bp, rank, t1, t2, t3))
            res.append((r[0] - base[0], r[1] - base[1], t1, t2, t3))
res.sort(key=lambda x: -min(x[0], x[1]))
print("RULE grid, best by worse half (delta vs 0.75/0.35):", flush=True)
for d0, d1, t1, t2, t3 in res[:10]:
    print(f"  first>{t1} second>{t2} third+>{t3 if t3 is not None else t2:.2f}: {d0:+.5f} / {d1:+.5f}", flush=True)
