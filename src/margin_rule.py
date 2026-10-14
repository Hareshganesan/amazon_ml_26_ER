"""CPU: margin rules on the raw e5-base dense-validation OOF (ce_oof.npz), with the validated 10% stage-3 mix.

Per record: p1 = best S1 probability, p2 = second best. Current decision: first/extra rule 0.75 / 0.35.
(a) abstain: reject when p1 - p2 < gap and p1 < cap (ambiguous record);
(b) clear winner: also accept when p1 > lo and p2 < 0.02 (no competitor).
Scored with the dense macro F0.5 on both halves of the sampled entities.

    python src/margin_rule.py ce_oof.npz train_truth.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fopt_probe import eval_keep  # noqa: E402

z = np.load(sys.argv[1], allow_pickle=True)
tt = pd.read_parquet(sys.argv[2])
truth_all = {s: set(m.split(",")) - {""} for s, m in zip(tt["s1_id"], tt["matched"])}
rec2ent = {r: s for s, v in truth_all.items() for r in v}
q, s, y, fold = z["q"], z["s"], z["y"], z["fold"]
p = (0.9 * z["p_plus"] + 0.1 * z["p_base"]).astype(np.float32)
ent_fold = pd.Series(fold[y == 1], index=s[y == 1]).groupby(level=0).first()
halves = []
for k in (0, 1):
    tk = {e: v for e, v in truth_all.items() if ent_fold.get(e, int(e.split("-")[-1]) % 2) == k}
    halves.append((tk, np.array(list(tk.keys()))))

o = np.lexsort((-p, q))
qo, so, po = q[o], s[o], p[o]
first = np.r_[True, qo[1:] != qo[:-1]]
pos = np.flatnonzero(first)
bq, bs, bp = qo[pos], so[pos], po[pos]
nxt = pos + 1
has2 = (nxt < len(qo)) & (np.r_[qo, [None]][np.minimum(nxt, len(qo))] == bq)
p2 = np.where(has2, po[np.minimum(nxt, len(qo) - 1)], 0.0).astype(np.float32)
te = np.array([rec2ent.get(x, "") for x in bq])
hm = [np.array([(t == "") or (t in tk) for t in te]) for tk, _ in halves]
rank = pd.DataFrame({"s": bs, "p": bp}).groupby("s")["p"].rank(ascending=False, method="first").to_numpy() - 1
keep0 = np.where(rank == 0, bp > 0.75, bp > 0.35)


def sc(keep):
    """Dense macro F0.5 of a keep-mask on each of the two entity halves."""
    return [eval_keep(bq, bs, keep & m, tk, rec2ent, ids) for (tk, ids), m in zip(halves, hm)]


base = sc(keep0)
print(f"base (mix 10%, rule 0.75/0.35): {base[0]:.5f} / {base[1]:.5f}; records {len(bq):,}, "
      f"with a 2nd candidate {has2.mean():.2%}", flush=True)
res = []
for gap in (0.05, 0.1, 0.2, 0.3):
    for cap in (0.8, 0.9, 0.97):
        k = keep0 & ~((bp - p2 < gap) & (bp < cap))
        r = sc(k)
        res.append((r[0] - base[0], r[1] - base[1], f"abstain gap<{gap} p1<{cap}", int((keep0 & ~k).sum())))
for lo in (0.2, 0.3, 0.4, 0.5, 0.6):
    for p2max in (0.01, 0.02, 0.05):
        k = keep0 | ((bp > lo) & (p2 < p2max))
        r = sc(k)
        res.append((r[0] - base[0], r[1] - base[1], f"clear winner p1>{lo} p2<{p2max}", int((k & ~keep0).sum())))
res.sort(key=lambda x: -min(x[0], x[1]))
for d0, d1, name, n in res:
    print(f"  {name:34s} changed {n:6d}: {d0:+.5f} / {d1:+.5f}", flush=True)
