"""CPU: best first/extra thresholds for BLENDS of the CE-stacked and stage-3 probabilities
p = (1-w)*p_plus + w*p_base on the raw e5-base dense-validation OOF, per w; dense macro F0.5 on both halves.
Question: does a 50/50 blend (France recipe) need lower thresholds than the single-model 0.75/0.35?

    python src/blend_thresh.py ce_oof.npz train_truth.parquet
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
for w in (0.0, 0.5, 0.3, 0.7, 1.0):
    p = ((1 - w) * z["p_plus"] + w * z["p_base"]).astype(np.float32)
    bq, bs, bp = best_per_record(q, s, p)
    te = np.array([rec2ent.get(x, "") for x in bq])
    hm = [np.array([(t == "") or (t in tk) for t in te]) for tk, _ in halves]
    rank = pd.DataFrame({"s": bs, "p": bp}).groupby("s")["p"].rank(ascending=False, method="first").to_numpy() - 1
    sc = lambda t1, t2: [eval_keep(bq, bs, np.where(rank == 0, bp > t1, bp > t2) & m, tk, rec2ent, ids)
                         for (tk, ids), m in zip(halves, hm)]
    ref = sc(0.75, 0.35)
    res = []
    for t1 in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        for t2 in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45):
            r = sc(t1, t2)
            res.append((min(r[0] - ref[0], r[1] - ref[1]), r, t1, t2))
    res.sort(key=lambda x: -x[0])
    b = res[0]
    print(f"w={w:.1f}: rule 0.75/0.35 = {ref[0]:.5f}/{ref[1]:.5f} | best first>{b[2]} extra>{b[3]} = {b[1][0]:.5f}/{b[1][1]:.5f} "
          f"(+{b[1][0]-ref[0]:.5f}/+{b[1][1]-ref[1]:.5f}) | next: " +
          ", ".join(f"{t1}/{t2} {min(r[0]-ref[0], r[1]-ref[1]):+.5f}" for _, r, t1, t2 in res[1:4]), flush=True)
