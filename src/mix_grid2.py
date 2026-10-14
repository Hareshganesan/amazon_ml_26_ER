"""CPU: 3-way mix on the raw e5-base dense-validation OOF: p = (1-wb-wc)*p_plus + wb*p_base + wc*ce (ce = CE score
alone, before stacking); first/extra rule 0.75/0.35; dense macro F0.5 on both halves.

    python src/mix_grid2.py ce_oof.npz train_truth.parquet
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
ref = None
res = []
for wb in (0.0, 0.1, 0.15, 0.2):
    for wc in (0.0, 0.05, 0.1, 0.2):
        p = ((1 - wb - wc) * z["p_plus"] + wb * z["p_base"] + wc * z["ce"]).astype(np.float32)
        bq, bs, bp = best_per_record(q, s, p)
        te = np.array([rec2ent.get(x, "") for x in bq])
        hm = [np.array([(t == "") or (t in tk) for t in te]) for tk, _ in halves]
        rank = pd.DataFrame({"s": bs, "p": bp}).groupby("s")["p"].rank(ascending=False, method="first").to_numpy() - 1
        keep = np.where(rank == 0, bp > 0.75, bp > 0.35)
        r = [eval_keep(bq, bs, keep & m, tk, rec2ent, ids) for (tk, ids), m in zip(halves, hm)]
        ref = ref or r
        res.append((r[0] - ref[0], r[1] - ref[1], wb, wc))
        print(f"wb={wb:.2f} wc={wc:.2f}: {r[0]:.5f} / {r[1]:.5f}  delta {r[0]-ref[0]:+.5f} / {r[1]-ref[1]:+.5f}", flush=True)
res.sort(key=lambda x: -min(x[0], x[1]))
print("BEST by worse half:", " | ".join(f"wb={wb} wc={wc}: {d0:+.5f}/{d1:+.5f}" for d0, d1, wb, wc in res[:5]))
