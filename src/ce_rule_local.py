"""Local (CPU, laptop): first/extra thresholds for the +CE model, on its saved dense-validation OOF (ce_oof.npz),
checked on both halves of the sampled entities (as rule2_probe.py)."""
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
for key in ("p_plus", "p_base"):
    bq, bs, bp = best_per_record(q, s, z[key])
    te = np.array([rec2ent.get(x, "") for x in bq])
    hm = [np.array([(t == "") or (t in tk) for t in te]) for tk, _ in halves]
    sc = lambda keep: [eval_keep(bq, bs, keep & m, tk, rec2ent, ids) for (tk, ids), m in zip(halves, hm)]
    rank = pd.DataFrame({"s": bs, "p": bp}).groupby("s")["p"].rank(ascending=False, method="first").to_numpy() - 1
    flat = {t: sc(bp > t) for t in (0.5, 0.55, 0.6, 0.65)}
    tb = max(flat, key=lambda t: sum(flat[t]))
    print(f"{key}: flat " + " ".join(f"{t}:{v[0]:.5f}/{v[1]:.5f}" for t, v in flat.items()) + f" | best {tb}", flush=True)
    rows = []
    for t1 in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
        for t2 in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
            h = sc(np.where(rank == 0, bp > t1, bp > t2))
            rows.append((t1, t2, h[0] - flat[tb][0], h[1] - flat[tb][1]))
    rows.sort(key=lambda r: -min(r[2], r[3]))
    for r in rows[:6]:
        print(f"   first>{r[0]:.2f} extra>{r[1]:.2f}: {r[2]:+.5f} {r[3]:+.5f}", flush=True)
