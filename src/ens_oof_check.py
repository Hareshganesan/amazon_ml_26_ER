"""CPU: does averaging two CE-stacked models' dense-validation predictions beat the better one alone?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fopt_probe import best_per_record, eval_keep  # noqa: E402

a, b = np.load(sys.argv[1], allow_pickle=True), np.load(sys.argv[2], allow_pickle=True)
tt = pd.read_parquet(sys.argv[3])
truth = {s: set(m.split(",")) - {""} for s, m in zip(tt["s1_id"], tt["matched"])}
rec2ent = {r: s for s, v in truth.items() for r in v}
ids = np.array(list(truth.keys()))
assert (a["q"] == b["q"]).all() and (a["s"] == b["s"]).all()
q, s = a["q"], a["s"]
for name, p in (("A (base)", a["p_plus"]), ("B (small)", b["p_plus"]), ("mean A+B", (a["p_plus"] + b["p_plus"]) / 2),
                ("0.7A+0.3B", 0.7 * a["p_plus"] + 0.3 * b["p_plus"])):
    bq, bs, bp = best_per_record(q, s, p)
    rank = pd.DataFrame({"s": bs, "p": bp}).groupby("s")["p"].rank(ascending=False, method="first").to_numpy() - 1
    flat = max(eval_keep(bq, bs, bp > t, truth, rec2ent, ids) for t in (0.55, 0.6, 0.65))
    rule = eval_keep(bq, bs, np.where(rank == 0, bp > 0.75, bp > 0.35), truth, rec2ent, ids)
    print(f"{name:10s} best flat {flat:.5f} | rule 0.75/0.35 {rule:.5f}", flush=True)
