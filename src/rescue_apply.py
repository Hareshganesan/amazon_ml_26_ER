"""Apply the empty-entity rescue rule to a run's test predictions (test_best.npz: best S1 + probability per record).

Macro F0.5 scores an S1 entity with true records but no prediction as 0. Rule: accept every record whose best S1
has p > t; then every S1 entity still without a record takes its most probable record among the records whose best
S1 it is, if that probability > t_low (one record per entity). Validated on CPU (decision_probe.py,
rescue_probe.py). Candidate pairs are unchanged (rescued pairs are always candidates).

    python src/rescue_apply.py --data $DATA --npz <run>/test_best.npz --t 0.5 --t_low 0.1 --out output
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def best_per_s1(idx, s, p):
    """Among record positions idx, the most probable record of each S1."""
    order = idx[np.lexsort((-p[idx], s[idx]))]  # by S1, most probable record first
    return order[np.r_[True, s[order][1:] != s[order][:-1]]] if len(order) else order


def rescue(s, q, p, t, t_low, t_low2=None):
    """Boolean mask over records: accepted at t, plus one rescued record per otherwise-empty S1; with t_low2, S1
    entities with exactly one accepted record also take their best remaining record if p > t_low2
    (rescue2_probe.py rule r4: +0.0002..0.0003 on normal and test-like validation)."""
    acc = p > t
    n_acc = np.bincount(s[acc], minlength=s.max() + 1)
    first = best_per_s1(np.flatnonzero(~acc & (p > t_low) & (n_acc[s] == 0)), s, p)
    keep = acc.copy()
    keep[first] = True
    n2 = 0
    if t_low2 is not None:
        second = best_per_s1(np.flatnonzero(~keep & (p > t_low2) & (n_acc[s] == 1)), s, p)
        keep[second] = True
        n2 = len(second)
    return keep, len(first), n2


def main(a):
    """Apply the threshold (and optional rescue variants) to a test npz; write one submission folder per setting."""
    d = Path(a.data) / "test"
    rd = lambda f, c: pd.read_csv(d / f, sep="\t", dtype=str, keep_default_na=False, usecols=c)
    te_ids = rd("test_source1.tsv", ["entity_id"])["entity_id"].to_numpy()
    qids = np.concatenate([rd(f"test_source{s}.tsv", ["entity_id"])["entity_id"].to_numpy() for s in (2, 3)])
    z = np.load(a.npz)
    s, q, p = z["s1_pos"].astype(np.int64), z["q_code"].astype(np.int64), z["prob"].astype(np.float32)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for t_low in a.t_low:
        keep, n, n2 = rescue(s, q, p, a.t, t_low, a.t_low2)
        s1a, qa = s[keep], q[keep]
        order = np.lexsort((qa, s1a))
        s1a, qa = s1a[order], qa[order]
        bounds = np.searchsorted(s1a, np.arange(len(te_ids) + 1))
        name = f"matching_results_t{a.t}_rescue{t_low}" + (f"_second{a.t_low2}" if a.t_low2 else "") + ".tsv"
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("source1_entity_id\tmatched_entity_ids\n")
            for j, e in enumerate(te_ids):
                fh.write(f"{e}\t{','.join(qids[qa[bounds[j]:bounds[j + 1]]])}\n")
        print(f"{name}: accepted at t={int((p > a.t).sum())}, rescued={n}, second records={n2}, "
              f"S1 with matches={int((np.diff(bounds) > 0).sum())} of {len(te_ids)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--t_low", type=float, nargs="+", default=[0.1])
    ap.add_argument("--t_low2", type=float, default=None, help="second record for 1-record entities (rule r4)")
    ap.add_argument("--out", default="output")
    main(ap.parse_args())
