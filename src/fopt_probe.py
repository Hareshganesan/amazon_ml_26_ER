"""CPU: F0.5-optimal per-entity decisions vs flat thresholds, on DENSE (test-like) validation.

Probability source = the ensemble we can also build on the test: best-per-record vote average of the saved stage-3
matchers (leak-free members only). Calibration (isotonic, record-level "is the best S1 the true one") is fitted on
fold 0 and applied to fold 1 and vice versa, so the score is out-of-sample. Also writes calib.json (fitted on all
records) for applying the rule to the test ensemble locally (fopt_apply.py).

    python src/fopt_probe.py --dir <dense train_pairs dir> --models <dir with matcher_*.txt> --members v14c v15 v15b v16b
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
from fopt import calibrate, fopt_keep  # noqa: E402
from metric import score  # noqa: E402

T0 = time.time()


def log(m):
    """Print a progress message prefixed with the seconds elapsed since the script started."""
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def best_per_record(q, s, p):
    """Keep only each record's highest-probability candidate; returns (q, s1, p) arrays."""
    o = np.lexsort((-p, q))
    first = np.r_[True, q[o][1:] != q[o][:-1]]
    return q[o][first], s[o][first], p[o][first]


def eval_keep(bq, bs, keep, truth, rec2ent, ids, seed=1):
    """Dense evaluation: background false matches on out-of-sample S1 are dropped, true records' ones moved."""
    in_sample = set(ids)
    rng = np.random.default_rng(seed)
    pred = {}
    for q, s in zip(bq[keep], bs[keep]):
        if s not in in_sample:
            if q not in rec2ent:
                continue
            s = ids[rng.integers(len(ids))]
        pred.setdefault(s, set()).add(q)
    return score(pred, truth)


def main(a):
    """Compare decision rules (threshold, first/extra, F0.5-optimal) on saved dense OOF predictions."""
    d = Path(a.dir)
    df = pd.read_parquet(d / "train_pairs.parquet")
    tt = pd.read_parquet(d / "train_truth.parquet")
    truth = {s: set(m.split(",")) - {""} for s, m in zip(tt["s1_id"], tt["matched"])}
    rec2ent = {r: s for s, v in truth.items() for r in v}
    ids = np.array(list(truth.keys()))
    q, s = df["q_id"].to_numpy(), df["s1_id"].to_numpy()
    fold_of = df.groupby("q_id")["fold"].first()
    sources = {"dense-own": df["oof"].to_numpy(np.float32)}
    mem = []
    for name in a.members:
        f = next(Path(a.models).rglob(f"matcher_{name}.txt"))
        m = lgb.Booster(model_file=str(f))
        mem.append(best_per_record(q, s, m.predict(df[m.feature_name()].to_numpy(np.float32)).astype(np.float32)))
    e = pd.concat([pd.DataFrame({"q": bq, "s": bs, "p": bp}) for bq, bs, bp in mem])
    g = e.groupby(["q", "s"], sort=False)["p"].sum().reset_index()
    g["p"] /= len(mem)
    log(f"{len(df)} pairs; ensemble of {a.members}")
    calib = {}
    for name, (bq, bs, bp) in {"ENSEMBLE": best_per_record(g["q"].to_numpy(), g["s"].to_numpy(),
                                                           g["p"].to_numpy(np.float32)),
                               "dense-own": best_per_record(q, s, sources["dense-own"])}.items():
        correct = np.array([rec2ent.get(x, "") == y for x, y in zip(bq, bs)], dtype=float)
        fo = fold_of.reindex(bq).to_numpy()
        qcal = np.zeros(len(bp))
        for k in (0, 1):
            iso = calibrate(bp[fo != k], correct[fo != k])
            qcal[fo == k] = iso.predict(bp[fo == k])
        iso_all = calibrate(bp, correct)
        calib[name] = {"x": iso_all.X_thresholds_.tolist(), "y": iso_all.y_thresholds_.tolist()}
        flat = {t: eval_keep(bq, bs, bp > t, truth, rec2ent, ids) for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)}
        tb = max(flat, key=flat.get)
        log(f"{name}: flat thresholds " + " ".join(f"{t}:{v:.5f}" for t, v in flat.items()) + f" | best t={tb}")
        for miss in (0.0, 0.03, 0.1, 0.3):
            keep = fopt_keep(bs, qcal, miss=miss)
            sc = eval_keep(bq, bs, keep, truth, rec2ent, ids)
            log(f"{name}: F0.5-optimal (miss={miss}) {sc:.5f} ({sc - flat[tb]:+.5f} vs best flat), "
                f"kept {int(keep.sum())} vs flat@{tb} {int((bp > tb).sum())}")
    Path("output").mkdir(exist_ok=True)
    json.dump(calib, open("output/calib.json", "w"))
    log("wrote output/calib.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--members", nargs="+", default=["v14c", "v15", "v15b", "v16b"])
    main(ap.parse_args())
