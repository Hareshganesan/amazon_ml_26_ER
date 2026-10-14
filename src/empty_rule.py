"""CPU: "no address + exact unique core name" rule, validated on DENSE training pairs, then located in the test pairs.

Segment (pair level, from the saved stage-3 features): addr_empty == 1 (either side has no address) and core_eq == 1
(same normalised core name) and the S1 core name is unique in its country (s1_name_cnt == log1p(1)).
Validation: stage-3 OOF (v18, dense) with the first/extra rule 0.75/0.35; extra acceptance of segment records with
p > t_low, scored with the dense macro F0.5 (eval_keep), overall, per country and per fold half.
Test: writes output/test_seg_pairs.npz (q, s of every test pair in the segment) for local application.

    python src/empty_rule.py --run <dir with train_pairs.parquet and test_parts/>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fopt_probe import eval_keep  # noqa: E402

META = ["q_id", "s1_id", "country", "y", "fold", "oof"]
UNIQ = np.float32(np.log1p(1))


def seg_mask(get):
    """Segment: record with no address whose core name equals the S1 core name and is unique among S1 entities."""
    return (get("addr_empty") == 1) & (get("core_eq") == 1) & np.isclose(get("s1_name_cnt"), UNIQ, atol=1e-4)


def best(q, p):
    """Row indices of each record's highest-probability candidate."""
    o = np.lexsort((-p, q))
    first = np.r_[True, q[o][1:] != q[o][:-1]]
    return o[first]


def main(a):
    """Measure the segment rule on dense training pairs (truth rate per probability band), then export test segment pairs."""
    run = Path(a.run)
    tp = next(run.rglob("train_pairs.parquet"))
    df = pd.read_parquet(tp)
    tt = pd.read_parquet(tp.parent / "train_truth.parquet")
    truth = {s: set(m.split(",")) - {""} for s, m in zip(tt["s1_id"], tt["matched"])}
    rec2ent = {r: s for s, v in truth.items() for r in v}
    ids = np.array(list(truth.keys()))
    feats = [c for c in df.columns if c not in META]
    print("features:", len(feats), "| segment columns present:",
          all(c in feats for c in ("addr_empty", "core_eq", "s1_name_cnt")), flush=True)
    seg_pair = seg_mask(lambda c: df[c].to_numpy())
    q, s, p, y = df["q_id"].to_numpy(), df["s1_id"].to_numpy(), df["oof"].to_numpy(np.float32), df["y"].to_numpy()
    idx = best(q, p)
    bq, bs, bp, by, seg = q[idx], s[idx], p[idx], y[idx], seg_pair[idx]
    ctry, fold = df["country"].to_numpy()[idx], df["fold"].to_numpy()[idx]
    rank = pd.DataFrame({"s": bs, "p": bp}).groupby("s")["p"].rank(ascending=False, method="first").to_numpy() - 1
    keep0 = np.where(rank == 0, bp > 0.75, bp > 0.35)
    print(f"records {len(bq):,}; segment records {seg.sum():,} ({seg.mean():.2%}); of them true best pair "
          f"{by[seg].mean():.3f}; accepted by the rule {keep0[seg].mean():.3f}", flush=True)
    for lo, hi in ((0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.75), (0.75, 1.01)):
        m = seg & (bp >= lo) & (bp < hi)
        mo = ~seg & (bp >= lo) & (bp < hi)
        print(f"  p in [{lo},{hi}): segment n={m.sum():6d} true={by[m].mean() if m.any() else 0:.3f} | "
              f"non-segment n={mo.sum():7d} true={by[mo].mean() if mo.any() else 0:.3f}", flush=True)
    for c in ("US", "India"):
        m = seg & (ctry == c)
        print(f"  {c}: segment {m.sum():,} records, true {by[m].mean():.3f}, accepted {keep0[m].mean():.3f}")
    halves = {"all": np.ones(len(bq), bool), "fold0": fold == 0, "fold1": fold == 1,
              "US": ctry == "US", "India": ctry == "India"}
    for name, h in halves.items():
        hid = np.unique(bs[h]) if name != "all" else ids
        hid = np.intersect1d(hid, ids)
        th = truth if name == "all" else {k: truth[k] for k in hid}
        base = eval_keep(bq[h], bs[h], keep0[h], th, rec2ent, hid)
        row = [f"base {base:.5f}"]
        for tl in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
            keep1 = keep0 | (seg & (bp > tl))
            row.append(f"t_low {tl}: {eval_keep(bq[h], bs[h], keep1[h], th, rec2ent, hid) - base:+.5f}")
        print(f"DENSE {name:6s} " + " | ".join(row), flush=True)

    # ---- test: every test pair in the segment ----
    parts = next(run.rglob("test_parts"))
    num = lambda f: int(f.stem.split("_")[1])
    fi = {c: feats.index(c) for c in ("addr_empty", "core_eq", "s1_name_cnt")}
    Q, S = [], []
    for f in sorted(parts.glob("chunk_*.npz"), key=num):
        z = np.load(f)
        X = z["X"]
        m = seg_mask(lambda c: X[:, fi[c]])
        Q.append(z["q"][m]); S.append(z["s"][m])
    Q, S = np.concatenate(Q), np.concatenate(S)
    Path("output").mkdir(exist_ok=True)
    np.savez_compressed("output/test_seg_pairs.npz", q=Q.astype(np.int64), s=S.astype(np.int32))
    print(f"test segment pairs: {len(Q):,} (records {len(np.unique(Q)):,}) -> output/test_seg_pairs.npz", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    main(ap.parse_args())
