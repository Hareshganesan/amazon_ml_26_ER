"""Test-behaviour gate for a sibling (stage-4) run before it may be submitted (lesson of 26 Sep: LB 0.964).

Compares the run's stage-4 test decisions with its OWN stage-3 decisions, overall and per country:
  - accepted records: stage 4 may accept at most MAX_EXTRA (2%) more than stage 3
  - covered S1 entities: stage 4 must not cover fewer than stage 3
  - empty S1 share: must stay >= SINGLETON_FLOOR (5.6% = train singleton rate)
v14c (failed on the LB) accepted +4.2% more records at t=0.8 and covered FEWER entities -> it fails this gate.

    python src/gate_check.py --ids runs/amz-export-ids/output/test_ids.npz --run <run output dir> --t3 0.8 --t4 <run best_t>
"""
import argparse
from pathlib import Path

import numpy as np

MAX_EXTRA, SINGLETON_FLOOR = 0.02, 0.056


def stats(z, t, country, n_s1):
    """Per country and overall: accepted records at threshold t, S1 entities covered, share of S1 left empty."""
    acc = z["prob"] > t
    s = z["s1_pos"][acc]
    out = {}
    for c in ["ALL"] + sorted(set(country)):
        m = np.ones(len(s), bool) if c == "ALL" else country[s] == c
        n_c = n_s1 if c == "ALL" else int((country == c).sum())
        cov = len(np.unique(s[m]))
        out[c] = (int(m.sum()), cov, 1 - cov / n_c)
    return out


def main(a):
    """Test gate: compare a candidate model's test decisions with its own stage 3 and print PASS/FAIL per country."""
    ids = np.load(a.ids)
    country = ids["s1_country"]
    run = Path(a.run)
    z3 = np.load(next(run.rglob("test_best_stage3.npz")))
    z4 = np.load(next(run.rglob("test_best.npz")))
    s3, s4 = stats(z3, a.t3, country, len(country)), stats(z4, a.t4, country, len(country))
    ok = True
    print(f"{'country':8s} {'stage3 acc':>12s} {'stage4 acc':>12s} {'extra':>7s} {'cov3':>10s} {'cov4':>10s} {'empty4':>7s}  verdict")
    for c in s3:
        (a3, c3, e3), (a4, c4, e4) = s3[c], s4[c]
        extra = a4 / max(a3, 1) - 1
        fails = []
        if extra > MAX_EXTRA:
            fails.append(f"+{extra:.1%} records > {MAX_EXTRA:.0%}")
        if c4 < c3:
            fails.append(f"covers {c3 - c4:,} fewer S1")
        if e4 < SINGLETON_FLOOR:
            fails.append(f"empty {e4:.3f} < {SINGLETON_FLOOR}")
        ok &= not fails
        print(f"{c:8s} {a3:12,d} {a4:12,d} {extra:+7.2%} {c3:10,d} {c4:10,d} {e4:7.3f}  {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    print("GATE:", "PASS - may be submitted (plain threshold, no rescue)" if ok else "FAIL - do not submit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--t3", type=float, default=0.8)
    ap.add_argument("--t4", type=float, required=True)
    main(ap.parse_args())
