"""Build an ensemble submission locally from several runs' best-per-record predictions (no test data needed).

Inputs: test_ids.npz (export_ids.py: S1 ids, S2+S3 record ids in run order), each run's stage-3 predictions
(test_best_stage3.npz or a stage-3 run's test_best.npz: s1_pos, q_code, prob), one run's candidate_pairs.tsv.
Averaging as in ensemble.py: per record, each run votes its best S1 with its probability (0 for other S1s);
the S1 with the highest mean wins if the mean > t. Candidates = the base candidate file + any matched pair not
already in it (every match must be a candidate). Checks: row counts/headers, one S1 per record, matches ⊆ candidates.

    python src/local_ensemble.py --ids test_ids.npz --npz a.npz b.npz --cands base/candidate_pairs.tsv --t 0.8 --out dir
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def main(a):
    """Build a submission from saved predictions.

    Averages the given models per record, applies the decision rule (global threshold, or first/extra thresholds with
    optional France-specific values), adds any matched pair missing from the candidate file, writes
    matching_results.tsv and candidate_pairs.tsv, and checks row counts, headers, one S1 per record and that every
    match is a candidate.
    """
    ids = np.load(a.ids)
    s1_ids, q_ids = ids["s1"], ids["q"]
    frames = []
    for f in a.npz:
        z = np.load(f)
        frames.append(pd.DataFrame({"q": z["q_code"].astype(np.int64), "s": z["s1_pos"].astype(np.int64),
                                    "p": z["prob"].astype(np.float32)}))
        print(f"{f}: {len(z['q_code']):,} records, accepted@{a.t}: {int((z['prob'] > a.t).sum()):,}")
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby(["q", "s"], sort=False)["p"].sum().reset_index()
    g["p"] /= len(a.npz)
    g = g.sort_values(["q", "p"], ascending=[True, False]).drop_duplicates("q")
    np.savez_compressed(Path(a.out).parent / (Path(a.out).name + "_best.npz"), s1_pos=g["s"].to_numpy(np.int32),
                        q_code=g["q"].to_numpy(np.int64), prob=g["p"].to_numpy(np.float32))
    if a.t_france is not None:  # per-country threshold for France only (no labels there -> LB test)
        fr = ids["s1_country"][g["s"].to_numpy()] == "France"
        thr = np.where(fr, a.t_france, a.t)
        keep = g["p"].to_numpy() > thr
        acc = g[keep]
        print(f"France threshold {a.t_france} (others {a.t}): kept {int(keep.sum()):,}; France records "
              f"{int((keep & fr).sum()):,} vs {int(((g['p'].to_numpy() > a.t) & fr).sum()):,} at the global threshold")
    elif a.t_first is not None:  # first/extra thresholds (rule2_probe.py: +0.0003 on both halves of dense validation)
        rank = g.groupby("s")["p"].rank(ascending=False, method="first").to_numpy() - 1
        keep = np.where(rank == 0, g["p"].to_numpy() > a.t_first, g["p"].to_numpy() > a.t_extra)
        if a.fr_first is not None:  # France-specific first/extra thresholds (blended French probabilities)
            fr = ids["s1_country"][g["s"].to_numpy()] == "France"
            kf = np.where(rank == 0, g["p"].to_numpy() > a.fr_first, g["p"].to_numpy() > a.fr_extra)
            print(f"France first>{a.fr_first} extra>{a.fr_extra}: French records kept {int((kf & fr).sum()):,} "
                  f"(vs {int((keep & fr).sum()):,} with the global rule)")
            keep = np.where(fr, kf, keep)
        if a.seg_npz:  # empty_rule.py: also accept segment pairs (no address + exact unique core name) above seg_t_low
            z = np.load(a.seg_npz)
            key = z["q"].astype(np.int64) * 2_000_000 + z["s"].astype(np.int64)
            seg = np.isin(g["q"].to_numpy(np.int64) * 2_000_000 + g["s"].to_numpy(np.int64), key)
            if a.seg_country:
                seg &= ids["s1_country"][g["s"].to_numpy()] == a.seg_country
            add = seg & (g["p"].to_numpy() > a.seg_t_low) & ~keep
            keep = keep | add
            print(f"segment rule ({a.seg_country or 'all countries'}, p > {a.seg_t_low}): best pairs in segment "
                  f"{int(seg.sum()):,}, newly accepted {int(add.sum()):,}")
        acc = g[keep]
        print(f"first/extra rule: first>{a.t_first}, extra>{a.t_extra}: kept {int(keep.sum()):,} "
              f"(flat t={a.t} would keep {int((g['p'] > a.t).sum()):,})")
    elif a.calib:  # F0.5-optimal per-entity decision with a calibration fitted on dense validation (fopt_probe.py)
        import json
        from fopt import fopt_keep
        c = json.load(open(a.calib))[a.calib_key]
        qcal = np.interp(g["p"].to_numpy(np.float64), c["x"], c["y"])
        keep = fopt_keep(g["s"].to_numpy(), qcal, miss=a.miss)
        acc = g[keep]
        print(f"F0.5-optimal decision ({a.calib_key}, miss={a.miss}): kept {int(keep.sum()):,} "
              f"(flat t={a.t} would keep {int((g['p'] > a.t).sum()):,})")
    elif a.t_low is not None:  # decision rule validated on dense data (dense_rule_probe.py)
        from rescue_apply import rescue
        keep, n1, n2 = rescue(g["s"].to_numpy(), g["q"].to_numpy(), g["p"].to_numpy(), a.t, a.t_low, a.t_low2)
        acc = g[keep]
        print(f"rescue rule: t={a.t} t_low={a.t_low} t_low2={a.t_low2}: rescued {n1:,}, second records {n2:,}")
    else:
        acc = g[g["p"] > a.t]
    print(f"ensemble: accepted {len(acc):,} records; S1 covered {acc['s'].nunique():,} of {len(s1_ids):,}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_s1 = [[] for _ in range(len(s1_ids))]
    for q, s in zip(acc["q"].to_numpy(), acc["s"].to_numpy()):
        per_s1[s].append(q_ids[q])
    with open(out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for e, lst in zip(s1_ids, per_s1):
            fh.write(f"{e}\t{','.join(sorted(lst))}\n")
    base = pd.read_csv(a.cands, sep="\t", dtype=str, keep_default_na=False)
    assert (base["source1_entity_id"].to_numpy() == s1_ids).all(), "candidate file S1 order differs"
    added = 0
    with open(out / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for e, c, lst in zip(s1_ids, base["candidate_entity_ids"].to_numpy(), per_s1):
            cs = c.split(",") if c else []
            extra = sorted(set(lst) - set(cs))
            added += len(extra)
            fh.write(f"{e}\t{','.join(cs + extra)}\n")
    print(f"candidates: base {a.cands} + {added:,} matched pairs not in it")
    m = pd.read_csv(out / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    c = pd.read_csv(out / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
    allm = [x for v in m["matched_entity_ids"] if v for x in v.split(",")]
    bad = sum(1 for x, y in zip(m["matched_entity_ids"], c["candidate_entity_ids"]) if x and not set(x.split(",")) <= set(y.split(",")))
    print(f"CHECKS rows {len(m)}/{len(c)} (expect {len(s1_ids)}), headers {list(m.columns)} {list(c.columns)}, "
          f"duplicate records {len(allm) - len(set(allm))}, matches outside candidates {bad}, "
          f"S1 with matches {(m['matched_entity_ids'] != '').sum():,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--fr_npz", nargs="+", default=None, help="models averaged for French records only")
    ap.add_argument("--cands", required=True)
    ap.add_argument("--t", type=float, default=0.8)
    ap.add_argument("--t_low", type=float, default=None, help="empty-entity rescue (only if validated on dense data)")
    ap.add_argument("--t_low2", type=float, default=None, help="second record for 1-record entities")
    ap.add_argument("--t_france", type=float, default=None, help="separate threshold for French S1 entities")
    ap.add_argument("--t_first", type=float, default=None, help="threshold for each entity's most probable record")
    ap.add_argument("--t_extra", type=float, default=0.55, help="threshold for its other records")
    ap.add_argument("--calib", default=None, help="calib.json from fopt_probe.py -> F0.5-optimal decisions")
    ap.add_argument("--calib_key", default="ENSEMBLE")
    ap.add_argument("--miss", type=float, default=0.03)
    ap.add_argument("--fr_first", type=float, default=None, help="France-only first-record threshold")
    ap.add_argument("--fr_extra", type=float, default=0.35, help="France-only extra-record threshold")
    ap.add_argument("--seg_npz", default=None, help="empty_rule.py output: test pairs in the no-address/exact-name segment")
    ap.add_argument("--seg_t_low", type=float, default=0.2)
    ap.add_argument("--seg_country", default="", help="apply the segment rule only in this country")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
