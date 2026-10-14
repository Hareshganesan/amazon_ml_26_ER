"""Average test predictions per country from different model sets -> one best-per-record npz for local_ensemble.py.

Same averaging as local_ensemble.py: per record, every model votes its best S1 with its probability (0 for other
S1s), the S1 with the highest mean wins. Records belong to exactly one country (blocking is per country).

    python src/country_mix.py --ids test_ids.npz --out mix.npz --usin a.npz b.npz --fr a.npz b.npz s3.npz
"""
import argparse

import numpy as np
import pandas as pd


def average(files):
    """Average models' best-per-record predictions: each votes its best S1 with its probability; the highest mean wins."""
    frames = []
    for f in files:
        z = np.load(f)
        frames.append(pd.DataFrame({"q": z["q_code"].astype(np.int64), "s": z["s1_pos"].astype(np.int64),
                                    "p": z["prob"].astype(np.float32)}))
    g = pd.concat(frames, ignore_index=True).groupby(["q", "s"], sort=False)["p"].sum().reset_index()
    g["p"] /= len(files)
    return g.sort_values(["q", "p"], ascending=[True, False]).drop_duplicates("q")


def main(a):
    """Average the US/India model set and the France model set separately, keep each record from its own country's set."""
    ctry = np.load(a.ids, allow_pickle=True)["s1_country"]
    us = average(a.usin)
    fr = average(a.fr)
    us = us[ctry[us["s"].to_numpy()] != "France"]
    fr = fr[ctry[fr["s"].to_numpy()] == "France"]
    g = pd.concat([us, fr], ignore_index=True)
    assert g["q"].is_unique, "a record appears in both country sets"
    np.savez_compressed(a.out, s1_pos=g["s"].to_numpy(np.int32), q_code=g["q"].to_numpy(np.int64),
                        prob=g["p"].to_numpy(np.float32))
    print(f"{a.out}: US/India {len(us):,} records from {len(a.usin)} models, France {len(fr):,} from {len(a.fr)} models")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--usin", nargs="+", required=True)
    ap.add_argument("--fr", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
