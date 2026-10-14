"""Local scorer: macro F0.5 per Source-1 entity, exactly as described in the problem statement.

Usage:
    python src/metric.py --pred output/matching_results.tsv --truth path/to/val_ground_truth.tsv
"""
import argparse

import pandas as pd


def parse_ids(value):
    """Turn a comma-separated ID cell (possibly empty/NaN) into a set of IDs."""
    if not isinstance(value, str) or not value.strip():
        return set()
    return {x.strip() for x in value.split(",") if x.strip()}


def f05_entity(pred, truth):
    """F0.5 for one Source-1 entity. Singleton rule: empty vs empty = 1.0, any pred on a singleton = 0.0."""
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def score(pred_map, truth_map):
    """Macro-average over every Source-1 entity in the truth set (missing predictions count as empty)."""
    scores = [f05_entity(pred_map.get(k, set()), v) for k, v in truth_map.items()]
    return sum(scores) / len(scores)


def load_map(path, id_col="source1_entity_id"):
    """Read a two-column TSV into {source1_entity_id: set of S2/S3 ids}."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    list_col = [c for c in df.columns if c != id_col][0]
    return dict(zip(df[id_col], df[list_col].map(parse_ids)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()
    truth = load_map(args.truth)
    pred = load_map(args.pred)
    n_single = sum(1 for v in truth.values() if not v)
    print(f"entities={len(truth)}  singletons={n_single} ({n_single / len(truth):.1%})")
    print(f"macro F0.5 = {score(pred, truth):.5f}")
