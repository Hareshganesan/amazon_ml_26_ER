"""Quick EDA for the Business Entity Resolution data. Light enough to run on any laptop.

Usage (from the folder that contains dataset/):
    python src/eda.py --data dataset
Answers the questions that decide our pipeline design:
  - sizes, countries, missing values per source
  - singleton rate (= score of an all-empty submission)
  - matches per S1 entity, S2 vs S3 split
  - does any S2/S3 record match more than one S1? (if not -> one-to-one assignment constraint)
  - do matched pairs always share the same country? (if yes -> block by country)
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def read(path):
    """Read a competition TSV with every column as a string and empty cells kept as ''."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def main(data):
    """Print dataset statistics: row counts, countries, empty fields, matches per S1 entity, singletons, multi-linked records."""
    data = Path(data)
    tr = {s: read(data / "train" / f"train_source{s}.tsv") for s in (1, 2, 3)}
    te = {s: read(data / "test" / f"test_source{s}.tsv") for s in (1, 2, 3)}
    gt = read(data / "train" / "train_ground_truth.tsv")

    for name, dfs in (("TRAIN", tr), ("TEST", te)):
        print(f"\n===== {name} =====")
        for s, df in dfs.items():
            empty = {c: int((df[c].str.strip() == "").sum()) for c in df.columns}
            print(f"source{s}: rows={len(df)} cols={list(df.columns)}")
            print(f"   countries={dict(Counter(df['country']))}")
            print(f"   empty cells={empty}")

    print("\n===== GROUND TRUTH =====")
    gt["ids"] = gt["matched_entity_ids"].map(lambda v: [x for x in v.split(",") if x])
    n = gt["ids"].map(len)
    print(f"S1 entities in GT: {len(gt)} (train_source1 rows: {len(tr[1])})")
    print(f"singleton rate: {(n == 0).mean():.2%}  <- all-empty submission would score about this")
    print(f"matches per entity distribution: {dict(sorted(Counter(n).items()))}")
    all_ids = [x for ids in gt["ids"] for x in ids]
    print(f"matched ids: {len(all_ids)}  S2={sum(x.startswith('S2') for x in all_ids)}  S3={sum(x.startswith('S3') for x in all_ids)}")
    dup = [k for k, c in Counter(all_ids).items() if c > 1]
    print(f"S2/S3 ids matched to >1 S1 entity: {len(dup)}  (0 => enforce one-to-one assignment)")
    for s in (2, 3):
        used = set(all_ids) & set(tr[s]["entity_id"])
        print(f"source{s}: {len(used)}/{len(tr[s])} records are matched to some S1 ({len(used) / len(tr[s]):.1%})")

    # country agreement between matched pairs
    country = {}
    for df in tr.values():
        country.update(zip(df["entity_id"], df["country"]))
    pairs = [(a, b) for a, ids in zip(gt["source1_entity_id"], gt["ids"]) for b in ids]
    same = sum(country.get(a) == country.get(b) for a, b in pairs)
    print(f"matched pairs with same country: {same}/{len(pairs)}")

    # show a few examples of positives to eyeball the noise
    name = {}
    for df in tr.values():
        name.update(zip(df["entity_id"], df["business_name"] + "  |  " + df["business_address"]))
    print("\n===== SAMPLE MATCHES =====")
    for a, ids in list(zip(gt["source1_entity_id"], gt["ids"]))[:40]:
        if ids:
            print(f"{a}: {name.get(a)}  [{country.get(a)}]")
            for b in ids:
                print(f"    {b}: {name.get(b)}  [{country.get(b)}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    main(ap.parse_args().data)
