"""Write the test id order used by every run (S1 ids; S2 then S3 record ids) to output/test_ids.npz."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

d = Path(sys.argv[1]) / "test"
rd = lambda f: pd.read_csv(d / f, sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id", "country"])
s1 = rd("test_source1.tsv")
q = pd.concat([rd(f"test_source{s}.tsv") for s in (2, 3)], ignore_index=True)
Path("output").mkdir(exist_ok=True)
np.savez_compressed("output/test_ids.npz", s1=s1["entity_id"].to_numpy(str), s1_country=s1["country"].to_numpy(str),
                    q=q["entity_id"].to_numpy(str))
print(len(s1), len(q))
