"""Merge the per-shard pickles written by generate_conformers_from_smiles --num_shards.

Usage:
    python scripts/merge_conformer_shards.py out_merged.pkl shard_*.pkl
"""

import pickle
import sys
from pathlib import Path


def main(out_path, shard_paths):
    merged = {}
    overlaps = 0
    for p in shard_paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        overlaps += len(set(d) & set(merged))
        merged.update(d)
        print(f"  {Path(p).name}: {len(d)} molecules")

    if overlaps:
        print(f"WARNING: {overlaps} SMILES appeared in more than one shard.")

    counts = [len(v) for v in merged.values()]
    with open(out_path, "wb") as f:
        pickle.dump(merged, f)
    print(
        f"Wrote {len(merged)} molecules ({sum(counts)} conformers, "
        f"min {min(counts)} / max {max(counts)} per molecule) to {out_path}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
