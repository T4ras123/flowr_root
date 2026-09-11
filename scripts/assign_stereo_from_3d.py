"""Label conformers in a generated pickle with the stereochemistry their coordinates encode.

The generator returned graphs without chiral tags, so conformers are geometrically
correct but read as stereo-unspecified. This assigns tags from the 3D coordinates;
it never moves an atom.

Usage:
    python scripts/assign_stereo_from_3d.py in.pkl out.pkl
"""

import pickle
import sys

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def main(in_path, out_path):
    with open(in_path, "rb") as f:
        data = pickle.load(f)

    n_total = n_failed = 0
    for smiles, mols in data.items():
        for mol in mols:
            if mol is None:
                continue
            n_total += 1
            try:
                Chem.AssignStereochemistryFrom3D(mol)
            except Exception:
                n_failed += 1

    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(
        f"{in_path}: labelled {n_total - n_failed}/{n_total} conformers "
        f"across {len(data)} molecules -> {out_path}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
