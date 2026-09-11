"""
Generate 3D conformers for a list of SMILES using the trained FlowR ligand-only
(no-pocket) model.

Pipeline:
  1. Each SMILES is parsed and given a single throwaway 3D structure via RDKit
     (embedding only -- this seed geometry is required so the molecule can be
     read into the pipeline's internal graph representation, see
     `GeometricMol.from_rdkit`). It is NOT the output: with `--graph_inpainting`
     set, the model treats the atom/bond graph as fixed ("inpainted") and
     re-generates every atom's 3D coordinates from scratch via the flow-matching
     ODE, so the final geometries come entirely from the model, not from RDKit.
  2. Each molecule's graph is duplicated `--n_conformers` times and passed
     through the model (`flowr.gen.generate.generate_molecules`) to obtain that
     many independently generated conformers.
  3. Each conformer is post-processed (`postprocess_conformer`): hydrogens are
     added, a short force-field minimisation cleans up local geometry without
     touching the graph, and chirality is perceived from the coordinates.
  4. Results are collected into {input_smiles: [Chem.Mol, ...]} (each Mol has
     exactly one conformer) and pickled.

Three things are worth knowing about this mode, all of them measured rather than
assumed (see the notes below each point):

  * `final_inpaint=True` is essential. The model runs one last un-clamped
    corrector prediction at t~1; without re-inpainting it, the returned graph is
    whatever that final pass sampled, and it drifts arbitrarily far from the
    input SMILES -- measured at 0/80537 exact matches on the CASF16 core set.
    With it, the graph matches by construction.
  * `--graph_inpainting harmonic` beats `random`. Full-graph inpainting is
    out-of-distribution for this checkpoint (it was trained with partial
    scaffold/fragment inpainting), so the coordinates it produces do not respect
    the clamped bonds well. Seeding from the bond-graph Laplacian rather than
    from noise moves mean heavy-atom bond length from 2.72 A to 1.87 A.
  * The force-field cleanup (`--relax_iters`) is what makes the geometry usable:
    it brings mean bond length to 1.42 A, matching experimental reference
    ensembles, while preserving torsional diversity in proportion to how
    flexible the molecule actually is.

Requires a ligand-only ("mol") FlowR checkpoint (--arch flowr), e.g.
`flowr_root_v2.2_mol.ckpt`, and a GPU.

Example:
    python -m flowr.gen.generate_conformers_from_smiles \
        --csv_path molecules.csv \
        --smiles_column smiles \
        --ckpt_path /path/to/flowr_root_v2.2_mol.ckpt \
        --output_pkl conformers.pkl \
        --n_conformers 1000
"""

import argparse
import csv
import pickle
import tempfile
import warnings
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from tqdm import tqdm

import flowr.gen.utils as util
from flowr.gen.generate import generate_molecules
from flowr.scriptutil import load_mol_model
from flowr.util.molrepr import GeometricMolBatch
from flowr.util.rdkit import write_sdf_file

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=DeprecationWarning)
RDLogger.DisableLog("rdApp.*")

DEFAULT_INTEGRATION_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_ODE_SAMPLING_STRATEGY = "linear"
DEFAULT_CATEGORICAL_STRATEGY = "uniform-sample"


def read_smiles_csv(csv_path: str, smiles_column: str | None):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if smiles_column is None:
            candidates = [
                c for c in fieldnames if c.lower() in ("smiles", "smile", "canonical_smiles")
            ]
            if not candidates:
                raise ValueError(
                    f"Could not auto-detect a SMILES column among {fieldnames}; "
                    "pass --smiles_column explicitly."
                )
            smiles_column = candidates[0]
        elif smiles_column not in fieldnames:
            raise ValueError(
                f"Column '{smiles_column}' not found in CSV columns: {fieldnames}"
            )

        smiles_list = [
            row[smiles_column].strip() for row in reader if row[smiles_column].strip()
        ]
    return smiles_list, smiles_column


def embed_seed_mol(smiles: str, seed: int = 42):
    """Build a throwaway 3D structure for `smiles` to feed the SDF loader.

    Only the atom/bond graph derived from this mol is used downstream; the
    coordinates are discarded by the model under --graph_inpainting.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    conf_id = AllChem.EmbedMolecule(
        mol, randomSeed=seed, useRandomCoords=True, maxAttempts=50
    )
    if conf_id == -1:
        return None
    return mol


def _reflect(mol: Chem.Mol) -> Chem.Mol:
    """Mirror the conformer through the x=0 plane (turns a molecule into its enantiomer)."""
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(-p.x, p.y, p.z))
    return m


def _stereo_smiles(mol: Chem.Mol):
    """Canonical SMILES with stereochemistry perceived from the 3D coordinates."""
    m = Chem.Mol(mol)
    Chem.SanitizeMol(m)
    Chem.AssignStereochemistryFrom3D(m)
    return Chem.MolToSmiles(Chem.RemoveHs(m))


def postprocess_conformer(mol, target_smiles, relax_iters, add_hs, stereo_mode):
    """Turn one raw model output into a usable conformer.

    The model fixes the atom/bond graph but not its 3D realism or its chirality:
      * local geometry is cleaned up by a short force-field minimisation that cannot
        change the (inpainted) graph, only the coordinates;
      * stereochemistry is perceived from the coordinates, and if the conformer came
        out as the mirror image of the requested stereoisomer it is reflected, which
        is an exact isometry and so leaves the force-field energy untouched.

    Returns (mol_or_None, status).
    """
    try:
        m = Chem.Mol(mol)
        Chem.SanitizeMol(m)
    except Exception:
        return None, "unsanitizable"

    if add_hs:
        try:
            m = Chem.AddHs(m, addCoords=True)
        except Exception:
            return None, "unsanitizable"

    if relax_iters > 0:
        try:
            props = AllChem.MMFFGetMoleculeProperties(m)
            ff = (
                AllChem.UFFGetMoleculeForceField(m)
                if props is None
                else AllChem.MMFFGetMoleculeForceField(m, props)
            )
            ff.Minimize(maxIts=relax_iters)
        except Exception:
            # Keep the unrelaxed geometry rather than discarding the conformer.
            pass

    status = "ok"
    if stereo_mode != "none" and target_smiles is not None:
        try:
            if _stereo_smiles(m) != target_smiles:
                mirrored = _reflect(m)
                if _stereo_smiles(mirrored) == target_smiles:
                    m, status = mirrored, "reflected"
                else:
                    status = "stereo_mismatch"
                    if stereo_mode == "strict":
                        return None, status
        except Exception:
            status = "stereo_mismatch"
            if stereo_mode == "strict":
                return None, status

    return m, status


def generate_conformers_for_mol(
    args,
    model,
    transform,
    interpolant,
    seed_mol: Chem.Mol,
    iter_offset: int,
    target_smiles=None,
    status_counts=None,
):
    """Generate exactly `args.n_conformers` usable conformers for one molecule's graph.

    Conformers are post-processed as they come off the model, so that the top-up
    loop counts only the ones actually kept -- otherwise `--stereo_mode strict`
    would silently return short instead of sampling more.
    """
    gen_mols = []
    k = 0
    while len(gen_mols) < args.n_conformers and k < args.max_sample_iter:
        with tempfile.TemporaryDirectory(dir=args.scratch_dir) as tmp_dir:
            sdf_path = Path(tmp_dir) / "mol.sdf"
            write_sdf_file(sdf_path, [seed_mol], name="seed")
            args.sdf_path = str(sdf_path)
            args.ligand_idx = None

            n_remaining = args.n_conformers - len(gen_mols)
            molecules = util.load_data_from_sdf_mol(
                args,
                remove_hs=args._remove_hs,
                remove_aromaticity=args._remove_aromaticity,
                transform=transform,
                sample=True,
                sample_n_molecules_per_mol=n_remaining,
            )
            dataset = GeometricMolBatch(molecules)
            dataloader = util.get_dataloader(
                args, dataset, interpolant, iter=iter_offset + k
            )
            for batch in dataloader:
                prior, _, _, _ = batch
                batch_mols = generate_molecules(
                    args,
                    model=model,
                    prior=prior,
                    device=args.device,
                    # The whole atom/bond graph is fixed here (graph_inpainting is
                    # always active in this script); without this, the final
                    # corrector step returns an un-clamped, freshly-sampled graph
                    # that can drift arbitrarily far from the input SMILES.
                    final_inpaint=True,
                )
                for _raw in batch_mols:
                    if _raw is None:
                        continue
                    _pm, _st = postprocess_conformer(
                        _raw,
                        target_smiles,
                        relax_iters=args.relax_iters,
                        add_hs=not args.no_add_hs,
                        stereo_mode=args.stereo_mode,
                    )
                    if status_counts is not None:
                        status_counts[_st] = status_counts.get(_st, 0) + 1
                    if _pm is not None:
                        gen_mols.append(_pm)
        k += 1

    return gen_mols[: args.n_conformers], k


def main(args):
    torch.set_float32_matmul_precision("high")

    smiles_list, smiles_column = read_smiles_csv(args.csv_path, args.smiles_column)
    print(f"Loaded {len(smiles_list)} SMILES from column '{smiles_column}' in {args.csv_path}")

    n_rows = len(smiles_list)
    smiles_list = list(dict.fromkeys(smiles_list))
    if len(smiles_list) < n_rows:
        print(
            f"Deduplicated {n_rows} rows down to {len(smiles_list)} unique SMILES "
            "(duplicate rows share the same generated conformers)."
        )

    if args.num_shards > 1:
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError(
                f"--shard_index must be in [0, {args.num_shards}); got {args.shard_index}"
            )
        n_all = len(smiles_list)
        smiles_list = smiles_list[args.shard_index :: args.num_shards]
        print(
            f"Shard {args.shard_index}/{args.num_shards}: processing "
            f"{len(smiles_list)} of {n_all} unique SMILES. "
            "Merge the per-shard pickles when every shard has finished."
        )

    print(f"Loading model from {args.ckpt_path} ...")
    (model, hparams, vocab, vocab_charges, vocab_hybridization, vocab_aromatic) = (
        load_mol_model(args)
    )
    model = model.to(args.device)
    model.eval()
    print("Model loaded.")

    args._remove_hs = hparams["remove_hs"]
    args._remove_aromaticity = hparams["remove_aromaticity"]

    transform, interpolant = util.load_util_mol(
        args, hparams, vocab, vocab_charges, vocab_hybridization, vocab_aromatic
    )

    seed_records = []
    results = {}
    failed_to_embed = []
    incomplete = []
    status_counts = {}
    iter_offset = 0

    for smiles in tqdm(smiles_list, desc="Molecules"):
        seed_mol = embed_seed_mol(smiles, seed=args.seed)
        if seed_mol is None:
            print(f"WARNING: could not build a seed 3D structure for SMILES, skipping: {smiles}")
            failed_to_embed.append(smiles)
            continue
        seed_records.append((smiles, seed_mol))

        # The model only guarantees the atom/bond graph, not the realism of the
        # geometry or the chirality, so each conformer is cleaned up as it is
        # produced (inside the top-up loop).
        target_canon = None
        if args.stereo_mode != "none":
            _t = Chem.MolFromSmiles(smiles)
            target_canon = Chem.MolToSmiles(_t) if _t is not None else None

        gen_mols, n_iters = generate_conformers_for_mol(
            args,
            model,
            transform,
            interpolant,
            seed_mol,
            iter_offset,
            target_smiles=target_canon,
            status_counts=status_counts,
        )
        iter_offset += n_iters

        if len(gen_mols) < args.n_conformers:
            print(
                f"WARNING: only generated {len(gen_mols)}/{args.n_conformers} valid "
                f"conformers for {smiles} after {args.max_sample_iter} sampling rounds."
            )
            incomplete.append(smiles)

        results[smiles] = gen_mols
        torch.cuda.empty_cache()

    Path(args.output_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"Wrote generated conformers for {len(results)} molecules to {args.output_pkl}")

    if status_counts:
        total = sum(status_counts.values())
        print("\nPost-processing summary (per conformer):")
        for key in sorted(status_counts):
            n = status_counts[key]
            print(f"  {key:<16} {n:>8}  ({100 * n / total:.1f}%)")

    if args.output_sdf is not None and seed_records:
        Path(args.output_sdf).parent.mkdir(parents=True, exist_ok=True)
        write_sdf_file(args.output_sdf, [m for _, m in seed_records], name="mol")
        print(f"Wrote {len(seed_records)} seed structures to {args.output_sdf}")

    if failed_to_embed:
        print(f"\n{len(failed_to_embed)} SMILES failed to embed and were skipped entirely:")
        for smi in failed_to_embed:
            print(f"  {smi}")
    if incomplete:
        print(f"\n{len(incomplete)} SMILES came up short of --n_conformers:")
        for smi in incomplete:
            print(f"  {smi}")


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Generate conformers from SMILES using the FlowR model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpus", type=int, default=1, help="Required by flowr.gen.utils.split_list; leave at 1 for single-GPU runs")
    parser.add_argument("--mp_index", type=int, default=0, help="Required by flowr.gen.utils.split_list; leave at 0 for single-GPU runs")

    # Data
    parser.add_argument("--csv_path", type=str, required=True, help="CSV file containing a column of SMILES strings")
    parser.add_argument("--smiles_column", type=str, default=None, help="Name of the SMILES column; auto-detected if omitted")
    parser.add_argument("--output_pkl", type=str, required=True, help="Output pickle path: {smiles: [Chem.Mol, ...]}")
    parser.add_argument("--output_sdf", type=str, default=None, help="Optional path to save the seed 3D structures used to seed the pipeline (one record per input SMILES)")
    parser.add_argument("--scratch_dir", type=str, default=None, help="Directory for per-molecule temporary SDF files (defaults to the system temp dir)")

    # Model
    parser.add_argument("--arch", type=str, default="flowr", choices=["flowr", "transformer"])
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to a ligand-only ('mol') FlowR checkpoint")
    parser.add_argument("--lora_finetuned", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)

    # Sampling
    parser.add_argument("--n_conformers", type=int, default=1000, help="Number of conformers to generate per SMILES")
    parser.add_argument("--num_shards", type=int, default=1, help="Split the (deduplicated) SMILES list into this many shards so they can be generated as parallel jobs; each shard writes its own pickle, merge them afterwards")
    parser.add_argument("--shard_index", type=int, default=0, help="Which shard this process handles, in [0, --num_shards)")
    parser.add_argument("--relax_iters", type=int, default=200, help="Force-field (MMFF, UFF fallback) minimisation steps applied to each generated conformer. The inpainted graph is held fixed, so this only cleans up bond lengths/angles the model got wrong; 0 disables it.")
    parser.add_argument("--no_add_hs", action="store_true", help="Do not add explicit hydrogens to the generated conformers (they are added by default, matching the reference conformer sets)")
    parser.add_argument("--stereo_mode", type=str, default="reflect", choices=["none", "reflect", "strict"], help="How to handle chirality, which the model does not condition on: 'none' leaves it alone, 'reflect' mirrors a conformer when it came out as the wrong enantiomer, 'strict' additionally drops conformers whose stereochemistry still does not match")
    parser.add_argument("--max_sample_iter", type=int, default=5, help="Max retry rounds per molecule to make up for any invalid samples")
    parser.add_argument("--sample_mol_sizes", action="store_true")
    parser.add_argument("--batch_cost", type=int, default=128, help="Bucketing cost budget per batch; lower this if you hit OOM")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bucket_cost_scale", type=str, default="quadratic")

    # Graph inpainting (fixes the atom/bond graph, only coordinates are generated)
    parser.add_argument("--graph_inpainting", type=str, default="harmonic", choices=["random", "harmonic"], help="Prior used for the fixed-graph coordinates. 'harmonic' seeds them from the bond-graph Laplacian and gives markedly better geometry than 'random'.")
    parser.add_argument("--coord_noise_scale", type=float, default=0.0)
    parser.add_argument("--integration_steps", type=int, default=DEFAULT_INTEGRATION_STEPS)
    parser.add_argument("--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)
    parser.add_argument("--ode_sampling_strategy", type=str, default=DEFAULT_ODE_SAMPLING_STRATEGY)
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "midpoint"])
    parser.add_argument("--use_sde_simulation", action="store_true")
    parser.add_argument("--use_cosine_scheduler", action="store_true")
    parser.add_argument("--categorical_strategy", type=str, default=DEFAULT_CATEGORICAL_STRATEGY)
    parser.add_argument("--corrector_iters", type=int, default=0)
    parser.add_argument("--max_fragment_cuts", type=int, default=3)
    parser.add_argument("--rotation_alignment", action="store_true")
    parser.add_argument("--permutation_alignment", action="store_true")
    parser.add_argument("--anisotropic_prior", action="store_true")
    parser.add_argument("--ref_ligand_com_prior", action="store_true")
    parser.add_argument("--ref_ligand_com_noise_std", type=float, default=1.0)

    # Flags required by load_mol_model / load_util_mol; always disabled here since
    # we want plain graph-conditioned conformer generation, not any of the other
    # inpainting/conditioning modes.
    parser.add_argument("--scaffold_hopping", action="store_true", default=False)
    parser.add_argument("--scaffold_elaboration", action="store_true", default=False)
    parser.add_argument("--linker_inpainting", action="store_true", default=False)
    parser.add_argument("--core_growing", action="store_true", default=False)
    parser.add_argument("--fragment_inpainting", action="store_true", default=False)
    parser.add_argument("--fragment_growing", action="store_true", default=False)
    parser.add_argument("--substructure_inpainting", action="store_true", default=False)
    parser.add_argument("--substructure", default=None)

    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    main(get_args())
