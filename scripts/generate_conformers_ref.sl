#!/bin/bash
#SBATCH -J flowr-conf-ref
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=/mnt/weka/vtarasov/code/flowr_root/slurm_outs/gen_conformers/ref_%A_%a.out
#SBATCH --error=/mnt/weka/vtarasov/code/flowr_root/slurm_outs/gen_conformers/ref_%A_%a.err

# The 1044-molecule ref set is split across 8 array tasks: a single job took
# ~21h and was cancelled part-way last time, losing everything. Merge afterwards:
#   python scripts/merge_conformer_shards.py \
#       casf16_ref_flowr_conformers.pkl casf16_ref_flowr_conformers.shard*.pkl

cd /home/vtarasov/code/flowr_root

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate flowr_root
# The cluster's login-shell PATH otherwise puts /opt/conda/bin ahead of the
# activated env, which resolves `python` to a build with no torch installed.
export PATH="$CONDA_PREFIX/bin:$PATH"
# rdkit's bundled RDPaths._share got hardlinked from a shared pkg cache with a
# stale absolute path baked in from a different flowr_root env (/opt/conda/envs).
# Overriding RDBASE forces RDConfig to use this env's own data dir instead.
export RDBASE="$CONDA_PREFIX/share/RDKit"
export PYTHONPATH="/home/vtarasov/code/flowr_root"

python -m flowr.gen.generate_conformers_from_smiles \
    --csv_path /mnt/weka/mbedrosian/data/casf16/casf16_ref_chembl3d_exact_intersection.csv \
    --smiles_column chembl3d_isomeric_smiles \
    --ckpt_path /home/vtarasov/code/flowr_root/flowr_root_v2.2_mol.ckpt \
    --output_pkl casf16_ref_flowr_conformers.shard${SLURM_ARRAY_TASK_ID}.pkl \
    --n_conformers 1000 \
    --stereo_mode strict \
    --max_sample_iter 20 \
    --num_shards 8 \
    --shard_index ${SLURM_ARRAY_TASK_ID}
