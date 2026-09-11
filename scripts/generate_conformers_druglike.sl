#!/bin/bash
#SBATCH -J flowr-conf-druglike
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/weka/vtarasov/code/flowr_root/slurm_outs/gen_conformers/druglike_%j.out
#SBATCH --error=/mnt/weka/vtarasov/code/flowr_root/slurm_outs/gen_conformers/druglike_%j.err

# 23 curated drug-like molecules. --max_sample_iter is higher than the CASF16
# runs because the set is small enough to afford it: strict-mode acceptance is
# ~2^(1-k) for k stereocentres, and dexamethasone has eight (~0.8%), so it will
# still be reported as a shortfall. Twelve of the 23 are achiral and cost one round.

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
    --csv_path /home/vtarasov/code/3DMolGen/druglike_curated_shortlist.tsv \
    --smiles_column smiles \
    --ckpt_path /home/vtarasov/code/flowr_root/flowr_root_v2.2_mol.ckpt \
    --output_pkl druglike_shortlist_flowr_conformers.pkl \
    --n_conformers 1000 \
    --stereo_mode strict \
    --max_sample_iter 40
