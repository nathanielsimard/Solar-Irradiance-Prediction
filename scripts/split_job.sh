#!/bin/bash
#SBATCH --time=60:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=10000M

module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index pandas

python run_split.py
