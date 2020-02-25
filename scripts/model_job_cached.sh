#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M

module load python/3.7
source /project/cq-training-1/project1/teams/team10/env/bin/activate

echo python run_train.py --skip_non_cached
python -V
date

echo cp /project/cq-training-1/project1/teams/team10/image_reader_cache.tar $SLURM_TMPDIR
cp /project/cq-training-1/project1/teams/team10/image_reader_cache.tar $SLURM_TMPDIR
date

echo tar -xf image_reader_cache.tar
cwd=$(pwd)
cd $SLURM_TMPDIR
pwd
tar -xf image_reader_cache.tar
cd $cwd
<<<<<<< HEAD:scripts/model_job_cached.sh
date

python run_model.py $@

