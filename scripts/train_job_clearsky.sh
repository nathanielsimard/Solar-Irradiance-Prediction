#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000M

module load python/3.7
source /project/cq-training-1/project1/teams/team10/env/bin/activate
#source /home/guest150/project1/bin/activate
echo python run_train.py --skip_non_cached
python -V
#python run_train.py --skip_non_cached
echo cp /project/cq-training-1/project1/teams/team10/image_reader_cache.tar $SLURM_TMPDIR
date
cp /project/cq-training-1/project1/teams/team10/image_reader_cache.tar $SLURM_TMPDIR
cwd=$(pwd)
date
cd $SLURM_TMPDIR
pwd
echo tar -xf image_reader_cache.tar
tar -xf image_reader_cache.tar
date
#du -d 1 .
cd $cwd
python run_train.py --skip_non_cached --model=CNN2DClearsky


