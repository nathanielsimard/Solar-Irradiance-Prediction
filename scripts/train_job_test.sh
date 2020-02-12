#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000M

module load python/3.7
source /project/cq-training-1/project1/teams/team10/env/bin/activate
echo cp /project/cq-training-1/project1/teams/team10/image_reader_cache.tar $SLURM_TMPDIR
date
cp /project/cq-training-1/project1/teams/team10/image_reader_cache.tar $SLURM_TMPDIR
pushd
date
cd $SLURM_TMPDIR
pwd
echo tar -xf image_reader_cache.tar
tar -xf image_reader_cache.tar
date
du -d 1 .
popd
#python run_train.py
