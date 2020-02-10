#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000M

module load python/3.7
source /project/cq-training-1/project1/teams/team10/env/bin/activate

python run_train.py
