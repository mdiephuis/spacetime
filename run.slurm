#!/bin/sh

#SBATCH -n 1
#SBATCH -p long
#SBATCH -t 72:00:00
#SBATCH --array=0-44

python kl_exp.py $SLURM_ARRAY_TASK_ID
