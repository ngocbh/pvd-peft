#!/bin/bash

#SBATCH --job-name=cage
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --mail-type=REQUEUE,FAIL,TIME_LIMIT
#SBATCH --output=logs/slurm/o_%A.out
#SBATCH --error=logs/slurm/o_%A.err
#SBATCH --partition=gpu,scavenge_gpu
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --constraint=v100|a5000

# Example wrapper, adjust to your system.

# you have to creat log folder first
mkdir -p logs/slurm/

date
hostname
pwd

source activate pvd

cd $SLURM_SUBMIT_DIR
pwd

echo $@
eval $@


echo "All done in sbatch."
date



