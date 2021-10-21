#!/bin/bash -l
#
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=my_first_project_slurm
#SBATCH --export=NONE

# load cuda module first
module load cuda/11.2.2

# do not export environment variables
unset SLURM_EXPORT_ENV

# jobs always start in $HOME
cd $HOME

# activate virtual environment
source ~/miniconda3/bin/activate dl_test

# run script
python mnist_minimal.py -o output.json
