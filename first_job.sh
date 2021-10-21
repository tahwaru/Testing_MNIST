#!/bin/bash -l
#
#PBS -l nodes=1:ppn=4:anygtx,walltime=24:00:00
#
# job name
#PBS -N my_first_project
#

# load cuda module first
module load cuda/10.2

# jobs always start in $HOME
cd $HOME

# activate virtual environment
source ~/miniconda3/bin/activate dl_test

# run script
python mnist_minimal.py -o output.json
