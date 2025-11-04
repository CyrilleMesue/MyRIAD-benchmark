#!/bin/bash
#SBATCH -A mvmatk
#SBATCH -p core40q
#SBATCH -n 40
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=njume24@itu.edu.tr

# Load module
module load Python

# module load Python/python-3.8.8-openmpi-3.1.6-intel-2017.4
# Activate virtual env
#source ~/.bashrc
#conda activate alzheimer
# Run Python script
python "BenchPipelineDualMayo.py"

# Deactivate the env
# conda deactivate
