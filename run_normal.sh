#!/bin/bash

#Set job requirements
#SBATCH --job-name=training
#SBATCH --time=00:05:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal

cd $HOME/experiments

echo $$
mkdir training`echo $$`
cd training`echo $$`

python /home/jbeek/LoadDataset.py
