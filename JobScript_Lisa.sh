#!/bin/bash

#Set job requirements
#SBATCH --job-name=training
#SBATCH --time=08:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

cd $HOME/experiments

echo $$
mkdir training`echo $$`
cd training`echo $$`


python /home/jbeek/LoadDataset.py --batch_size=4 --num_epochs=100
python /home/jbeek/LoadDataset.py --batch_size=8 --num_epochs=100
python /home/jbeek/LoadDataset.py --batch_size=16 --num_epochs=100
python /home/jbeek/LoadDataset.py --batch_size=32 --num_epochs=100
