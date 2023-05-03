#!/bin/bash -l
#SBATCH --job-name=gpu
#SBATCH --partition=gpu
#SBATCH --nodelist=icsnode09
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclusive


python main.py