#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=16G


module load cuda/11.2

python main_vis.py 
