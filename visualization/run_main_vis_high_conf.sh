#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=6G


#module load cuda/11.2

python main_vis_high_conf.py 
