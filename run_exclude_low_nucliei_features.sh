#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=15G


# module load cuda/11.2

python exclude_low_nucliei_features.py --feature_dir /cbica/home/tianyu/dataset/Penn_NIH_Combine_Features/h5_files --slide_dir /cbica/home/tianyu/dataset/penn_nih_combine_slides --output_dir /cbica/home/tianyu/dataset/processed_features_nuclei
