#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=20G


module load cuda/11.2

python create_patches_fp.py --source /cbica/home/tianyu/GBM_NIH_All --save_dir /cbica/home/tianyu/dataset/NIH_Extracted_Patches/extracted_patches_GBM_NIH_All_TCGA_Presets --patch_size 256 --preset tcga.csv --seg --patch --stitch