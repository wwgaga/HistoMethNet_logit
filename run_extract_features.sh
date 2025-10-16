#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=20G


module load cuda/11.2


python extract_features_fp.py --data_h5_dir /cbica/home/tianyu/dataset/NIH_Extracted_Patches/extracted_patches_GBM_NIH_All_bwh_resection_Presets --data_slide_dir /cbica/home/tianyu/GBM_NIH_All --csv_path /cbica/home/tianyu/dataset/NIH_Extracted_Patches/extracted_patches_GBM_NIH_All_bwh_resection_Presets/process_list_autogen.csv --feat_dir /cbica/home/tianyu/dataset/NIH_features_extracted --batch_size 512 --slide_ext .ndpi
