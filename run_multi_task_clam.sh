#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=20G


module load cuda/11.2

python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_dna_subtyping_CLAM_UNI_full_set --weighted_sample --bag_loss ce --inst_loss svm --task task_3_cell_type_classification --model_type clam_mt --log_data --subtyping --data_root_dir /cbica/home/tianyu/dataset/Natalie_Cohort_UNI_Features --embed_dim 1024

python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_dna_subtyping_CLAM_UNI --weighted_sample --bag_loss ce --inst_loss svm --task task_3_cell_type_classification --model_type clam_mt  --log_data --subtyping --data_root_dir /Users/yu/Downloads/Pathology_features --embed_dim 1024