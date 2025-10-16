#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=20G


# module load cuda/11.2

python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code All_data_Ken_class_contrast_uni_2_feature_trans_classifer_no_feature_mlps --weighted_sample --bag_loss ce --cell_property cell_type --inst_loss svm --task task_3_cell_type_classification --embed_dim 1536 --model_type rrt  --log_data --subtyping --data_root_dir /cbica/home/tianyu/dataset/Penn_NIH_Combine_Features --embed_dim 1024 --all_shortcut --crmsa_mlp --epeg_k=13 --crmsa_k=3 --crmsa_heads=1 --n_trans_layers=2 --da_act=tanh --k_start=6

