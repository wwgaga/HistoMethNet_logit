#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=15G


# module load cuda/11.2

# python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_dna_subtyping_RRT_Mixbag_Stem_like --weighted_sample --bag_loss ce --cell_property stem --inst_loss svm --task task_3_cell_type_classification --model_type rrt  --log_data --subtyping --data_root_dir /cbica/home/tianyu/dataset/Natalie_Cohort_UNI_Features --embed_dim 1024 --all_shortcut --crmsa_mlp --epeg_k=13 --crmsa_k=3 --crmsa_heads=1 --n_trans_layers=2 --da_act=tanh 

python eval.py --k 10 --models_exp_code All_data_Ken_class_contrast_uni_2_feature_trans_classifer_no_feature_mlps_s1 --splits_dir splits/task_2_TCGA_NIH_Class_100 --cell_property cell_type --save_exp_code tcga_results_Ken_classes_contrastive_transmlp_uni2_feature --task task_3_cell_type_classification --model_type rrt --results_dir results --data_root_dir /cbica/home/tianyu/dataset/TCGA_selected_slides_features_extracted_UNI_2 --data_csv dataset_csv/matched_data_TCGA_NIH.csv --embed_dim 1536 
