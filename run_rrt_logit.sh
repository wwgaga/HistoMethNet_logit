#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=20G

# Example runner for RRT with logistic regression head.
# Edit paths/flags for your environment, then run:
#   bash run_rrt_logit.sh

# module load cuda/11.2

python main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 5 \
  --exp_code exp_rrt_logit_baseline \
  --weighted_sample \
  --bag_loss ce \
  --cell_property cell_type \
  --inst_loss svm \
  --task task_3_cell_type_classification \
  --model_type rrt_logit \
  --log_data \
  --subtyping \
  --data_root_dir /path/to/features \
  --embed_dim 1024 \
  --all_shortcut \
  --crmsa_mlp \
  --epeg_k=13 \
  --crmsa_k=3 \
  --crmsa_heads=1 \
  --n_trans_layers=2 \
  --da_act=tanh

