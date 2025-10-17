#!/bin/bash
# Minimal runner for Logit-Only (no encoder) 8-type classification
# Edit paths/flags for your environment, then run:
#   bash run_logit_only.sh

python main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 5 \
  --exp_code exp_logit_only_baseline \
  --weighted_sample \
  --bag_loss ce \
  --cell_property cell_type \
  --task task_3_cell_type_classification \
  --model_type logit_only \
  --log_data \
  --subtyping \
  --data_root_dir /path/to/features \
  --embed_dim 1536

