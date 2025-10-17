#!/usr/bin/env bash
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --job-name=rrt_logit_ken_8type
#SBATCH --time=26:00:00
#SBATCH --gres=gpu:l40:1
#SBATCH --mem-per-gpu=48G
#SBATCH --cpus-per-gpu=32
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riga.wu@pennmedicine.upenn.edu

set -euo pipefail

# 8-type Ken classification with RRT + Logistic Regression head
# Using embedding dimension 1536 with logistic regression classifier

# Create logs directory if it doesn't exist
mkdir -p logs

# Load CUDA + conda environment
module load cuda/11.8
source /cbica/software/external/python/anaconda/3/etc/profile.d/conda.sh
conda activate uni2_env

# Install ALL required dependencies
echo "Installing all required dependencies..."
pip install --quiet \
    scipy \
    scikit-learn \
    h5py \
    torchmetrics \
    einops \
    opencv-python \
    matplotlib \
    seaborn \
    openslide-python \
    histolab \
    timm \
    tqdm \
    pyyaml \
    wandb \
    pillow

# Test critical imports before running main script
echo "Testing imports..."
python -c "
import sys
import torch
import scipy
import sklearn
import h5py
import torchmetrics
import einops
import cv2
import matplotlib
import openslide
import timm
import wandb
print('✓ All critical imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
" || { echo "Import test failed! Check dependencies."; exit 1; }

# Basic env info
echo "HOST: $(hostname)"
echo "CUDA: $(nvcc --version 2>/dev/null | tail -n1 || true)"
nvidia-smi || true
python -V || true
echo "=========================================="
echo "Starting RRT + Logistic Regression Ken 8-type classification"
echo "Using 30% of dataset with embed_dim=1536"
echo "Model type: rrt_logit"
echo "=========================================="

# Disable wandb to avoid authentication issues
export WANDB_MODE=offline

# Set current directory as working directory and results directory
WORKING_DIR=/gpfs/fs001/cbica/home/wurig/projects/HistoMethNet_logit
RESULTS_DIR=${WORKING_DIR}/results

# Create results directory if it doesn't exist
mkdir -p ${RESULTS_DIR}

# Change to the working directory
cd ${WORKING_DIR}

srun --cpu-bind=none python main.py \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 10 \
    --exp_code task_dna_subtyping_RRT_LOGIT_NIH_and_Penn_Ken_8type_30pct \
    --weighted_sample \
    --bag_loss ce \
    --cell_property cell_type \
    --inst_loss svm \
    --task task_3_cell_type_classification \
    --model_type rrt_logit \
    --subtyping \
    --data_root_dir /gpfs/fs001/cbica/projects/Path_UPenn_GBM/tianyu_files/dataset/Penn_NIH_Combine_Features \
    --split_dir /cbica/home/tianyu/projects/dna-rrt-patch-only/splits/task_3_cell_type_classification_100 \
    --results_dir ${RESULTS_DIR} \
    --embed_dim 1536 \
    --max_epochs 50 \
    --label_frac 0.3 \
    --all_shortcut \
    --crmsa_mlp \
    --epeg_k 13 \
    --crmsa_k 3 \
    --crmsa_heads 1 \
    --n_trans_layers 2 \
    --da_act tanh \
    --k_start 7

echo "=========================================="
echo "[INFO] Training completed."
echo "Results stored at: ${RESULTS_DIR}/task_dna_subtyping_RRT_LOGIT_NIH_and_Penn_Ken_8type_30pct_s1/"
echo "Logs stored at: ${WORKING_DIR}/logs/"
echo "==========================================="