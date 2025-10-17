#!/bin/bash
# vim: ft=slurm
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=rrt_logit_ken_8type_cpu
#SBATCH --time=26:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G 
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riga.wu@pennmedicine.upenn.edu

set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs

# Load conda environment (no CUDA needed)
source /cbica/software/external/python/anaconda/3/etc/profile.d/conda.sh
conda activate uni2_env

# Pin threading to the CPUs you requested
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=offline

# (Optional) Improve CPU binding on some clusters
export KMP_AFFINITY=granularity=fine,compact,1,0 || true

# Install dependencies
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

# Test imports
python - <<'PY'
import torch, scipy, sklearn, h5py, torchmetrics, einops, cv2, matplotlib, openslide, timm, wandb
print('✓ All critical imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Number of CPU threads: {torch.get_num_threads()}')
print('Running in CPU‑only mode')
PY

# Basic environment info
hostname
lscpu | grep 'Model name' | head -n1
nproc
free -h | grep Mem | awk '{print $2}'
python -V || true

# Disable wandb to avoid authentication issues
export WANDB_MODE=offline
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Define directories
WORKING_DIR=/gpfs/fs001/cbica/home/wurig/projects/HistoMethNet_logit
RESULTS_DIR=${WORKING_DIR}/results
mkdir -p ${RESULTS_DIR}
cd ${WORKING_DIR}

# Run the training script
srun --cpu-bind=none python main.py \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 10 \
    --exp_code task_dna_subtyping_RRT_LOGIT_NIH_and_Penn_Ken_8type_30pct_CPU \
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

# Final messages
echo "=========================================="
echo "[INFO] Training completed (CPU‑only mode)."
echo "Results stored at: ${RESULTS_DIR}/task_dna_subtyping_RRT_LOGIT_NIH_and_Penn_Ken_8type_30pct_CPU_s1/"
echo "Logs stored at: ${WORKING_DIR}/logs/"
echo "==========================================="
