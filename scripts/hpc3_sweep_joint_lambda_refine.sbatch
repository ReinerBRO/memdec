#!/bin/bash
#SBATCH -J memdec_refn
#SBATCH -p acd_ue
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/jzhu997/MemoryDecoder

set -euo pipefail

mkdir -p ./logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./logs/sweep_joint_lambda_refine_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true
set +u
source /data/user/jzhu997/envs/memorydecoder/bin/activate
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/data/user/jzhu997/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/output_layout.sh"

PROC_DATASET=/data/user/jzhu997/MemoryDecoder/dataset/wikitext-gpt2
BASE_MODEL=/data/user/jzhu997/cache/Models/gpt2
MEMDEC_MODEL=/data/user/jzhu997/cache/Models/MemoryDecoder-gpt2-small

for L in 0.72 0.74 0.76 0.78 0.80
do
  echo "[INFO] evaluating lambda=${L}"
  python -m evaluate_joint \
    --do_test \
    --model_name_or_path "${BASE_MODEL}" \
    --dataset_name "${PROC_DATASET}" \
    --dataset_split_name test \
    --per_device_eval_batch_size 16 \
    --output_dir "$(result_dir "joint_lambda_refine_${L}")" \
    --knn_temp 1 \
    --lmbda "${L}" \
    --knn_generator_path "${MEMDEC_MODEL}" \
    --report_to none
done
