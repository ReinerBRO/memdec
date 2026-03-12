#!/bin/bash
#SBATCH -J memdec_gpt2s
#SBATCH -p acd_ue
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/jzhu997/MemoryDecoder

set -euo pipefail

mkdir -p ./logs ./dataset
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./logs/reproduce_gpt2_small_${TIMESTAMP}_${SLURM_JOB_ID}.log"
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
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/output_layout.sh"

RAW_DATASET=/data/user/jzhu997/cache/data/wikitext-103-raw
PROC_DATASET=/data/user/jzhu997/MemoryDecoder/dataset/wikitext-gpt2
BASE_MODEL=/data/user/jzhu997/cache/Models/gpt2
MEMDEC_MODEL=/data/user/jzhu997/cache/Models/MemoryDecoder-gpt2-small
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-$(result_dir "base_gpt2_small")}
JOINT_OUTPUT_DIR=${JOINT_OUTPUT_DIR:-$(result_dir "joint_gpt2_small")}

echo "[INFO] host=$(hostname)"
echo "[INFO] workdir=$PWD"
echo "[INFO] log_file=$LOG_FILE"
nvidia-smi || true

if [ ! -f "${PROC_DATASET}/dstore_summary.json" ]; then
  echo "[INFO] preprocessing dataset into ${PROC_DATASET}"
  python utils/preprocess_dataset.py \
    --dataset_name "${RAW_DATASET}" \
    --dataset_config_name wikitext-103-raw-v1 \
    --tokenizer_path "${BASE_MODEL}" \
    --output_dir "${PROC_DATASET}" \
    --num_proc 8
else
  echo "[INFO] using existing processed dataset at ${PROC_DATASET}"
fi

echo "[INFO] running base evaluation"
python -m train_base \
  --model_name_or_path "${BASE_MODEL}" \
  --dataset_name "${PROC_DATASET}" \
  --per_device_eval_batch_size 16 \
  --do_eval \
  --eval_subset test \
  --output_dir "${BASE_OUTPUT_DIR}" \
  --report_to none

echo "[INFO] running joint evaluation"
python -m evaluate_joint \
  --do_test \
  --model_name_or_path "${BASE_MODEL}" \
  --dataset_name "${PROC_DATASET}" \
  --dataset_split_name test \
  --per_device_eval_batch_size 16 \
  --output_dir "${JOINT_OUTPUT_DIR}" \
  --knn_temp 1 \
  --lmbda 0.55 \
  --knn_generator_path "${MEMDEC_MODEL}" \
  --report_to none

echo "[INFO] reproduction finished"
