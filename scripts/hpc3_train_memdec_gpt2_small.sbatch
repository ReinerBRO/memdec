#!/bin/bash
#SBATCH -J memdec_train
#SBATCH -p acd_u
#SBATCH -n 32
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/jzhu997/MemoryDecoder

set -euo pipefail

mkdir -p ./logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./logs/train_memdec_gpt2_small_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

module load cuda/12.2 2>/dev/null || module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
set +u
source /data/user/jzhu997/envs/memorydecoder/bin/activate
set -u

ENV_ROOT=/data/user/jzhu997/envs/memorydecoder
PYTHON_BIN="${ENV_ROOT}/bin/python"
NVIDIA_LIB_DIRS="$(${PYTHON_BIN} - <<'PY'
from pathlib import Path
from sysconfig import get_paths

root = Path(get_paths()["purelib"]) / "nvidia"
lib_dirs = sorted(str(path) for path in root.glob("*/lib") if path.is_dir())
print(":".join(lib_dirs))
PY
)"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
if [ -n "${NVIDIA_LIB_DIRS}" ]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
fi
export HF_HOME=/data/user/jzhu997/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_IB_DISABLE=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/output_layout.sh"
NUM_GPUS=${NUM_GPUS:-4}
CPU_THREADS_PER_PROC=${CPU_THREADS_PER_PROC:-8}
export OMP_NUM_THREADS="${CPU_THREADS_PER_PROC}"

HPC2_CACHE=/hpc2hdd/home/jzhu997/cache
HPC3_CACHE=/data/user/jzhu997/cache
TRAIN_MODEL_HPC2=${HPC2_CACHE}/Models/gpt2-finetuned-wikitext103
TRAIN_MODEL_HPC3=${HPC3_CACHE}/Models/gpt2-finetuned-wikitext103
PROC_DATASET=/data/user/jzhu997/MemoryDecoder/dataset/wikitext-gpt2
KNN_SAVE_PATH=/data/user/jzhu997/MemoryDecoder/dstore/gpt2-xl-wikitext103-train/knn_gpt2_train_1600.arrow
RUN_NAME=${RUN_NAME:-train_memdec_gpt2_small}
OUTPUT_DIR=${OUTPUT_DIR:-$(ckpt_dir "${RUN_NAME}")}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}
NUM_EPOCHS=${NUM_EPOCHS:-70}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
ALPHA=${ALPHA:-0.5}
ENABLE_TRACKING=${ENABLE_TRACKING:-1}
TRACKING_BACKEND=${TRACKING_BACKEND:-tensorboard}
TRACKING_PROJECT_NAME=${TRACKING_PROJECT_NAME:-memdec_train}
TRACKING_DIR=${TRACKING_DIR:-$(tb_dir "train_memdec_gpt2_small")}
TRACKING_RUN_NAME=${TRACKING_RUN_NAME:-${RUN_NAME}_${SLURM_JOB_ID}}
AUTO_RESUME=${AUTO_RESUME:-0}
LATEST_CKPT_LINK=$(ckpt_latest_link "${RUN_NAME}")

echo "[INFO] host=$(hostname)"
echo "[INFO] workdir=$PWD"
echo "[INFO] log_file=$LOG_FILE"
echo "[INFO] enable_tracking=${ENABLE_TRACKING}"
echo "[INFO] tracking_backend=${TRACKING_BACKEND}"
echo "[INFO] tracking_dir=${TRACKING_DIR}"
echo "[INFO] run_name=${RUN_NAME}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] nvidia_lib_dirs=${NVIDIA_LIB_DIRS:-none}"
echo "[INFO] num_gpus=${NUM_GPUS}"
echo "[INFO] cpu_threads_per_proc=${CPU_THREADS_PER_PROC}"
nvidia-smi || true

if [ ! -d "${TRAIN_MODEL_HPC3}" ]; then
  echo "[INFO] pulling training model from HPC2"
  mkdir -p "${HPC3_CACHE}/Models"
  rsync -avh --delete "jzhu997@hpc2login.hpc.hkust-gz.edu.cn:${TRAIN_MODEL_HPC2}/" "${TRAIN_MODEL_HPC3}/"
fi

test -d "${TRAIN_MODEL_HPC3}" || { echo "[ERROR] missing training model ${TRAIN_MODEL_HPC3}"; exit 1; }
test -d "${PROC_DATASET}" || { echo "[ERROR] missing processed dataset ${PROC_DATASET}"; exit 1; }
test -f "${KNN_SAVE_PATH}" || { echo "[ERROR] missing kNN signal file ${KNN_SAVE_PATH}"; exit 1; }

mkdir -p "${OUTPUT_DIR}" "${TRACKING_DIR}"

TRACKING_ARGS=()
if [ "${ENABLE_TRACKING}" = "1" ]; then
  TRACKING_ARGS=(
    --with_tracking
    --report_to "${TRACKING_BACKEND}"
    --project_name "${TRACKING_PROJECT_NAME}"
    --run_name "${TRACKING_RUN_NAME}"
    --tracking_dir "${TRACKING_DIR}"
  )
fi

TRAIN_LIMIT_ARGS=()
if [ -n "${MAX_TRAIN_STEPS}" ]; then
  TRAIN_LIMIT_ARGS=(--max_train_steps "${MAX_TRAIN_STEPS}")
fi

RESUME_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
elif [ "${AUTO_RESUME}" = "1" ]; then
  RESUME_RUN_ROOT=$(latest_run_dir "${RUN_NAME}")
  LATEST_CHECKPOINT=$(latest_epoch_checkpoint "${RESUME_RUN_ROOT}")
  if [ -n "${LATEST_CHECKPOINT}" ]; then
    echo "[INFO] resuming from ${LATEST_CHECKPOINT}"
    RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CHECKPOINT}")
  fi
fi

accelerate launch --multi_gpu --num_machines 1 --num_processes "${NUM_GPUS}" --mixed_precision bf16 \
  -m train_memdec \
  --model_name_or_path "${TRAIN_MODEL_HPC3}" \
  --dataset_name "${PROC_DATASET}" \
  --dataset_split_name train \
  --knn_save_path "${KNN_SAVE_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type linear \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  "${TRAIN_LIMIT_ARGS[@]}" \
  --seed 42 \
  --alpha "${ALPHA}" \
  --checkpointing_steps epoch \
  --logging_steps 20 \
  "${TRACKING_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

update_latest_link "${OUTPUT_DIR}" "${LATEST_CKPT_LINK}"
echo "[INFO] Memory Decoder training finished"
