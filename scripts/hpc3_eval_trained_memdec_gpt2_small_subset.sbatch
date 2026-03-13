#!/bin/bash
#SBATCH -J memdec_eval
#SBATCH -p acd_u
#SBATCH -n 48
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/jzhu997/MemoryDecoder

set -euo pipefail

mkdir -p ./logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUBSET_TAG=${SUBSET_TAG:-train1of8}
STUDENT_TAG=${STUDENT_TAG:-gpt2_small}
LOG_FILE="./logs/eval_trained_memdec_${STUDENT_TAG}_${SUBSET_TAG}_${TIMESTAMP}_${SLURM_JOB_ID}.log"
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
CPU_THREADS_PER_PROC=${CPU_THREADS_PER_PROC:-12}
export OMP_NUM_THREADS="${CPU_THREADS_PER_PROC}"

BASE_MODEL=${BASE_MODEL:-/data/user/jzhu997/cache/Models/gpt2}
PROC_DATASET=/data/user/jzhu997/MemoryDecoder/dataset/wikitext-gpt2-${SUBSET_TAG}
TRAIN_RUN_NAME=${TRAIN_RUN_NAME:-train_memdec_${STUDENT_TAG}_${SUBSET_TAG}}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-$(latest_run_dir "${TRAIN_RUN_NAME}")}
BASE_EVAL_OUTPUT_DIR=${BASE_EVAL_OUTPUT_DIR:-$(result_dir "base_gpt2_small_${SUBSET_TAG}")}
JOINT_EVAL_OUTPUT_DIR=${JOINT_EVAL_OUTPUT_DIR:-$(result_dir "eval_trained_memdec_${STUDENT_TAG}_${SUBSET_TAG}")}
LMBDA=${LMBDA:-0.80}
KNN_TEMP=${KNN_TEMP:-1.0}
EVAL_SPLIT=${EVAL_SPLIT:-validation}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-1}
PIN_MEMORY=${PIN_MEMORY:-0}
ENABLE_TRACKING=${ENABLE_TRACKING:-1}
TRACKING_BACKEND=${TRACKING_BACKEND:-tensorboard}
BASE_TRACKING_DIR=${BASE_TRACKING_DIR:-$(tb_dir "base_gpt2_small_${SUBSET_TAG}")}
JOINT_TRACKING_DIR=${JOINT_TRACKING_DIR:-$(tb_dir "eval_trained_memdec_${STUDENT_TAG}_${SUBSET_TAG}")}
BASE_RUN_NAME=${BASE_RUN_NAME:-base_gpt2_small_${SUBSET_TAG}_${SLURM_JOB_ID}}
JOINT_RUN_NAME=${JOINT_RUN_NAME:-${STUDENT_TAG}_${SUBSET_TAG}_eval_${SLURM_JOB_ID}}

echo "[INFO] host=$(hostname)"
echo "[INFO] workdir=$PWD"
echo "[INFO] log_file=$LOG_FILE"
echo "[INFO] subset_tag=${SUBSET_TAG}"
echo "[INFO] student_tag=${STUDENT_TAG}"
echo "[INFO] eval_split=${EVAL_SPLIT}"
echo "[INFO] dataloader_num_workers=${DATALOADER_NUM_WORKERS}"
echo "[INFO] dataloader_prefetch_factor=${DATALOADER_PREFETCH_FACTOR}"
echo "[INFO] pin_memory=${PIN_MEMORY}"
echo "[INFO] enable_tracking=${ENABLE_TRACKING}"
echo "[INFO] tracking_backend=${TRACKING_BACKEND}"
echo "[INFO] base_tracking_dir=${BASE_TRACKING_DIR}"
echo "[INFO] joint_tracking_dir=${JOINT_TRACKING_DIR}"
echo "[INFO] train_run_name=${TRAIN_RUN_NAME}"
echo "[INFO] train_output_dir=${TRAIN_OUTPUT_DIR}"
echo "[INFO] nvidia_lib_dirs=${NVIDIA_LIB_DIRS:-none}"
echo "[INFO] num_gpus=${NUM_GPUS}"
echo "[INFO] cpu_threads_per_proc=${CPU_THREADS_PER_PROC}"
nvidia-smi || true

MEMDEC_MODEL=${MEMDEC_MODEL:-$(latest_epoch_checkpoint "${TRAIN_OUTPUT_DIR}")}

test -d "${BASE_MODEL}" || { echo "[ERROR] missing base model ${BASE_MODEL}"; exit 1; }
test -d "${PROC_DATASET}" || { echo "[ERROR] missing processed dataset ${PROC_DATASET}"; exit 1; }
test -n "${TRAIN_OUTPUT_DIR}" || { echo "[ERROR] no training run found for ${TRAIN_RUN_NAME}"; exit 1; }
test -n "${MEMDEC_MODEL}" || { echo "[ERROR] no trained checkpoint found under ${TRAIN_OUTPUT_DIR}"; exit 1; }
test -d "${MEMDEC_MODEL}" || { echo "[ERROR] missing trained checkpoint ${MEMDEC_MODEL}"; exit 1; }

mkdir -p "${BASE_EVAL_OUTPUT_DIR}" "${JOINT_EVAL_OUTPUT_DIR}"
mkdir -p "${BASE_TRACKING_DIR}" "${JOINT_TRACKING_DIR}"

PIN_MEMORY_ARGS=()
if [ "${PIN_MEMORY}" != "1" ]; then
  PIN_MEMORY_ARGS+=(--no_pin_memory)
fi

BASE_TRACKING_ARGS=()
JOINT_TRACKING_ARGS=()
if [ "${ENABLE_TRACKING}" = "1" ]; then
  BASE_TRACKING_ARGS=(
    --report_to "${TRACKING_BACKEND}"
    --logging_dir "${BASE_TRACKING_DIR}"
    --run_name "${BASE_RUN_NAME}"
  )
  JOINT_TRACKING_ARGS=(
    --with_tracking
    --report_to "${TRACKING_BACKEND}"
    --project_name memdec_eval
    --run_name "${JOINT_RUN_NAME}"
    --tracking_dir "${JOINT_TRACKING_DIR}"
  )
fi

accelerate launch --multi_gpu --num_machines 1 --num_processes "${NUM_GPUS}" --mixed_precision bf16 \
  -m train_base \
  --model_name_or_path "${BASE_MODEL}" \
  --dataset_name "${PROC_DATASET}" \
  --per_device_eval_batch_size 16 \
  --do_eval \
  --eval_subset "${EVAL_SPLIT}" \
  --output_dir "${BASE_EVAL_OUTPUT_DIR}" \
  "${BASE_TRACKING_ARGS[@]}"

accelerate launch --multi_gpu --num_machines 1 --num_processes "${NUM_GPUS}" --mixed_precision bf16 \
  -m evaluate_joint \
  --do_test \
  --model_name_or_path "${BASE_MODEL}" \
  --dataset_name "${PROC_DATASET}" \
  --dataset_split_name "${EVAL_SPLIT}" \
  --per_device_eval_batch_size 16 \
  --output_dir "${JOINT_EVAL_OUTPUT_DIR}" \
  --knn_temp "${KNN_TEMP}" \
  --lmbda "${LMBDA}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
  --knn_generator_path "${MEMDEC_MODEL}" \
  "${JOINT_TRACKING_ARGS[@]}" \
  "${PIN_MEMORY_ARGS[@]}"

echo "[INFO] trained Memory Decoder evaluation finished"
