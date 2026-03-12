#!/bin/bash
#SBATCH -J memdec_knn
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
LOG_FILE="./logs/build_knn_signals_gpt2_small_${SUBSET_TAG}_${TIMESTAMP}_${SLURM_JOB_ID}.log"
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
NUM_GPUS=${NUM_GPUS:-4}
CPU_THREADS_PER_PROC=${CPU_THREADS_PER_PROC:-12}
export OMP_NUM_THREADS="${CPU_THREADS_PER_PROC}"

HPC2_CACHE=/hpc2hdd/home/jzhu997/cache
HPC3_CACHE=/data/user/jzhu997/cache
RAW_DATASET=${HPC3_CACHE}/data/wikitext-103-raw
FULL_PROC_DATASET=/data/user/jzhu997/MemoryDecoder/dataset/wikitext-gpt2-full
PROC_DATASET=/data/user/jzhu997/MemoryDecoder/dataset/wikitext-gpt2-${SUBSET_TAG}
TRAIN_FRACTION=${TRAIN_FRACTION:-0.125}
SUBSET_SEED=${SUBSET_SEED:-42}
SUBSET_BUCKETS=${SUBSET_BUCKETS:-128}
SOURCE_MODEL_HPC2=${HPC2_CACHE}/Models/gpt2-xl-finetuned-wikitext103
SOURCE_MODEL_HPC3=${HPC3_CACHE}/Models/gpt2-xl-finetuned-wikitext103
DSTORE_DIR=/data/user/jzhu997/MemoryDecoder/dstore/gpt2-xl-wikitext103-${SUBSET_TAG}
OUTPUT_DIR=/data/user/jzhu997/MemoryDecoder/outputs/build_knn_signals_gpt2_small_${SUBSET_TAG}
DIMENSION=1600
BATCH_SIZE_EVAL=${BATCH_SIZE_EVAL:-16}
BATCH_SIZE_KNN=${BATCH_SIZE_KNN:-8192}
K=${K:-1024}
KNN_TEMP=${KNN_TEMP:-16.0}
PROBE=${PROBE:-32}
NCENTROIDS=${NCENTROIDS:-4096}
CODE_SIZE=${CODE_SIZE:-64}
NUM_KEYS_TO_ADD=${NUM_KEYS_TO_ADD:-10000000}

DSTORE_PATH="${DSTORE_DIR}/dstore_gpt2_train_${DIMENSION}.arrow"
VAL_PATH="${DSTORE_DIR}/train_vals.pkl"
INDEX_PATH="${DSTORE_DIR}/train_${DIMENSION}.index"
OUTPUT_PATH="${DSTORE_DIR}/knn_gpt2_train_${DIMENSION}.arrow"
STEP1_MARKER="${DSTORE_DIR}/.step1_complete"
STEP2_MARKER="${DSTORE_DIR}/.step2_complete"
STEP3_MARKER="${DSTORE_DIR}/.step3_complete"

echo "[INFO] host=$(hostname)"
echo "[INFO] workdir=$PWD"
echo "[INFO] log_file=$LOG_FILE"
echo "[INFO] subset_tag=${SUBSET_TAG}"
echo "[INFO] train_fraction=${TRAIN_FRACTION}"
echo "[INFO] subset_seed=${SUBSET_SEED}"
echo "[INFO] subset_buckets=${SUBSET_BUCKETS}"
echo "[INFO] nvidia_lib_dirs=${NVIDIA_LIB_DIRS:-none}"
echo "[INFO] num_gpus=${NUM_GPUS}"
echo "[INFO] cpu_threads_per_proc=${CPU_THREADS_PER_PROC}"
nvidia-smi || true

if [ ! -f "${FULL_PROC_DATASET}/dstore_summary.json" ]; then
  echo "[INFO] preprocessing full dataset into ${FULL_PROC_DATASET}"
  python utils/preprocess_dataset.py \
    --dataset_name "${RAW_DATASET}" \
    --dataset_config_name wikitext-103-raw-v1 \
    --tokenizer_path /data/user/jzhu997/cache/Models/gpt2 \
    --output_dir "${FULL_PROC_DATASET}" \
    --num_proc 8
else
  echo "[INFO] using existing full processed dataset at ${FULL_PROC_DATASET}"
fi

if [ ! -f "${PROC_DATASET}/subset_manifest.json" ]; then
  echo "[INFO] building subset dataset into ${PROC_DATASET}"
  rm -rf "${PROC_DATASET}"
  python utils/subsample_processed_dataset.py \
    --input_dir "${FULL_PROC_DATASET}" \
    --output_dir "${PROC_DATASET}" \
    --train_fraction "${TRAIN_FRACTION}" \
    --seed "${SUBSET_SEED}" \
    --num_buckets "${SUBSET_BUCKETS}"
else
  echo "[INFO] using existing subset dataset at ${PROC_DATASET}"
fi

if [ ! -d "${SOURCE_MODEL_HPC3}" ]; then
  echo "[INFO] pulling source model from HPC2"
  mkdir -p "${HPC3_CACHE}/Models"
  rsync -avh --delete "jzhu997@hpc2login.hpc.hkust-gz.edu.cn:${SOURCE_MODEL_HPC2}/" "${SOURCE_MODEL_HPC3}/"
fi

if [ ! -d "${SOURCE_MODEL_HPC3}" ]; then
  echo "[ERROR] missing source model at ${SOURCE_MODEL_HPC3}"
  exit 1
fi

python - <<'PY'
import faiss
print("faiss_file", faiss.__file__)
print("num_gpus", faiss.get_num_gpus())
PY

mkdir -p "${DSTORE_DIR}" "${OUTPUT_DIR}"

if [ ! -f "${STEP1_MARKER}" ]; then
  echo "[INFO] step 1/3 saving datastore embeddings with ${NUM_GPUS} GPUs"
  rm -f "${DSTORE_PATH}" "${INDEX_PATH}" "${VAL_PATH}" "${OUTPUT_PATH}"
  rm -f "${STEP1_MARKER}" "${STEP2_MARKER}" "${STEP3_MARKER}"
  accelerate launch --multi_gpu --num_machines 1 --num_processes "${NUM_GPUS}" --mixed_precision bf16 \
    -m train_base \
    --model_name_or_path "${SOURCE_MODEL_HPC3}" \
    --dataset_name "${PROC_DATASET}" \
    --do_eval \
    --eval_subset train \
    --per_device_eval_batch_size "${BATCH_SIZE_EVAL}" \
    --output_dir "${OUTPUT_DIR}" \
    --dstore_dir "${DSTORE_DIR}" \
    --save_knnlm_dstore \
    --report_to none
  test -f "${DSTORE_PATH}" || { echo "[ERROR] missing datastore ${DSTORE_PATH}"; exit 1; }
  touch "${STEP1_MARKER}"
else
  echo "[INFO] step 1/3 datastore already exists at ${DSTORE_PATH}"
fi

test -f "${DSTORE_PATH}" || { echo "[ERROR] missing datastore ${DSTORE_PATH}"; exit 1; }

if [ ! -f "${STEP2_MARKER}" ]; then
  echo "[INFO] step 2/3 building faiss index"
  rm -f "${INDEX_PATH}" "${VAL_PATH}" "${OUTPUT_PATH}"
  rm -f "${STEP2_MARKER}" "${STEP3_MARKER}"
  python -m knn_utils.build_index \
    --dstore_path "${DSTORE_PATH}" \
    --num_keys_to_add_at_a_time "${NUM_KEYS_TO_ADD}" \
    --ncentroids "${NCENTROIDS}" \
    --code_size "${CODE_SIZE}" \
    --probe "${PROBE}"
  test -f "${INDEX_PATH}" || { echo "[ERROR] missing index ${INDEX_PATH}"; exit 1; }
  test -f "${VAL_PATH}" || { echo "[ERROR] missing vals ${VAL_PATH}"; exit 1; }
  touch "${STEP2_MARKER}"
else
  echo "[INFO] step 2/3 index already exists at ${INDEX_PATH}"
fi

test -f "${INDEX_PATH}" || { echo "[ERROR] missing index ${INDEX_PATH}"; exit 1; }
test -f "${VAL_PATH}" || { echo "[ERROR] missing vals ${VAL_PATH}"; exit 1; }

if [ ! -f "${STEP3_MARKER}" ]; then
  echo "[INFO] step 3/3 saving kNN distributions with ${NUM_GPUS} GPUs"
  rm -f "${OUTPUT_PATH}" "${STEP3_MARKER}"
  accelerate launch --multi_gpu --num_machines 1 --num_processes "${NUM_GPUS}" --mixed_precision bf16 \
    -m knn_utils.saveKNNMulti \
    --model_path "${SOURCE_MODEL_HPC3}" \
    --dstore_path "${DSTORE_PATH}" \
    --val_path "${VAL_PATH}" \
    --index_path "${INDEX_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --k "${K}" \
    --knn_temp "${KNN_TEMP}" \
    --probe "${PROBE}" \
    --batch_size "${BATCH_SIZE_KNN}" \
    --ignore_first True \
    --knn_gpu
  test -f "${OUTPUT_PATH}" || { echo "[ERROR] missing knn distributions ${OUTPUT_PATH}"; exit 1; }
  touch "${STEP3_MARKER}"
else
  echo "[INFO] step 3/3 kNN distributions already exist at ${OUTPUT_PATH}"
fi

test -f "${OUTPUT_PATH}" || { echo "[ERROR] missing knn distributions ${OUTPUT_PATH}"; exit 1; }

echo "[INFO] KNN signal pipeline finished"
