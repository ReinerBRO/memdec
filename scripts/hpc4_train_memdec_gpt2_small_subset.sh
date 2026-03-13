#!/usr/bin/env bash

set -euo pipefail

DEV_NODE=${DEV_NODE:-miku}
OWNER_REF_DIR=${OWNER_REF_DIR:-/data/user/user06}
MINICONDA_HOME=${MINICONDA_HOME:-/data/user/user06/miniconda3}
ENV_NAME=${ENV_NAME:-MemoryDecoder}
REPO_ROOT=${REPO_ROOT:-/data/user/user06/MemoryDecoder}

SUBSET_TAG=${SUBSET_TAG:-train1of8}
STUDENT_TAG=${STUDENT_TAG:-gpt2_small}
RUN_NAME=${RUN_NAME:-train_memdec_${STUDENT_TAG}_${SUBSET_TAG}}
NUM_NPUS=${NUM_NPUS:-8}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
CPU_THREADS_PER_PROC=${CPU_THREADS_PER_PROC:-1}

TRAIN_MODEL=${TRAIN_MODEL:-/data/user/user06/cache/Models/gpt2-finetuned-wikitext103}
PROC_DATASET=${PROC_DATASET:-${REPO_ROOT}/dataset/wikitext-gpt2-${SUBSET_TAG}}
KNN_SAVE_PATH=${KNN_SAVE_PATH:-${REPO_ROOT}/dstore/gpt2-xl-wikitext103-${SUBSET_TAG}/knn_gpt2_train_1600.arrow}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}
NUM_EPOCHS=${NUM_EPOCHS:-15}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
ALPHA=${ALPHA:-0.5}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-2}
PIN_MEMORY=${PIN_MEMORY:-0}
ENABLE_TRACKING=${ENABLE_TRACKING:-1}
TRACKING_BACKEND=${TRACKING_BACKEND:-tensorboard}
AUTO_RESUME=${AUTO_RESUME:-0}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
MIXED_PRECISION=${MIXED_PRECISION:-no}

REMOTE_VARS_FILE=$(mktemp)
trap 'rm -f "${REMOTE_VARS_FILE}"' EXIT

{
  printf 'PROJECT_ROOT=%q\n' "${REPO_ROOT}"
  printf 'OWNER_REF_DIR=%q\n' "${OWNER_REF_DIR}"
  printf 'MINICONDA_HOME=%q\n' "${MINICONDA_HOME}"
  printf 'ENV_NAME=%q\n' "${ENV_NAME}"
  printf 'RUN_NAME=%q\n' "${RUN_NAME}"
  printf 'SUBSET_TAG=%q\n' "${SUBSET_TAG}"
  printf 'STUDENT_TAG=%q\n' "${STUDENT_TAG}"
  printf 'NUM_NPUS=%q\n' "${NUM_NPUS}"
  printf 'ASCEND_RT_VISIBLE_DEVICES=%q\n' "${ASCEND_RT_VISIBLE_DEVICES}"
  printf 'CPU_THREADS_PER_PROC=%q\n' "${CPU_THREADS_PER_PROC}"
  printf 'TRAIN_MODEL=%q\n' "${TRAIN_MODEL}"
  printf 'PROC_DATASET=%q\n' "${PROC_DATASET}"
  printf 'KNN_SAVE_PATH=%q\n' "${KNN_SAVE_PATH}"
  printf 'TRAIN_BATCH_SIZE=%q\n' "${TRAIN_BATCH_SIZE}"
  printf 'EVAL_BATCH_SIZE=%q\n' "${EVAL_BATCH_SIZE}"
  printf 'GRAD_ACCUM=%q\n' "${GRAD_ACCUM}"
  printf 'NUM_EPOCHS=%q\n' "${NUM_EPOCHS}"
  printf 'MAX_TRAIN_STEPS=%q\n' "${MAX_TRAIN_STEPS}"
  printf 'LEARNING_RATE=%q\n' "${LEARNING_RATE}"
  printf 'ALPHA=%q\n' "${ALPHA}"
  printf 'DATALOADER_NUM_WORKERS=%q\n' "${DATALOADER_NUM_WORKERS}"
  printf 'DATALOADER_PREFETCH_FACTOR=%q\n' "${DATALOADER_PREFETCH_FACTOR}"
  printf 'PIN_MEMORY=%q\n' "${PIN_MEMORY}"
  printf 'ENABLE_TRACKING=%q\n' "${ENABLE_TRACKING}"
  printf 'TRACKING_BACKEND=%q\n' "${TRACKING_BACKEND}"
  printf 'AUTO_RESUME=%q\n' "${AUTO_RESUME}"
  printf 'RESUME_FROM_CHECKPOINT=%q\n' "${RESUME_FROM_CHECKPOINT}"
  printf 'MIXED_PRECISION=%q\n' "${MIXED_PRECISION}"
} > "${REMOTE_VARS_FILE}"

{
  cat "${REMOTE_VARS_FILE}"
  cat <<'EOS'
set -euo pipefail

OWNER_UID="$(stat -c '%u' "${OWNER_REF_DIR}")"
OWNER_GID="$(stat -c '%g' "${OWNER_REF_DIR}")"

ensure_owner_dir() {
  if [ "$(id -u)" = "0" ]; then
    install -d -m 755 -o "${OWNER_UID}" -g "${OWNER_GID}" "$1"
  else
    mkdir -p "$1"
  fi
}

ensure_owner_path() {
  local target="$1"
  local rel="${target#"${PROJECT_ROOT}/"}"
  local path="${PROJECT_ROOT}"
  IFS='/' read -r -a parts <<< "${rel}"
  for part in "${parts[@]}"; do
    [ -n "${part}" ] || continue
    path="${path}/${part}"
    ensure_owner_dir "${path}"
  done
}

fix_owner() {
  if [ "$(id -u)" = "0" ]; then
    chown -R "${OWNER_UID}:${OWNER_GID}" "$@" 2>/dev/null || true
  fi
}

ensure_owner_path "${PROJECT_ROOT}/logs"
ensure_owner_path "${PROJECT_ROOT}/ckpt"
ensure_owner_path "${PROJECT_ROOT}/tb_logs"
ensure_owner_path "${PROJECT_ROOT}/wandb_logs"

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source "${MINICONDA_HOME}/etc/profile.d/conda.sh"
set -u

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/output_layout.sh"

OUTPUT_DIR=${OUTPUT_DIR:-$(ckpt_dir "${RUN_NAME}")}
TRACKING_DIR=${TRACKING_DIR:-$(tb_dir "${RUN_NAME}")}
LATEST_CKPT_LINK=$(ckpt_latest_link "${RUN_NAME}")
LOG_TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJECT_ROOT}/logs/${RUN_NAME}_${LOG_TS}.log"
LAUNCH_SCRIPT="${PROJECT_ROOT}/logs/${RUN_NAME}_${LOG_TS}.launch.sh"

ensure_owner_path "${OUTPUT_DIR}"
ensure_owner_path "${TRACKING_DIR}"
export PROJECT_ROOT OWNER_REF_DIR MINICONDA_HOME ENV_NAME RUN_NAME NUM_NPUS
export ASCEND_RT_VISIBLE_DEVICES CPU_THREADS_PER_PROC TRAIN_MODEL PROC_DATASET KNN_SAVE_PATH
export TRAIN_BATCH_SIZE EVAL_BATCH_SIZE GRAD_ACCUM NUM_EPOCHS MAX_TRAIN_STEPS LEARNING_RATE ALPHA
export DATALOADER_NUM_WORKERS DATALOADER_PREFETCH_FACTOR PIN_MEMORY ENABLE_TRACKING
export TRACKING_BACKEND AUTO_RESUME RESUME_FROM_CHECKPOINT MIXED_PRECISION OUTPUT_DIR TRACKING_DIR
export LATEST_CKPT_LINK LOG_FILE LAUNCH_SCRIPT

cleanup() {
  local exit_code=$?
  local paths=()

  [ -n "${OUTPUT_DIR:-}" ] && paths+=("${OUTPUT_DIR}")
  [ -n "${TRACKING_DIR:-}" ] && paths+=("${TRACKING_DIR}")
  [ -n "${LOG_FILE:-}" ] && paths+=("${LOG_FILE}")
  [ -n "${LAUNCH_SCRIPT:-}" ] && paths+=("${LAUNCH_SCRIPT}")

  if [ "${exit_code}" -eq 0 ] && [ -n "${LATEST_CKPT_LINK:-}" ] && [ -n "${OUTPUT_DIR:-}" ]; then
    ln -sfn "${OUTPUT_DIR}" "${LATEST_CKPT_LINK}"
    paths+=("${LATEST_CKPT_LINK}")
  fi

  if [ "${#paths[@]}" -gt 0 ]; then
    fix_owner "${paths[@]}"
  fi

  exit "${exit_code}"
}
trap cleanup EXIT

cat > "${LAUNCH_SCRIPT}" <<'INNER'
#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source "${MINICONDA_HOME}/etc/profile.d/conda.sh"
set -u
conda activate "${ENV_NAME}"

cd "${PROJECT_ROOT}"
source scripts/output_layout.sh

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${CPU_THREADS_PER_PROC}"
export HF_HOME="${OWNER_REF_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HCCL_CONNECT_TIMEOUT=1800
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

mkdir -p "$(dirname "${LOG_FILE}")" "${OUTPUT_DIR}" "${TRACKING_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "[INFO] host=$(hostname)"
echo "[INFO] run_name=${RUN_NAME}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] tracking_dir=${TRACKING_DIR}"
echo "[INFO] train_model=${TRAIN_MODEL}"
echo "[INFO] proc_dataset=${PROC_DATASET}"
echo "[INFO] knn_save_path=${KNN_SAVE_PATH}"
echo "[INFO] ascend_visible_devices=${ASCEND_RT_VISIBLE_DEVICES}"

python - <<'PY'
import torch, torch_npu
print("NPU_AVAILABLE", torch.npu.is_available())
print("NPU_COUNT", torch.npu.device_count())
PY

CMD=(
  accelerate launch --multi_gpu --num_machines 1 --num_processes "${NUM_NPUS}" --mixed_precision "${MIXED_PRECISION}" -m train_memdec
  --model_name_or_path "${TRAIN_MODEL}"
  --dataset_name "${PROC_DATASET}"
  --dataset_split_name train
  --knn_save_path "${KNN_SAVE_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --learning_rate "${LEARNING_RATE}"
  --lr_scheduler_type linear
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
  --num_train_epochs "${NUM_EPOCHS}"
  --seed 42
  --alpha "${ALPHA}"
  --checkpointing_steps epoch
  --logging_steps 20
)

if [ -n "${MAX_TRAIN_STEPS}" ]; then
  CMD+=(--max_train_steps "${MAX_TRAIN_STEPS}")
fi

if [ "${ENABLE_TRACKING}" = "1" ]; then
  CMD+=(
    --with_tracking
    --report_to "${TRACKING_BACKEND}"
    --project_name memdec_train
    --run_name "${RUN_NAME}"
    --tracking_dir "${TRACKING_DIR}"
  )
fi

if [ "${PIN_MEMORY}" != "1" ]; then
  CMD+=(--no_pin_memory)
fi

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
elif [ "${AUTO_RESUME}" = "1" ]; then
  RESUME_RUN_ROOT="$(latest_run_dir "${RUN_NAME}")"
  RESUME_CKPT="$(latest_epoch_checkpoint "${RESUME_RUN_ROOT}")"
  if [ -n "${RESUME_CKPT}" ]; then
    echo "[INFO] resuming from ${RESUME_CKPT}"
    CMD+=(--resume_from_checkpoint "${RESUME_CKPT}")
  fi
fi

"${CMD[@]}"
INNER

chmod 755 "${LAUNCH_SCRIPT}"
bash "${LAUNCH_SCRIPT}"
echo "[INFO] latest checkpoint link -> ${LATEST_CKPT_LINK}"
echo "[INFO] log_file=${LOG_FILE}"
EOS
} | ssh "${DEV_NODE}" 'bash -s'
