#!/usr/bin/env bash

set -euo pipefail

DEV_NODE=${DEV_NODE:-miku}
OWNER_REF_DIR=${OWNER_REF_DIR:-/data/user/user06}
MINICONDA_HOME=${MINICONDA_HOME:-/data/user/user06/miniconda3}
ENV_NAME=${ENV_NAME:-MemoryDecoder}
REPO_ROOT=${REPO_ROOT:-/data/user/user06/MemoryDecoder}

SUBSET_TAG=${SUBSET_TAG:-train1of8}
STUDENT_TAG=${STUDENT_TAG:-gpt2_small}
TRAIN_RUN_NAME=${TRAIN_RUN_NAME:-train_memdec_${STUDENT_TAG}_${SUBSET_TAG}}
NUM_NPUS=${NUM_NPUS:-8}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
CPU_THREADS_PER_PROC=${CPU_THREADS_PER_PROC:-1}

BASE_MODEL=${BASE_MODEL:-/data/user/user06/cache/Models/gpt2}
PROC_DATASET=${PROC_DATASET:-${REPO_ROOT}/dataset/wikitext-gpt2-${SUBSET_TAG}}
MEMDEC_MODEL=${MEMDEC_MODEL:-}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-}
BASE_EVAL_OUTPUT_DIR=${BASE_EVAL_OUTPUT_DIR:-}
JOINT_EVAL_OUTPUT_DIR=${JOINT_EVAL_OUTPUT_DIR:-}
BASE_TRACKING_DIR=${BASE_TRACKING_DIR:-}
JOINT_TRACKING_DIR=${JOINT_TRACKING_DIR:-}
BASE_RUN_NAME=${BASE_RUN_NAME:-}
JOINT_RUN_NAME=${JOINT_RUN_NAME:-}

EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
EVAL_SPLIT=${EVAL_SPLIT:-validation}
LMBDA=${LMBDA:-0.80}
KNN_TEMP=${KNN_TEMP:-1.0}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-1}
PIN_MEMORY=${PIN_MEMORY:-0}
ENABLE_TRACKING=${ENABLE_TRACKING:-1}
TRACKING_BACKEND=${TRACKING_BACKEND:-tensorboard}
MIXED_PRECISION=${MIXED_PRECISION:-no}
JOINT_ENABLE_TRACKING=${JOINT_ENABLE_TRACKING:-0}

REMOTE_VARS_FILE=$(mktemp)
trap 'rm -f "${REMOTE_VARS_FILE}"' EXIT

{
  printf 'PROJECT_ROOT=%q\n' "${REPO_ROOT}"
  printf 'OWNER_REF_DIR=%q\n' "${OWNER_REF_DIR}"
  printf 'MINICONDA_HOME=%q\n' "${MINICONDA_HOME}"
  printf 'ENV_NAME=%q\n' "${ENV_NAME}"
  printf 'SUBSET_TAG=%q\n' "${SUBSET_TAG}"
  printf 'STUDENT_TAG=%q\n' "${STUDENT_TAG}"
  printf 'TRAIN_RUN_NAME=%q\n' "${TRAIN_RUN_NAME}"
  printf 'NUM_NPUS=%q\n' "${NUM_NPUS}"
  printf 'ASCEND_RT_VISIBLE_DEVICES=%q\n' "${ASCEND_RT_VISIBLE_DEVICES}"
  printf 'CPU_THREADS_PER_PROC=%q\n' "${CPU_THREADS_PER_PROC}"
  printf 'BASE_MODEL=%q\n' "${BASE_MODEL}"
  printf 'PROC_DATASET=%q\n' "${PROC_DATASET}"
  printf 'MEMDEC_MODEL=%q\n' "${MEMDEC_MODEL}"
  printf 'TRAIN_OUTPUT_DIR=%q\n' "${TRAIN_OUTPUT_DIR}"
  printf 'BASE_EVAL_OUTPUT_DIR=%q\n' "${BASE_EVAL_OUTPUT_DIR}"
  printf 'JOINT_EVAL_OUTPUT_DIR=%q\n' "${JOINT_EVAL_OUTPUT_DIR}"
  printf 'BASE_TRACKING_DIR=%q\n' "${BASE_TRACKING_DIR}"
  printf 'JOINT_TRACKING_DIR=%q\n' "${JOINT_TRACKING_DIR}"
  printf 'BASE_RUN_NAME=%q\n' "${BASE_RUN_NAME}"
  printf 'JOINT_RUN_NAME=%q\n' "${JOINT_RUN_NAME}"
  printf 'EVAL_BATCH_SIZE=%q\n' "${EVAL_BATCH_SIZE}"
  printf 'EVAL_SPLIT=%q\n' "${EVAL_SPLIT}"
  printf 'LMBDA=%q\n' "${LMBDA}"
  printf 'KNN_TEMP=%q\n' "${KNN_TEMP}"
  printf 'DATALOADER_NUM_WORKERS=%q\n' "${DATALOADER_NUM_WORKERS}"
  printf 'DATALOADER_PREFETCH_FACTOR=%q\n' "${DATALOADER_PREFETCH_FACTOR}"
  printf 'PIN_MEMORY=%q\n' "${PIN_MEMORY}"
  printf 'ENABLE_TRACKING=%q\n' "${ENABLE_TRACKING}"
  printf 'TRACKING_BACKEND=%q\n' "${TRACKING_BACKEND}"
  printf 'MIXED_PRECISION=%q\n' "${MIXED_PRECISION}"
  printf 'JOINT_ENABLE_TRACKING=%q\n' "${JOINT_ENABLE_TRACKING}"
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
ensure_owner_path "${PROJECT_ROOT}/results"
ensure_owner_path "${PROJECT_ROOT}/tb_logs"
ensure_owner_path "${PROJECT_ROOT}/wandb_logs"

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source "${MINICONDA_HOME}/etc/profile.d/conda.sh"
set -u

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/output_layout.sh"

if [ -z "${TRAIN_OUTPUT_DIR}" ]; then
  TRAIN_OUTPUT_DIR="$(latest_run_dir "${TRAIN_RUN_NAME}")"
fi
if [ -z "${MEMDEC_MODEL}" ] && [ -n "${TRAIN_OUTPUT_DIR}" ]; then
  MEMDEC_MODEL="$(latest_epoch_checkpoint "${TRAIN_OUTPUT_DIR}")"
fi
if [ -z "${BASE_EVAL_OUTPUT_DIR}" ]; then
  BASE_EVAL_OUTPUT_DIR="$(result_dir "base_gpt2_small_${SUBSET_TAG}")"
fi
if [ -z "${JOINT_EVAL_OUTPUT_DIR}" ]; then
  JOINT_EVAL_OUTPUT_DIR="$(result_dir "eval_trained_memdec_${STUDENT_TAG}_${SUBSET_TAG}")"
fi
if [ -z "${BASE_TRACKING_DIR}" ]; then
  BASE_TRACKING_DIR="$(tb_dir "base_gpt2_small_${SUBSET_TAG}")"
fi
if [ -z "${JOINT_TRACKING_DIR}" ]; then
  JOINT_TRACKING_DIR="$(tb_dir "eval_trained_memdec_${STUDENT_TAG}_${SUBSET_TAG}")"
fi
if [ -z "${BASE_RUN_NAME}" ]; then
  BASE_RUN_NAME="base_gpt2_small_${SUBSET_TAG}"
fi
if [ -z "${JOINT_RUN_NAME}" ]; then
  JOINT_RUN_NAME="${STUDENT_TAG}_${SUBSET_TAG}_eval"
fi

test -d "${BASE_MODEL}" || { echo "[ERROR] missing base model ${BASE_MODEL}"; exit 1; }
test -d "${PROC_DATASET}" || { echo "[ERROR] missing processed dataset ${PROC_DATASET}"; exit 1; }
test -n "${TRAIN_OUTPUT_DIR}" || { echo "[ERROR] no training run found for ${TRAIN_RUN_NAME}"; exit 1; }
test -n "${MEMDEC_MODEL}" || { echo "[ERROR] no trained checkpoint found under ${TRAIN_OUTPUT_DIR}"; exit 1; }
test -d "${MEMDEC_MODEL}" || { echo "[ERROR] missing trained checkpoint ${MEMDEC_MODEL}"; exit 1; }

ensure_owner_path "${BASE_EVAL_OUTPUT_DIR}"
ensure_owner_path "${JOINT_EVAL_OUTPUT_DIR}"
ensure_owner_path "${BASE_TRACKING_DIR}"
ensure_owner_path "${JOINT_TRACKING_DIR}"

LOG_TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJECT_ROOT}/logs/eval_trained_memdec_${STUDENT_TAG}_${SUBSET_TAG}_${LOG_TS}.log"
LAUNCH_SCRIPT="${PROJECT_ROOT}/logs/eval_trained_memdec_${STUDENT_TAG}_${SUBSET_TAG}_${LOG_TS}.launch.sh"

export PROJECT_ROOT OWNER_REF_DIR MINICONDA_HOME ENV_NAME SUBSET_TAG STUDENT_TAG TRAIN_RUN_NAME
export NUM_NPUS ASCEND_RT_VISIBLE_DEVICES CPU_THREADS_PER_PROC BASE_MODEL PROC_DATASET MEMDEC_MODEL
export BASE_EVAL_OUTPUT_DIR JOINT_EVAL_OUTPUT_DIR BASE_TRACKING_DIR JOINT_TRACKING_DIR
export BASE_RUN_NAME JOINT_RUN_NAME EVAL_BATCH_SIZE EVAL_SPLIT LMBDA KNN_TEMP
export DATALOADER_NUM_WORKERS DATALOADER_PREFETCH_FACTOR PIN_MEMORY ENABLE_TRACKING TRACKING_BACKEND MIXED_PRECISION JOINT_ENABLE_TRACKING
export LOG_FILE LAUNCH_SCRIPT

cleanup() {
  local exit_code=$?
  local paths=()

  [ -n "${BASE_EVAL_OUTPUT_DIR:-}" ] && paths+=("${BASE_EVAL_OUTPUT_DIR}")
  [ -n "${JOINT_EVAL_OUTPUT_DIR:-}" ] && paths+=("${JOINT_EVAL_OUTPUT_DIR}")
  [ -n "${BASE_TRACKING_DIR:-}" ] && paths+=("${BASE_TRACKING_DIR}")
  [ -n "${JOINT_TRACKING_DIR:-}" ] && paths+=("${JOINT_TRACKING_DIR}")
  [ -n "${LOG_FILE:-}" ] && paths+=("${LOG_FILE}")
  [ -n "${LAUNCH_SCRIPT:-}" ] && paths+=("${LAUNCH_SCRIPT}")

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

mkdir -p "$(dirname "${LOG_FILE}")" "${BASE_EVAL_OUTPUT_DIR}" "${JOINT_EVAL_OUTPUT_DIR}" "${BASE_TRACKING_DIR}" "${JOINT_TRACKING_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "[INFO] host=$(hostname)"
echo "[INFO] train_run_name=${TRAIN_RUN_NAME}"
echo "[INFO] memdec_model=${MEMDEC_MODEL}"
echo "[INFO] base_output_dir=${BASE_EVAL_OUTPUT_DIR}"
echo "[INFO] joint_output_dir=${JOINT_EVAL_OUTPUT_DIR}"

BASE_REPORT_TO=none
JOINT_TRACKING_ARGS=()
if [ "${ENABLE_TRACKING}" = "1" ]; then
  BASE_REPORT_TO="${TRACKING_BACKEND}"
fi

if [ "${JOINT_ENABLE_TRACKING}" = "1" ]; then
  JOINT_TRACKING_ARGS=(
    --with_tracking
    --report_to "${TRACKING_BACKEND}"
    --project_name memdec_eval
    --run_name "${JOINT_RUN_NAME}"
    --tracking_dir "${JOINT_TRACKING_DIR}"
  )
fi

JOINT_PIN_MEMORY_ARGS=()
if [ "${PIN_MEMORY}" != "1" ]; then
  JOINT_PIN_MEMORY_ARGS+=(--no_pin_memory)
fi

BASE_CMD=(
  accelerate launch --multi_gpu --num_machines 1 --num_processes "${NUM_NPUS}" --mixed_precision "${MIXED_PRECISION}" -m train_base
  --model_name_or_path "${BASE_MODEL}"
  --dataset_name "${PROC_DATASET}"
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
  --do_eval
  --eval_subset "${EVAL_SPLIT}"
  --output_dir "${BASE_EVAL_OUTPUT_DIR}"
  --overwrite_output_dir
  --logging_dir "${BASE_TRACKING_DIR}"
  --run_name "${BASE_RUN_NAME}"
  --report_to "${BASE_REPORT_TO}"
)

JOINT_CMD=(
  accelerate launch --multi_gpu --num_machines 1 --num_processes "${NUM_NPUS}" --mixed_precision "${MIXED_PRECISION}" -m evaluate_joint
  --do_test
  --model_name_or_path "${BASE_MODEL}"
  --dataset_name "${PROC_DATASET}"
  --dataset_split_name "${EVAL_SPLIT}"
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
  --output_dir "${JOINT_EVAL_OUTPUT_DIR}"
  --knn_temp "${KNN_TEMP}"
  --lmbda "${LMBDA}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
  --knn_generator_path "${MEMDEC_MODEL}"
  "${JOINT_TRACKING_ARGS[@]}"
  "${JOINT_PIN_MEMORY_ARGS[@]}"
)

"${BASE_CMD[@]}"
"${JOINT_CMD[@]}"

python - <<'PY'
import json
import os
import re
from pathlib import Path

base_dir = Path(os.environ["BASE_EVAL_OUTPUT_DIR"])
joint_dir = Path(os.environ["JOINT_EVAL_OUTPUT_DIR"])
log_file = Path(os.environ["LOG_FILE"])
memdec_model = os.environ["MEMDEC_MODEL"]

base_metrics_path = base_dir / "eval_results.json"
base_perplexity = None
if base_metrics_path.exists():
    base_metrics = json.loads(base_metrics_path.read_text())
    base_perplexity = base_metrics.get("perplexity")

log_text = log_file.read_text(errors="ignore")
lm_matches = re.findall(r"lm perplexity:\s*([0-9.eE+-]+)", log_text)
joint_matches = re.findall(r"joint perplexity:\s*([0-9.eE+-]+)", log_text)

summary = {
    "base_perplexity": float(base_perplexity) if base_perplexity is not None else None,
    "lm_perplexity": float(lm_matches[-1]) if lm_matches else None,
    "joint_perplexity": float(joint_matches[-1]) if joint_matches else None,
    "memdec_model": memdec_model,
}

(joint_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
INNER

chmod 755 "${LAUNCH_SCRIPT}"
bash "${LAUNCH_SCRIPT}"
echo "[INFO] log_file=${LOG_FILE}"
EOS
} | ssh "${DEV_NODE}" 'bash -s'
