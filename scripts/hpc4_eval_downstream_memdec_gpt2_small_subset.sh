#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-hpc4-test}
DEV_NODE=${DEV_NODE:-miku}
OWNER_REF_DIR=${OWNER_REF_DIR:-/data/user/user06}
MINICONDA_HOME=${MINICONDA_HOME:-/data/user/user06/miniconda3}
ENV_NAME=${ENV_NAME:-MemoryDecoder}
REPO_ROOT=${REPO_ROOT:-/data/user/user06/MemoryDecoder}

SUBSET_TAG=${SUBSET_TAG:-train1of8}
STUDENT_TAG=${STUDENT_TAG:-gpt2_small}
TRAIN_RUN_NAME=${TRAIN_RUN_NAME:-train_memdec_${STUDENT_TAG}_${SUBSET_TAG}}
BASE_MODEL=${BASE_MODEL:-/data/user/user06/cache/Models/gpt2}
TASK_ROOT=${TASK_ROOT:-/data/user/user06/cache/data/memorydecoder_knn_prompt_task_data}
YAHOO_DATASET_PATH=${YAHOO_DATASET_PATH:-${TASK_ROOT}/yahoo/yahoo_answers_topics}
TASK_DATA_HF_ENDPOINT=${TASK_DATA_HF_ENDPOINT:-https://hf-mirror.com}
PREP_TASK_DATA=${PREP_TASK_DATA:-1}

TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-}
MEMDEC_MODEL=${MEMDEC_MODEL:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-}

BATCH_SIZE=${BATCH_SIZE:-32}
DTYPE=${DTYPE:-bfloat16}
MAX_EXAMPLES=${MAX_EXAMPLES:-}
YAHOO_NUM_SHARDS=${YAHOO_NUM_SHARDS:-8}

REMOTE_VARS_FILE=$(mktemp)
trap 'rm -f "${REMOTE_VARS_FILE}"' EXIT

{
  printf 'DEV_NODE=%q\n' "${DEV_NODE}"
  printf 'OWNER_REF_DIR=%q\n' "${OWNER_REF_DIR}"
  printf 'MINICONDA_HOME=%q\n' "${MINICONDA_HOME}"
  printf 'ENV_NAME=%q\n' "${ENV_NAME}"
  printf 'PROJECT_ROOT=%q\n' "${REPO_ROOT}"
  printf 'SUBSET_TAG=%q\n' "${SUBSET_TAG}"
  printf 'STUDENT_TAG=%q\n' "${STUDENT_TAG}"
  printf 'TRAIN_RUN_NAME=%q\n' "${TRAIN_RUN_NAME}"
  printf 'BASE_MODEL=%q\n' "${BASE_MODEL}"
  printf 'TASK_ROOT=%q\n' "${TASK_ROOT}"
  printf 'YAHOO_DATASET_PATH=%q\n' "${YAHOO_DATASET_PATH}"
  printf 'TASK_DATA_HF_ENDPOINT=%q\n' "${TASK_DATA_HF_ENDPOINT}"
  printf 'PREP_TASK_DATA=%q\n' "${PREP_TASK_DATA}"
  printf 'TRAIN_OUTPUT_DIR=%q\n' "${TRAIN_OUTPUT_DIR}"
  printf 'MEMDEC_MODEL=%q\n' "${MEMDEC_MODEL}"
  printf 'OUTPUT_ROOT=%q\n' "${OUTPUT_ROOT}"
  printf 'BATCH_SIZE=%q\n' "${BATCH_SIZE}"
  printf 'DTYPE=%q\n' "${DTYPE}"
  printf 'MAX_EXAMPLES=%q\n' "${MAX_EXAMPLES}"
  printf 'YAHOO_NUM_SHARDS=%q\n' "${YAHOO_NUM_SHARDS}"
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

fix_owner() {
  if [ "$(id -u)" = "0" ]; then
    chown -R "${OWNER_UID}:${OWNER_GID}" "$@" 2>/dev/null || true
  fi
}

ensure_owner_dir "${PROJECT_ROOT}/logs"
ensure_owner_dir "${PROJECT_ROOT}/results"

set +u
source "${MINICONDA_HOME}/etc/profile.d/conda.sh"
set -u
conda activate "${ENV_NAME}"
cd "${PROJECT_ROOT}"
source scripts/output_layout.sh

if [ -z "${TRAIN_OUTPUT_DIR}" ]; then
  TRAIN_OUTPUT_DIR="$(latest_run_dir "${TRAIN_RUN_NAME}")"
fi
if [ -z "${MEMDEC_MODEL}" ] && [ -n "${TRAIN_OUTPUT_DIR}" ]; then
  MEMDEC_MODEL="$(latest_epoch_checkpoint "${TRAIN_OUTPUT_DIR}")"
fi
if [ -z "${OUTPUT_ROOT}" ]; then
  OUTPUT_ROOT="$(result_dir "downstream_memdec_base_gpt2_with_memdec_${STUDENT_TAG}_${SUBSET_TAG}_formal")"
fi

test -d "${BASE_MODEL}" || { echo "[ERROR] missing base model ${BASE_MODEL}"; exit 1; }
test -n "${TRAIN_OUTPUT_DIR}" || { echo "[ERROR] no training run found for ${TRAIN_RUN_NAME}"; exit 1; }
test -d "${MEMDEC_MODEL}" || { echo "[ERROR] missing memdec model ${MEMDEC_MODEL}"; exit 1; }

if [ "${PREP_TASK_DATA}" = "1" ] && [ ! -d "${TASK_ROOT}" ]; then
  echo "[INFO] preparing downstream task data under ${TASK_ROOT}"
  mkdir -p "$(dirname "${TASK_ROOT}")"
  python utils/prepare_downstream_task_data.py \
    --output_dir "${TASK_ROOT}" \
    --hf_endpoint "${TASK_DATA_HF_ENDPOINT}"
fi

test -d "${TASK_ROOT}" || { echo "[ERROR] missing task root ${TASK_ROOT}"; exit 1; }
test -d "${YAHOO_DATASET_PATH}" || { echo "[ERROR] missing yahoo dataset ${YAHOO_DATASET_PATH}"; exit 1; }

ensure_owner_dir "${OUTPUT_ROOT}"

LOG_TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJECT_ROOT}/logs/eval_downstream_memdec_${STUDENT_TAG}_${SUBSET_TAG}_${LOG_TS}.log"
LAUNCH_SCRIPT="${PROJECT_ROOT}/logs/eval_downstream_memdec_${STUDENT_TAG}_${SUBSET_TAG}_${LOG_TS}.launch.sh"

cat > "${LAUNCH_SCRIPT}" <<INNER
#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:\${LD_LIBRARY_PATH:-}
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source "${MINICONDA_HOME}/etc/profile.d/conda.sh"
set -u
conda activate "${ENV_NAME}"

cd "${PROJECT_ROOT}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${OWNER_REF_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="\${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="\${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HCCL_CONNECT_TIMEOUT=1800
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
OWNER_UID="${OWNER_UID}"
OWNER_GID="${OWNER_GID}"

run_as_owner() {
  if [ "\$(id -u)" = "\${OWNER_UID}" ]; then
    "\$@"
  else
    setpriv --reuid="\${OWNER_UID}" --regid="\${OWNER_GID}" --clear-groups \
      env HOME="${OWNER_REF_DIR}" USER="user06" LOGNAME="user06" \
      LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}" \
      PYTHONPATH="\${PYTHONPATH:-}" \
      "\$@"
  fi
}

mkdir -p "$(dirname "${LOG_FILE}")" "${OUTPUT_ROOT}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "[INFO] host=\$(hostname)"
echo "[INFO] base_model=${BASE_MODEL}"
echo "[INFO] train_output_dir=${TRAIN_OUTPUT_DIR}"
echo "[INFO] memdec_model=${MEMDEC_MODEL}"
echo "[INFO] task_root=${TASK_ROOT}"
echo "[INFO] yahoo_dataset_path=${YAHOO_DATASET_PATH}"
echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] batch_size=${BATCH_SIZE}"
echo "[INFO] dtype=${DTYPE}"

run_task() {
  local chip="\$1"
  local task="\$2"
  shift 2
  local max_examples_args=()
  if [ -n "${MAX_EXAMPLES}" ]; then
    max_examples_args+=(--max_examples "${MAX_EXAMPLES}")
  fi

  run_as_owner env ASCEND_RT_VISIBLE_DEVICES="\${chip}" ASCEND_DEVICE_ID=0 \
    python utils/evaluate_downstream_memdec.py \
      --base_model "${BASE_MODEL}" \
      --memdec_model "${MEMDEC_MODEL}" \
      --task "\${task}" \
      --task_root "${TASK_ROOT}" \
      --yahoo_dataset_path "${YAHOO_DATASET_PATH}" \
      --output_dir "${OUTPUT_ROOT}" \
      --batch_size "${BATCH_SIZE}" \
      --dtype "${DTYPE}" \
      "\${max_examples_args[@]}" \
      "\$@"
}

run_task 0 sst2 &
PID0=\$!
run_task 1 mr &
PID1=\$!
run_task 2 cr &
PID2=\$!
run_task 3 rt &
PID3=\$!
run_task 4 hyp &
PID4=\$!
run_task 5 cb &
PID5=\$!
run_task 6 rte &
PID6=\$!
run_task 7 agn &
PID7=\$!
wait \$PID0 \$PID1 \$PID2 \$PID3 \$PID4 \$PID5 \$PID6 \$PID7

YAHOO_PIDS=()
for chip in \$(seq 0 $((YAHOO_NUM_SHARDS - 1))); do
  run_task "\${chip}" yahoo --shard_index "\${chip}" --num_shards "${YAHOO_NUM_SHARDS}" &
  YAHOO_PIDS+=("\$!")
done

for pid in "\${YAHOO_PIDS[@]}"; do
  wait "\${pid}"
done

OUTPUT_ROOT_PATH="${OUTPUT_ROOT}" YAHOO_NUM_SHARDS="${YAHOO_NUM_SHARDS}" run_as_owner python - <<'PY'
import json
import os
from csv import writer
from pathlib import Path

output_root = Path(os.environ["OUTPUT_ROOT_PATH"])
yahoo_num_shards = int(os.environ["YAHOO_NUM_SHARDS"])
tasks = ["sst2", "mr", "cr", "rt", "hyp", "cb", "rte", "agn", "yahoo"]
display_names = {
    "sst2": "SST2",
    "mr": "MR",
    "cr": "CR",
    "rt": "RT",
    "hyp": "HYP",
    "cb": "CB",
    "rte": "RTE",
    "agn": "AGN",
    "yahoo": "Yahoo",
}

results = []
for task in tasks:
    if task != "yahoo":
        results.append(json.loads((output_root / f"{task}.json").read_text()))
        continue

    shard_paths = [
        output_root / f"yahoo.shard{idx}of{yahoo_num_shards}.json"
        for idx in range(yahoo_num_shards)
    ]
    if shard_paths[0].exists():
        shard_items = [json.loads(path.read_text()) for path in shard_paths]
        total_examples = sum(item["num_examples"] for item in shard_items)
        base_correct = sum(item["base_correct"] for item in shard_items)
        memdec_correct = sum(item["memdec_correct"] for item in shard_items)
        merged = dict(shard_items[0])
        merged["num_examples"] = total_examples
        merged["base_correct"] = base_correct
        merged["memdec_correct"] = memdec_correct
        merged["base_dcpmi_acc"] = round(100.0 * base_correct / total_examples, 4)
        merged["memdec_dcpmi_acc"] = round(100.0 * memdec_correct / total_examples, 4)
        merged["shard_index"] = None
        merged["num_shards"] = yahoo_num_shards
        (output_root / "yahoo.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
        results.append(merged)
    else:
        results.append(json.loads((output_root / "yahoo.json").read_text()))

avg_base = sum(item["base_dcpmi_acc"] for item in results) / len(results)
avg_memdec = sum(item["memdec_dcpmi_acc"] for item in results) / len(results)
summary = {
    "tasks": results,
    "avg_base_dcpmi_acc": round(avg_base, 4),
    "avg_memdec_dcpmi_acc": round(avg_memdec, 4),
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
(output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

csv_path = output_root / "results_table.csv"
with csv_path.open("w", newline="") as f:
    csv_writer = writer(f)
    csv_writer.writerow(["Method", "SST2", "MR", "CR", "RT", "HYP", "CB", "RTE", "AGN", "Yahoo", "Avg"])
    csv_writer.writerow(["base"] + [f'{item["base_dcpmi_acc"]:.4f}' for item in results] + [f"{avg_base:.4f}"])
    csv_writer.writerow(["+MemDec"] + [f'{item["memdec_dcpmi_acc"]:.4f}' for item in results] + [f"{avg_memdec:.4f}"])

def fmt_cell(base_value, memdec_value, target):
    best = max(base_value, memdec_value)
    value = base_value if target == "base" else memdec_value
    text = f"{value:.2f}"
    if abs(value - best) < 1e-9:
        return f"**{text}**"
    return text

md_lines = [
    "| Method | SST2 | MR | CR | RT | HYP | CB | RTE | AGN | Yahoo | Avg |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]

base_row = ["base"]
memdec_row = ["+MemDec"]
for item in results:
    base_row.append(fmt_cell(item["base_dcpmi_acc"], item["memdec_dcpmi_acc"], "base"))
    memdec_row.append(fmt_cell(item["base_dcpmi_acc"], item["memdec_dcpmi_acc"], "memdec"))

base_row.append(fmt_cell(avg_base, avg_memdec, "base"))
memdec_row.append(fmt_cell(avg_base, avg_memdec, "memdec"))

md_lines.append("| " + " | ".join(base_row) + " |")
md_lines.append("| " + " | ".join(memdec_row) + " |")
md_lines.append("")
md_lines.append("| Task | #Examples | Alpha | Delta |")
md_lines.append("|---|---:|---:|---:|")
for item in results:
    delta = item["memdec_dcpmi_acc"] - item["base_dcpmi_acc"]
    md_lines.append(
        f'| {display_names[item["task"]]} | {item["num_examples"]} | {item["alpha"]:.2f} | {delta:+.4f} |'
    )
md_lines.append(f"| Avg | - | - | {avg_memdec - avg_base:+.4f} |")
(output_root / "results_table.md").write_text("\n".join(md_lines) + "\n")
PY
INNER

chmod 755 "${LAUNCH_SCRIPT}"
ssh "${DEV_NODE}" "bash ${LAUNCH_SCRIPT}"
fix_owner "${OUTPUT_ROOT}" "${LOG_FILE}" "${LAUNCH_SCRIPT}"
echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] log_file=${LOG_FILE}"
EOS
} | ssh "${REMOTE_HOST}" 'bash -s'
