#!/bin/bash
#SBATCH -J memdec_ds
#SBATCH -p acd_u
#SBATCH -n 16
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/jzhu997/MemoryDecoder

set -euo pipefail

mkdir -p ./logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./logs/eval_downstream_memdec_l6_subset_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

module load cuda/12.2 2>/dev/null || module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
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

BASE_MODEL=${BASE_MODEL:-/data/user/jzhu997/cache/Models/gpt2}
SUBSET_TAG=${SUBSET_TAG:-train1of8}
STUDENT_TAG=${STUDENT_TAG:-gpt2_small_l6}
TRAIN_RUN_NAME=${TRAIN_RUN_NAME:-train_memdec_${STUDENT_TAG}_${SUBSET_TAG}}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-$(latest_run_dir "${TRAIN_RUN_NAME}")}
MEMDEC_MODEL=${MEMDEC_MODEL:-$(latest_epoch_checkpoint "${TRAIN_OUTPUT_DIR}")}
TASK_ROOT=${TASK_ROOT:-/data/user/jzhu997/cache/data/memorydecoder_knn_prompt_task_data}
YAHOO_DATASET_PATH=${YAHOO_DATASET_PATH:-${TASK_ROOT}/yahoo/yahoo_answers_topics}
OUTPUT_ROOT=${OUTPUT_ROOT:-$(result_dir "downstream_memdec_base_gpt2_with_memdec_gpt2_small_l6_train1of8")}
BATCH_SIZE=${BATCH_SIZE:-32}
DTYPE=${DTYPE:-bfloat16}

echo "[INFO] host=$(hostname)"
echo "[INFO] log_file=$LOG_FILE"
echo "[INFO] base_model=${BASE_MODEL}"
echo "[INFO] train_run_name=${TRAIN_RUN_NAME}"
echo "[INFO] train_output_dir=${TRAIN_OUTPUT_DIR}"
echo "[INFO] memdec_model=${MEMDEC_MODEL}"
echo "[INFO] task_root=${TASK_ROOT}"
echo "[INFO] yahoo_dataset_path=${YAHOO_DATASET_PATH}"
echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] batch_size=${BATCH_SIZE}"
echo "[INFO] dtype=${DTYPE}"
nvidia-smi || true

test -d "${BASE_MODEL}" || { echo "[ERROR] missing base model ${BASE_MODEL}"; exit 1; }
test -d "${MEMDEC_MODEL}" || { echo "[ERROR] missing memdec model ${MEMDEC_MODEL}"; exit 1; }
test -d "${TASK_ROOT}" || { echo "[ERROR] missing task root ${TASK_ROOT}"; exit 1; }
test -d "${YAHOO_DATASET_PATH}" || { echo "[ERROR] missing yahoo dataset ${YAHOO_DATASET_PATH}"; exit 1; }

mkdir -p "${OUTPUT_ROOT}"

run_task() {
  local gpu="$1"
  local task="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" python utils/evaluate_downstream_memdec.py \
    --base_model "${BASE_MODEL}" \
    --memdec_model "${MEMDEC_MODEL}" \
    --task "${task}" \
    --task_root "${TASK_ROOT}" \
    --yahoo_dataset_path "${YAHOO_DATASET_PATH}" \
    --output_dir "${OUTPUT_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --dtype "${DTYPE}"
}

run_task 0 sst2 &
PID0=$!
run_task 1 mr &
PID1=$!
run_task 2 cr &
PID2=$!
run_task 3 rt &
PID3=$!
wait $PID0 $PID1 $PID2 $PID3

run_task 0 hyp &
PID0=$!
run_task 1 cb &
PID1=$!
run_task 2 rte &
PID2=$!
run_task 3 agn &
PID3=$!
wait $PID0 $PID1 $PID2 $PID3

run_task 3 yahoo

OUTPUT_ROOT_PATH="${OUTPUT_ROOT}" python - <<'PY'
import json
import os
from csv import writer
from pathlib import Path

output_root = Path(os.environ["OUTPUT_ROOT_PATH"])
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
    path = output_root / f"{task}.json"
    data = json.loads(path.read_text())
    results.append(data)

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
    csv_writer.writerow(
        ["base"]
        + [f'{item["base_dcpmi_acc"]:.4f}' for item in results]
        + [f"{avg_base:.4f}"]
    )
    csv_writer.writerow(
        ["+MemDec"]
        + [f'{item["memdec_dcpmi_acc"]:.4f}' for item in results]
        + [f"{avg_memdec:.4f}"]
    )

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
