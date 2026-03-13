#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

REPO_ROOT=${REPO_ROOT:-${DEFAULT_REPO_ROOT}}
DATE_TAG=${DATE_TAG:-$(date +%m_%d)}
RUN_TAG=${RUN_TAG:-$(date +%m_%d_%H%M%S)}
RESULTS_ROOT=${RESULTS_ROOT:-${REPO_ROOT}/results}
TB_ROOT=${TB_ROOT:-${REPO_ROOT}/tb_logs}
WANDB_ROOT=${WANDB_ROOT:-${REPO_ROOT}/wandb_logs}
CKPT_ROOT=${CKPT_ROOT:-${REPO_ROOT}/ckpt}

mkdir -p "${RESULTS_ROOT}" "${TB_ROOT}" "${WANDB_ROOT}" "${CKPT_ROOT}"

export WANDB_DIR="${WANDB_DIR:-${WANDB_ROOT}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_ROOT}/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_ROOT}/config}"

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

result_dir() {
  printf '%s/%s_%s' "${RESULTS_ROOT}" "$1" "${DATE_TAG}"
}

tb_dir() {
  printf '%s/%s_%s' "${TB_ROOT}" "$1" "${DATE_TAG}"
}

ckpt_dir() {
  printf '%s/%s_%s' "${CKPT_ROOT}" "$1" "${RUN_TAG}"
}

ckpt_latest_link() {
  printf '%s/%s_latest' "${CKPT_ROOT}" "$1"
}

update_latest_link() {
  local target="$1"
  local link_path="$2"
  ln -sfn "${target}" "${link_path}"
}

latest_run_dir() {
  local run_name="$1"
  local latest_link
  local latest_dir

  latest_link="$(ckpt_latest_link "${run_name}")"
  if [ -e "${latest_link}" ]; then
    (cd "${latest_link}" >/dev/null 2>&1 && pwd -P)
    return 0
  fi

  latest_dir=$(find "${CKPT_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "${run_name}_*" | sort | tail -n 1 || true)
  if [ -n "${latest_dir}" ]; then
    printf '%s' "${latest_dir}"
  fi
}

latest_epoch_checkpoint() {
  local run_root="$1"

  if [ -z "${run_root}" ] || [ ! -d "${run_root}" ]; then
    return 0
  fi

  find "${run_root}" -maxdepth 1 -type d -name 'epoch_*' | sort -V | tail -n 1 || true
}
