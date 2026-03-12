#!/bin/bash

REPO_ROOT=${REPO_ROOT:-/data/user/jzhu997/MemoryDecoder}
DATE_TAG=${DATE_TAG:-$(date +%m_%d)}
RESULTS_ROOT=${RESULTS_ROOT:-${REPO_ROOT}/results}
TB_ROOT=${TB_ROOT:-${REPO_ROOT}/tb_logs}
WANDB_ROOT=${WANDB_ROOT:-${REPO_ROOT}/wandb_logs}

mkdir -p "${RESULTS_ROOT}" "${TB_ROOT}" "${WANDB_ROOT}"

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
