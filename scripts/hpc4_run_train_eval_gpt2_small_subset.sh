#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

cd "${REPO_ROOT}"

bash scripts/hpc4_train_memdec_gpt2_small_subset.sh
bash scripts/hpc4_eval_trained_memdec_gpt2_small_subset.sh
