#!/bin/bash

set -euo pipefail

cd /data/user/jzhu997/MemoryDecoder

if [ -n "${BUILD_JOB_ID:-}" ]; then
  build_job="${BUILD_JOB_ID}"
else
  build_job=$(sbatch --parsable scripts/hpc3_build_faiss_from_source.sbatch)
fi
knn_job=$(sbatch --parsable --dependency=afterok:${build_job} scripts/hpc3_build_knn_signals_gpt2_small.sbatch)
train_job=$(sbatch --parsable --dependency=afterok:${knn_job} scripts/hpc3_train_memdec_gpt2_small.sbatch)
eval_job=$(sbatch --parsable --dependency=afterok:${train_job} scripts/hpc3_eval_trained_memdec_gpt2_small.sbatch)

echo "build_faiss=${build_job}"
echo "build_knn=${knn_job}"
echo "train_memdec=${train_job}"
echo "eval_memdec=${eval_job}"
