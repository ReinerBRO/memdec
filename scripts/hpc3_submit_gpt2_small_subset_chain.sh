#!/bin/bash

set -euo pipefail

cd /data/user/jzhu997/MemoryDecoder

SUBSET_TAG=${SUBSET_TAG:-train1of8}
STUDENT_TAG=${STUDENT_TAG:-gpt2_small}
TRAIN_FRACTION=${TRAIN_FRACTION:-0.125}
SUBSET_SEED=${SUBSET_SEED:-42}
SUBSET_BUCKETS=${SUBSET_BUCKETS:-128}
DSTORE_DIR=/data/user/jzhu997/MemoryDecoder/dstore/gpt2-xl-wikitext103-${SUBSET_TAG}
KNN_READY_MARKER=${DSTORE_DIR}/.step3_complete
KNN_ARROW=${DSTORE_DIR}/knn_gpt2_train_1600.arrow
FORCE_REBUILD_KNN=${FORCE_REBUILD_KNN:-0}

train_dependency_args=()
if [ "${FORCE_REBUILD_KNN}" = "1" ] || [ ! -f "${KNN_READY_MARKER}" ] || [ ! -s "${KNN_ARROW}" ]; then
  knn_job=$(sbatch --parsable \
    --export=ALL,SUBSET_TAG="${SUBSET_TAG}",TRAIN_FRACTION="${TRAIN_FRACTION}",SUBSET_SEED="${SUBSET_SEED}",SUBSET_BUCKETS="${SUBSET_BUCKETS}" \
    scripts/hpc3_build_knn_signals_gpt2_small_subset.sbatch)
  train_dependency_args=(--dependency=afterok:${knn_job})
else
  knn_job=reused
  echo "[INFO] reusing existing KNN artifacts under ${DSTORE_DIR}"
fi

train_job=$(sbatch --parsable \
  "${train_dependency_args[@]}" \
  --export=ALL,SUBSET_TAG="${SUBSET_TAG}",STUDENT_TAG="${STUDENT_TAG}" \
  scripts/hpc3_train_memdec_gpt2_small_subset.sbatch)
eval_job=$(sbatch --parsable \
  --dependency=afterok:${train_job} \
  --export=ALL,SUBSET_TAG="${SUBSET_TAG}",STUDENT_TAG="${STUDENT_TAG}" \
  scripts/hpc3_eval_trained_memdec_gpt2_small_subset.sbatch)

echo "subset_tag=${SUBSET_TAG}"
echo "student_tag=${STUDENT_TAG}"
echo "build_knn=${knn_job}"
echo "train_memdec=${train_job}"
echo "eval_memdec=${eval_job}"
