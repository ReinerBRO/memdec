#!/bin/bash
#SBATCH -J faiss_src
#SBATCH -p acd_ue
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/jzhu997/MemoryDecoder

set -euo pipefail

mkdir -p ./logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./logs/build_faiss_from_source_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

module load gcc/11.5 2>/dev/null || module load gcc/12.5 2>/dev/null || module load gcc 2>/dev/null || true
module load cuda/12.2 2>/dev/null || module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
set +u
source /data/user/jzhu997/envs/memorydecoder/bin/activate
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PIP_NO_INPUT=1
export PATH=/usr/local/bin:/data/user/jzhu997/bin:${PATH}
export CC=$(command -v gcc)
export CXX=$(command -v g++)

ENV_ROOT=/data/user/jzhu997/envs/memorydecoder
PYTHON_BIN="${ENV_ROOT}/bin/python"
PIP_BIN="${ENV_ROOT}/bin/pip"
CMAKE_BIN="${ENV_ROOT}/bin/cmake"
if [ ! -x "${CMAKE_BIN}" ] && [ -x /usr/local/bin/cmake ]; then
  CMAKE_BIN=/usr/local/bin/cmake
fi
if [ ! -x "${CMAKE_BIN}" ]; then
  CMAKE_BIN=$(command -v cmake)
fi
export CMAKE_BIN
SRC_ROOT=/data/user/jzhu997/src/faiss-1.12.0
BUILD_ROOT=/data/user/jzhu997/build/faiss-1.12.0
WHEELHOUSE=/data/user/jzhu997/wheelhouse
TOOLS_ROOT=/data/user/jzhu997/tools
CMAKE_WHEEL_ROOT=${TOOLS_ROOT}/cmake-wheel
HPC2_SRC=/hpc2hdd/home/jzhu997/src/faiss-1.12.0
HPC2_WHEELHOUSE=/hpc2hdd/home/jzhu997/wheelhouse

mkdir -p /data/user/jzhu997/src /data/user/jzhu997/build /data/user/jzhu997/wheelhouse /data/user/jzhu997/tools

SITE_PACKAGES="$(${PYTHON_BIN} - <<'PY'
from sysconfig import get_paths
print(get_paths()["purelib"])
PY
)"
NVIDIA_LIB_DIRS="$(${PYTHON_BIN} - <<'PY'
from pathlib import Path
from sysconfig import get_paths

root = Path(get_paths()["purelib"]) / "nvidia"
lib_dirs = sorted(str(path) for path in root.glob("*/lib") if path.is_dir())
print(":".join(lib_dirs))
PY
)"
if [ -n "${NVIDIA_LIB_DIRS}" ]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
fi
CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")
CUDA_ARCH="$(${PYTHON_BIN} - <<'PY'
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"{major}{minor}")
else:
    print("")
PY
)"
NEED_NEW_CMAKE="$(${PYTHON_BIN} - <<'PY'
import os
import subprocess

cmake_bin = os.environ["CMAKE_BIN"]
try:
    first_line = subprocess.check_output([cmake_bin, "--version"], text=True).splitlines()[0]
    version = first_line.split()[-1]
    parts = tuple(int(x) for x in version.split(".")[:2])
    print("1" if parts < (3, 24) else "0")
except Exception:
    print("1")
PY
)"

if [ "${NEED_NEW_CMAKE}" = "1" ]; then
  echo "[INFO] extracting newer cmake into ${CMAKE_WHEEL_ROOT}"
  rsync -avh "jzhu997@hpc2login.hpc.hkust-gz.edu.cn:${HPC2_WHEELHOUSE}/" "${WHEELHOUSE}/"
  CMAKE_WHEEL=$(find "${WHEELHOUSE}" -maxdepth 1 -name 'cmake-*.whl' | sort | tail -n 1 || true)
  test -n "${CMAKE_WHEEL}" || { echo "[ERROR] no cmake wheel found under ${WHEELHOUSE}"; exit 1; }
  rm -rf "${CMAKE_WHEEL_ROOT}"
  mkdir -p "${CMAKE_WHEEL_ROOT}"
  "${PYTHON_BIN}" - "${CMAKE_WHEEL}" "${CMAKE_WHEEL_ROOT}" <<'PY'
import shutil
import sys
import zipfile
from pathlib import Path

wheel_path = Path(sys.argv[1])
extract_root = Path(sys.argv[2])
with zipfile.ZipFile(wheel_path) as zf:
    zf.extractall(extract_root)
PY
  chmod +x "${CMAKE_WHEEL_ROOT}/cmake/data/bin/cmake" "${CMAKE_WHEEL_ROOT}/cmake/data/bin/cpack" "${CMAKE_WHEEL_ROOT}/cmake/data/bin/ctest"
  CMAKE_BIN="${CMAKE_WHEEL_ROOT}/cmake/data/bin/cmake"
  export CMAKE_BIN
fi

echo "[INFO] host=$(hostname)"
echo "[INFO] workdir=$PWD"
echo "[INFO] log_file=$LOG_FILE"
echo "[INFO] env_root=${ENV_ROOT}"
echo "[INFO] site_packages=${SITE_PACKAGES}"
echo "[INFO] nvidia_lib_dirs=${NVIDIA_LIB_DIRS:-none}"
echo "[INFO] cuda_root=${CUDA_ROOT}"
echo "[INFO] cuda_arch=${CUDA_ARCH:-auto}"
echo "[INFO] python=$(${PYTHON_BIN} -V)"
echo "[INFO] cmake=${CMAKE_BIN}"
echo "[INFO] cmake_version=$(${CMAKE_BIN} --version | head -n 1)"
echo "[INFO] cc=${CC}"
echo "[INFO] cxx=${CXX}"
echo "[INFO] gcc_version=$(gcc --version | head -n 1)"
echo "[INFO] gxx_version=$(g++ --version | head -n 1)"
command -v nvcc
nvcc --version

if [ ! -d "${SRC_ROOT}" ]; then
  echo "[INFO] pulling Faiss source from HPC2"
  rsync -avh "jzhu997@hpc2login.hpc.hkust-gz.edu.cn:${HPC2_SRC}/" "${SRC_ROOT}/"
else
  echo "[INFO] using existing source tree at ${SRC_ROOT}"
fi

if ! command -v swig >/dev/null 2>&1; then
  echo "[INFO] installing SWIG wheel into ${ENV_ROOT}"
  rsync -avh "jzhu997@hpc2login.hpc.hkust-gz.edu.cn:${HPC2_WHEELHOUSE}/" "${WHEELHOUSE}/"
  SWIG_WHEEL=$(find "${WHEELHOUSE}" -maxdepth 1 -name 'swig-*.whl' | sort | tail -n 1 || true)
  test -n "${SWIG_WHEEL}" || { echo "[ERROR] no SWIG wheel found under ${WHEELHOUSE}"; exit 1; }
  "${PIP_BIN}" install --no-deps "${SWIG_WHEEL}"
else
  echo "[INFO] swig already available at $(command -v swig)"
fi

command -v swig
swig -version

BACKUP_ROOT="${ENV_ROOT}/faiss_backup_${TIMESTAMP}"
mkdir -p "${BACKUP_ROOT}/lib" "${BACKUP_ROOT}/site-packages" "${BACKUP_ROOT}/include" "${BACKUP_ROOT}/conda-meta"

shopt -s nullglob
for path in "${ENV_ROOT}"/lib/libfaiss*.so* "${ENV_ROOT}"/lib/libfaiss_c*.so*; do
  [ -e "${path}" ] || continue
  echo "[INFO] backing up $(basename "${path}")"
  mv "${path}" "${BACKUP_ROOT}/lib/"
done
for path in "${SITE_PACKAGES}"/faiss "${SITE_PACKAGES}"/faiss-* "${SITE_PACKAGES}"/faiss_*; do
  [ -e "${path}" ] || continue
  echo "[INFO] backing up $(basename "${path}")"
  mv "${path}" "${BACKUP_ROOT}/site-packages/"
done
for path in "${ENV_ROOT}"/include/faiss; do
  [ -e "${path}" ] || continue
  echo "[INFO] backing up $(basename "${path}")"
  mv "${path}" "${BACKUP_ROOT}/include/"
done
for path in "${ENV_ROOT}"/conda-meta/faiss-gpu-*.json "${ENV_ROOT}"/conda-meta/libfaiss-*.json; do
  [ -e "${path}" ] || continue
  echo "[INFO] backing up $(basename "${path}")"
  mv "${path}" "${BACKUP_ROOT}/conda-meta/"
done
shopt -u nullglob

rm -rf "${BUILD_ROOT}"

CMAKE_ARGS=(
  -S "${SRC_ROOT}"
  -B "${BUILD_ROOT}"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="${ENV_ROOT}"
  -DCUDAToolkit_ROOT="${CUDA_ROOT}"
  -DBUILD_SHARED_LIBS=ON
  -DBUILD_TESTING=OFF
  -DFAISS_ENABLE_GPU=ON
  -DFAISS_ENABLE_PYTHON=ON
  -DFAISS_ENABLE_CUVS=OFF
  -DFAISS_OPT_LEVEL=avx2
  -DPython_EXECUTABLE="${PYTHON_BIN}"
)

if [ -n "${CUDA_ARCH}" ]; then
  CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}")
fi

echo "[INFO] configuring Faiss with CMake"
"${CMAKE_BIN}" "${CMAKE_ARGS[@]}"

echo "[INFO] building Faiss"
"${CMAKE_BIN}" --build "${BUILD_ROOT}" --parallel 16

echo "[INFO] installing Faiss shared libraries"
"${CMAKE_BIN}" --install "${BUILD_ROOT}"

echo "[INFO] installing Python bindings"
(
  cd "${BUILD_ROOT}/faiss/python"
  "${PYTHON_BIN}" setup.py install
)

echo "[INFO] running Faiss smoke tests"
"${PYTHON_BIN}" - <<'PY'
import faiss
import faiss.contrib.torch_utils
print("faiss_file", faiss.__file__)
print("faiss_version", getattr(faiss, "__version__", "unknown"))
print("num_gpus", faiss.get_num_gpus())
PY

echo "[INFO] checking linked libraries"
ldd "${SITE_PACKAGES}/faiss/_swigfaiss.so" | sed -n '1,80p'

echo "[INFO] Faiss source build completed"
