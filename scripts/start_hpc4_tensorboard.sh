#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REMOTE_HOST=${REMOTE_HOST:-hpc4-test}
PROJECT_ROOT=${PROJECT_ROOT:-/data/user/user06/MemoryDecoder}
MINICONDA_HOME=${MINICONDA_HOME:-/data/user/user06/miniconda3}
ENV_NAME=${ENV_NAME:-MemoryDecoder}
TB_LOGDIR=${TB_LOGDIR:-${PROJECT_ROOT}/tb_logs/train_memdec_gpt2_small_train1of8_03_13}
REMOTE_PORT=${REMOTE_PORT:-6006}
LOCAL_PORT=${LOCAL_PORT:-16006}
TB_SESSION=${TB_SESSION:-memorydecoder_tb_train1of8}
REMOTE_LOG_FILE=${REMOTE_LOG_FILE:-${PROJECT_ROOT}/logs/tensorboard_train1of8.log}
LOCAL_RUN_DIR=${LOCAL_RUN_DIR:-${SCRIPT_DIR}/logs}
LOCAL_TUNNEL_LOG=${LOCAL_TUNNEL_LOG:-${LOCAL_RUN_DIR}/tensorboard_tunnel_${LOCAL_PORT}.log}
LOCAL_TUNNEL_PIDFILE=${LOCAL_TUNNEL_PIDFILE:-${LOCAL_RUN_DIR}/tensorboard_tunnel_${LOCAL_PORT}.pid}
LOCAL_TUNNEL_LABEL=${LOCAL_TUNNEL_LABEL:-memorydecoder_tb_${LOCAL_PORT}}
LOCAL_TUNNEL_PATH=${LOCAL_TUNNEL_PATH:-/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin}
LOCAL_TUNNEL_STDOUT_LOG=${LOCAL_TUNNEL_STDOUT_LOG:-/tmp/${LOCAL_TUNNEL_LABEL}.out}
LOCAL_TUNNEL_STDERR_LOG=${LOCAL_TUNNEL_STDERR_LOG:-/tmp/${LOCAL_TUNNEL_LABEL}.err}

mkdir -p "${LOCAL_RUN_DIR}"

if ! command -v ssh >/dev/null 2>&1; then
  echo "[local] ssh not found" >&2
  exit 1
fi

ssh "${REMOTE_HOST}" bash -s -- \
  "${PROJECT_ROOT}" \
  "${MINICONDA_HOME}" \
  "${ENV_NAME}" \
  "${TB_LOGDIR}" \
  "${REMOTE_PORT}" \
  "${TB_SESSION}" \
  "${REMOTE_LOG_FILE}" <<'EOS'
set -euo pipefail

PROJECT_ROOT=$1
MINICONDA_HOME=$2
ENV_NAME=$3
TB_LOGDIR=$4
REMOTE_PORT=$5
TB_SESSION=$6
REMOTE_LOG_FILE=$7

if [ ! -d "${TB_LOGDIR}" ]; then
  echo "[remote] missing logdir: ${TB_LOGDIR}" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "[remote] tmux not found" >&2
  exit 1
fi

mkdir -p "$(dirname "${REMOTE_LOG_FILE}")"

if tmux has-session -t "${TB_SESSION}" 2>/dev/null; then
  echo "[remote] reusing tmux session ${TB_SESSION}"
else
  tmux new-session -d -s "${TB_SESSION}" \
    "source \"${MINICONDA_HOME}/etc/profile.d/conda.sh\" && \
     conda activate \"${ENV_NAME}\" && \
     exec tensorboard --logdir \"${TB_LOGDIR}\" --host 127.0.0.1 --port \"${REMOTE_PORT}\" \
       > \"${REMOTE_LOG_FILE}\" 2>&1"
  echo "[remote] started tmux session ${TB_SESSION}"
fi

sleep 2
if tmux has-session -t "${TB_SESSION}" 2>/dev/null; then
  tmux capture-pane -pt "${TB_SESSION}:0" | tail -n 20 || true
else
  echo "[remote] tensorboard session exited early; tailing ${REMOTE_LOG_FILE}" >&2
  tail -n 40 "${REMOTE_LOG_FILE}" >&2 || true
  exit 1
  fi
EOS

start_local_tunnel_with_launchctl() {
  local uid launchctl_job launchctl_cmd
  uid=$(id -u)
  launchctl_job="gui/${uid}/${LOCAL_TUNNEL_LABEL}"

  if launchctl print "${launchctl_job}" 2>/dev/null | grep -q "state = running"; then
    echo "[local] reusing launchctl tunnel ${LOCAL_TUNNEL_LABEL} on 127.0.0.1:${LOCAL_PORT}"
    return 0
  fi

  if command -v lsof >/dev/null 2>&1 && lsof -tiTCP:"${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[local] port ${LOCAL_PORT} is already in use" >&2
    exit 1
  fi

  launchctl remove "${LOCAL_TUNNEL_LABEL}" >/dev/null 2>&1 || true
  launchctl_cmd="export PATH=\"${LOCAL_TUNNEL_PATH}:\$PATH\"; exec ssh -N -o ExitOnForwardFailure=yes -L ${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT} ${REMOTE_HOST}"
  launchctl submit -l "${LOCAL_TUNNEL_LABEL}" \
    -o "${LOCAL_TUNNEL_STDOUT_LOG}" \
    -e "${LOCAL_TUNNEL_STDERR_LOG}" \
    -- /bin/zsh -lc "${launchctl_cmd}"

  sleep 3
  if ! launchctl print "${launchctl_job}" 2>/dev/null | grep -q "state = running"; then
    echo "[local] launchctl tunnel failed to start; tailing ${LOCAL_TUNNEL_STDERR_LOG}" >&2
    tail -n 40 "${LOCAL_TUNNEL_STDERR_LOG}" >&2 || true
    exit 1
  fi

  echo "[local] started launchctl tunnel ${LOCAL_TUNNEL_LABEL} on 127.0.0.1:${LOCAL_PORT}"
}

start_local_tunnel_with_nohup() {
  if [ -f "${LOCAL_TUNNEL_PIDFILE}" ]; then
    existing_pid=$(cat "${LOCAL_TUNNEL_PIDFILE}")
    if kill -0 "${existing_pid}" >/dev/null 2>&1; then
      echo "[local] reusing ssh tunnel on 127.0.0.1:${LOCAL_PORT} (pid ${existing_pid})"
      return 0
    fi
    rm -f "${LOCAL_TUNNEL_PIDFILE}"
  fi

  if command -v lsof >/dev/null 2>&1 && lsof -tiTCP:"${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[local] port ${LOCAL_PORT} is already in use" >&2
    exit 1
  fi

  nohup ssh -N -o ExitOnForwardFailure=yes \
    -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    "${REMOTE_HOST}" > "${LOCAL_TUNNEL_LOG}" 2>&1 &
  tunnel_pid=$!
  echo "${tunnel_pid}" > "${LOCAL_TUNNEL_PIDFILE}"

  sleep 2
  if ! kill -0 "${tunnel_pid}" >/dev/null 2>&1; then
    echo "[local] ssh tunnel exited early; tailing ${LOCAL_TUNNEL_LOG}" >&2
    tail -n 40 "${LOCAL_TUNNEL_LOG}" >&2 || true
    rm -f "${LOCAL_TUNNEL_PIDFILE}"
    exit 1
  fi

  echo "[local] started ssh tunnel on 127.0.0.1:${LOCAL_PORT} (pid ${tunnel_pid})"
}

if command -v launchctl >/dev/null 2>&1 && [ "$(uname -s)" = "Darwin" ]; then
  start_local_tunnel_with_launchctl
else
  start_local_tunnel_with_nohup
fi

if ! curl -sSf --max-time 5 "http://127.0.0.1:${LOCAL_PORT}" >/dev/null 2>&1; then
  echo "[local] tunnel is up, but TensorBoard did not answer on port ${LOCAL_PORT} yet" >&2
  if command -v launchctl >/dev/null 2>&1 && [ "$(uname -s)" = "Darwin" ]; then
    echo "[local] tailing ${LOCAL_TUNNEL_STDERR_LOG}" >&2
    tail -n 20 "${LOCAL_TUNNEL_STDERR_LOG}" >&2 || true
  else
    echo "[local] tailing ${LOCAL_TUNNEL_LOG}" >&2
    tail -n 20 "${LOCAL_TUNNEL_LOG}" >&2 || true
  fi
  exit 1
else
  echo "[local] verified TensorBoard on 127.0.0.1:${LOCAL_PORT}"
fi

echo
echo "TensorBoard URL: http://127.0.0.1:${LOCAL_PORT}"
echo "Remote logdir: ${TB_LOGDIR}"
echo "Remote tmux session: ${TB_SESSION}"
echo "Remote log file: ${REMOTE_LOG_FILE}"
echo "Local tunnel label: ${LOCAL_TUNNEL_LABEL}"
echo "Local tunnel pid file: ${LOCAL_TUNNEL_PIDFILE}"
echo "Local tunnel log file: ${LOCAL_TUNNEL_LOG}"
echo "Local tunnel stdout log: ${LOCAL_TUNNEL_STDOUT_LOG}"
echo "Local tunnel stderr log: ${LOCAL_TUNNEL_STDERR_LOG}"
