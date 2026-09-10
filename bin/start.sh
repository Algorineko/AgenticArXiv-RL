#!/usr/bin/env bash
set -euo pipefail

# ====== 定位仓库根目录 ======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_DIR="${ROOT_DIR}/AgenticArxiv"
FRONTEND_DIR="${ROOT_DIR}/AgenticArxivWeb"

# ====== 参数（可通过环境变量覆盖） ======
B_HOST="${BACKEND_HOST:-0.0.0.0}"
B_PORT="${BACKEND_PORT:-8000}"
B_RELOAD="${BACKEND_RELOAD:-1}"

F_HOST="${FRONTEND_HOST:-0.0.0.0}"
F_PORT="${FRONTEND_PORT:-5173}"

RUN_DIR="${ROOT_DIR}/.run"
mkdir -p "${RUN_DIR}"

USE_LOGS=0
# 兼容：默认后台；--logs 写日志；--no-logs 直出到 /dev/null；--reload/--no-reload 覆盖
while [[ $# -gt 0 ]]; do
  case "$1" in
    --logs) USE_LOGS=1; shift ;;
    --no-logs) USE_LOGS=0; shift ;;
    --reload) B_RELOAD=1; shift ;;
    --no-reload) B_RELOAD=0; shift ;;
    *)
      echo "[all] Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

require_cmd() {
  local c="$1"
  if ! command -v "${c}" >/dev/null 2>&1; then
    echo "[all] ERROR: command not found: ${c}" >&2
    exit 1
  fi
}

[[ -d "${BACKEND_DIR}" ]] || { echo "[all] ERROR: backend dir not found: ${BACKEND_DIR}" >&2; exit 1; }
[[ -d "${FRONTEND_DIR}" ]] || { echo "[all] ERROR: frontend dir not found: ${FRONTEND_DIR}" >&2; exit 1; }

# activate venv if present
VENV="${ROOT_DIR}/.venv"
if [[ -d "${VENV}" ]]; then
  source "${VENV}/bin/activate"
fi

require_cmd uvicorn
require_cmd npm
require_cmd setsid

BACKEND_PGID_FILE="${RUN_DIR}/backend.pgid"
FRONTEND_PGID_FILE="${RUN_DIR}/frontend.pgid"

is_running_pgid() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  local pgid
  pgid="$(cat "$f" 2>/dev/null || true)"
  [[ -n "$pgid" ]] || return 1
  # 进程组存在：kill -0 -PGID
  kill -0 "-${pgid}" >/dev/null 2>&1
}

start_backend() {
  if is_running_pgid "${BACKEND_PGID_FILE}"; then
    echo "[all] backend already running (pgid=$(cat "${BACKEND_PGID_FILE}"))"
    return 0
  fi

  local cmd
  if [[ "${B_RELOAD}" == "1" ]]; then
    cmd="exec uvicorn api.app:app --host '${B_HOST}' --port '${B_PORT}' --reload"
  else
    cmd="exec uvicorn api.app:app --host '${B_HOST}' --port '${B_PORT}'"
  fi

  if [[ "${USE_LOGS}" == "1" ]]; then
    local log="${RUN_DIR}/backend.log"
    : > "${log}"
    (
      cd "${BACKEND_DIR}"
      # setsid：让后端成为独立“会话/进程组”，$! 即 PGID
      setsid bash -lc "${cmd}" >> "${log}" 2>&1 < /dev/null &
      echo $! > "${BACKEND_PGID_FILE}"
    )
    echo "[all] backend started (pgid=$(cat "${BACKEND_PGID_FILE}")), log: ${log}"
  else
    (
      cd "${BACKEND_DIR}"
      setsid bash -lc "${cmd}" > /dev/null 2>&1 < /dev/null &
      echo $! > "${BACKEND_PGID_FILE}"
    )
    echo "[all] backend started (pgid=$(cat "${BACKEND_PGID_FILE}"))"
  fi
}

start_frontend() {
  if is_running_pgid "${FRONTEND_PGID_FILE}"; then
    echo "[all] frontend already running (pgid=$(cat "${FRONTEND_PGID_FILE}"))"
    return 0
  fi

  local cmd="exec npm run dev -- --host '${F_HOST}' --port '${F_PORT}'"

  if [[ "${USE_LOGS}" == "1" ]]; then
    local log="${RUN_DIR}/frontend.log"
    : > "${log}"
    (
      cd "${FRONTEND_DIR}"
      setsid bash -lc "${cmd}" >> "${log}" 2>&1 < /dev/null &
      echo $! > "${FRONTEND_PGID_FILE}"
    )
    echo "[all] frontend started (pgid=$(cat "${FRONTEND_PGID_FILE}")), log: ${log}"
  else
    (
      cd "${FRONTEND_DIR}"
      setsid bash -lc "${cmd}" > /dev/null 2>&1 < /dev/null &
      echo $! > "${FRONTEND_PGID_FILE}"
    )
    echo "[all] frontend started (pgid=$(cat "${FRONTEND_PGID_FILE}"))"
  fi
}

echo "[all] repo: ${ROOT_DIR}"
echo "[all] backend:  http://${B_HOST}:${B_PORT}   (reload=${B_RELOAD})"
echo "[all] frontend: http://${F_HOST}:${F_PORT}"
echo "[all] mode: background"
echo

start_backend
start_frontend

echo
echo "[all] done."
echo "[all] pgid files: ${BACKEND_PGID_FILE} , ${FRONTEND_PGID_FILE}"
echo "[all] shutdown:   ${SCRIPT_DIR}/shutdown.sh"
