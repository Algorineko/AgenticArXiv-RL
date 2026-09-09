#!/usr/bin/env bash

# 若服务未正常关闭而占用端口, 查看并kill其进程
# sudo lsof -i :8000 | grep LISTEN

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${ROOT_DIR}/.run"

B_PORT="${BACKEND_PORT:-8000}"
F_PORT="${FRONTEND_PORT:-5173}"

BACKEND_PGID_FILE="${RUN_DIR}/backend.pgid"
FRONTEND_PGID_FILE="${RUN_DIR}/frontend.pgid"

stop_group() {
  local name="$1"
  local f="$2"

  if [[ ! -f "$f" ]]; then
    echo "[stop] ${name}: no pgid file (${f}), skip."
    return 0
  fi

  local pgid
  pgid="$(cat "$f" 2>/dev/null || true)"
  if [[ -z "${pgid}" ]]; then
    echo "[stop] ${name}: empty pgid file, remove."
    rm -f "$f"
    return 0
  fi

  if ! kill -0 "-${pgid}" >/dev/null 2>&1; then
    echo "[stop] ${name}: not running (pgid=${pgid}), remove file."
    rm -f "$f"
    return 0
  fi

  echo "[stop] ${name}: TERM pgid=${pgid}"
  kill "-${pgid}" >/dev/null 2>&1 || true

  # 等待最多 8 秒
  for _ in {1..80}; do
    if ! kill -0 "-${pgid}" >/dev/null 2>&1; then
      echo "[stop] ${name}: stopped."
      rm -f "$f"
      return 0
    fi
    sleep 0.1
  done

  echo "[stop] ${name}: KILL pgid=${pgid}"
  kill -KILL "-${pgid}" >/dev/null 2>&1 || true

  # 再等最多 3 秒
  for _ in {1..30}; do
    if ! kill -0 "-${pgid}" >/dev/null 2>&1; then
      echo "[stop] ${name}: killed."
      rm -f "$f"
      return 0
    fi
    sleep 0.1
  done

  echo "[stop] ${name}: still alive after KILL (pgid=${pgid})" >&2
  return 1
}

echo "[stop] repo: ${ROOT_DIR}"
echo

stop_group "backend"  "${BACKEND_PGID_FILE}" || true
stop_group "frontend" "${FRONTEND_PGID_FILE}" || true

# 兜底：如果端口仍被占用，按端口杀
kill_by_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids="$(lsof -ti "TCP:${port}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      echo "[stop] fallback: killing pids on port ${port}: ${pids}"
      kill ${pids} >/dev/null 2>&1 || true
      sleep 0.5
      kill -9 ${pids} >/dev/null 2>&1 || true
    fi
  fi
}

kill_by_port "${B_PORT}" || true
kill_by_port "${F_PORT}" || true

echo
echo "[stop] done."