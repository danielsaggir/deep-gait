#!/usr/bin/env bash
# Stop any process listening on the Express API port so dev:all / run:all can bind.
# Default port matches webapp/server (see .env PORT=).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${SKIP_FREE_API_PORT:-}" == "1" ]]; then
  echo "[free-api-port] SKIP_FREE_API_PORT=1 — leaving existing listeners on the API port."
  exit 0
fi

if ! command -v lsof >/dev/null 2>&1; then
  echo "[free-api-port] Warning: lsof not found; cannot free the API port automatically." >&2
  exit 0
fi

ENV_FILE="${ROOT}/webapp/server/.env"
PORT=3001
if [[ -f "$ENV_FILE" ]]; then
  line="$(grep -E '^[[:space:]]*PORT=' "$ENV_FILE" | tail -n1 || true)"
  if [[ -n "$line" ]]; then
    val="${line#*=}"
    val="${val//\"/}"
    val="${val//\'/}"
    val="${val//$'\r'/}"
    val="${val// /}"
    if [[ "$val" =~ ^[0-9]+$ ]] && (( val > 0 && val < 65536 )); then
      PORT="$val"
    fi
  fi
fi

listeners() {
  lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
}

pids="$(listeners)"
if [[ -z "$pids" ]]; then
  exit 0
fi

echo "[free-api-port] Port ${PORT} is in use (PID(s): ${pids//$'\n' / }). Stopping listener(s) so the API can start…"
kill $pids 2>/dev/null || true
sleep 1

pids="$(listeners)"
if [[ -n "$pids" ]]; then
  echo "[free-api-port] Sending SIGKILL to PID(s): ${pids//$'\n' / }…"
  kill -9 $pids 2>/dev/null || true
  sleep 0.5
fi

pids="$(listeners)"
if [[ -n "$pids" ]]; then
  echo "[free-api-port] Could not free port ${PORT}; stop the process manually or use another PORT in webapp/server/.env" >&2
  exit 1
fi

echo "[free-api-port] Port ${PORT} is free."
