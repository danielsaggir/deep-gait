#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/scripts/free-api-port.sh" ]]; then
  bash "${ROOT}/scripts/free-api-port.sh" || true
fi

export PYTHONPATH="${ROOT}/server${PYTHONPATH:+:${PYTHONPATH}}"
exec npx concurrently -k -n server,client -c cyan,magenta \
  "npm run dev -w deepgait-server" \
  "npm run dev -w deepgait-client"
