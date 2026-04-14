#!/usr/bin/env bash
# Start MongoDB (Docker) if needed, then the API + Vite dev client together.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MONGO_CONTAINER="${MONGO_CONTAINER:-deepgait-mongo}"
MONGO_IMAGE="${MONGO_IMAGE:-mongo:7}"

mongod_ready() {
  if command -v nc >/dev/null 2>&1; then
    nc -z 127.0.0.1 27017 >/dev/null 2>&1
    return $?
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:27017 -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  echo "[run-all] Install netcat (\`nc\`) or ensure \`lsof\` is available to probe port 27017." >&2
  return 1
}

wait_for_mongo() {
  local i
  for i in $(seq 1 45); do
    if mongod_ready; then
      return 0
    fi
    sleep 1
  done
  return 1
}

if [[ "${SKIP_MONGO_DOCKER:-}" == "1" ]]; then
  echo "[run-all] SKIP_MONGO_DOCKER=1 — not starting Docker MongoDB (use your own MONGODB_URI)."
else
  if mongod_ready; then
    echo "[run-all] MongoDB already accepting connections on 127.0.0.1:27017."
  else
    if ! command -v docker >/dev/null 2>&1; then
      echo "[run-all] MongoDB is not reachable on 127.0.0.1:27017 and Docker is not installed." >&2
      echo "[run-all] Start MongoDB yourself, or install Docker, or set SKIP_MONGO_DOCKER=1 if using a remote URI." >&2
      exit 1
    fi
    echo "[run-all] Starting MongoDB container (${MONGO_CONTAINER})…"
    if docker container inspect "${MONGO_CONTAINER}" >/dev/null 2>&1; then
      docker start "${MONGO_CONTAINER}" >/dev/null
    else
      docker run -d --name "${MONGO_CONTAINER}" -p 27017:27017 "${MONGO_IMAGE}" >/dev/null
    fi
    if ! wait_for_mongo; then
      echo "[run-all] Timed out waiting for MongoDB on 127.0.0.1:27017." >&2
      exit 1
    fi
    echo "[run-all] MongoDB is ready."
  fi
fi

if [[ ! -f "${ROOT}/webapp/server/.env" ]]; then
  echo "[run-all] Missing webapp/server/.env — copy webapp/server/.env.example and set REPO_ROOT, PYTHON_BIN, CHECKPOINT_PATH." >&2
  exit 1
fi

echo "[run-all] Starting API + Vite (Ctrl+C stops both)…"
exec npm run dev:all
