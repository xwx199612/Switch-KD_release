#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${VLM_PORT:-8000}"
for _ in $(seq 1 180); do
  if docker exec "${VLM_CONTAINER_NAME:-vlm-online-dbild-runtime-120}" curl --fail --silent --show-error http://127.0.0.1:8000/ready > "$ROOT/reports/health.json"; then
    cat "$ROOT/reports/health.json"
    exit 0
  fi
  sleep 2
done
docker logs "${VLM_CONTAINER_NAME:-vlm-online-dbild-runtime-120}" > "$ROOT/reports/runtime.log" 2>&1 || true
exit 1
