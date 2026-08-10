#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${VLM_IMAGE_NAME:-vlm-online-dbild-runtime:1.2.0}"
NAME="${VLM_CONTAINER_NAME:-vlm-online-dbild-runtime-120}"
PORT="${VLM_PORT:-8000}"
mkdir -p "$ROOT/output"
chmod a+rwx "$ROOT/output"
docker run -d --gpus all --network none --name "$NAME" -p "${PORT}:8000" \
  -v "$ROOT/models/student:/models/student:ro" \
  -v "$ROOT/models/adapter:/models/adapter:ro" \
  -v "$ROOT/config/runtime.yaml:/config/runtime.yaml:ro" \
  -v "$ROOT/samples:/data:ro" \
  -v "$ROOT/output:/output" \
  "$IMAGE" > "$ROOT/reports/container_id.txt"
docker inspect "$NAME" > "$ROOT/reports/container_start.json"
echo "container=$NAME port=$PORT" | tee "$ROOT/reports/container_start.txt"
