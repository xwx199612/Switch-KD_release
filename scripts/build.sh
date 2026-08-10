#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${VLM_IMAGE_NAME:-vlm-online-dbild-runtime:1.2.0}"
cd "$ROOT"
docker build --progress=plain -t "$IMAGE" . 2>&1 | tee reports/build.txt
docker image inspect "$IMAGE" > reports/image_inspect.json
echo "image=$IMAGE" >> reports/build.txt
