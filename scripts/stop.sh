#!/usr/bin/env bash
set -Eeuo pipefail
NAME="${VLM_CONTAINER_NAME:-vlm-online-dbild-runtime-120}"
docker stop "$NAME"
docker rm "$NAME"
