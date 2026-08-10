#!/usr/bin/env bash
set -Eeuo pipefail
python3.11 /app/scripts/validate_assets.py
exec "$@"
