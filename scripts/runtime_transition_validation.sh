#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${VLM_CONTAINER_NAME:-vlm-online-dbild-runtime-120}"
REPORT="$ROOT/reports/transition_runtime_test.txt"
mkdir -p "$ROOT/reports"
: > "$REPORT"

sample_vram() {
  docker exec "$CONTAINER" nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '
}
run_with_vram() {
  local label="$1"; shift
  local before after peak
  before="$(sample_vram)"
  ( while :; do sample_vram; sleep 0.2; done ) > "$ROOT/reports/.vram_sample" &
  local monitor=$!
  "$@"
  kill "$monitor" 2>/dev/null || true
  wait "$monitor" 2>/dev/null || true
  after="$(sample_vram)"
  peak="$(tr ' ' '\n' < "$ROOT/reports/.vram_sample" | awk 'NF && $1 ~ /^[0-9]+$/ {if ($1>m)m=$1} END{print m+0}')"
  echo "$label idle_or_before_mib=$before peak_mib=$peak after_mib=$after" >> "$REPORT"
}

echo "container=$CONTAINER" >> "$REPORT"
echo "ready=$(docker exec "$CONTAINER" curl --fail --silent http://127.0.0.1:8000/ready)" >> "$REPORT"
run_with_vram parsing docker exec "$CONTAINER" curl --fail --silent -X POST http://127.0.0.1:8000/infer/parsing \
  -F 'image=@/data/sample_ui_0001.png;type=image/png' -F 'instruction=List visible controls.' > "$ROOT/reports/transition_validation_parsing.json"
run_with_vram transition docker exec "$CONTAINER" curl --fail --silent -X POST http://127.0.0.1:8000/infer/transition \
  -F 'before_image=@/data/transition/before.jpg;type=image/jpeg' \
  -F 'after_image=@/data/transition/after.jpg;type=image/jpeg' \
  -F 'instruction=Describe the UI state transition from before to after.' > "$ROOT/reports/inference_transition.json"
docker exec "$CONTAINER" curl --fail --silent -X POST http://127.0.0.1:8000/infer/transition \
  -F 'before_image=@/data/transition/after.jpg;type=image/jpeg' \
  -F 'after_image=@/data/transition/before.jpg;type=image/jpeg' \
  -F 'instruction=Describe the UI state transition from before to after.' > "$ROOT/reports/inference_transition_reversed.json"

python3 - "$ROOT/reports/inference_transition.json" "$ROOT/reports/inference_transition_reversed.json" "$REPORT" <<'PY'
import json, sys
forward, reverse = [json.load(open(p)) for p in sys.argv[1:3]]
assert forward["mode"] == reverse["mode"] == "transition"
assert forward["inference_debug"]["image_count"] == reverse["inference_debug"]["image_count"] == 2
assert forward["inference_debug"]["model_instance_id"] == reverse["inference_debug"]["model_instance_id"]
assert isinstance(forward["raw_output"], str) and forward["raw_output"].strip()
t = forward["transition"]
assert set(t) == {"before", "after"}
for state in ("before", "after"):
    assert set(t[state]) == {"elements", "focus_path"}
    assert all(isinstance(item, str) for item in t[state]["elements"])
    assert all(isinstance(item, str) for item in t[state]["focus_path"])
dbg = forward["inference_debug"]
assert len(dbg["image_grid_thw"]) == 2
with open(sys.argv[3], "a", encoding="utf-8") as out:
    out.write(f"transition_elapsed_seconds={forward['elapsed_seconds']}\n")
    out.write(f"input_ids_shape={dbg.get('input_ids_shape')}\n")
    out.write(f"pixel_values_shape={dbg.get('pixel_values_shape')}\n")
    out.write(f"image_grid_thw={dbg.get('image_grid_thw')}\n")
    out.write("before_after_order=PASS\n")
    out.write("transition_schema=PASS\n")
    out.write("shared_model_instance=PASS\n")
PY

echo "forbidden_field_statuses=" >> "$REPORT"
for field in prompt prompt_template system_prompt output_mode max_new_tokens do_sample temperature top_p generation_config images image; do
  code=$(docker exec "$CONTAINER" curl --silent --output /dev/null --write-out '%{http_code}' -X POST http://127.0.0.1:8000/infer/transition \
    -F 'before_image=@/data/transition/before.jpg;type=image/jpeg' -F 'after_image=@/data/transition/after.jpg;type=image/jpeg' \
    -F 'instruction=x' -F "$field=forbidden")
  test "$code" = 422
  echo "$field=$code" >> "$REPORT"
done
echo "forbidden_fields=PASS" >> "$REPORT"
echo "max_active_inferences=$(docker logs "$CONTAINER" 2>&1 | rg -o 'max_active_inferences=[0-9]+' | sort -u | tr '\n' ' ')" >> "$REPORT"
