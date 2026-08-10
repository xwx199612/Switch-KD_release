#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${VLM_CONTAINER_NAME:-vlm-online-dbild-runtime-120}"
CURL=(docker exec "$CONTAINER" curl --fail --silent --show-error)

"${CURL[@]}" http://127.0.0.1:8000/ready > "$ROOT/reports/health.json"
"${CURL[@]}" -X POST http://127.0.0.1:8000/infer/parsing \
  -F 'image=@/data/sample_ui_0001.png;type=image/png' \
  -F 'instruction=List all visible interactive UI elements.' > "$ROOT/reports/inference_parsing.json"
"${CURL[@]}" -X POST http://127.0.0.1:8000/infer/text \
  -F 'image=@/data/sample_ui_0001.png;type=image/png' \
  -F 'instruction=Describe the current screen.' > "$ROOT/reports/inference_text.json"
"${CURL[@]}" -X POST http://127.0.0.1:8000/infer \
  -F 'image=@/data/sample_ui_0001.png;type=image/png' \
  -F 'instruction=List all visible interactive UI elements.' > "$ROOT/reports/inference_legacy.json"
"${CURL[@]}" -X POST http://127.0.0.1:8000/infer/transition \
  -F 'before_image=@/data/transition/before.jpg;type=image/jpeg' \
  -F 'after_image=@/data/transition/after.jpg;type=image/jpeg' \
  -F 'instruction=Describe the UI state transition from before to after.' > "$ROOT/reports/inference_transition.json"

python3 - "$ROOT/reports/health.json" "$ROOT/reports/inference_parsing.json" "$ROOT/reports/inference_text.json" "$ROOT/reports/inference_legacy.json" "$ROOT/reports/inference_transition.json" <<'PY' | tee "$ROOT/reports/schema_validation.txt"
import json
import sys

ready, parsing, text, legacy, transition = [json.load(open(path)) for path in sys.argv[1:]]
assert ready["status"] == "ready"
assert ready["model_loaded"] is True
assert ready["model_load_count"] == 1
assert ready["supported_output_modes"] == ["parsing", "text", "transition"]
assert ready["endpoints"]["transition"] == "/infer/transition"
assert parsing["usable"] is True
assert parsing["coordinate_system"] == "normalized_0_1000"
assert isinstance(parsing["elements"], list)
assert chr(96) * 3 not in parsing["raw_output"]
assert text["usable"] is True and isinstance(text["text"], str) and text["text"].strip()
assert isinstance(text["raw_output"], str) and text["raw_output"].strip()
assert legacy["coordinate_system"] == parsing["coordinate_system"]
assert transition["mode"] == "transition"
assert transition["usable"] is True
assert isinstance(transition["raw_output"], str) and transition["raw_output"].strip()
assert isinstance(transition["transition"], dict)
assert set(transition["transition"]) == {"before", "after"}
for state in ("before", "after"):
    assert set(transition["transition"][state]) == {"elements", "focus_path"}
    assert all(isinstance(item, str) for item in transition["transition"][state]["elements"])
    assert all(isinstance(item, str) for item in transition["transition"][state]["focus_path"])
ids = {
    parsing["inference_debug"]["model_instance_id"],
    text["inference_debug"]["model_instance_id"],
    legacy["inference_debug"]["model_instance_id"],
    transition["inference_debug"]["model_instance_id"],
}
assert len(ids) == 1
for element in parsing["elements"]:
    assert isinstance(element.get("bbox_norm"), list) and len(element["bbox_norm"]) == 4
    assert all(isinstance(x, (int, float)) and 0 <= x <= 1000 for x in element["bbox_norm"])
    assert isinstance(element.get("focused"), bool)
print("model_load_count=1")
print("shared_model_instance=PASS")
print("parsing_schema=PASS")
print("text_non_empty=PASS")
print("legacy_parsing_alias=PASS")
print("transition_schema=PASS")
print("transition_multi_image=PASS")
PY

: > "$ROOT/reports/forbidden_fields.txt"
for route in parsing text transition; do
  fields=(prompt prompt_template system_prompt output_mode max_new_tokens do_sample temperature top_p generation_config)
  test "$route" = transition && fields+=(images image)
  for field in "${fields[@]}"; do
    args=()
    if test "$route" = transition; then
      args=(-F 'before_image=@/data/transition/before.jpg;type=image/jpeg' -F 'after_image=@/data/transition/after.jpg;type=image/jpeg')
    else
      args=(-F 'image=@/data/sample_ui_0001.png;type=image/png')
    fi
    code=$(docker exec "$CONTAINER" curl --silent --output /dev/null --write-out '%{http_code}' \
      -X POST "http://127.0.0.1:8000/infer/$route" \
      "${args[@]}" -F 'instruction=List visible controls.' -F "$field=forbidden" || true)
    test "$code" = 422
    echo "$route $field=$code" >> "$ROOT/reports/forbidden_fields.txt"
  done
done
cat "$ROOT/reports/forbidden_fields.txt"

for field in before_image after_image; do
  args=(-F 'instruction=Describe transition.')
  test "$field" = before_image && args+=(-F 'after_image=@/data/transition/after.jpg;type=image/jpeg')
  test "$field" = after_image && args+=(-F 'before_image=@/data/transition/before.jpg;type=image/jpeg')
  code=$(docker exec "$CONTAINER" curl --silent --output /dev/null --write-out '%{http_code}' -X POST \
    http://127.0.0.1:8000/infer/transition "${args[@]}" || true)
  test "$code" = 422
  echo "missing_$field=$code" >> "$ROOT/reports/forbidden_fields.txt"
done

for field in before_image after_image; do
  args=(-F 'instruction=Describe transition.')
  test "$field" = before_image && args+=(-F 'before_image=@/data/request.json;type=application/octet-stream' -F 'after_image=@/data/transition/after.jpg;type=image/jpeg')
  test "$field" = after_image && args+=(-F 'before_image=@/data/transition/before.jpg;type=image/jpeg' -F 'after_image=@/data/request.json;type=application/octet-stream')
  code=$(docker exec "$CONTAINER" curl --silent --output /dev/null --write-out '%{http_code}' -X POST \
    http://127.0.0.1:8000/infer/transition "${args[@]}" || true)
  test "$code" -ge 400
  echo "corrupt_$field=$code" >> "$ROOT/reports/forbidden_fields.txt"
done
