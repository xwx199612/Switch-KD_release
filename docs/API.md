# API

## Endpoints

- `GET /health`: process liveness only.
- `GET /ready`: readiness after one shared Student model, processor, adapter and projector are loaded.
- `POST /infer/parsing`: fixed parsing mode.
- `POST /infer/text`: fixed text mode.
- `POST /infer/transition`: fixed two-image UI transition mode.
- `POST /infer`: deprecated compatibility alias of `POST /infer/parsing`; it cannot dynamically select a mode.

All inference endpoints use multipart form-data:

- `image`: required PNG/JPEG file, maximum 10 MiB and 20,000,000 decoded pixels.
- `instruction`: preferred user instruction.
- `query`: legacy alias. If both are present, `instruction` wins.
- `request_id`: optional identifier.

Unknown fields are rejected with HTTP 422. This includes `prompt`, `prompt_template`, `system_prompt`, `output_mode`, `max_new_tokens`, `do_sample`, `temperature`, `top_p`, `generation_config`, `images`, and `image` on transition requests. The caller cannot select the mode or override the complete prompt.

## Parsing request

```bash
curl -X POST http://127.0.0.1:8000/infer/parsing \
  -F 'image=@samples/sample_ui_0001.png;type=image/png' \
  -F 'instruction=List all visible interactive UI elements on this screen.'
```

The response contains `raw_output`, `usable`, `parse_error`, `coordinate_system`, `elements`, `inference_debug`, `id`, `query`, and `elapsed_seconds`. The structured contract is:

```json
{
  "coordinate_system": "normalized_0_1000",
  "elements": [
    {
      "text": "Settings",
      "bbox_norm": [100, 200, 300, 280],
      "focused": true
    }
  ]
}
```

`bbox_norm` values are in 0–1000, `focused` is boolean, and `elements` is a list. Model output determines the actual elements.

## Text request

```bash
curl -X POST http://127.0.0.1:8000/infer/text \
  -F 'image=@samples/sample_ui_0001.png;type=image/png' \
  -F 'instruction=Describe the current screen.'
```

The response contains non-empty `text` and `raw_output`, plus `usable`, `id`, `query`, `elapsed_seconds`, and `inference_debug`. Text mode does not apply the parsing JSON processor and does not require `coordinate_system`, `elements`, `bbox_norm`, or `focused`.

Both modes use the same application-startup model, processor, adapter/projector and semaphore. Generation is deterministic with `do_sample=false` and `max_new_tokens=2048`. The endpoint selects the prompt mode internally via the existing `compose_prompt(..., output_mode=...)` contract.

## Transition request

`POST /infer/transition` requires two PNG/JPEG multipart files: `before_image` and `after_image`. Each is limited to 10 MiB and 20,000,000 decoded pixels; each is decoded, EXIF-transposed, and converted to RGB. `before_image` is always Image 1 and `after_image` is always Image 2. `instruction` takes precedence over the legacy `query` alias; when neither is supplied, the fixed default is `Describe the UI state transition from the before image to the after image.`

```bash
curl -X POST http://127.0.0.1:8000/infer/transition \
  -F 'before_image=@samples/transition/before.jpg;type=image/jpeg' \
  -F 'after_image=@samples/transition/after.jpg;type=image/jpeg' \
  -F 'instruction=Describe the UI state transition from before to after.'
```

The request creates one Qwen3-VL user message containing Image 1, Image 2, then the transition prompt, followed by one processor call and one `model.generate` call. A parse failure returns `usable=false`, a non-null `parse_error`, and the original `raw_output`; it never fabricates changes. Transition shares the single model, processor, generation settings, GPU device map, and inference semaphore with parsing and text. It does not trigger a second deployment, and can use more VRAM than a single-image request.

## Readiness example

```json
{
  "status": "ready",
  "model_loaded": true,
  "model_load_count": 1,
  "model_instance_id": "opaque-runtime-id",
  "adapter_loaded": true,
  "projector_restored": true,
  "supported_output_modes": ["parsing", "text"],
  "endpoints": {
    "text": "/infer/text",
    "parsing": "/infer/parsing"
  },
  "merged_artifact_mode": "4bit_base_bf16_adapter",
  "device_map": {"model.language_model.layers": "cuda:3"}
}
```

The service uses one Uvicorn worker, one shared semaphore, a 300-second queue timeout and a 600-second inference timeout. Models are loaded once during FastAPI lifespan startup; requests never call `from_pretrained` or adapter loading.
