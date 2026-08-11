"""Offline, single-worker Qwen3-VL runtime with shared parsing/text endpoints."""

from __future__ import annotations

import asyncio
import io
import os
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI, HTTPException, Request, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from .bbox_grounding_inference import BBoxGroundingInferenceEngine
from .config_schema import load_config
from .data_manifest import VlmSample
from .output_processors import ParsingOutputProcessor, TransitionOutputProcessor
from .parsing_output_parser import COORDINATE_SYSTEM_NORMALIZED_0_1000
from .parsing_generation_stopper import RepeatedTokenBlockStoppingCriteria
from .prompt_composer import TRANSITION_PROMPT_TEMPLATE, compose_prompt, compose_transition_prompt
from .runtime_validation import summarize_model_precision, validate_loaded_precision
from .stage_teacher_precompute import _load_teacher_image
from .state_fingerprint import SAME_STATE_THRESHOLD
from .state_registry import StateRegistry
from .state_tracker import StateObservationError, StateTracker, build_state_observation

DEFAULT_QUERY = "List all visible interactive UI elements on this screen."
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
QUEUE_TIMEOUT_SECONDS = 300
INFERENCE_TIMEOUT_SECONDS = 600
SUPPORTED_OUTPUT_MODES = ("parsing", "text", "transition")
Mode = Literal["parsing", "text"]
FORBIDDEN_FIELDS = {"prompt", "prompt_template", "system_prompt", "output_mode", "max_new_tokens",
                    "do_sample", "temperature", "top_p", "generation_config", "images", "image"}


class InferenceFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruction: str | None = None
    query: str | None = None
    request_id: str | None = None


@dataclass
class RuntimeContext:
    config: Any
    engine: BBoxGroundingInferenceEngine
    model: Any
    processor: Any
    semaphore: threading.BoundedSemaphore
    model_instance_id: str
    processor_instance_id: str
    model_load_count: int
    ready: bool
    summary: dict[str, Any]
    active_inferences: int = 0
    max_active_inferences: int = 0
    active_lock: threading.Lock = field(default_factory=threading.Lock)


class InferenceQueueTimeout(RuntimeError):
    """Raised when the shared inference queue cannot be acquired in time."""


def _hard_runtime_checks(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    runtime = raw.get("runtime") or {}
    expected = {
        "merged_artifact_mode": "4bit_base_bf16_adapter",
        "device_map": "auto",
        "max_new_tokens": 2048,
        "do_sample": False,
        "max_concurrent_inference": 1,
    }
    for key, value in expected.items():
        if runtime.get(key) != value:
            raise RuntimeError(f"hard check failed: runtime.{key} must be {value!r}")
    if (raw.get("pipeline") or {}).get("output_mode") != "parsing":
        raise RuntimeError("hard check failed: pipeline.output_mode must be parsing")
    student = raw.get("student") or {}
    if student.get("merged_artifact_mode") != expected["merged_artifact_mode"]:
        raise RuntimeError("hard check failed: student.merged_artifact_mode mismatch")
    if student.get("device_map") != "auto" or student.get("quantization") != "4bit":
        raise RuntimeError("hard check failed: Student must use device_map=auto and quantization=4bit")
    if any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")):
        raise RuntimeError("hard check failed: Hugging Face offline mode is incomplete")
    if any(shutil.which(name) is None for name in ("cc", "g++", "make")):
        raise RuntimeError("hard check failed: C/C++ compiler toolchain is unavailable")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError("hard check failed: CUDA GPU is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("hard check failed: BF16 is not supported by the available GPU")
    import bitsandbytes  # noqa: F401
    import triton  # noqa: F401
    return runtime


def _gpu_inventory() -> list[dict[str, Any]]:
    import torch
    return [
        {"index": i, "name": torch.cuda.get_device_name(i),
         "total_memory": torch.cuda.get_device_properties(i).total_memory}
        for i in range(torch.cuda.device_count())
    ]


def _device_map(model: Any) -> dict[str, Any]:
    value = getattr(model, "hf_device_map", None)
    if value is None and getattr(model, "base_model", None) is not None:
        value = getattr(model.base_model, "hf_device_map", None)
    return {
        key: (f"cuda:{item}" if isinstance(item, int) else item)
        for key, item in dict(value or {}).items()
    }


def _runtime_load() -> tuple[Any, BBoxGroundingInferenceEngine, dict[str, Any]]:
    config_path = Path(os.environ["VLM_CONFIG_PATH"])
    runtime = _hard_runtime_checks(config_path)
    config = load_config(config_path)
    if config.student.merged_artifact_mode != "4bit_base_bf16_adapter":
        raise RuntimeError("hard check failed: loaded config artifact mode mismatch")
    engine = BBoxGroundingInferenceEngine.from_pipeline_config(config)
    summary = summarize_model_precision(engine.model)
    validate_loaded_precision(config, summary)
    device_map = _device_map(engine.model)
    devices = {str(value) for value in device_map.values()}
    cuda_devices = sorted(value for value in devices if value.startswith("cuda"))
    if not cuda_devices or any(value in {"cpu", "disk"} for value in devices):
        raise RuntimeError(f"hard check failed: model device map is not GPU-resident: {device_map}")
    summary.update({
        "student_model_id": "Qwen3-VL-8B-Instruct",
        "student_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "adapter_sha256": "e43c568e23f3f19a313f6b1cc65e0d9d9c0bcbc17554d856bce1621657f85e99",
        "merged_artifact_mode": runtime["merged_artifact_mode"],
        "output_mode": "route_selected",
        "device_map_setting": runtime["device_map"],
        "hf_device_map": device_map,
        "gpu_inventory": _gpu_inventory(),
        "bf16_compute": True,
        "max_new_tokens": runtime["max_new_tokens"],
        "do_sample": runtime["do_sample"],
        "max_concurrent_inference": runtime["max_concurrent_inference"],
        "adapter_loaded": summary.get("peft_model_mounted", False),
        "projector_restored": True,
    })
    return config, engine, summary


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime_context = None
    app.state.transition_inferencer = None
    app.state.state_observer = None
    app.state.debug_tracker = None
    config, engine, summary = _runtime_load()
    context = RuntimeContext(
        config=config,
        engine=engine,
        model=engine.model,
        processor=engine.processor,
        semaphore=threading.BoundedSemaphore(1),
        model_instance_id=str(id(engine.model)),
        processor_instance_id=str(id(engine.processor)),
        model_load_count=1,
        ready=True,
        summary=summary,
    )
    app.state.runtime_context = context
    app.state.transition_inferencer = create_transition_inferencer(context)
    app.state.state_observer = create_state_observer(context)
    print(f"model_instance_id={context.model_instance_id}", flush=True)
    print(f"processor_instance_id={context.processor_instance_id}", flush=True)
    print("model_load_count=1", flush=True)
    print("supported_output_modes=parsing,text,transition", flush=True)
    yield
    context.ready = False
    app.state.transition_inferencer = None
    app.state.state_observer = None
    app.state.debug_tracker = None
    app.state.runtime_context = None


app = FastAPI(title="VLM Online DBiLD runtime", lifespan=lifespan)


def _context() -> RuntimeContext:
    context = getattr(app.state, "runtime_context", None)
    if context is None or not context.ready:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return context


def _transition_inferencer():
    inferencer = getattr(app.state, "transition_inferencer", None)
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Transition inferencer is not ready")
    return inferencer


def create_state_registry(threshold: float = SAME_STATE_THRESHOLD) -> StateRegistry:
    """Create a session-local registry using the shared runtime inferencer."""
    return StateRegistry(
        threshold=threshold,
        transition_inferencer=_transition_inferencer(),
    )


def _state_observer():
    observer = getattr(app.state, "state_observer", None)
    if observer is None:
        raise HTTPException(status_code=503, detail="State observer is not ready")
    return observer


def create_state_tracker(threshold: float = SAME_STATE_THRESHOLD) -> StateTracker:
    """Create a session-local tracker with a fresh registry."""
    return StateTracker(
        registry=StateRegistry(threshold=threshold),
        observer=_state_observer(),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    context = _context()
    return {
        "status": "ready",
        "model_loaded": True,
        "model_load_count": context.model_load_count,
        "model_instance_id": context.model_instance_id,
        "adapter_loaded": bool(context.summary["adapter_loaded"]),
        "projector_restored": bool(context.summary["projector_restored"]),
        "supported_output_modes": list(SUPPORTED_OUTPUT_MODES),
        "endpoints": {"parsing": "/infer/parsing", "text": "/infer/text",
                       "transition": "/infer/transition", "legacy": "/infer"},
        "merged_artifact_mode": "4bit_base_bf16_adapter",
        "device_map": context.summary["hf_device_map"],
    }


async def _parse_request(request: Request) -> tuple[RuntimeContext, InferenceFields, bytes]:
    context = _context()
    form = await request.form()
    allowed = {"image", "instruction", "query", "request_id"}
    unknown = sorted(set(form.keys()) - allowed)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown or forbidden fields: {unknown}")
    image = form.get("image")
    if not isinstance(image, (UploadFile, StarletteUploadFile)):
        raise HTTPException(status_code=422, detail="image must be a multipart file")
    try:
        fields = InferenceFields.model_validate({
            key: form.get(key) for key in allowed
            if key != "image" and form.get(key) is not None
        })
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    instruction = fields.instruction if fields.instruction is not None else fields.query
    instruction = (instruction or DEFAULT_QUERY).strip()
    if not instruction:
        raise HTTPException(status_code=422, detail="instruction/query must be non-empty")
    content = await image.read()
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image is empty or exceeds 10 MB")
    try:
        with Image.open(io.BytesIO(content)) as checked:
            if checked.format not in {"PNG", "JPEG"}:
                raise HTTPException(status_code=400, detail="image must be PNG or JPEG")
            checked.verify()
        with Image.open(io.BytesIO(content)) as checked:
            if checked.width * checked.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=400, detail="image exceeds maximum decoded pixel count")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid PNG/JPEG image") from exc
    return context, fields, content


def _decode_image(content: bytes, field_name: str) -> Image.Image:
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.format not in {"PNG", "JPEG"}:
                raise ValueError(f"{field_name} must be PNG or JPEG")
            image = ImageOps.exif_transpose(source).convert("RGB")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"{field_name} exceeds maximum decoded pixel count")
            if image.width <= 0 or image.height <= 0:
                raise ValueError(f"{field_name} is empty")
            return image
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}: {exc}") from exc


async def _parse_transition_request(request: Request) -> tuple[RuntimeContext, InferenceFields, bytes, bytes]:
    context = _context()
    form = await request.form()
    allowed = {"before_image", "after_image", "instruction", "query", "request_id"}
    unknown = sorted(set(form.keys()) - allowed)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown or forbidden fields: {unknown}")
    image_parts = [
        (key, value)
        for key, value in form.multi_items()
        if key in {"before_image", "after_image"}
    ]
    if len(image_parts) != 2:
        raise HTTPException(status_code=422, detail="transition requires exactly two images")
    before_parts = [value for key, value in image_parts if key == "before_image"]
    after_parts = [value for key, value in image_parts if key == "after_image"]
    if len(before_parts) != 1:
        raise HTTPException(status_code=422, detail="before_image is required and must be a multipart file")
    if len(after_parts) != 1:
        raise HTTPException(status_code=422, detail="after_image is required and must be a multipart file")
    before, after = before_parts[0], after_parts[0]
    if not isinstance(before, (UploadFile, StarletteUploadFile)):
        raise HTTPException(status_code=422, detail="before_image is required and must be a multipart file")
    if not isinstance(after, (UploadFile, StarletteUploadFile)):
        raise HTTPException(status_code=422, detail="after_image is required and must be a multipart file")
    try:
        fields = InferenceFields.model_validate({key: form.get(key) for key in ("instruction", "query", "request_id")
                                                 if form.get(key) is not None})
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    instruction = (fields.instruction if fields.instruction is not None else fields.query)
    instruction = (instruction or "Describe the UI state transition from the before image to the after image.").strip()
    if not instruction:
        raise HTTPException(status_code=422, detail="instruction/query must be non-empty")
    before_content = await before.read()
    after_content = await after.read()
    if not before_content or len(before_content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="before_image is empty or exceeds 10 MB")
    if not after_content or len(after_content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="after_image is empty or exceeds 10 MB")
    _decode_image(before_content, "before_image")
    _decode_image(after_content, "after_image")
    return context, fields, before_content, after_content


def _infer_sync(context: RuntimeContext, image_bytes: bytes, instruction: str, mode: Mode) -> dict[str, Any]:
    if mode == "parsing":
        return _infer_parsing_sync(context, image_bytes, instruction)

    with tempfile.NamedTemporaryFile(suffix=".image") as handle:
        handle.write(image_bytes)
        handle.flush()
        image = _load_teacher_image(Path(handle.name), context.config.training.image_resize)
    prompt = compose_prompt(instruction, output_mode=mode)
    raw_output = context.engine.generate_raw(image, prompt, 2048)
    debug = dict(context.engine.last_debug)
    debug.update({
        "mode": mode,
        "model_instance_id": context.model_instance_id,
        "processor_instance_id": context.processor_instance_id,
        "generation_kwargs": {"do_sample": False, "max_new_tokens": 2048},
    })
    if mode == "text":
        return {
            "text": raw_output,
            "raw_output": raw_output,
            "usable": bool(raw_output),
            "inference_debug": debug,
        }
    raise AssertionError(f"unsupported non-parsing mode: {mode!r}")


def _infer_parsing_sync(
    context: RuntimeContext,
    image_bytes: bytes,
    instruction: str = DEFAULT_QUERY,
) -> dict[str, Any]:
    """Run the existing single-image parsing inference core."""
    with tempfile.NamedTemporaryFile(suffix=".image") as handle:
        handle.write(image_bytes)
        handle.flush()
        image = _load_teacher_image(Path(handle.name), context.config.training.image_resize)
    prompt = compose_prompt(instruction, output_mode="parsing")
    repetition_stopper = RepeatedTokenBlockStoppingCriteria()
    raw_output = context.engine.generate_raw(
        image,
        prompt,
        2048,
        stopping_criteria=repetition_stopper,
    )
    debug = dict(context.engine.last_debug)
    debug.update({
        "mode": "parsing",
        "model_instance_id": context.model_instance_id,
        "processor_instance_id": context.processor_instance_id,
        "generation_kwargs": {"do_sample": False, "max_new_tokens": 2048},
        "repetition_stop_triggered": repetition_stopper.triggered,
        "repetition_block_length": repetition_stopper.block_length,
    })
    parsed = ParsingOutputProcessor().process(
        sample=VlmSample(id="request", image="", query=instruction),
        raw_output=raw_output,
        backend_result={},
    )
    return {
        "raw_output": raw_output,
        "usable": bool(parsed.get("usable")),
        "parse_error": parsed.get("parse_error"),
        "elements": parsed.get("elements", []),
        "coordinate_system": COORDINATE_SYSTEM_NORMALIZED_0_1000,
        "inference_debug": debug,
    }


def _infer_transition_sync(
    context: RuntimeContext,
    before_bytes: bytes,
    after_bytes: bytes,
    instruction: str | None,
) -> dict[str, Any]:
    before = _decode_image(before_bytes, "before_image")
    after = _decode_image(after_bytes, "after_image")
    prompt = (compose_transition_prompt(instruction)
              if instruction is not None else TRANSITION_PROMPT_TEMPLATE)
    raw_output = context.engine.generate_raw([before, after], prompt, 2048)
    debug = dict(context.engine.last_debug)
    debug.update({"mode": "transition", "image_count": 2,
                  "model_instance_id": context.model_instance_id,
                  "processor_instance_id": context.processor_instance_id,
                  "generation_kwargs": {"do_sample": False, "max_new_tokens": 2048}})
    parsed = TransitionOutputProcessor().process(
        sample=VlmSample(id="request", image="", query=instruction), raw_output=raw_output, backend_result={})
    return {"raw_output": raw_output, "usable": bool(parsed.get("usable")),
            "parse_error": parsed.get("parse_error"), "transition": parsed.get("transition"),
            "inference_debug": debug}


def create_transition_inferencer(
    context: RuntimeContext,
    instruction: str | None = None,
):
    """Bind the loaded runtime context to the existing transition inference core."""
    normalized_instruction = instruction.strip() if instruction is not None else None

    def infer_transition(before_image: bytes, after_image: bytes) -> dict[str, Any]:
        return _run_guarded_inference(
            context,
            _infer_transition_sync,
            before_image,
            after_image,
            normalized_instruction,
        )

    return infer_transition


def create_state_observer(context: RuntimeContext):
    """Bind single-image parsing observation to the shared runtime guard."""
    def observe(image: bytes) -> dict[str, Any]:
        parsing_result = _run_guarded_inference(
            context,
            _infer_parsing_sync,
            image,
            DEFAULT_QUERY,
        )
        if not parsing_result.get("usable"):
            detail = parsing_result.get("parse_error")
            suffix = f": {detail}" if detail else ""
            raise StateObservationError(f"Parsing inference unusable{suffix}")
        return build_state_observation(parsing_result.get("elements", []))

    return observe


def _run_guarded_inference(context: RuntimeContext, inference, *args):
    _acquire_inference_slot(context)
    return _execute_guarded_inference(context, inference, *args)


def _acquire_inference_slot(context: RuntimeContext) -> None:
    if not context.semaphore.acquire(timeout=QUEUE_TIMEOUT_SECONDS):
        raise InferenceQueueTimeout("Inference queue timeout")


def _execute_guarded_inference(context: RuntimeContext, inference, *args):
    try:
        with context.active_lock:
            context.active_inferences += 1
            context.max_active_inferences = max(context.max_active_inferences, context.active_inferences)
            print(f"active_inferences={context.active_inferences} max_active_inferences={context.max_active_inferences}", flush=True)
        return inference(context, *args)
    finally:
        with context.active_lock:
            context.active_inferences -= 1
            print(f"active_inferences={context.active_inferences}", flush=True)
        context.semaphore.release()


async def _acquire_inference_slot_async(context: RuntimeContext) -> None:
    deadline = asyncio.get_running_loop().time() + QUEUE_TIMEOUT_SECONDS
    while True:
        if context.semaphore.acquire(blocking=False):
            return
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise InferenceQueueTimeout("Inference queue timeout")
        await asyncio.sleep(min(0.01, remaining))


def _consume_worker_result(worker: asyncio.Future) -> None:
    if not worker.cancelled():
        worker.exception()


async def _run_guarded_inference_async(context: RuntimeContext, inference, *args):
    await _acquire_inference_slot_async(context)
    worker = asyncio.ensure_future(
        asyncio.to_thread(_execute_guarded_inference, context, inference, *args)
    )
    try:
        return await asyncio.wait_for(
            asyncio.shield(worker),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except (asyncio.CancelledError, asyncio.TimeoutError):
        worker.add_done_callback(_consume_worker_result)
        raise


async def _infer_endpoint(request: Request, mode: Mode) -> dict[str, Any]:
    context, fields, content = await _parse_request(request)
    instruction = fields.instruction if fields.instruction is not None else fields.query
    instruction = (instruction or DEFAULT_QUERY).strip()
    started = time.perf_counter()
    print(f"endpoint=/infer/{mode} model_instance_id={context.model_instance_id}", flush=True)
    try:
        result = await _run_guarded_inference_async(context, _infer_sync, content, instruction, mode)
    except InferenceQueueTimeout as exc:
        raise HTTPException(status_code=503, detail="Inference queue timeout") from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Inference timeout") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    result.update({
        "id": fields.request_id or "request-id",
        "query": instruction,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    })
    return result


async def _infer_transition_endpoint(request: Request) -> dict[str, Any]:
    context, fields, before, after = await _parse_transition_request(request)
    instruction = (fields.instruction if fields.instruction is not None else fields.query)
    instruction = (instruction or "Describe the UI state transition from the before image to the after image.").strip()
    started = time.perf_counter()
    try:
        result = await _run_guarded_inference_async(
            context,
            _infer_transition_sync,
            before,
            after,
            instruction,
        )
    except InferenceQueueTimeout as exc:
        raise HTTPException(status_code=503, detail="Inference queue timeout") from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Inference timeout") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    result.update({"id": fields.request_id or "request-id", "query": instruction,
                   "mode": "transition", "elapsed_seconds": round(time.perf_counter() - started, 3)})
    return result


@app.post("/infer/parsing")
async def infer_parsing(request: Request) -> dict[str, Any]:
    return await _infer_endpoint(request, "parsing")


@app.post("/infer/text")
async def infer_text(request: Request) -> dict[str, Any]:
    return await _infer_endpoint(request, "text")


@app.post("/infer/transition")
async def infer_transition(request: Request) -> dict[str, Any]:
    return await _infer_transition_endpoint(request)


@app.post("/infer")
async def infer_legacy(request: Request) -> dict[str, Any]:
    return await _infer_endpoint(request, "parsing")


def _debug_tracker_response(tracker: StateTracker, resolution) -> dict[str, Any]:
    observation = tracker.current_observation
    if observation is None:
        raise RuntimeError("Debug tracker has no current observation")
    return {
        "state_id": resolution.state_id,
        "is_new": resolution.is_new,
        "score": resolution.score,
        "observation": observation,
        "registry_size": len(tracker.registry),
    }


@app.post("/debug/tracker/start")
async def debug_tracker_start(request: Request) -> dict[str, Any]:
    if getattr(app.state, "debug_tracker", None) is not None:
        raise HTTPException(status_code=409, detail="Debug tracker already exists; reset it first")
    _, _, image_bytes = await _parse_request(request)
    tracker = create_state_tracker()
    app.state.debug_tracker = tracker
    resolution = await asyncio.to_thread(tracker.start, image_bytes)
    return _debug_tracker_response(tracker, resolution)


@app.post("/debug/tracker/step")
async def debug_tracker_step(request: Request) -> dict[str, Any]:
    tracker = getattr(app.state, "debug_tracker", None)
    if tracker is None:
        raise HTTPException(status_code=409, detail="Debug tracker does not exist; call start first")
    _, _, image_bytes = await _parse_request(request)
    resolution = await asyncio.to_thread(tracker.step, image_bytes)
    return _debug_tracker_response(tracker, resolution)


@app.post("/debug/tracker/reset")
async def debug_tracker_reset() -> dict[str, bool]:
    app.state.debug_tracker = None
    return {"ok": True}
    
    
@app.post("/debug/state-registry/resolve")
async def debug_state_registry_resolve(request: Request) -> dict[str, Any]:
    _, _, before_bytes, after_bytes = await _parse_transition_request(request)

    registry = create_state_registry()

    result = await asyncio.to_thread(
        registry.resolve_images,
        before_bytes,
        after_bytes,
    )

    return {
        "before": {
            "state_id": result.before.state_id,
            "is_new": result.before.is_new,
            "score": result.before.score,
        },
        "after": {
            "state_id": result.after.state_id,
            "is_new": result.after.is_new,
            "score": result.after.score,
        },
        "registry_size": len(registry),
    }
