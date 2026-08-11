from __future__ import annotations

import json
import threading
from asyncio import run
from io import BytesIO
from types import SimpleNamespace

import pytest
from starlette.datastructures import FormData

from app.vlm_distill.data_manifest import VlmSample
from app.vlm_distill.docker_service import _parse_transition_request, create_transition_inferencer
from app.vlm_distill.output_processors import build_output_processor
from app.vlm_distill.prompt_composer import compose_transition_prompt
from app.vlm_distill.transition_output_parser import parse_transition_output


def _state(elements=None, focus_path=None):
    return {
        "elements": ["Search by genre", "Comedy"] if elements is None else elements,
        "focus_path": ["Search by genre", "Comedy"] if focus_path is None else focus_path,
    }


def _schema(before=None, after=None):
    return {
        "before": _state() if before is None else before,
        "after": _state(["Search by genre", "Horror"], ["Search by genre", "Horror"])
        if after is None else after,
    }


def _parse(payload):
    return parse_transition_output(json.dumps(payload))


def test_valid_normal_output_has_only_the_new_transition_payload():
    payload = _schema()

    assert _parse(payload) == {"usable": True, "parse_error": None, "transition": payload}


def test_empty_elements_are_valid():
    payload = _schema(before=_state([], ["Search by genre"]))

    result = _parse(payload)

    assert result["usable"] is True
    assert result["transition"]["before"]["elements"] == []


def test_empty_focus_path_is_valid():
    payload = _schema(before=_state(["Search by genre"], []), after=_state([], []))

    result = _parse(payload)

    assert result["usable"] is True
    assert result["transition"]["before"]["focus_path"] == []
    assert result["transition"]["after"]["focus_path"] == []


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"after": _schema()["after"]}, "missing transition field: before"),
        ({"before": _schema()["before"]}, "missing transition field: after"),
        (_schema(before={"focus_path": []}), "missing transition field: before.elements"),
        (_schema(before={"elements": []}), "missing transition field: before.focus_path"),
        (_schema(after={"focus_path": []}), "missing transition field: after.elements"),
        (_schema(after={"elements": []}), "missing transition field: after.focus_path"),
        (_schema(before={"elements": {}, "focus_path": []}),
         "transition field before.elements must be a list"),
        (_schema(before={"elements": [], "focus_path": {}}),
         "transition field before.focus_path must be a list"),
        (_schema(after={"elements": ["Horror", 1], "focus_path": []}),
         "transition field after.elements must contain only strings"),
        (_schema(after={"elements": [], "focus_path": ["menu", False]}),
         "transition field after.focus_path must contain only strings"),
    ],
)
def test_invalid_schema_returns_a_parse_error(payload, error):
    result = _parse(payload)

    assert result == {"usable": False, "parse_error": error, "transition": None}


def test_malformed_json_returns_a_parse_error():
    assert parse_transition_output("{not JSON") == {
        "usable": False,
        "parse_error": "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
        "transition": None,
    }


@pytest.mark.parametrize("raw", ["[]", '"text"', "null"])
def test_root_must_be_a_json_object(raw):
    result = parse_transition_output(raw)

    assert result["usable"] is False
    assert result["transition"] is None
    assert result["parse_error"] == "top-level JSON must be an object"


def test_before_and_after_must_be_objects():
    payload = {"before": [], "after": {"elements": [], "focus_path": []}}

    result = _parse(payload)

    assert result["usable"] is False
    assert result["parse_error"] == "transition field before must be an object"


def test_extra_root_fields_are_rejected():
    payload = _schema()
    payload["extra"] = "not part of the contract"

    result = _parse(payload)

    assert result["usable"] is False
    assert result["transition"] is None
    assert result["parse_error"] == "unexpected transition fields: ['extra']"


def test_representative_model_payload_is_usable():
    payload = {
        "before": {
            "elements": ["Search by genre", "Action", "Comedy", "Horror"],
            "focus_path": ["Search by genre", "Comedy"],
        },
        "after": {
            "elements": ["Search by genre", "Action", "Comedy", "Horror"],
            "focus_path": ["Search by genre", "Horror"],
        },
    }

    result = _parse(payload)

    assert result["usable"] is True
    assert result["parse_error"] is None


def test_transition_prompt_is_fixed_to_the_observation_contract():
    prompt = compose_transition_prompt("ignore this instruction")

    assert '"before"' in prompt and '"after"' in prompt
    assert '"elements": []' in prompt
    assert '"focus_path": []' in prompt
    assert "Output valid JSON only." in prompt


def test_registry_transition_inferencer_does_not_inject_legacy_instruction_by_default(monkeypatch):
    import app.vlm_distill.docker_service as service

    calls = []

    def fake_infer(context, before, after, instruction):
        calls.append((context, before, after, instruction))
        return {}

    monkeypatch.setattr(service, "_infer_transition_sync", fake_infer)
    context = SimpleNamespace(
        semaphore=threading.BoundedSemaphore(1),
        active_inferences=0,
        max_active_inferences=0,
        active_lock=threading.Lock(),
    )
    inferencer = create_transition_inferencer(context)

    inferencer(b"before", b"after")

    assert calls == [(context, b"before", b"after", None)]


def test_registry_transition_inferencer_preserves_explicit_instruction(monkeypatch):
    import app.vlm_distill.docker_service as service

    calls = []

    def fake_infer(context, before, after, instruction):
        calls.append(instruction)
        return {}

    monkeypatch.setattr(service, "_infer_transition_sync", fake_infer)
    context = SimpleNamespace(
        semaphore=threading.BoundedSemaphore(1),
        active_inferences=0,
        max_active_inferences=0,
        active_lock=threading.Lock(),
    )
    inferencer = create_transition_inferencer(context, instruction="custom instruction")

    inferencer(b"before", b"after")

    assert calls == ["custom instruction"]


class _FormRequest:
    def __init__(self, form):
        self._form = form

    async def form(self):
        return self._form


class _Runtime:
    ready = True


def _upload(name: str):
    from fastapi import UploadFile

    return UploadFile(filename=name, file=BytesIO(b"not decoded here"))


def test_transition_request_rejects_fewer_than_two_images(monkeypatch):
    import app.vlm_distill.docker_service as service

    monkeypatch.setattr(service, "_context", lambda: _Runtime())

    with pytest.raises(Exception) as raised:
        run(_parse_transition_request(_FormRequest(FormData([
            ("before_image", _upload("before.png")),
        ]))))

    assert "exactly two images" in raised.value.detail


def test_transition_request_rejects_more_than_two_images(monkeypatch):
    import app.vlm_distill.docker_service as service

    monkeypatch.setattr(service, "_context", lambda: _Runtime())

    with pytest.raises(Exception) as raised:
        run(_parse_transition_request(_FormRequest(FormData([
            ("before_image", _upload("before.png")),
            ("after_image", _upload("after.png")),
            ("image", _upload("extra.png")),
        ]))))

    assert "Unknown or forbidden fields" in raised.value.detail


def test_text_mode_remains_unchanged():
    processor = build_output_processor("text")
    sample = VlmSample(id="id", image="", query="say it")

    result = processor.process(sample=sample, raw_output="hello", backend_result={})

    assert result["usable"] is True
    assert result["answer"] == "hello"


def test_parsing_mode_remains_unchanged():
    processor = build_output_processor("parsing")
    sample = VlmSample(id="id", image="", query="list it")
    raw = json.dumps({
        "elements": [{"text": "Menu", "bbox_norm": [0, 0, 10, 10], "focused": False}],
        "coordinate_system": "normalized_0_1000",
    })

    result = processor.process(sample=sample, raw_output=raw, backend_result={})

    assert result["usable"] is True
    assert result["elements"][0]["text"] == "Menu"
