from __future__ import annotations

from asyncio import run
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import app.vlm_distill.docker_service as service


class FakeEngine:
    model = object()
    processor = object()


def test_lifespan_creates_shared_inferencer_and_session_local_registries(monkeypatch):
    load_calls = []
    context_seen = []

    def fake_runtime_load():
        load_calls.append(True)
        return SimpleNamespace(), FakeEngine(), {
            "adapter_loaded": True,
            "hf_device_map": {"model": "cuda:0"},
        }

    def fake_transition(context, before, after, instruction):
        context_seen.append((context, before, after, instruction))
        return {"usable": False, "parse_error": "not used", "transition": None}

    monkeypatch.setattr(service, "_runtime_load", fake_runtime_load)
    monkeypatch.setattr(service, "_infer_transition_sync", fake_transition)

    async def exercise_lifespan():
        async with service.lifespan(service.app):
            context = service.app.state.runtime_context
            shared = service.app.state.transition_inferencer
            registry_a = service.create_state_registry()
            registry_b = service.create_state_registry()

            assert context is not None
            assert shared is not None
            assert registry_a is not registry_b
            assert registry_a.transition_inferencer is shared
            assert registry_b.transition_inferencer is shared

            registry_a.resolve({"elements": ["A"], "focus_path": []})
            registry_b.resolve({"elements": ["B"], "focus_path": []})
            assert list(registry_a.states) == ["S001"]
            assert list(registry_b.states) == ["S001"]

            shared(b"before", b"after")
            assert context_seen == [(context, b"before", b"after", None)]
            assert context.active_inferences == 0
            assert context.max_active_inferences == 1
            assert context.semaphore.acquire(blocking=False) is True
            context.semaphore.release()

        assert service.app.state.transition_inferencer is None
        assert service.app.state.runtime_context is None

    run(exercise_lifespan())
    assert len(load_calls) == 1
    assert not hasattr(service.app.state, "state_registry")


def test_transition_inferencer_accessor_returns_consistent_503_when_unready():
    service.app.state.transition_inferencer = None

    with pytest.raises(HTTPException) as raised:
        service.create_state_registry()

    assert raised.value.status_code == 503
    assert raised.value.detail == "Transition inferencer is not ready"
