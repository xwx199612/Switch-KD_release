from __future__ import annotations

import pytest

from app.vlm_distill.state_fingerprint import build_state_fingerprint
from app.vlm_distill.state_registry import (
    RegisteredState,
    StateRegistry,
    StateResolution,
    TransitionStateResolution,
)


def observation(elements, focus_path=()):
    return {"elements": list(elements), "focus_path": list(focus_path)}


def test_empty_registry_creates_s001():
    registry = StateRegistry()

    result = registry.resolve(observation(["Home"], ["Home"]))

    assert result == StateResolution("S001", True, 1.0)
    assert len(registry) == 1


def test_distinct_observation_creates_s002():
    registry = StateRegistry()
    registry.resolve(observation(["Home"]))

    result = registry.resolve(observation(["Settings"]))

    assert result.state_id == "S002"
    assert result.is_new is True
    assert len(registry) == 2


def test_same_observation_resolves_existing_s001():
    registry = StateRegistry()
    registry.resolve(observation(["Home"], ["Home"]))

    result = registry.resolve(observation(["Home"], ["Home"]))

    assert result == StateResolution("S001", False, 1.0)
    assert len(registry) == 1


def test_same_elements_with_different_focused_child_resolves_same_state():
    registry = StateRegistry()
    before = observation(["Menu", "Comedy", "Horror"], ["Menu", "Comedy"])
    after = observation(["Menu", "Comedy", "Horror"], ["Menu", "Horror"])

    before_result = registry.resolve(before)
    after_result = registry.resolve(after)

    assert before_result.state_id == "S001"
    assert after_result.state_id == "S001"
    assert after_result.is_new is False
    assert len(registry) == 1


def test_comedy_to_horror_transition_creates_only_one_state():
    elements = ["Search by genre", "Comedy", "Horror"]
    registry = StateRegistry()

    result = registry.resolve_transition({
        "before": observation(elements, ["Search by genre", "Comedy"]),
        "after": observation(elements, ["Search by genre", "Horror"]),
    })

    assert result.before.state_id == "S001"
    assert result.before.is_new is True
    assert result.after.state_id == "S001"
    assert result.after.is_new is False
    assert len(registry) == 1


def test_clearly_different_screens_create_different_ids():
    registry = StateRegistry()
    network = observation(["Wi-Fi", "Ethernet", "Proxy"], ["Network", "Wi-Fi"])
    display = observation(["Brightness", "Contrast", "Picture Mode"], ["Display", "Brightness"])

    first = registry.resolve(network)
    second = registry.resolve(display)

    assert first.state_id == "S001"
    assert second.state_id == "S002"
    assert len(registry) == 2


def test_below_threshold_observation_creates_new_state():
    registry = StateRegistry(threshold=0.80)
    registry.resolve(observation(["a", "b", "c", "d"]))

    result = registry.resolve(observation(["x", "y", "z"]))

    assert result.state_id == "S002"
    assert result.is_new is True


def test_above_threshold_observation_resolves_existing_state():
    registry = StateRegistry(threshold=0.70)
    registry.resolve(observation(["a", "b", "c", "d"]))

    result = registry.resolve(observation(["a", "b", "c", "d", "new"]))

    assert result.state_id == "S001"
    assert result.is_new is False
    assert result.score == pytest.approx(0.85 * 0.8 + 0.15)


def test_custom_threshold_changes_resolution_result():
    input_observation = observation(["a", "b", "c"])

    strict = StateRegistry(threshold=0.80)
    strict.resolve(observation(["a", "b"]))
    strict_result = strict.resolve(input_observation)

    permissive = StateRegistry(threshold=0.50)
    permissive.resolve(observation(["a", "b"]))
    permissive_result = permissive.resolve(input_observation)

    assert strict_result.is_new is True
    assert permissive_result == StateResolution("S001", False, pytest.approx(0.85 * (2 / 3) + 0.15))


def test_equal_best_scores_choose_earliest_registered_state():
    registry = StateRegistry(threshold=0.50)
    registry.resolve(observation(["a", "b"]))
    registry.resolve(observation(["a", "c"]))

    result = registry.resolve(observation(["a"]))

    assert result.state_id == "S001"
    assert result.is_new is False


def test_state_ids_increment_deterministically():
    registry = StateRegistry()

    results = [registry.resolve(observation([letter])) for letter in ("a", "b", "c")]

    assert [result.state_id for result in results] == ["S001", "S002", "S003"]


def test_reset_clears_states_and_restarts_numbering():
    registry = StateRegistry()
    registry.resolve(observation(["a"]))
    registry.resolve(observation(["b"]))

    registry.reset()
    result = registry.resolve(observation(["c"]))

    assert len(registry) == 1
    assert result.state_id == "S001"


@pytest.mark.parametrize(
    "input_observation",
    [observation([], []), observation(["Empty focus"], [])],
)
def test_empty_elements_and_empty_focus_path_are_supported(input_observation):
    registry = StateRegistry()

    first = registry.resolve(input_observation)
    second = registry.resolve(input_observation)

    assert first.state_id == second.state_id == "S001"
    assert len(registry) == 1


def test_existing_match_does_not_modify_canonical_fingerprint():
    registry = StateRegistry(threshold=0.65)
    registry.resolve(observation(["a", "b", "c", "d"], ["Root", "a"]))
    original = registry.states["S001"].fingerprint

    result = registry.resolve(observation(["a", "b", "c", "d", "new"], ["Other", "new"]))

    assert result.state_id == "S001"
    assert registry.states["S001"].fingerprint == original


def test_resolve_transition_resolves_before_then_after_with_same_registry():
    registry = StateRegistry()

    result = registry.resolve_transition({
        "before": observation(["Home"], ["Home"]),
        "after": observation(["Settings"], ["Settings"]),
    })

    assert result == TransitionStateResolution(
        before=StateResolution("S001", True, 1.0),
        after=StateResolution("S002", True, 1.0),
    )
    assert len(registry) == 2


def test_states_are_read_only_and_registered_fingerprints_are_canonical():
    registry = StateRegistry()
    registry.resolve(observation(["Home"]))

    assert isinstance(registry.states["S001"], RegisteredState)
    with pytest.raises(TypeError):
        registry.states["S002"] = RegisteredState(
            "S002", build_state_fingerprint(observation(["Other"]))
        )
