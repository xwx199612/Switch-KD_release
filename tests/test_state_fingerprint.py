from __future__ import annotations

import pytest

from app.vlm_distill.state_fingerprint import (
    SAME_STATE_THRESHOLD,
    StateFingerprint,
    build_state_fingerprint,
    focus_context_similarity,
    jaccard_similarity,
    match_state_fingerprints,
    match_transition_states,
    normalize_element,
)


def observation(elements, focus_path):
    return {"elements": elements, "focus_path": focus_path}


def test_normalization_handles_case_and_outer_whitespace():
    assert normalize_element("  Search by Genre ") == "search by genre"


def test_normalization_collapses_repeated_whitespace():
    assert normalize_element("Search\t  by\n genre") == "search by genre"


def test_normalization_applies_unicode_nfkc():
    assert normalize_element(" ＡＢＣ　１２３ ") == "abc 123"


def test_normalization_does_not_apply_synonyms():
    assert normalize_element("Wi-Fi") != normalize_element("Wireless Network")


def test_duplicate_elements_collapse_in_fingerprint():
    fingerprint = build_state_fingerprint(
        observation(["Comedy", " comedy ", "Horror"], [])
    )

    assert fingerprint.elements == frozenset({"comedy", "horror"})


def test_focus_path_splits_context_and_focused_element():
    fingerprint = build_state_fingerprint(
        observation(["Settings", "Network", "Wi-Fi"], ["Settings", "Network", "Wi-Fi"])
    )

    assert fingerprint.focus_context == ("settings", "network")
    assert fingerprint.focused_element == "wi-fi"


def test_one_element_focus_path_has_empty_context():
    fingerprint = build_state_fingerprint(observation(["Home"], ["Home"]))

    assert fingerprint.focus_context == ()
    assert fingerprint.focused_element == "home"


def test_empty_focus_path_has_no_focus():
    fingerprint = build_state_fingerprint(observation(["Home"], []))

    assert fingerprint.focus_context == ()
    assert fingerprint.focused_element is None


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ({"a", "b"}, {"a", "b"}, 1.0),
        ({"a"}, {"b"}, 0.0),
        ({"a", "b"}, {"b", "c"}, 1 / 3),
        (set(), set(), 1.0),
        (set(), {"a"}, 0.0),
    ],
)
def test_jaccard_similarity(left, right, expected):
    assert jaccard_similarity(left, right) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (("settings", "network"), ("settings", "network"), 1.0),
        (("settings", "network"), ("network",), 0.5),
        (("settings", "network"), ("settings", "display"), 0.0),
        ((), (), 1.0),
        ((), ("settings",), 0.0),
    ],
)
def test_focus_context_similarity(left, right, expected):
    assert focus_context_similarity(left, right) == pytest.approx(expected)


def test_identical_screen_and_focus_match():
    fingerprint = build_state_fingerprint(
        observation(["Home", "Settings"], ["Settings"])
    )

    result = match_state_fingerprints(fingerprint, fingerprint)

    assert result.same_state is True
    assert result.state_score == pytest.approx(1.0)
    assert result.element_score == pytest.approx(1.0)
    assert result.context_score == pytest.approx(1.0)
    assert result.focus_changed is False
    assert result.focus_before == "settings"
    assert result.focus_after == "settings"


def test_focus_child_change_does_not_reduce_state_score():
    before = observation(
        ["Search by genre", "Comedy", "Horror"],
        ["Search by genre", "Comedy"],
    )
    after = observation(
        ["Search by genre", "Comedy", "Horror"],
        ["Search by genre", "Horror"],
    )

    result = match_transition_states({"before": before, "after": after})

    assert result.element_score == pytest.approx(1.0)
    assert result.context_score == pytest.approx(1.0)
    assert result.state_score == pytest.approx(1.0)
    assert result.same_state is True
    assert result.focus_changed is True
    assert result.focus_before == "comedy"
    assert result.focus_after == "horror"


def test_representative_comedy_to_horror_transition_matches():
    elements = [
        "Search movies, shows, cast and more", "Search by genre", "Action",
        "Adventure", "Animated", "Comedy", "Crime", "Documentary", "Drama",
        "Family", "Fantasy", "Game Shows", "Historical", "Horror", "Musicals",
        "Mystery", "Reality TV", "Romance", "Sci-fi", "Thrillers",
    ]
    result = match_transition_states({
        "before": observation(elements, ["Search by genre", "Comedy"]),
        "after": observation(elements, ["Search by genre", "Horror"]),
    })

    assert result == result.__class__(
        same_state=True,
        state_score=1.0,
        element_score=1.0,
        context_score=1.0,
        focus_changed=True,
        focus_before="comedy",
        focus_after="horror",
    )


def test_substantially_different_elements_fall_below_default_threshold():
    before = build_state_fingerprint(observation(["a", "b", "c", "d"], []))
    after = build_state_fingerprint(observation(["x", "y", "z"], []))

    result = match_state_fingerprints(before, after)

    assert SAME_STATE_THRESHOLD == 0.80
    assert result.state_score < SAME_STATE_THRESHOLD
    assert result.same_state is False


def test_custom_threshold_can_be_supplied():
    before = build_state_fingerprint(observation(["a", "b"], []))
    after = build_state_fingerprint(observation(["a", "b", "c"], []))

    result = match_state_fingerprints(before, after, threshold=0.5)

    assert result.state_score == pytest.approx(0.85 * (2 / 3) + 0.15)
    assert result.same_state is True


def test_fingerprint_types_are_frozen_dataclasses():
    fingerprint = StateFingerprint(frozenset({"a"}), (), None)

    with pytest.raises(AttributeError):
        fingerprint.focused_element = "b"
