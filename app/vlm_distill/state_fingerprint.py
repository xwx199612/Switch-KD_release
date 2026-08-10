"""Deterministic pairwise matching for transition state observations.

This module deliberately operates after transition parsing.  It does not
participate in prompt construction, VLM inference, or transition response
serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any


SAME_STATE_THRESHOLD = 0.80


@dataclass(frozen=True)
class StateFingerprint:
    """Normalized, pairwise-comparable representation of one observation."""

    elements: frozenset[str]
    focus_context: tuple[str, ...]
    focused_element: str | None


@dataclass(frozen=True)
class StateMatchResult:
    """Scores and focus movement for a pair of state fingerprints."""

    same_state: bool
    state_score: float
    element_score: float
    context_score: float
    focus_changed: bool
    focus_before: str | None
    focus_after: str | None


def normalize_element(text: str) -> str:
    """Apply conservative, language-agnostic normalization to one label."""
    if not isinstance(text, str):
        raise TypeError("state element must be a string")
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    return re.sub(r"\s+", " ", normalized)


def build_state_fingerprint(observation: dict[str, Any]) -> StateFingerprint:
    """Build a fingerprint from one already-parsed transition observation."""
    elements = observation["elements"]
    focus_path = observation["focus_path"]
    if not isinstance(elements, list) or not isinstance(focus_path, list):
        raise TypeError("observation elements and focus_path must be lists")

    normalized_path = tuple(normalize_element(item) for item in focus_path)
    return StateFingerprint(
        elements=frozenset(normalize_element(item) for item in elements),
        focus_context=normalized_path[:-1],
        focused_element=normalized_path[-1] if normalized_path else None,
    )


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Return Jaccard similarity, including safe empty-set behavior."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def focus_context_similarity(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """Compare focus contexts by their common leaf-side (suffix) path."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    common_suffix_length = 0
    for left, right in zip(reversed(a), reversed(b)):
        if left != right:
            break
        common_suffix_length += 1
    return common_suffix_length / max(len(a), len(b))


def match_state_fingerprints(
    a: StateFingerprint,
    b: StateFingerprint,
    threshold: float = SAME_STATE_THRESHOLD,
) -> StateMatchResult:
    """Match two fingerprints without penalizing a changed focused child."""
    element_score = jaccard_similarity(a.elements, b.elements)
    context_score = focus_context_similarity(a.focus_context, b.focus_context)
    state_score = 0.85 * element_score + 0.15 * context_score
    focus_changed = a.focused_element != b.focused_element
    return StateMatchResult(
        same_state=state_score >= threshold,
        state_score=state_score,
        element_score=element_score,
        context_score=context_score,
        focus_changed=focus_changed,
        focus_before=a.focused_element,
        focus_after=b.focused_element,
    )


def match_transition_states(
    transition: dict[str, Any],
    threshold: float = SAME_STATE_THRESHOLD,
) -> StateMatchResult:
    """Match the parsed ``before`` and ``after`` observations of a transition."""
    before = build_state_fingerprint(transition["before"])
    after = build_state_fingerprint(transition["after"])
    return match_state_fingerprints(before, after, threshold=threshold)


__all__ = [
    "SAME_STATE_THRESHOLD",
    "StateFingerprint",
    "StateMatchResult",
    "build_state_fingerprint",
    "focus_context_similarity",
    "jaccard_similarity",
    "match_state_fingerprints",
    "match_transition_states",
    "normalize_element",
]
